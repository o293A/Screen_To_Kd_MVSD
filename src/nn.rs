//! Pure-Rust inference of EasyOCR's `english_g2` recognizer (VGG + 2x BiLSTM + CTC).
//!
//! The network weights are embedded in the executable (`assets/crnn_weights.bin`,
//! produced by `tools/convert_weights.py`; BatchNorm layers are already fused into
//! the convolutions). No Python, no PyTorch, no model download: the program is a
//! single self-contained file.
//!
//! Convolutions are lowered to matrix products (im2col + `matrixmultiply::sgemm`,
//! which dispatches at run time to AVX/FMA/SSE kernels), so reading a crop takes
//! a few milliseconds.

use image::{imageops::FilterType, GrayImage};
use std::sync::OnceLock;

static WEIGHTS_BLOB: &[u8] = include_bytes!("../assets/crnn_weights.bin");

/// EasyOCR english_g2 alphabet (index 0 is the CTC blank, the characters start at 1).
pub const CHARSET: &str =
    "0123456789!\"#$%&'()*+,-./:;<=>?@[\\]^_`{|}~ \u{20AC}ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz";
const NUM_CLASSES: usize = 97;
const IMG_H: u32 = 64;
const HIDDEN: usize = 256;

struct Conv {
    w: Vec<f32>, // (cout, cin*kh*kw)
    b: Vec<f32>,
    cin: usize,
    cout: usize,
    kh: usize,
    kw: usize,
    pad: usize,
}

struct Lstm {
    w_ih: Vec<f32>, // (4*HIDDEN, in)
    w_hh: Vec<f32>, // (4*HIDDEN, HIDDEN)
    b: Vec<f32>,    // (4*HIDDEN)
}

struct BiLstm {
    fwd: Lstm,
    rev: Lstm,
    lin_w: Vec<f32>, // (256, 512)
    lin_b: Vec<f32>,
}

pub struct Crnn {
    convs: Vec<Conv>,
    rnn: Vec<BiLstm>,
    pred_w: Vec<f32>,
    pred_b: Vec<f32>,
    chars: Vec<char>,
}

struct Reader<'a> {
    data: &'a [u8],
    pos: usize,
}

impl Reader<'_> {
    fn take(&mut self, n: usize) -> Vec<f32> {
        let bytes = &self.data[self.pos..self.pos + 4 * n];
        self.pos += 4 * n;
        bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    }
}

/// A recognized line of text.
#[derive(Clone, Debug)]
pub struct Recognition {
    pub text: String,
    /// 0.0 - 1.0 (EasyOCR-style confidence).
    pub confidence: f32,
}

impl Crnn {
    pub fn global() -> &'static Crnn {
        static CELL: OnceLock<Crnn> = OnceLock::new();
        CELL.get_or_init(Crnn::load)
    }

    fn load() -> Crnn {
        let mut r = Reader { data: WEIGHTS_BLOB, pos: 0 };
        let specs: [(usize, usize, usize, usize, usize); 7] = [
            (1, 32, 3, 3, 1),
            (32, 64, 3, 3, 1),
            (64, 128, 3, 3, 1),
            (128, 128, 3, 3, 1),
            (128, 256, 3, 3, 1),
            (256, 256, 3, 3, 1),
            (256, 256, 2, 2, 0),
        ];
        let convs = specs
            .iter()
            .map(|&(cin, cout, kh, kw, pad)| Conv {
                w: r.take(cout * cin * kh * kw),
                b: r.take(cout),
                cin,
                cout,
                kh,
                kw,
                pad,
            })
            .collect();
        let mut rnn = Vec::new();
        for _ in 0..2 {
            let mut lstm = || Lstm {
                w_ih: r.take(4 * HIDDEN * HIDDEN),
                w_hh: r.take(4 * HIDDEN * HIDDEN),
                b: r.take(4 * HIDDEN),
            };
            let fwd = lstm();
            let rev = lstm();
            rnn.push(BiLstm { fwd, rev, lin_w: r.take(HIDDEN * 2 * HIDDEN), lin_b: r.take(HIDDEN) });
        }
        let pred_w = r.take(NUM_CLASSES * HIDDEN);
        let pred_b = r.take(NUM_CLASSES);
        debug_assert_eq!(r.pos, WEIGHTS_BLOB.len());
        Crnn { convs, rnn, pred_w, pred_b, chars: CHARSET.chars().collect() }
    }

    /// Reads one line of text from a grayscale image (already padded by the caller).
    /// `allow`: restricts the output alphabet (None = everything).
    pub fn recognize(&self, gray: &GrayImage, allow: Option<&str>) -> Recognition {
        let (w, h) = gray.dimensions();
        if w < 4 || h < 4 {
            return Recognition { text: String::new(), confidence: 0.0 };
        }
        let new_w = ((IMG_H as f32) * (w as f32) / (h as f32)).ceil().max(16.0) as u32;
        let resized = image::imageops::resize(gray, new_w, IMG_H, FilterType::CatmullRom);
        let input: Vec<f32> = resized.as_raw().iter().map(|&p| (p as f32 / 255.0 - 0.5) / 0.5).collect();
        let logits = self.forward(&input, new_w as usize);
        self.decode(&logits, allow)
    }

    fn forward(&self, input: &[f32], width: usize) -> Vec<f32> {
        // ---- VGG feature extractor ------------------------------------------
        let (mut h, mut w) = (IMG_H as usize, width);
        let mut x = input.to_vec();
        let mut c = 1usize;
        // conv0 + relu + pool 2x2
        x = self.conv(&self.convs[0], &x, h, w);
        c = self.convs[0].cout.max(c);
        (x, h, w) = maxpool(&x, c, h, w, 2, 2);
        // conv1 + relu + pool 2x2
        x = self.conv(&self.convs[1], &x, h, w);
        c = self.convs[1].cout;
        (x, h, w) = maxpool(&x, c, h, w, 2, 2);
        // conv2, conv3 + pool (2,1)
        x = self.conv(&self.convs[2], &x, h, w);
        x = self.conv(&self.convs[3], &x, h, w);
        c = self.convs[3].cout;
        (x, h, w) = maxpool(&x, c, h, w, 2, 1);
        // conv4, conv5 (BN fused) + pool (2,1)
        x = self.conv(&self.convs[4], &x, h, w);
        x = self.conv(&self.convs[5], &x, h, w);
        c = self.convs[5].cout;
        (x, h, w) = maxpool(&x, c, h, w, 2, 1);
        // conv6: 2x2, no padding
        x = self.conv(&self.convs[6], &x, h, w);
        c = self.convs[6].cout;
        h -= 1;
        w -= 1;

        // ---- (c, h, w) -> sequence (w, c), averaged over h ------------------
        let mut seq = vec![0f32; w * c];
        let inv = 1.0 / h as f32;
        for ch in 0..c {
            for y in 0..h {
                let row = &x[(ch * h + y) * w..(ch * h + y) * w + w];
                for (t, v) in row.iter().enumerate() {
                    seq[t * c + ch] += v * inv;
                }
            }
        }

        // ---- BiLSTM x2 -------------------------------------------------------
        let t = w;
        let mut feats = seq;
        for layer in &self.rnn {
            let fwd = run_lstm(&layer.fwd, &feats, t, false);
            let rev = run_lstm(&layer.rev, &feats, t, true);
            let mut cat = vec![0f32; t * 2 * HIDDEN];
            for i in 0..t {
                cat[i * 2 * HIDDEN..i * 2 * HIDDEN + HIDDEN].copy_from_slice(&fwd[i * HIDDEN..(i + 1) * HIDDEN]);
                cat[i * 2 * HIDDEN + HIDDEN..(i + 1) * 2 * HIDDEN]
                    .copy_from_slice(&rev[i * HIDDEN..(i + 1) * HIDDEN]);
            }
            feats = linear(&cat, t, 2 * HIDDEN, &layer.lin_w, &layer.lin_b, HIDDEN);
        }
        linear(&feats, t, HIDDEN, &self.pred_w, &self.pred_b, NUM_CLASSES)
    }

    fn conv(&self, cv: &Conv, input: &[f32], h: usize, w: usize) -> Vec<f32> {
        let (ho, wo) = (h + 2 * cv.pad + 1 - cv.kh, w + 2 * cv.pad + 1 - cv.kw);
        let n = ho * wo;
        let k = cv.cin * cv.kh * cv.kw;
        let mut col = vec![0f32; k * n];
        for ci in 0..cv.cin {
            for ky in 0..cv.kh {
                for kx in 0..cv.kw {
                    let krow = (ci * cv.kh + ky) * cv.kw + kx;
                    let dst = &mut col[krow * n..(krow + 1) * n];
                    for y in 0..ho {
                        let sy = y + ky;
                        if sy < cv.pad || sy - cv.pad >= h {
                            continue;
                        }
                        let src_row = &input[(ci * h + (sy - cv.pad)) * w..(ci * h + (sy - cv.pad)) * w + w];
                        for xo in 0..wo {
                            let sx = xo + kx;
                            if sx >= cv.pad && sx - cv.pad < w {
                                dst[y * wo + xo] = src_row[sx - cv.pad];
                            }
                        }
                    }
                }
            }
        }
        let mut out = vec![0f32; cv.cout * n];
        // SAFETY: slices are sized exactly m*k, k*n and m*n for row-major matrices.
        unsafe {
            matrixmultiply::sgemm(
                cv.cout, k, n, 1.0, cv.w.as_ptr(), k as isize, 1, col.as_ptr(), n as isize, 1, 0.0,
                out.as_mut_ptr(), n as isize, 1,
            );
        }
        for (co, row) in out.chunks_mut(n).enumerate() {
            let b = cv.b[co];
            for v in row {
                *v = (*v + b).max(0.0);
            }
        }
        out
    }

    fn decode(&self, logits: &[f32], allow: Option<&str>) -> Recognition {
        let t = logits.len() / NUM_CLASSES;
        let mut allowed = [true; NUM_CLASSES];
        if let Some(list) = allow {
            for (i, flag) in allowed.iter_mut().enumerate().skip(1) {
                *flag = list.contains(self.chars[i - 1]);
            }
        }
        let mut best_idx = Vec::with_capacity(t);
        let mut best_prob = Vec::with_capacity(t);
        for step in 0..t {
            let row = &logits[step * NUM_CLASSES..(step + 1) * NUM_CLASSES];
            let max = row.iter().cloned().fold(f32::MIN, f32::max);
            let mut sum = 0f32;
            let mut probs = [0f32; NUM_CLASSES];
            for (i, &v) in row.iter().enumerate() {
                let p = (v - max).exp();
                if allowed[i] {
                    probs[i] = p;
                    sum += p;
                }
            }
            let (mut bi, mut bp) = (0usize, 0f32);
            for (i, &p) in probs.iter().enumerate() {
                if p > bp {
                    bp = p;
                    bi = i;
                }
            }
            best_idx.push(bi);
            best_prob.push(if sum > 0.0 { bp / sum } else { 0.0 });
        }
        // CTC greedy collapse
        let mut text = String::new();
        let mut probs = Vec::new();
        let mut prev = usize::MAX;
        for (i, &idx) in best_idx.iter().enumerate() {
            if idx != 0 && idx != prev {
                text.push(self.chars[idx - 1]);
                probs.push(best_prob[i]);
            }
            prev = idx;
        }
        let confidence = if probs.is_empty() {
            0.0
        } else {
            let prod: f32 = probs.iter().product();
            prod.powf(2.0 / (probs.len() as f32).sqrt())
        };
        Recognition { text, confidence }
    }
}

fn maxpool(x: &[f32], c: usize, h: usize, w: usize, ph: usize, pw: usize) -> (Vec<f32>, usize, usize) {
    let (ho, wo) = (h / ph, w / pw);
    let mut out = vec![f32::MIN; c * ho * wo];
    for ch in 0..c {
        for y in 0..ho {
            for dy in 0..ph {
                let src = &x[(ch * h + y * ph + dy) * w..(ch * h + y * ph + dy) * w + w];
                let dst = &mut out[(ch * ho + y) * wo..(ch * ho + y) * wo + wo];
                for (xo, d) in dst.iter_mut().enumerate() {
                    for dx in 0..pw {
                        *d = d.max(src[xo * pw + dx]);
                    }
                }
            }
        }
    }
    (out, ho, wo)
}

/// y = x * W^T + b, with x (t, k) and W (n, k).
fn linear(x: &[f32], t: usize, k: usize, w: &[f32], b: &[f32], n: usize) -> Vec<f32> {
    let mut out = vec![0f32; t * n];
    // SAFETY: row-major buffers of exactly t*k, n*k (read transposed) and t*n elements.
    unsafe {
        matrixmultiply::sgemm(
            t, k, n, 1.0, x.as_ptr(), k as isize, 1, w.as_ptr(), 1, k as isize, 0.0, out.as_mut_ptr(), n as isize, 1,
        );
    }
    for row in out.chunks_mut(n) {
        for (v, bias) in row.iter_mut().zip(b) {
            *v += bias;
        }
    }
    out
}

fn dot(a: &[f32], b: &[f32]) -> f32 {
    let mut acc = [0f32; 8];
    let (ca, cb) = (a.chunks_exact(8), b.chunks_exact(8));
    let (ra, rb) = (ca.remainder(), cb.remainder());
    for (x, y) in ca.zip(cb) {
        for i in 0..8 {
            acc[i] += x[i] * y[i];
        }
    }
    let mut s: f32 = acc.iter().sum();
    for (x, y) in ra.iter().zip(rb) {
        s += x * y;
    }
    s
}

fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// One LSTM direction. Gate order (PyTorch): input, forget, cell, output.
fn run_lstm(l: &Lstm, x: &[f32], t: usize, reverse: bool) -> Vec<f32> {
    let k = x.len() / t;
    let pre = linear(x, t, k, &l.w_ih, &l.b, 4 * HIDDEN);
    let mut out = vec![0f32; t * HIDDEN];
    let mut h = vec![0f32; HIDDEN];
    let mut c = vec![0f32; HIDDEN];
    let mut gates = vec![0f32; 4 * HIDDEN];
    for step in 0..t {
        let ti = if reverse { t - 1 - step } else { step };
        for (g, gate) in gates.iter_mut().enumerate() {
            *gate = pre[ti * 4 * HIDDEN + g] + dot(&l.w_hh[g * HIDDEN..(g + 1) * HIDDEN], &h);
        }
        for j in 0..HIDDEN {
            let i_g = sigmoid(gates[j]);
            let f_g = sigmoid(gates[HIDDEN + j]);
            let g_g = gates[2 * HIDDEN + j].tanh();
            let o_g = sigmoid(gates[3 * HIDDEN + j]);
            c[j] = f_g * c[j] + i_g * g_g;
            h[j] = o_g * c[j].tanh();
        }
        out[ti * HIDDEN..(ti + 1) * HIDDEN].copy_from_slice(&h);
    }
    out
}
