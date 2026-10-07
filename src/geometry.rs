//! Geometry-based correction of look-alike characters in a player name.
//!
//! OCR engines confuse characters whose shapes are nearly identical and differ only in SIZE or
//! in a small detail: "o" vs "0", "z" vs "Z" vs "2", "1" vs "l". The pixels of the name crop
//! settle these questions where the recognizer cannot:
//!
//! * height: a lowercase letter without ascender (o z c s v w x) stops at the x-height, while
//!   digits and capitals reach the full cap height;
//! * top edge: a "Z" has a flat full-width top bar, a "2" a curved top;
//! * width: a "1" carries a flag and is wider than the plain bar of an "l" or "I".
//!
//! A correction is only applied when the alignment between the text and the glyphs of the crop
//! is verified and unique; whenever anything is doubtful the text is returned unchanged.

use crate::imgproc as ip;
use image::RgbImage;

#[derive(Clone, Copy, PartialEq, Eq)]
enum Top {
    Tall,
    Low,
    Any,
}

fn is_tall_char(c: char) -> bool {
    "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZbdfhklij".contains(c)
}
fn is_low_char(c: char) -> bool {
    "acemnoruvwxzgpqy".contains(c)
}
fn is_descender(c: char) -> bool {
    "gjpqy,;".contains(c)
}

fn top_class(c: char) -> Top {
    if "oO0z".contains(c) {
        Top::Any
    } else if is_tall_char(c) {
        Top::Tall
    } else if is_low_char(c) {
        Top::Low
    } else {
        Top::Any
    }
}

#[derive(Clone, Copy)]
struct Cluster {
    x0: usize,
    x1: usize,
    top: Top, // Tall or Low
    descends: bool,
    top_coverage: f32,
}

impl Cluster {
    fn width(&self) -> usize {
        self.x1 - self.x0 + 1
    }
}

struct Reference {
    cap_height: usize,
}

fn analyze(crop: &RgbImage) -> Option<(Vec<Cluster>, Reference)> {
    let fg = ip::foreground(crop);
    let gl = ip::glyphs(&fg);
    if gl.len() < 2 {
        return None;
    }
    let tops: Vec<f32> = gl.iter().map(|g| g.y0 as f32).collect();
    let mut bottoms: Vec<f32> = gl.iter().map(|g| g.y1 as f32).collect();
    let tall_top = tops.iter().cloned().fold(f32::MAX, f32::min) as i64;
    bottoms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = bottoms.len();
    let baseline = (if n % 2 == 1 { bottoms[n / 2] } else { (bottoms[n / 2 - 1] + bottoms[n / 2]) / 2.0 }) as i64;
    let mut lower: Vec<f32> = tops.iter().cloned().filter(|&t| t >= (tall_top + 2) as f32).collect();
    if lower.is_empty() {
        return None; // only one top level: no x-height reference in this name
    }
    lower.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let m = lower.len();
    let low_top = (if m % 2 == 1 { lower[m / 2] } else { (lower[m / 2 - 1] + lower[m / 2]) / 2.0 }) as i64;
    let cap_height = baseline - tall_top + 1;
    if cap_height < 8 || low_top - tall_top < 2 || (low_top - tall_top) as f32 > 0.6 * cap_height as f32 {
        return None; // implausible geometry: do not trust it
    }
    let midpoint = (tall_top + low_top) as f32 / 2.0;
    let descend_margin = 2.max((0.15 * cap_height as f32) as i64);
    let clusters = gl
        .iter()
        .map(|g| {
            let width = g.width();
            let covered = (g.x0..=g.x1).filter(|&x| fg.d[g.y0 * fg.w + x]).count();
            Cluster {
                x0: g.x0,
                x1: g.x1,
                top: if g.y0 as f32 <= midpoint { Top::Tall } else { Top::Low },
                descends: g.y1 as i64 >= baseline + descend_margin,
                top_coverage: covered as f32 / width as f32,
            }
        })
        .collect();
    Some((clusters, Reference { cap_height: cap_height as usize }))
}

const MAX_GROUP: usize = 3;

fn group_ok(chars: &[char], cl: &Cluster) -> bool {
    let tops: Vec<Top> = chars.iter().map(|&c| top_class(c)).collect();
    if cl.top == Top::Low && tops.contains(&Top::Tall) {
        return false;
    }
    if cl.top == Top::Tall && tops.iter().all(|t| *t == Top::Low) {
        return false;
    }
    let has_desc = chars.iter().any(|&c| is_descender(c));
    if cl.descends && !(has_desc || tops.contains(&Top::Any)) {
        return false;
    }
    if has_desc && !cl.descends {
        return false;
    }
    true
}

fn alignments(text: &[char], clusters: &[Cluster], limit: usize) -> Vec<Vec<(usize, usize)>> {
    fn walk(
        text: &[char],
        clusters: &[Cluster],
        ci: usize,
        pos: usize,
        path: &mut Vec<(usize, usize)>,
        found: &mut Vec<Vec<(usize, usize)>>,
        limit: usize,
    ) {
        if found.len() >= limit {
            return;
        }
        let (n, m) = (text.len(), clusters.len());
        if ci == m {
            if pos == n {
                found.push(path.clone());
            }
            return;
        }
        let (rc, rch) = (m - ci, n - pos);
        if rch < rc || rch > rc * MAX_GROUP {
            return;
        }
        for size in 1..=MAX_GROUP {
            let end = pos + size;
            if end > n {
                break;
            }
            if group_ok(&text[pos..end], &clusters[ci]) {
                path.push((pos, end));
                walk(text, clusters, ci + 1, end, path, found, limit);
                path.pop();
            }
        }
    }
    let mut found = Vec::new();
    walk(text, clusters, 0, 0, &mut Vec::new(), &mut found, limit);
    found
}

fn case_pair(lower: char) -> Option<char> {
    match lower {
        'c' => Some('C'),
        'o' => Some('O'),
        's' => Some('S'),
        'v' => Some('V'),
        'w' => Some('W'),
        'x' => Some('X'),
        'z' => Some('Z'),
        _ => None,
    }
}

fn correct_char(ch: char, cl: &Cluster, reference: &Reference, neighbors: &[char]) -> char {
    let lower = ch.to_lowercase().next().unwrap_or(ch);
    if "oO0z2Z".contains(ch) || case_pair(lower).is_some() {
        if cl.top == Top::Low {
            if case_pair(lower).is_some() {
                return lower;
            }
            return if "oO0".contains(ch) { 'o' } else { ch };
        }
        // full-height glyph
        if "oO0zZ2".contains(ch) {
            if lower == 'z' || ch == '2' {
                return if cl.top_coverage >= 0.9 { 'Z' } else { '2' };
            }
            if ch == 'o' {
                return if neighbors.iter().any(|c| c.is_ascii_digit()) { '0' } else { 'O' };
            }
            return ch;
        }
        if let Some(up) = case_pair(lower) {
            return up;
        }
        return ch;
    }
    if ch == '1' || "lI|".contains(ch) {
        if cl.top == Top::Tall {
            let thr = 4.max((0.30f64 * reference.cap_height as f64).round_ties_even() as usize);
            let wide = cl.width() >= thr;
            if wide && "lI|".contains(ch) {
                return '1';
            }
            if !wide && ch == '1' {
                let letters: Vec<&char> = neighbors.iter().filter(|c| c.is_alphabetic()).collect();
                let upper = !letters.is_empty() && letters.iter().all(|c| c.is_uppercase());
                return if upper { 'I' } else { 'l' };
            }
        }
    }
    ch
}

/// `text` with look-alike characters corrected from the geometry of `crop`.
pub fn refine_name(text: &str, crop: &RgbImage) -> String {
    if text.is_empty() || text.contains(' ') {
        return text.to_string();
    }
    if !text.chars().any(|c| "oO0zZ2cCsSvVwWxX1lI|".contains(c)) {
        return text.to_string();
    }
    let Some((clusters, reference)) = analyze(crop) else { return text.to_string() };
    let chars: Vec<char> = text.chars().collect();
    let aligns = alignments(&chars, &clusters, 6);
    if aligns.is_empty() {
        return text.to_string();
    }
    let mut out = chars.clone();
    for ci in 0..clusters.len() {
        let first = aligns[0][ci];
        if aligns.iter().any(|a| a[ci] != first) {
            continue; // this cluster's own split is ambiguous
        }
        let (start, end) = first;
        if end - start != 1 {
            continue; // a merged cluster hides which character is which
        }
        let mut neighbors = Vec::new();
        if start > 0 {
            neighbors.push(chars[start - 1]);
        }
        if end < chars.len() {
            neighbors.push(chars[end]);
        }
        out[start] = correct_char(chars[start], &clusters[ci], &reference, &neighbors);
    }
    let refined: String = out.into_iter().collect();
    if refined != text {
        crate::info_log!("Name geometry: \"{text}\" corrected to \"{refined}\".");
    }
    refined
}
