//! Image processing: loading, player-row segmentation, crops, KD splitting, OCR variants.
//!
//! No OCR here. Row backgrounds are flat saturated blocks of colour (green or red); an HSV
//! "row background" mask isolates them (text, icons and separator lines are lighter or far
//! less saturated). A row's colour is ALWAYS that of its background, never that of its text.
//! No pixel position is hard-coded: everything is relative, so the screenshot size does not
//! matter.

use crate::config as cfg;
use crate::model::Color;
use image::{imageops::FilterType, GrayImage, Rgb, RgbImage};
use std::path::Path;

// ---------------------------------------------------------------------------------------------
// Loading
// ---------------------------------------------------------------------------------------------
/// Loads any supported image as 8-bit RGB. Returns None (+ a warning) if unreadable.
pub fn load_image(path: &Path) -> Option<RgbImage> {
    let name = path.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
    let dynimg = match image::open(path) {
        Ok(i) => i,
        Err(e) => {
            crate::warn_log!("Unreadable image ({name}): {e}");
            return None;
        }
    };
    if dynimg.color().has_alpha() {
        let rgba = dynimg.to_rgba8();
        let mean_alpha: f64 =
            rgba.pixels().map(|p| p[3] as f64).sum::<f64>() / (255.0 * (rgba.width() * rgba.height()) as f64);
        let mut out = RgbImage::new(rgba.width(), rgba.height());
        for (x, y, p) in rgba.enumerate_pixels() {
            // Near-zero alpha everywhere = a "transparent" export: ignore the alpha channel.
            let a = if mean_alpha < 0.05 { 1.0 } else { p[3] as f32 / 255.0 };
            out.put_pixel(
                x,
                y,
                Rgb([(p[0] as f32 * a) as u8, (p[1] as f32 * a) as u8, (p[2] as f32 * a) as u8]),
            );
        }
        return Some(out);
    }
    let rgb = dynimg.to_rgb8();
    if rgb.width() == 0 || rgb.height() == 0 {
        crate::warn_log!("Unreadable image ({name}): empty image.");
        return None;
    }
    Some(rgb)
}

/// 128-bit fingerprint of the pixels (duplicate-screenshot detection).
pub fn image_fingerprint(img: &RgbImage) -> (u64, u64) {
    let (mut a, mut b) = (0xcbf29ce484222325u64 ^ img.width() as u64, 0x84222325cbf29ce4u64 ^ img.height() as u64);
    for &v in img.as_raw() {
        a = (a ^ v as u64).wrapping_mul(0x100000001b3);
        b = (b.rotate_left(5) ^ v as u64).wrapping_mul(0x9E3779B97F4A7C15);
    }
    (a, b)
}

// ---------------------------------------------------------------------------------------------
// Colour helpers (bit-compatible with OpenCV's 8-bit conversions)
// ---------------------------------------------------------------------------------------------
#[inline]
pub fn gray_of(p: &Rgb<u8>) -> u8 {
    ((p[0] as u32 * 4899 + p[1] as u32 * 9617 + p[2] as u32 * 1868 + 8192) >> 14) as u8
}

pub fn rgb_to_gray(img: &RgbImage) -> GrayImage {
    let data: Vec<u8> = img.pixels().map(gray_of).collect();
    GrayImage::from_raw(img.width(), img.height(), data).unwrap()
}

/// (H 0-179, S 0-255, V 0-255)
#[inline]
fn hsv(p: &Rgb<u8>) -> (u8, u8, u8) {
    let (r, g, b) = (p[0] as i32, p[1] as i32, p[2] as i32);
    let v = r.max(g).max(b);
    let mn = r.min(g).min(b);
    let diff = v - mn;
    if v == 0 || diff == 0 {
        return (0, 0, v as u8);
    }
    let s = ((255 * diff) as f32 / v as f32).round() as i32;
    let num = if v == r {
        g - b
    } else if v == g {
        2 * diff + (b - r)
    } else {
        4 * diff + (r - g)
    };
    let mut h = (30.0 * num as f32 / diff as f32).round() as i32;
    if h < 0 {
        h += 180;
    }
    if h >= 180 {
        h -= 180;
    }
    (h as u8, s as u8, v as u8)
}

pub struct Masks {
    pub w: usize,
    pub h: usize,
    pub green: Vec<bool>,
    pub red: Vec<bool>,
    pub value: Vec<u8>,
}

pub fn background_masks(img: &RgbImage) -> Masks {
    let (w, h) = (img.width() as usize, img.height() as usize);
    let mut green = vec![false; w * h];
    let mut red = vec![false; w * h];
    let mut value = vec![0u8; w * h];
    for (i, p) in img.pixels().enumerate() {
        let (hue, sat, val) = hsv(p);
        value[i] = val;
        let base = sat >= cfg::BG_MIN_SATURATION && val >= cfg::BG_MIN_VALUE && val <= cfg::BG_MAX_VALUE;
        if !base {
            continue;
        }
        green[i] = hue >= cfg::GREEN_HUE.0 && hue <= cfg::GREEN_HUE.1;
        red[i] = cfg::RED_HUE.iter().any(|&(lo, hi)| hue >= lo && hue <= hi);
    }
    Masks { w, h, green, red, value }
}

fn median_f32(v: &mut [f32]) -> f32 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n == 0 {
        return 0.0;
    }
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

// ---------------------------------------------------------------------------------------------
// Row segmentation
// ---------------------------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct Band {
    pub y0: usize,
    pub y1: usize,
    pub x0: usize,
    pub x1: usize,
    pub color: Color,
}

impl Band {
    pub fn height(&self) -> usize {
        self.y1 - self.y0 + 1
    }
    pub fn width(&self) -> usize {
        self.x1 - self.x0 + 1
    }
}

#[derive(Clone, Debug, Default)]
pub struct Layout {
    pub rows: Vec<Band>,
    pub banner: Option<Band>,
}

/// Consecutive runs of the same non-zero class: (start, end, class).
fn runs(classes: &[u8]) -> Vec<(usize, usize, u8)> {
    let mut out = Vec::new();
    let mut start: Option<usize> = None;
    let mut current = 0u8;
    for idx in 0..=classes.len() {
        let value = if idx < classes.len() { classes[idx] } else { 0 };
        if let Some(s) = start {
            if value != current {
                out.push((s, idx - 1, current));
                start = None;
            }
        }
        if start.is_none() && value != 0 {
            start = Some(idx);
            current = value;
        }
    }
    out
}

fn column_extent(mask: &[bool], w: usize, y0: usize, y1: usize) -> Option<(usize, usize)> {
    let n = (y1 - y0 + 1) as f32;
    let mut first = None;
    let mut last = 0;
    for x in 0..w {
        let c = (y0..=y1).filter(|&y| mask[y * w + x]).count() as f32;
        if c >= 0.5 * n {
            if first.is_none() {
                first = Some(x);
            }
            last = x;
        }
    }
    first.map(|f| (f, last))
}

fn split_on_separators(m: &Masks, both: &[bool], y0: usize, y1: usize) -> Vec<(usize, usize)> {
    let Some((x0, x1)) = column_extent(both, m.w, y0, y1) else {
        return vec![];
    };
    let margin = 3.min((x1 - x0) / 4);
    let (xa, xb) = (x0 + margin, x1 - margin);
    let mut profile: Vec<f32> = Vec::with_capacity(y1 - y0 + 1);
    for y in y0..=y1 {
        let mut row: Vec<f32> = (xa..=xb).map(|x| m.value[y * m.w + x] as f32).collect();
        profile.push(median_f32(&mut row));
    }
    let base = median_f32(&mut profile.clone());
    let threshold = (cfg::SEPARATOR_MIN_RATIO * base).max(base + cfg::SEPARATOR_MIN_DELTA);
    let mut pieces = Vec::new();
    let mut start: Option<usize> = None;
    for offset in 0..=profile.len() {
        let line = if offset < profile.len() { profile[offset] >= threshold } else { true };
        if !line && start.is_none() {
            start = Some(offset);
        } else if line {
            if let Some(s) = start {
                pieces.push((y0 + s, y0 + offset - 1));
                start = None;
            }
        }
    }
    pieces
}

/// Detects the banner and the player rows of a results screenshot.
pub fn detect_layout(img: &RgbImage) -> Layout {
    let (width, height) = (img.width() as usize, img.height() as usize);
    if height < 20 || width < 20 {
        return Layout::default();
    }
    let m = background_masks(img);
    let both: Vec<bool> = m.green.iter().zip(&m.red).map(|(a, b)| *a || *b).collect();
    let g_counts: Vec<usize> = (0..height).map(|y| m.green[y * width..(y + 1) * width].iter().filter(|v| **v).count()).collect();
    let r_counts: Vec<usize> = (0..height).map(|y| m.red[y * width..(y + 1) * width].iter().filter(|v| **v).count()).collect();
    let peak = g_counts.iter().chain(r_counts.iter()).copied().max().unwrap_or(0);
    if peak < cfg::MIN_BAND_WIDTH_PX.max((cfg::MIN_BAND_WIDTH_RATIO * width as f32) as usize) {
        return Layout::default();
    }
    let classes: Vec<u8> = (0..height)
        .map(|y| {
            let strongest = g_counts[y].max(r_counts[y]);
            if strongest as f32 >= cfg::BAND_LINE_MIN_FILL * peak as f32 {
                if g_counts[y] >= r_counts[y] {
                    1
                } else {
                    2
                }
            } else {
                0
            }
        })
        .collect();

    let mut bands: Vec<Band> = Vec::new();
    for (sy0, sy1, _) in runs(&classes) {
        for (y0, y1) in split_on_separators(&m, &both, sy0, sy1) {
            let Some((ex0, ex1)) = column_extent(&both, width, y0, y1) else { continue };
            let (mut gp, mut rp) = (0usize, 0usize);
            for y in y0..=y1 {
                for x in ex0..=ex1 {
                    gp += m.green[y * width + x] as usize;
                    rp += m.red[y * width + x] as usize;
                }
            }
            bands.push(Band { y0, y1, x0: ex0, x1: ex1, color: if gp >= rp { Color::Green } else { Color::Red } });
        }
    }
    if bands.is_empty() {
        return Layout::default();
    }
    let mut heights: Vec<f32> = bands.iter().map(|b| b.height() as f32).collect();
    let median_h = median_f32(&mut heights);
    let min_h = 6.0f32.max(cfg::MIN_ROW_HEIGHT_RATIO * median_h);
    bands.retain(|b| b.height() as f32 >= min_h);
    if bands.is_empty() {
        return Layout::default();
    }
    bands.sort_by_key(|b| b.y0);

    // Chains of nearly-adjacent bands: the table is the longest chain.
    let mut chains: Vec<Vec<Band>> = Vec::new();
    for band in bands {
        if let Some(chain) = chains.last_mut() {
            let prev = *chain.last().unwrap();
            let gap = band.y0 as f32 - prev.y1 as f32 - 1.0;
            let limit = cfg::CHAIN_GAP_MIN_PX.max(cfg::CHAIN_GAP_RATIO * prev.height().min(band.height()) as f32);
            if gap <= limit {
                chain.push(band);
                continue;
            }
        }
        chains.push(vec![band]);
    }
    let table_idx = (0..chains.len()).max_by_key(|&i| (chains[i].len(), i)).unwrap();
    let table = &chains[table_idx];
    let banner = if table_idx > 0 && chains[table_idx - 1].len() == 1 { Some(chains[table_idx - 1][0]) } else { None };

    let mut x0s: Vec<f32> = table.iter().map(|b| b.x0 as f32).collect();
    let mut x1s: Vec<f32> = table.iter().map(|b| b.x1 as f32).collect();
    let (x0, x1) = (median_f32(&mut x0s) as usize, median_f32(&mut x1s) as usize);
    let rows = table.iter().map(|b| Band { x0, x1, ..*b }).collect();
    Layout { rows, banner }
}

// ---------------------------------------------------------------------------------------------
// Foreground mask, clusters, crops
// ---------------------------------------------------------------------------------------------
pub struct Mask {
    pub w: usize,
    pub h: usize,
    pub d: Vec<bool>,
}

impl Mask {
    fn col_any(&self, x: usize) -> bool {
        (0..self.h).any(|y| self.d[y * self.w + x])
    }
    fn col_count_range(&self, xa: usize, xb: usize) -> usize {
        let mut n = 0;
        for y in 0..self.h {
            for x in xa..=xb {
                n += self.d[y * self.w + x] as usize;
            }
        }
        n
    }
    /// Rows (min, max) that carry foreground within columns [xa, xb].
    pub fn row_span(&self, xa: usize, xb: usize) -> Option<(usize, usize)> {
        let rows: Vec<usize> = (0..self.h).filter(|&y| (xa..=xb).any(|x| self.d[y * self.w + x])).collect();
        if rows.is_empty() {
            None
        } else {
            Some((rows[0], *rows.last().unwrap()))
        }
    }
}

/// Mask of "text / icon" pixels: gray-level distance from the median background.
pub fn foreground(region: &RgbImage) -> Mask {
    let gray: Vec<u8> = region.pixels().map(gray_of).collect();
    let mut sorted = gray.clone();
    sorted.sort_unstable();
    let n = sorted.len();
    let bg = if n == 0 {
        0
    } else if n % 2 == 1 {
        sorted[n / 2] as i32
    } else {
        ((sorted[n / 2 - 1] as i32) + (sorted[n / 2] as i32)) / 2
    };
    let d = gray.iter().map(|&g| (g as i32 - bg).abs() >= cfg::FOREGROUND_DELTA).collect();
    Mask { w: region.width() as usize, h: region.height() as usize, d }
}

pub fn column_clusters(fg: &Mask, min_gap: usize, min_pixels: usize) -> Vec<(usize, usize)> {
    let occupied: Vec<usize> = (0..fg.w).filter(|&x| fg.col_any(x)).collect();
    if occupied.is_empty() {
        return vec![];
    }
    let mut clusters = Vec::new();
    let (mut start, mut prev) = (occupied[0], occupied[0]);
    for &x in &occupied[1..] {
        if x - prev - 1 > min_gap {
            clusters.push((start, prev));
            start = x;
        }
        prev = x;
    }
    clusters.push((start, prev));
    clusters.into_iter().filter(|&(a, b)| fg.col_count_range(a, b) >= min_pixels).collect()
}

/// Sub-image with INCLUSIVE-exclusive bounds clipped to the image (None when empty).
fn sub_image(img: &RgbImage, x0: usize, y0: usize, x1_excl: usize, y1_excl: usize) -> Option<RgbImage> {
    let x1 = x1_excl.min(img.width() as usize);
    let y1 = y1_excl.min(img.height() as usize);
    if x0 >= x1 || y0 >= y1 {
        return None;
    }
    Some(image::imageops::crop_imm(img, x0 as u32, y0 as u32, (x1 - x0) as u32, (y1 - y0) as u32).to_image())
}

fn tight_crop(region: &RgbImage, fg: &Mask, xa: usize, xb: usize) -> Option<RgbImage> {
    let (ya, yb) = fg.row_span(xa, xb)?;
    let text_h = yb - ya + 1;
    let margin_y = 2.max((0.35 * text_h as f32) as usize);
    let margin_x = 3.max((0.6 * text_h as f32) as usize);
    sub_image(region, xa.saturating_sub(margin_x), ya.saturating_sub(margin_y), (xb + margin_x + 1).min(fg.w), (yb + margin_y + 1).min(fg.h))
}

/// A cluster that spans (almost) the whole row height is a frame / scroll-bar line, never text
/// (digits and letters are far shorter than a row).
fn without_frame_lines(fg: &Mask, clusters: Vec<(usize, usize)>) -> Vec<(usize, usize)> {
    clusters
        .into_iter()
        .filter(|&(a, b)| match fg.row_span(a, b) {
            Some((y0, y1)) => ((y1 - y0 + 1) as f32) < 0.8 * fg.h as f32,
            None => false,
        })
        .collect()
}

/// Name area (left column, before the device icon).
pub fn crop_name_region(img: &RgbImage, row: &Band) -> Option<RgbImage> {
    let w = row.width() as f32;
    let xa = row.x0 + (cfg::NAME_REGION.0 * w) as usize;
    let xb = row.x0 + (cfg::NAME_REGION.1 * w) as usize;
    let region = sub_image(img, xa, row.y0, xb + 1, row.y1 + 1)?;
    let fg = foreground(&region);
    let min_gap = 6.max((0.05 * w) as usize);
    let clusters = without_frame_lines(&fg, column_clusters(&fg, min_gap, cfg::CLUSTER_MIN_PIXELS));
    let first = clusters.first()?;
    tight_crop(&region, &fg, first.0, first.1)
}

/// KD area = the rightmost cluster of pixels on the row (the device and ping icons form other
/// clusters further left and are excluded without relying on a fixed position).
pub fn crop_kd_region(img: &RgbImage, row: &Band) -> Option<RgbImage> {
    let w = row.width() as f32;
    let start = row.x0 + (cfg::KD_SEARCH_START * w) as usize;
    if let Some(region) = sub_image(img, start, row.y0, row.x1 + 1, row.y1 + 1) {
        let fg = foreground(&region);
        let min_gap = 4.max((cfg::CLUSTER_GAP_RATIO * w) as usize);
        let clusters = without_frame_lines(&fg, column_clusters(&fg, min_gap, cfg::CLUSTER_MIN_PIXELS));
        if let Some(last) = clusters.last() {
            if let Some(c) = tight_crop(&region, &fg, last.0, last.1) {
                return Some(c);
            }
        }
    }
    let fa = row.x0 + (cfg::KD_FALLBACK_REGION.0 * w) as usize;
    let fallback = sub_image(img, fa, row.y0, row.x1 + 1, row.y1 + 1)?;
    let ffg = foreground(&fallback);
    if ffg.d.iter().filter(|v| **v).count() < cfg::CLUSTER_MIN_PIXELS {
        return None;
    }
    tight_crop(&fallback, &ffg, 0, fallback.width() as usize - 1)
}

// ---------------------------------------------------------------------------------------------
// KD structure: "kills / deaths" split around the "/"
// ---------------------------------------------------------------------------------------------
#[derive(Clone, Copy, Debug)]
pub struct Glyph {
    pub x0: usize,
    pub x1: usize,
    pub y0: usize,
    pub y1: usize,
}

impl Glyph {
    pub fn width(&self) -> usize {
        self.x1 - self.x0 + 1
    }
    pub fn height(&self) -> usize {
        self.y1 - self.y0 + 1
    }
}

/// Glyphs of the mask (empty columns = separation), left to right.
pub fn glyphs(fg: &Mask) -> Vec<Glyph> {
    column_clusters(fg, 0, cfg::CLUSTER_MIN_PIXELS)
        .into_iter()
        .filter_map(|(xa, xb)| fg.row_span(xa, xb).map(|(y0, y1)| Glyph { x0: xa, x1: xb, y0, y1 }))
        .collect()
}

fn find_slash(gl: &[Glyph]) -> Option<usize> {
    if gl.len() < 3 {
        return None;
    }
    let text_h = gl.iter().map(|g| g.height()).max().unwrap() as f32;
    let (mut best_idx, mut best_score) = (0usize, -1i64);
    for idx in 1..gl.len() - 1 {
        let left = gl[idx].x0 as i64 - gl[idx - 1].x1 as i64 - 1;
        let right = gl[idx + 1].x0 as i64 - gl[idx].x1 as i64 - 1;
        let score = left.min(right);
        if score > best_score {
            best_idx = idx;
            best_score = score;
        }
    }
    if (best_score as f32) < 1.0f32.max(cfg::KD_SPLIT_MIN_GAP_RATIO * text_h) {
        return None;
    }
    if (gl[best_idx].height() as f32) < cfg::KD_SLASH_MIN_HEIGHT_RATIO * text_h {
        return None;
    }
    Some(best_idx)
}

fn number_crop(crop: &RgbImage, fg: &Mask, gl: &[Glyph], lo: i64, hi: i64) -> Option<RgbImage> {
    let xa = gl[0].x0 as i64;
    let xb = gl[gl.len() - 1].x1 as i64;
    let ya = gl.iter().map(|g| g.y0).min().unwrap();
    let yb = gl.iter().map(|g| g.y1).max().unwrap();
    let text_h = (yb - ya + 1) as f32;
    let margin_y = 2.max((0.35 * text_h) as i64);
    let margin_x = 2.max((0.6 * text_h) as i64);
    // Margin capped at half the free space: the anti-aliasing halo of the neighbouring "/"
    // must never enter the crop (it would be read as a "1" or a "7").
    let left_margin = margin_x.min(1.max((xa - lo).div_euclid(2)));
    let right_margin = margin_x.min(1.max((hi - xb).div_euclid(2)));
    let x_start = lo.max(xa - left_margin).max(0) as usize;
    let x_end = (hi.min(xb + right_margin) + 1).max(0) as usize;
    let y_start = (ya as i64 - margin_y).max(0) as usize;
    let y_end = ((yb as i64 + margin_y + 1) as usize).min(fg.h);
    sub_image(crop, x_start, y_start, x_end, y_end)
}

pub struct KdSplit {
    pub kills: RgbImage,
    pub deaths: RgbImage,
}

/// Splits "kills / deaths" into two number crops (the central "/" excluded), or None when the
/// (number, "/", number) structure is not recognized.
pub fn split_kd_crop(crop: &RgbImage) -> Option<KdSplit> {
    let fg = foreground(crop);
    let gl = glyphs(&fg);
    let slash = find_slash(&gl)?;
    let width = fg.w as i64;
    let sg = gl[slash];
    let kills = number_crop(crop, &fg, &gl[..slash], 0, sg.x0 as i64 - 1)?;
    let deaths = number_crop(crop, &fg, &gl[slash + 1..], sg.x1 as i64 + 1, width - 1)?;
    Some(KdSplit { kills, deaths })
}

/// "- / -": two very short horizontal dashes on either side of a "/" (disconnected player).
/// Detected from GEOMETRY, without OCR.
pub fn is_placeholder_crop(crop: &RgbImage) -> bool {
    let gl = glyphs(&foreground(crop));
    if gl.len() < 3 {
        return false;
    }
    let text_h = gl.iter().map(|g| g.height()).max().unwrap() as f32;
    if text_h < 5.0 {
        return false;
    }
    let is_dash = |g: &Glyph| g.height() as f32 <= cfg::KD_DASH_MAX_HEIGHT_RATIO * text_h && g.width() >= g.height();
    let dashes = gl.iter().filter(|g| is_dash(g)).count();
    let tall = gl.len() - dashes;
    dashes >= 2 && tall == 1
}

// ---------------------------------------------------------------------------------------------
// Recognition variants
// ---------------------------------------------------------------------------------------------
fn percentile(sorted: &[u8], q: f64) -> f32 {
    let pos = q / 100.0 * (sorted.len() - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = (lo + 1).min(sorted.len() - 1);
    let frac = pos - lo as f64;
    (sorted[lo] as f64 * (1.0 - frac) + sorted[hi] as f64 * frac) as f32
}

fn stretch_contrast(gray: &GrayImage) -> GrayImage {
    let mut sorted = gray.as_raw().clone();
    sorted.sort_unstable();
    let lo = percentile(&sorted, 0.5);
    let mut hi = percentile(&sorted, 99.9);
    if hi - lo < 30.0 {
        hi = lo + 30.0;
    }
    let scale = 255.0 / (hi - lo);
    let data = gray.as_raw().iter().map(|&v| ((v as f32 - lo) * scale).clamp(0.0, 255.0) as u8).collect();
    GrayImage::from_raw(gray.width(), gray.height(), data).unwrap()
}

fn blur3(gray: &GrayImage) -> GrayImage {
    let (w, h) = (gray.width() as i32, gray.height() as i32);
    let px = |x: i32, y: i32| -> u32 {
        // reflect-101 border
        let r = |v: i32, n: i32| if n == 1 { 0 } else if v < 0 { -v } else if v >= n { 2 * n - 2 - v } else { v };
        gray.as_raw()[(r(y, h) * w + r(x, w)) as usize] as u32
    };
    let k = [1u32, 2, 1];
    let mut out = vec![0u8; (w * h) as usize];
    for y in 0..h {
        for x in 0..w {
            let mut s = 0u32;
            for (dy, ky) in k.iter().enumerate() {
                for (dx, kx) in k.iter().enumerate() {
                    s += px(x + dx as i32 - 1, y + dy as i32 - 1) * ky * kx;
                }
            }
            out[(y * w + x) as usize] = ((s + 8) >> 4) as u8;
        }
    }
    GrayImage::from_raw(w as u32, h as u32, out).unwrap()
}

fn otsu(gray: &GrayImage) -> GrayImage {
    let mut hist = [0f64; 256];
    for &v in gray.as_raw() {
        hist[v as usize] += 1.0;
    }
    let total: f64 = hist.iter().sum();
    let sum_all: f64 = hist.iter().enumerate().map(|(i, c)| i as f64 * c).sum();
    let (mut w0, mut sum0, mut best, mut thr) = (0f64, 0f64, -1f64, 0usize);
    for t in 0..256 {
        w0 += hist[t];
        if w0 == 0.0 {
            continue;
        }
        let w1 = total - w0;
        if w1 == 0.0 {
            break;
        }
        sum0 += t as f64 * hist[t];
        let (m0, m1) = (sum0 / w0, (sum_all - sum0) / w1);
        let between = w0 * w1 * (m0 - m1) * (m0 - m1);
        if between > best {
            best = between;
            thr = t;
        }
    }
    let data = gray.as_raw().iter().map(|&v| if v as usize > thr { 255 } else { 0 }).collect();
    GrayImage::from_raw(gray.width(), gray.height(), data).unwrap()
}

/// Pads with the (median) background colour. Replicating the border is forbidden: it would
/// stretch edge pixels (a leftover "/", a neighbouring icon) into vertical bars read as "1"/"7".
fn pad(gray: &GrayImage, pad: u32) -> GrayImage {
    let mut sorted = gray.as_raw().clone();
    sorted.sort_unstable();
    let n = sorted.len();
    let bg = if n % 2 == 1 { sorted[n / 2] } else { ((sorted[n / 2 - 1] as u32 + sorted[n / 2] as u32) / 2) as u8 };
    let mut out = GrayImage::from_pixel(gray.width() + 2 * pad, gray.height() + 2 * pad, image::Luma([bg]));
    image::imageops::overlay(&mut out, gray, pad as i64, pad as i64);
    out
}

/// Grayscale variants of a crop, from most to least reliable: stretched contrast (light text on
/// dark), inverted (dark on light), Otsu binary (black on white).
pub fn build_ocr_variants(crop: &RgbImage) -> Vec<(&'static str, GrayImage)> {
    let gray = rgb_to_gray(crop);
    let big = image::imageops::resize(&gray, gray.width() * cfg::OCR_SCALE, gray.height() * cfg::OCR_SCALE, FilterType::CatmullRom);
    let stretched = stretch_contrast(&big);
    let smoothed = blur3(&stretched);
    let ot = otsu(&smoothed);
    let mean: f64 = ot.as_raw().iter().map(|&v| v as f64).sum::<f64>() / ot.as_raw().len() as f64;
    let binary = if mean < 127.0 {
        GrayImage::from_raw(ot.width(), ot.height(), ot.as_raw().iter().map(|&v| 255 - v).collect()).unwrap()
    } else {
        ot
    };
    let inverted = GrayImage::from_raw(stretched.width(), stretched.height(), stretched.as_raw().iter().map(|&v| 255 - v).collect()).unwrap();
    vec![
        ("contrast", pad(&stretched, cfg::OCR_PADDING)),
        ("inverted", pad(&inverted, cfg::OCR_PADDING)),
        ("binary", pad(&binary, cfg::OCR_PADDING)),
    ]
}
