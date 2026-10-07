//! OCR layer: runs the embedded recognizer on contrast variants of a crop, votes between the
//! readings, and reconstructs "kills/deaths" even from degraded output.

use crate::config as cfg;
use crate::imgproc as ip;
use crate::nn::Crnn;
use image::RgbImage;

#[derive(Clone, Debug)]
pub struct OcrReading {
    pub text: String,
    pub confidence: f32,
}

/// Feeds `visit` with the readings of a crop (most reliable variant first); `visit` returns
/// true to stop early (recognition is only paid for as long as it is needed).
pub fn for_each_reading(crop: &RgbImage, allow: Option<&str>, mut visit: impl FnMut(OcrReading) -> bool) {
    if crop.width() == 0 || crop.height() == 0 {
        return;
    }
    let net = Crnn::global();
    for (_label, gray) in ip::build_ocr_variants(crop) {
        let r = net.recognize(&gray, allow);
        if visit(OcrReading { text: r.text, confidence: r.confidence }) {
            break;
        }
    }
}

// ===========================================================================================
// KD parsing
// ===========================================================================================
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KdParse {
    pub kills: u32,
    pub deaths: u32,
    /// true: two distinct numbers were read; false: reconstructed by inference (glued digits...).
    pub explicit: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Tok {
    Digit(char),
    Sep(char),
    Vertical,
    Blank,
}

fn lookalike_digit(c: char) -> Option<char> {
    Some(match c {
        'O' | 'o' | 'Q' | 'D' => '0',
        'S' | 's' => '5',
        'B' => '8',
        'Z' | 'z' => '2',
        'G' => '6',
        'g' | 'q' => '9',
        'T' => '7',
        'A' => '4',
        _ => return None,
    })
}

fn is_strong_sep(c: char) -> bool {
    "/\\\u{2215}\u{2044}".contains(c)
}

fn tokenize(text: &str) -> Vec<Tok> {
    text.chars()
        .map(|c| {
            // full-width digits
            let c = if ('\u{ff10}'..='\u{ff19}').contains(&c) { (b'0' + (c as u32 - 0xff10) as u8) as char } else { c };
            if c.is_ascii_digit() {
                Tok::Digit(c)
            } else if let Some(d) = lookalike_digit(c) {
                Tok::Digit(d)
            } else if is_strong_sep(c) || ",.;:\u{b7}\u{2022}'`\u{b4}\u{2018}\u{2019}-\u{2013}\u{2014}\u{2212}\u{2010}\u{2011}_~".contains(c) {
                Tok::Sep(c)
            } else if "lIi!|\u{a6}\u{2502}\u{2503}".contains(c) {
                Tok::Vertical
            } else {
                Tok::Blank
            }
        })
        .collect()
}

fn neighbor_is_digit(toks: &[Tok], idx: usize, step: i32) -> bool {
    let mut j = idx as i32 + step;
    while j >= 0 && (j as usize) < toks.len() && toks[j as usize] == Tok::Blank {
        j += step;
    }
    j >= 0 && (j as usize) < toks.len() && matches!(toks[j as usize], Tok::Digit(_))
}

fn in_range(digits: &str) -> bool {
    let max_len = cfg::MAX_STAT_VALUE.to_string().len();
    !digits.is_empty() && digits.len() <= max_len && digits.parse::<u32>().map(|v| v <= cfg::MAX_STAT_VALUE).unwrap_or(false)
}

fn splits_of_single_group(digits: &str) -> Vec<(String, String)> {
    let mut found = Vec::new();
    for cut in 1..digits.len() {
        let (l, r) = (&digits[..cut], &digits[cut..]);
        if !(in_range(l) && in_range(r)) {
            continue;
        }
        if (l.len() > 1 && l.starts_with('0')) || (r.len() > 1 && r.starts_with('0')) {
            continue;
        }
        found.push((l.to_string(), r.to_string()));
    }
    found
}

/// Reconstructs (kills, deaths) from an OCR text: "10/4", "10 4", "10 | 4", "10 l 4", "1O/4"...
pub fn parse_kd_detailed(text: &str) -> Option<KdParse> {
    if text.trim().is_empty() {
        return None;
    }
    let toks = tokenize(text);
    // 1. ambiguous verticals: a separator between two numbers, otherwise a "1"
    let mut resolved: Vec<Tok> = Vec::new();
    for (idx, t) in toks.iter().enumerate() {
        resolved.push(match t {
            Tok::Vertical => {
                if neighbor_is_digit(&toks, idx, -1) && neighbor_is_digit(&toks, idx, 1) {
                    Tok::Sep('|')
                } else {
                    Tok::Digit('1')
                }
            }
            other => *other,
        });
    }
    // 2. digit groups + separators between groups
    let mut groups: Vec<String> = Vec::new();
    let mut between: Vec<Vec<char>> = Vec::new();
    let mut in_group = false;
    let mut pending: Vec<char> = Vec::new();
    for t in &resolved {
        match t {
            Tok::Digit(c) => {
                if in_group {
                    groups.last_mut().unwrap().push(*c);
                } else {
                    if !groups.is_empty() {
                        between.push(std::mem::take(&mut pending));
                    }
                    pending = Vec::new();
                    groups.push(c.to_string());
                    in_group = true;
                }
            }
            other => {
                in_group = false;
                if let Tok::Sep(c) = other {
                    pending.push(*c);
                }
            }
        }
    }
    if groups.is_empty() {
        return None;
    }
    let build = |l: &str, r: &str, explicit: bool| -> Option<KdParse> {
        if in_range(l) && in_range(r) {
            Some(KdParse { kills: l.parse().ok()?, deaths: r.parse().ok()?, explicit })
        } else {
            None
        }
    };
    if groups.len() == 2 {
        return build(&groups[0], &groups[1], true);
    }
    if groups.len() == 1 {
        let splits = splits_of_single_group(&groups[0]);
        if splits.len() == 1 {
            return build(&splits[0].0, &splits[0].1, false);
        }
        return None;
    }
    for idx in 0..groups.len() - 1 {
        if between.get(idx).map(|s| s.iter().any(|c| is_strong_sep(*c))).unwrap_or(false) {
            return build(&groups[idx], &groups[idx + 1], true);
        }
    }
    if groups.len() == 3 && (groups[1] == "1" || groups[1] == "7") {
        return build(&groups[0], &groups[2], false);
    }
    None
}

/// Number read from a crop holding a single value (kills OR deaths).
pub fn parse_number(text: &str) -> Option<u32> {
    if text.trim().is_empty() {
        return None;
    }
    let mut digits = String::new();
    for t in tokenize(text) {
        match t {
            Tok::Digit(c) => digits.push(c),
            Tok::Vertical => digits.push('1'),
            Tok::Sep(_) => return None,
            Tok::Blank => {}
        }
    }
    if digits.is_empty() || !in_range(&digits) || (digits.len() > 1 && digits.starts_with('0')) {
        return None;
    }
    digits.parse().ok()
}

/// "- / -": the game shows no KD at all (disconnected player).
pub fn is_kd_placeholder(text: &str) -> bool {
    !text.trim().is_empty() && !text.chars().any(|c| c.is_ascii_digit()) && text.chars().any(|c| "-\u{2013}\u{2014}\u{2212}_".contains(c))
}

// ===========================================================================================
// Names
// ===========================================================================================
pub fn clean_ocr_name(text: &str) -> String {
    let collapsed = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut cleaned = collapsed.trim().to_string();
    // a leading ghost "~" is a documented anti-aliasing artifact on the crop's edge
    if cleaned.chars().count() > 1 && cleaned.starts_with('~') {
        cleaned = cleaned[1..].trim().to_string();
    }
    cleaned
}

#[derive(Clone, Debug)]
pub struct NameRead {
    pub name: String,
    /// OCR confidence 0-100.
    pub score: f32,
}

/// Best reading of a name: votes across the contrast variants on the cleaned text; two
/// identical readings are enough to stop, otherwise the most confident reading wins.
pub fn best_name_reading(crop: &RgbImage) -> (Option<NameRead>, Vec<String>) {
    let mut tallies: Vec<(String, u32, f32)> = Vec::new();
    let mut raws = Vec::new();
    for_each_reading(crop, None, |r| {
        raws.push(r.text.clone());
        let cleaned = clean_ocr_name(&r.text);
        if cleaned.is_empty() {
            return false;
        }
        let conf = r.confidence.max(0.01);
        if let Some(t) = tallies.iter_mut().find(|t| t.0 == cleaned) {
            t.1 += 1;
            t.2 += conf;
            t.1 >= 2
        } else {
            tallies.push((cleaned, 1, conf));
            false
        }
    });
    if tallies.is_empty() {
        return (None, raws);
    }
    tallies.sort_by(|a, b| (b.1, b.2).partial_cmp(&(a.1, a.2)).unwrap());
    let (name, count, conf) = tallies.remove(0);
    let score = (100.0 * conf / count.max(1) as f32).min(100.0);
    (Some(NameRead { name, score }), raws)
}

// ===========================================================================================
// KD reading
// ===========================================================================================
#[derive(Clone, Debug)]
pub struct KdResult {
    pub kills: u32,
    pub deaths: u32,
    pub explicit: bool,
    pub raw_texts: Vec<String>,
}

struct NumberVote {
    value: Option<u32>,
    raws: Vec<String>,
    contested: bool,
}

/// Reads an isolated number across several contrast variants (2 matching readings = done).
fn vote_number(crop: &RgbImage) -> NumberVote {
    let mut tallies: Vec<(u32, u32, f32)> = Vec::new();
    let mut raws = Vec::new();
    for_each_reading(crop, Some(cfg::NUMBER_ALLOWLIST), |r| {
        raws.push(r.text.clone());
        let Some(v) = parse_number(&r.text) else { return false };
        let conf = r.confidence.max(0.01);
        if let Some(t) = tallies.iter_mut().find(|t| t.0 == v) {
            t.1 += 1;
            t.2 += conf;
            t.1 >= 2
        } else {
            tallies.push((v, 1, conf));
            false
        }
    });
    if tallies.is_empty() {
        return NumberVote { value: None, raws, contested: false };
    }
    tallies.sort_by(|a, b| (b.1, b.2).partial_cmp(&(a.1, a.2)).unwrap());
    let contested = tallies.len() > 1 && tallies[0].1 < 2;
    NumberVote { value: Some(tallies[0].0), raws, contested }
}

/// Structural path: the two numbers around the "/" are isolated and read separately, so the
/// "/" can never be mistaken for a "1" glued to the next number ("4 / 5" read as "4 15").
fn read_kd_by_parts(crop: &RgbImage) -> (Option<KdResult>, Vec<String>) {
    let Some(split) = ip::split_kd_crop(crop) else { return (None, vec![]) };
    let (k, d) = rayon::join(|| vote_number(&split.kills), || vote_number(&split.deaths));
    let mut raws: Vec<String> = k.raws.iter().map(|t| format!("K:{t}")).collect();
    raws.extend(d.raws.iter().map(|t| format!("D:{t}")));
    let (Some(kv), Some(dv)) = (k.value, d.value) else { return (None, raws) };
    if k.contested || d.contested {
        crate::warn_log!("Unreliable KD reading: kills {:?}, deaths {:?} -> {kv}/{dv}.", k.raws, d.raws);
    }
    (Some(KdResult { kills: kv, deaths: dv, explicit: true, raw_texts: raws.clone() }), raws)
}

/// Reads a KD crop: structural path first, then the whole "k / d" string with voting.
pub fn read_kd(crop: &RgbImage) -> (Option<KdResult>, Vec<String>) {
    let (by_parts, part_raws) = read_kd_by_parts(crop);
    if by_parts.is_some() {
        return (by_parts, part_raws);
    }
    let mut tallies: Vec<((u32, u32), u32, u32, f32)> = Vec::new(); // (kd, count, explicit, conf)
    let mut raws = part_raws;
    for_each_reading(crop, Some(cfg::KD_ALLOWLIST), |r| {
        raws.push(r.text.clone());
        let Some(p) = parse_kd_detailed(&r.text) else { return false };
        let conf = r.confidence.max(0.01);
        let key = (p.kills, p.deaths);
        let idx = match tallies.iter().position(|t| t.0 == key) {
            Some(i) => i,
            None => {
                tallies.push((key, 0, 0, 0.0));
                tallies.len() - 1
            }
        };
        tallies[idx].1 += 1;
        tallies[idx].2 += p.explicit as u32;
        tallies[idx].3 += conf;
        tallies[idx].2 >= 2
    });
    if tallies.is_empty() {
        return (None, raws);
    }
    tallies.sort_by(|a, b| (b.2, b.3).partial_cmp(&(a.2, a.3)).unwrap());
    let ((kills, deaths), _, explicit, _) = tallies[0];
    if explicit > 0 && tallies[1..].iter().any(|t| t.2 > 0) {
        crate::warn_log!("Conflicting KD readings: kept {kills}/{deaths} (raw readings: {raws:?}).");
    }
    (Some(KdResult { kills, deaths, explicit: explicit > 0, raw_texts: raws.clone() }), raws)
}
