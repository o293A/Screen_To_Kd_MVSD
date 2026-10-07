//! Chronological ordering of screenshots from the date in their file name.
//!
//! Only names containing `YYYY-MM-DD` followed by a time (`HHMMSS`, or `HH-MM-SS` / `HH.MM.SS` /
//! `HH:MM:SS`) are dated, whatever the text before it ("Screenshot ...", "Capture d'écran ...",
//! "Captura de pantalla ...", "Bildschirmfoto ...", "Cattura schermata ...", ...). The oldest
//! comes first: earlier date first, then earlier time of the day (000000 = midnight is the
//! earliest, 235959 the latest). Files whose name carries no date keep their own position.

use std::path::{Path, PathBuf};

/// (year, month, day, hour, minute, second)
pub type Stamp = (u32, u32, u32, u32, u32, u32);

fn digits(b: &[u8], at: usize, n: usize) -> Option<u32> {
    if at + n > b.len() || !b[at..at + n].iter().all(|c| c.is_ascii_digit()) {
        return None;
    }
    Some(std::str::from_utf8(&b[at..at + n]).ok()?.parse().ok()?)
}

fn is_sep(c: u8) -> bool {
    matches!(c, b' ' | b'_' | b'-' | b'.' | b'T' | b'\t')
}

fn parse_at(b: &[u8], i: usize) -> Option<Stamp> {
    // not glued to a longer number on the left
    if i > 0 && b[i - 1].is_ascii_digit() {
        return None;
    }
    let (y, mo, d) = (digits(b, i, 4)?, digits(b, i + 5, 2)?, digits(b, i + 8, 2)?);
    if b[i + 4] != b'-' || b[i + 7] != b'-' || !(1..=12).contains(&mo) || !(1..=31).contains(&d) {
        return None;
    }
    let mut p = i + 10;
    if p >= b.len() || !is_sep(b[p]) {
        return None;
    }
    while p < b.len() && is_sep(b[p]) {
        p += 1;
    }
    // HHMMSS
    let (h, mi, s, end) = if let (Some(h), Some(mi), Some(s)) = (digits(b, p, 2), digits(b, p + 2, 2), digits(b, p + 4, 2)) {
        (h, mi, s, p + 6)
    } else if let (Some(h), Some(mi), Some(s)) = (digits(b, p, 2), digits(b, p + 3, 2), digits(b, p + 6, 2)) {
        if !matches!(b[p + 2], b'-' | b'.' | b':' | b'_') || !matches!(b[p + 5], b'-' | b'.' | b':' | b'_') {
            return None;
        }
        (h, mi, s, p + 8)
    } else {
        return None;
    };
    if h > 23 || mi > 59 || s > 59 || (end < b.len() && b[end].is_ascii_digit()) {
        return None;
    }
    Some((y, mo, d, h, mi, s))
}

/// The date/time carried by a file name, if it has the expected form.
pub fn parse_stamp(file_name: &str) -> Option<Stamp> {
    let stem = Path::new(file_name).file_stem()?.to_string_lossy().to_string();
    let b = stem.as_bytes();
    if b.len() < 17 {
        return None;
    }
    (0..=b.len() - 10).find_map(|i| parse_at(b, i))
}

/// Sorts the dated files chronologically (oldest first) inside the slots they occupy; undated
/// files stay exactly where they were.
pub fn sort_paths(paths: &[PathBuf]) -> Vec<PathBuf> {
    let stamps: Vec<Option<Stamp>> =
        paths.iter().map(|p| p.file_name().and_then(|n| parse_stamp(&n.to_string_lossy()))).collect();
    let slots: Vec<usize> = (0..paths.len()).filter(|&i| stamps[i].is_some()).collect();
    let mut dated: Vec<(Stamp, usize)> = slots.iter().map(|&i| (stamps[i].unwrap(), i)).collect();
    dated.sort(); // stamp, then original position (stable for identical stamps)
    let mut out = paths.to_vec();
    for (slot, (_, original)) in slots.iter().zip(dated) {
        out[*slot] = paths[original].clone();
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_all_languages() {
        for n in [
            "Capture_decran_2026-10-04_212418.png",
            "Capture d'écran 2026-10-06 070359.png",
            "Screenshot 2026-10-06 070359.png",
            "Captura de pantalla 2026-10-06 070359.png",
            "Bildschirmfoto 2026-10-06 070359.png",
            "Cattura schermata 2026-10-06 070359.png",
        ] {
            assert!(parse_stamp(n).is_some(), "{n}");
        }
        assert_eq!(parse_stamp("Capture_decran_2026-10-04_212418.png"), Some((2026, 10, 4, 21, 24, 18)));
        assert_eq!(parse_stamp("1.png"), None);
        assert_eq!(parse_stamp("test.png"), None);
    }

    #[test]
    fn orders_dates_then_times() {
        let p: Vec<PathBuf> = [
            "S 2026-11-04 000000.png",
            "S 2026-09-04 235959.png",
            "S 2026-10-04 212418.png",
            "S 2026-10-04 000000.png",
        ]
        .iter()
        .map(PathBuf::from)
        .collect();
        let s: Vec<String> = sort_paths(&p).iter().map(|x| x.to_string_lossy().to_string()).collect();
        assert_eq!(
            s,
            ["S 2026-09-04 235959.png", "S 2026-10-04 000000.png", "S 2026-10-04 212418.png", "S 2026-11-04 000000.png"]
        );
    }

    #[test]
    fn undated_files_keep_their_slot() {
        let p: Vec<PathBuf> = ["b 2026-10-05 010101.png", "x.png", "a 2026-10-04 010101.png"].iter().map(PathBuf::from).collect();
        let s: Vec<String> = sort_paths(&p).iter().map(|x| x.to_string_lossy().to_string()).collect();
        assert_eq!(s, ["a 2026-10-04 010101.png", "x.png", "b 2026-10-05 010101.png"]);
    }
}
