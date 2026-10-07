//! Real screenshots (tests/samples) read end to end and compared to the verified expected text.

use kd_ocr::{config::Settings, pipeline};
use std::path::PathBuf;

fn run(dir: &str, settings: Settings) -> String {
    let root = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/samples").join(dir);
    let mut files: Vec<String> = std::fs::read_dir(&root)
        .unwrap()
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| pipeline::is_supported(p))
        .map(|p| p.to_string_lossy().to_string())
        .collect();
    files.sort();
    let paths = pipeline::expand_paths(&files);
    let (_a, out) = pipeline::process_images(&paths, &settings, &|_, _, _| {});
    out.standings
}

fn strip_time(s: &str) -> String {
    s.lines()
        .filter(|l| !l.contains(" at ") || l.contains(':') && !l.chars().next().map_or(false, |c| c.is_ascii_digit()))
        .map(|l| l.trim_end())
        .collect::<Vec<_>>()
        .join("\n")
        .trim()
        .to_string()
}

fn check(dir: &str) {
    let expected = std::fs::read_to_string(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/samples").join(dir).join("expected.txt")).unwrap();
    assert_eq!(strip_time(&run(dir, Settings::default())), strip_time(&expected), "{dir}");
}

#[test]
fn sample_with_team_changes() {
    check("changement");
}

#[test]
fn sample_without_team_changes() {
    check("sans_changement");
}

/// 4v4: the hidden players of each game are on a separate "continuation" screenshot, which
/// must be attached to the game right before it (dates in the file names, any order given).
#[test]
fn hidden_players_are_merged_in_the_right_game() {
    let d = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/samples/partiel");
    let mut files: Vec<String> = std::fs::read_dir(&d).unwrap().map(|e| e.unwrap().path().to_string_lossy().to_string()).collect();
    files.sort();
    files.reverse(); // newest first: the date sort must restore the order
    let paths = pipeline::expand_paths(&files);
    let s = Settings { per_match: true, ..Settings::default() };
    let (_a, out) = pipeline::process_images(&paths, &s, &|_, _, _| {});
    let text = out.standings;
    for line in [
        "cata: 8/2 (4.00 KD) 👑", "18ms: 7/3 (2.33 KD)", "isma: 5/2 (2.50 KD)", "Sin: 3/4 (0.75 KD)",
        "Nemo: 5/6 (0.83 KD) 🥈", "Jerrylina: 3/5 (0.60 KD)", "Astro_Amzz: 2/6 (0.33 KD)", "bluffar: 1/6 (0.17 KD)",
        "cata: 11/5 (2.20 KD) 👑", "Sin: 6/7 (0.86 KD)", "isma: 5/4 (1.25 KD)", "18ms: 5/6 (0.83 KD)",
        "bluffar: 7/6 (1.17 KD) 🥈", "Nemo: 7/6 (1.17 KD) 🥈", "Jerrylina: 4/7 (0.57 KD)", "Astro_Amzz: 4/8 (0.50 KD)",
    ] {
        assert!(text.contains(line), "missing `{line}` in:\n{text}");
    }
}
