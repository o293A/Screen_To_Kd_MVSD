//! Central configuration: constants calibrated on the sample screenshots + settings.txt.

use std::path::{Path, PathBuf};

pub const APP_VERSION: &str = "1.0";
pub const SUPPORTED_EXTENSIONS: [&str; 5] = ["png", "jpg", "jpeg", "bmp", "webp"];

// ---- Row background colours (OpenCV-style HSV: H 0-179, S/V 0-255) ----------------------
pub const GREEN_HUE: (u8, u8) = (45, 100);
pub const RED_HUE: [(u8, u8); 2] = [(0, 12), (168, 179)];
pub const BG_MIN_SATURATION: u8 = 90;
pub const BG_MIN_VALUE: u8 = 35;
pub const BG_MAX_VALUE: u8 = 140;

// ---- Row segmentation ---------------------------------------------------------------------
pub const MIN_BAND_WIDTH_PX: usize = 40;
pub const MIN_BAND_WIDTH_RATIO: f32 = 0.15;
pub const BAND_LINE_MIN_FILL: f32 = 0.45;
pub const SEPARATOR_MIN_RATIO: f32 = 1.30;
pub const SEPARATOR_MIN_DELTA: f32 = 25.0;
pub const MIN_ROW_HEIGHT_RATIO: f32 = 0.40;
pub const CHAIN_GAP_RATIO: f32 = 0.25;
pub const CHAIN_GAP_MIN_PX: f32 = 4.0;

// ---- Columns (fractions of a row's width) ---------------------------------------------------
pub const NAME_REGION: (f32, f32) = (0.02, 0.58);
pub const KD_SEARCH_START: f32 = 0.62;
pub const KD_FALLBACK_REGION: (f32, f32) = (0.80, 1.0);
pub const FOREGROUND_DELTA: i32 = 35;
pub const CLUSTER_GAP_RATIO: f32 = 0.03;
pub const CLUSTER_MIN_PIXELS: usize = 6;

pub const KD_SPLIT_MIN_GAP_RATIO: f32 = 0.20;
pub const KD_SLASH_MIN_HEIGHT_RATIO: f32 = 0.60;
pub const KD_DASH_MAX_HEIGHT_RATIO: f32 = 0.40;

// ---- OCR ----------------------------------------------------------------------------------
pub const OCR_SCALE: u32 = 3;
pub const OCR_PADDING: u32 = 10;
pub const KD_ALLOWLIST: &str = "0123456789/|\\-lI,.: ";
pub const NUMBER_ALLOWLIST: &str = "0123456789";
pub const MAX_STAT_VALUE: u32 = 999;

/// A screenshot with no banner and at most this many rows is a "continuation" screenshot
/// (the last players of a 4v4 table hidden by the scroll area).
pub const PARTIAL_MAX_ROWS: usize = 4;

/// Folder of the program (next to the executable), used for settings.txt and the logs.
pub fn base_dir() -> PathBuf {
    std::env::current_exe()
        .ok()
        .and_then(|p| p.parent().map(Path::to_path_buf))
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn settings_path() -> PathBuf {
    base_dir().join("settings.txt")
}

// ===========================================================================================
// settings.txt
// ===========================================================================================
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Settings {
    /// One results block per game instead of totals summed over every game.
    pub per_match: bool,
    /// true  = the team that wins the games is the GREEN team (banner colour).
    /// false = the GREEN team is the one shown with the small GREEN row bands on the
    ///         screenshots, the RED team the one shown with the small RED bands (the result
    ///         of the game plays no role: only the team names change).
    pub green_team_is_game_winner: bool,
    pub auto_copy: bool,
    pub deduplicate_images: bool,
}

impl Default for Settings {
    fn default() -> Self {
        Settings { per_match: false, green_team_is_game_winner: true, auto_copy: true, deduplicate_images: false }
    }
}

const KEYS: [(&str, &str); 4] = [
    (
        "PER_MATCH",
        "One results block per game, stacked one after another, instead of a single block\n\
         with totals summed across every game.",
    ),
    (
        "GREEN_TEAM_IS_GAME_WINNER",
        "true  = the team that WINS the games is the GREEN team (banner colour decides).\n\
         false = the GREEN team is the one shown with the small GREEN row bands on the\n\
                 screenshots and the RED team the one shown with the small RED bands.\n\
                 Victory/defeat plays no role here: only the team names change.",
    ),
    (
        "AUTO_COPY_RESULT",
        "Automatically copy the results text to the clipboard once the analysis finishes\n\
         (graphical interface only).",
    ),
    (
        "DEDUPLICATE_IMAGES",
        "true  = skip screenshots that are strictly pixel-identical to another one already\n\
                 analyzed (avoids counting the same game twice).\n\
         false = every screenshot counts as one game, copies included.",
    ),
];

/// Keys of older versions that are silently ignored.
const LEGACY_KEYS: [&str; 2] = ["USE_TRACKED_LIST", "WINNER_IS_GREEN"];

fn parse_bool(s: &str) -> Option<bool> {
    match s.trim().to_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Some(true),
        "false" | "0" | "no" | "off" => Some(false),
        _ => None,
    }
}

impl Settings {
    fn get(&self, key: &str) -> bool {
        match key {
            "PER_MATCH" => self.per_match,
            "GREEN_TEAM_IS_GAME_WINNER" => self.green_team_is_game_winner,
            "AUTO_COPY_RESULT" => self.auto_copy,
            _ => self.deduplicate_images,
        }
    }

    fn set(&mut self, key: &str, v: bool) {
        match key {
            "PER_MATCH" => self.per_match = v,
            "GREEN_TEAM_IS_GAME_WINNER" => self.green_team_is_game_winner = v,
            "AUTO_COPY_RESULT" => self.auto_copy = v,
            _ => self.deduplicate_images = v,
        }
    }

    pub fn file_text(&self) -> String {
        let mut out = String::from(
            "# kd_ocr - startup settings\n#\n# Replace \"true\" or \"false\" after the \"=\" sign, save, then (re)start the\n\
             # program. A missing line, a typo, or a value that is neither true nor false falls back\n\
             # to its default value (shown below), with a warning in kd_ocr.log.\n\n",
        );
        for (key, comment) in KEYS {
            for line in comment.lines() {
                out.push_str(&format!("# {line}\n"));
            }
            out.push_str(&format!("{key} = {}\n\n", if self.get(key) { "true" } else { "false" }));
        }
        out.trim_end().to_string() + "\n"
    }

    pub fn parse(text: &str) -> Settings {
        let mut s = Settings::default();
        for (n, raw) in text.lines().enumerate() {
            let line = raw.trim().trim_start_matches('\u{feff}');
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let Some((k, v)) = line.split_once('=') else {
                crate::warn_log!("settings.txt line {}: \"{line}\" skipped (expected KEY = true/false).", n + 1);
                continue;
            };
            let key = k.trim().to_uppercase();
            if LEGACY_KEYS.contains(&key.as_str()) {
                continue;
            }
            if !KEYS.iter().any(|(name, _)| *name == key) {
                crate::warn_log!("settings.txt line {}: unknown setting \"{key}\" skipped.", n + 1);
                continue;
            }
            match parse_bool(v) {
                Some(b) => s.set(&key, b),
                None => crate::warn_log!(
                    "settings.txt line {}: \"{}\" is neither true nor false for {key}: default kept.",
                    n + 1,
                    v.trim()
                ),
            }
        }
        s
    }

    /// Reads settings.txt (created with the commented defaults when missing). Never fails.
    pub fn load(path: Option<&Path>) -> Settings {
        let target = path.map(Path::to_path_buf).unwrap_or_else(settings_path);
        match std::fs::read_to_string(&target) {
            Ok(text) => Settings::parse(&text),
            Err(_) => {
                let d = Settings::default();
                if let Err(e) = std::fs::write(&target, d.file_text()) {
                    crate::warn_log!("Could not create {}: {e}", target.display());
                }
                d
            }
        }
    }
}
