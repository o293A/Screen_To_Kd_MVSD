//! Shared types.

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Color {
    Green,
    Red,
}

impl Color {
    pub fn opposite(self) -> Color {
        match self {
            Color::Green => Color::Red,
            Color::Red => Color::Green,
        }
    }
}

/// A player read from ONE screenshot.
#[derive(Clone, Debug)]
pub struct PlayerScore {
    pub name: String,
    pub color: Color,
    pub kills: u32,
    pub deaths: u32,
    pub row_index: usize,
    /// OCR confidence of the name (0 - 100).
    pub name_score: f32,
}

/// Result of analyzing one screenshot.
#[derive(Clone, Debug, Default)]
pub struct ScreenshotResult {
    pub source: String,
    pub players: Vec<PlayerScore>,
    /// Colour of the winning rows on THIS screenshot (None = unknown).
    pub winner_color: Option<Color>,
    pub rows_detected: usize,
    /// A "continuation" screenshot: only the last rows of the table (players hidden by the
    /// scroll area, e.g. 4v4), no banner. It is merged into the previous full screenshot.
    pub is_partial: bool,
    /// Files merged into this one (partial screenshots).
    pub merged_sources: Vec<String>,
}
