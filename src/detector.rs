//! Player detection on one screenshot: name, background colour, KD of every row.
//!
//! The name and the KD of a row come from the SAME band of pixels, so they can never be
//! swapped between two rows.

use crate::config as cfg;
use crate::geometry::refine_name;
use crate::imgproc as ip;
use crate::model::{Color, PlayerScore, ScreenshotResult};
use crate::ocr;
use image::RgbImage;
use rayon::prelude::*;

fn read_row(image: &RgbImage, row: &ip::Band, index: usize, label: &str) -> Option<PlayerScore> {
    let Some(name_crop) = ip::crop_name_region(image, row) else {
        crate::info_log!("{label}: no text found in the name area.");
        return None;
    };
    let (read, _raws) = ocr::best_name_reading(&name_crop);
    let Some(read) = read else {
        crate::info_log!("{label}: no readable name: row skipped.");
        return None;
    };
    let name = refine_name(&read.name, &name_crop);

    let kd_crop = ip::crop_kd_region(image, row);
    if let Some(c) = &kd_crop {
        if ip::is_placeholder_crop(c) {
            // "- / -" = disconnected player: treated as if absent from the screen.
            crate::debug_log!("{label}: empty KD \"- / -\" for \"{name}\": player ignored.");
            return None;
        }
    }
    let (kd, kd_raws) = match &kd_crop {
        Some(c) => ocr::read_kd(c),
        None => (None, vec![]),
    };
    let Some(kd) = kd else {
        if kd_raws.iter().any(|t| ocr::is_kd_placeholder(t)) {
            crate::debug_log!("{label}: empty KD \"- / -\" for \"{name}\": player ignored.");
            return None;
        }
        crate::warn_log!("{label}: KD not found for \"{name}\", read {kd_raws:?}: player skipped.");
        return None;
    };
    if !kd.explicit {
        crate::warn_log!(
            "{label}: KD for \"{name}\" reconstructed from glued digits ({:?}) -> {}/{}: double-check it.",
            kd.raw_texts,
            kd.kills,
            kd.deaths
        );
    }
    Some(PlayerScore { name, color: row.color, kills: kd.kills, deaths: kd.deaths, row_index: index, name_score: read.score })
}

/// Analyzes one screenshot. Never panics over an image problem: warnings instead.
pub fn detect_players(image: &RgbImage, source: &str) -> ScreenshotResult {
    let mut result = ScreenshotResult { source: source.to_string(), ..Default::default() };
    let layout = ip::detect_layout(image);
    result.rows_detected = layout.rows.len();
    if layout.rows.is_empty() {
        crate::warn_log!("{source}: no player row detected (is this really a results screen?).");
        return result;
    }
    match layout.banner {
        Some(b) => result.winner_color = Some(b.color),
        None if layout.rows.len() <= cfg::PARTIAL_MAX_ROWS => {
            // Only the last rows of the table (players hidden by the scroll area): a
            // continuation of the previous screenshot, merged later by the pipeline.
            result.is_partial = true;
        }
        None => {
            crate::warn_log!(
                "{source}: VICTORY/DEFEAT banner not found: assuming the green rows won this game."
            );
            result.winner_color = Some(Color::Green);
        }
    }

    let rows: Vec<Option<PlayerScore>> = layout
        .rows
        .par_iter()
        .enumerate()
        .map(|(i, row)| {
            let label = format!("{source} row {}", i + 1);
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| read_row(image, row, i, &label))).unwrap_or_else(|_| {
                crate::warn_log!("{label}: unexpected error: row skipped.");
                None
            })
        })
        .collect();
    let mut seen: Vec<String> = Vec::new();
    for p in rows.into_iter().flatten() {
        if seen.contains(&p.name) {
            crate::warn_log!("{source}: \"{}\" was already read on this image (same name on two rows): duplicate skipped.", p.name);
            continue;
        }
        seen.push(p.name.clone());
        result.players.push(p);
    }
    if result.players.is_empty() {
        crate::warn_log!("{source}: no player found.");
    }
    result
}
