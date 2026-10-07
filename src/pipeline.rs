//! Pipeline with no interface: images -> readings -> teams -> results text.
//!
//! Shared by the command line and the graphical interface. Split in two so that changing a
//! display option (one game per image, green team rule) never needs the OCR to run again:
//! `analyze` reads the screenshots once, `build_output` turns the readings into text.

use crate::config::{self as cfg, Settings};
use crate::detector::detect_players;
use crate::formatter::{format_report, format_standings};
use crate::imgproc as ip;
use crate::log::{self, Level};
use crate::model::ScreenshotResult;
use crate::teams::{TeamAssignment, TeamBuilder};
use rayon::prelude::*;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};

/// Everything the OCR produced for a set of screenshots (before any team decision).
#[derive(Clone, Debug, Default)]
pub struct Analysis {
    /// One entry per GAME (continuation screenshots are already merged into their game).
    pub screens: Vec<ScreenshotResult>,
    /// Files skipped (unreadable, duplicates, orphan continuation screenshots).
    pub skipped: Vec<String>,
    pub log: Vec<(Level, String)>,
}

#[derive(Clone, Debug, Default)]
pub struct RunOutput {
    pub standings: String,
    pub report: String,
    pub assignment: TeamAssignment,
    pub log: Vec<(Level, String)>,
}

impl RunOutput {
    pub fn ok(&self) -> bool {
        !self.standings.is_empty()
    }
}

/// Expands folders into their images (alphabetical order); files are kept as-is. The result is
/// then put in chronological order when the file names carry a date (see `datesort`).
pub fn expand_paths(items: &[String]) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    for item in items {
        let p = PathBuf::from(item.trim().trim_matches('"'));
        if p.is_dir() {
            let mut files: Vec<PathBuf> = std::fs::read_dir(&p)
                .map(|rd| rd.filter_map(|e| e.ok().map(|e| e.path())).filter(|f| is_supported(f)).collect())
                .unwrap_or_default();
            files.sort();
            paths.extend(files);
        } else {
            paths.push(p);
        }
    }
    crate::datesort::sort_paths(&paths)
}

pub fn is_supported(p: &Path) -> bool {
    p.extension().map(|e| cfg::SUPPORTED_EXTENSIONS.contains(&e.to_string_lossy().to_lowercase().as_str())).unwrap_or(false)
}

/// Reads every screenshot. `progress(done, total, file_name)` is called as images complete.
pub fn analyze(paths: &[PathBuf], dedupe: bool, progress: &(dyn Fn(usize, usize, &str) + Sync)) -> Analysis {
    let mut out = Analysis::default();
    let paths = crate::datesort::sort_paths(paths);
    let total = paths.len();
    let done = AtomicUsize::new(0);

    // 1. decode (parallel)
    let decoded: Vec<Option<(image::RgbImage, (u64, u64))>> = paths
        .par_iter()
        .map(|p| {
            let img = ip::load_image(p);
            img.map(|i| {
                let fp = ip::image_fingerprint(&i);
                (i, fp)
            })
        })
        .collect();

    // 2. duplicates (sequential, keeps the first)
    let mut jobs: Vec<(usize, &image::RgbImage)> = Vec::new();
    let mut seen: HashMap<(u64, u64), String> = HashMap::new();
    for (i, d) in decoded.iter().enumerate() {
        let name = file_name(&paths[i]);
        let Some((img, fp)) = d else {
            out.skipped.push(name);
            continue;
        };
        if dedupe {
            if let Some(first) = seen.get(fp) {
                crate::warn_log!("{name} is pixel-for-pixel identical to {first}: duplicate skipped (otherwise its scores would be counted twice).");
                out.skipped.push(name);
                continue;
            }
            seen.insert(*fp, name.clone());
        }
        jobs.push((i, img));
    }

    // 3. read (parallel over screenshots, and over the rows inside each one)
    let results: Vec<ScreenshotResult> = jobs
        .par_iter()
        .map(|(i, img)| {
            let name = file_name(&paths[*i]);
            let r = detect_players(img, &name);
            let n = done.fetch_add(1, Ordering::SeqCst) + 1;
            progress(n, jobs.len().max(total.min(jobs.len())), &name);
            r
        })
        .collect();

    // 4. continuation screenshots (hidden players of a 4v4) join the game right before them
    let mut games: Vec<ScreenshotResult> = Vec::new();
    for r in results {
        if !r.is_partial {
            games.push(r);
            continue;
        }
        if r.players.is_empty() {
            out.skipped.push(r.source.clone());
            continue;
        }
        match games.last_mut() {
            Some(game) => {
                let mut added = Vec::new();
                for p in r.players {
                    if game.players.iter().any(|g| g.name == p.name) {
                        crate::info_log!("{}: \"{}\" is already on {}: not counted twice.", r.source, p.name, game.source);
                        continue;
                    }
                    added.push(p.name.clone());
                    let mut p = p;
                    p.row_index = game.players.len();
                    game.players.push(p);
                }
                crate::info_log!(
                    "{}: continuation of {} (players hidden by the scroll): added {}.",
                    r.source,
                    game.source,
                    added.join(", ")
                );
                game.merged_sources.push(r.source.clone());
            }
            None => {
                crate::warn_log!("{}: continuation screenshot with no earlier game screenshot to attach it to: skipped.", r.source);
                out.skipped.push(r.source.clone());
            }
        }
    }

    unify_names(&mut games);
    out.screens = games;
    out
}

fn file_name(p: &Path) -> String {
    p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_else(|| p.display().to_string())
}

/// Look-alike-insensitive key (case, o/0, l/I/1, s/5, z/2).
fn fold_key(name: &str) -> String {
    name.chars()
        .map(|c| match c.to_lowercase().next().unwrap_or(c) {
            '0' => 'o',
            '1' | 'i' => 'l',
            '5' => 's',
            '2' => 'z',
            x => x,
        })
        .collect()
}

/// The same player must have the same spelling on every screenshot: when several readings of a
/// name differ only by look-alike characters (and never appear together on one screenshot),
/// the most frequent reading wins everywhere.
fn unify_names(games: &mut [ScreenshotResult]) {
    // key -> spelling -> (count, confidence sum)
    let mut table: HashMap<String, Vec<(String, u32, f32)>> = HashMap::new();
    for g in games.iter() {
        for p in &g.players {
            let spellings = table.entry(fold_key(&p.name)).or_default();
            match spellings.iter_mut().find(|s| s.0 == p.name) {
                Some(s) => {
                    s.1 += 1;
                    s.2 += p.name_score;
                }
                None => spellings.push((p.name.clone(), 1, p.name_score)),
            }
        }
    }
    let mut rename: HashMap<String, String> = HashMap::new();
    for (key, spellings) in &table {
        if spellings.len() < 2 {
            continue;
        }
        let together = games.iter().any(|g| {
            g.players.iter().filter(|p| fold_key(&p.name) == *key).map(|p| &p.name).collect::<std::collections::HashSet<_>>().len() > 1
        });
        if together {
            continue; // two real players whose names differ by look-alikes: leave them alone
        }
        let best = spellings.iter().max_by(|a, b| (a.1, a.2).partial_cmp(&(b.1, b.2)).unwrap()).unwrap();
        for s in spellings {
            if s.0 != best.0 {
                crate::info_log!("Name \"{}\" unified to \"{}\" (same player, different readings).", s.0, best.0);
                rename.insert(s.0.clone(), best.0.clone());
            }
        }
    }
    if rename.is_empty() {
        return;
    }
    for g in games.iter_mut() {
        for p in g.players.iter_mut() {
            if let Some(n) = rename.get(&p.name) {
                p.name = n.clone();
            }
        }
    }
}

/// Teams, totals and text from already-read screenshots (instant: no OCR).
pub fn build_output(analysis: &Analysis, settings: &Settings) -> RunOutput {
    let mut builder = TeamBuilder::new(settings.green_team_is_game_winner);
    for s in &analysis.screens {
        if !s.players.is_empty() {
            builder.add_screenshot(s);
        }
    }
    let assignment = builder.resolve();
    let standings = if assignment.entries.is_empty() { String::new() } else { format_standings(&assignment, settings.per_match) };
    let report = format_report(&assignment);
    let mut log = analysis.log.clone();
    log.extend(log::drain());
    RunOutput { standings, report, assignment, log }
}

/// Full run: analyze then build.
pub fn process_images(paths: &[PathBuf], settings: &Settings, progress: &(dyn Fn(usize, usize, &str) + Sync)) -> (Analysis, RunOutput) {
    let mut analysis = analyze(paths, settings.deduplicate_images, progress);
    analysis.log = log::drain();
    let output = build_output(&analysis, settings);
    (analysis, output)
}
