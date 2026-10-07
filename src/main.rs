#![cfg_attr(all(windows, not(debug_assertions), feature = "gui"), windows_subsystem = "windows")]
//! Screen To Kd: reads RESULTS screenshots and prints a paste-ready Red/Green match summary.
//!
//!   kd_ocr                         opens the window
//!   kd_ocr 1.png 2.png             opens the window with these images already analyzed
//!   kd_ocr 1.png 2.png --no-gui    prints to the terminal
//!   kd_ocr folder/ --no-gui --report --per-match --green-rows

use kd_ocr::config::Settings;
use kd_ocr::log;
use kd_ocr::pipeline;
use std::io::Write;
use std::path::PathBuf;

fn usage() -> &'static str {
    "Usage: kd_ocr [images or folders...] [options]\n\
     \n\
     Options:\n\
       --no-gui            print to the terminal instead of opening the window\n\
       --report            also print totals, teams and the decision criterion\n\
       --per-match         one results block per game instead of combined totals\n\
       --green-rows        green team = players shown with the small GREEN bands (instead of the game winner)\n\
       --winner-green      green team = the game winner (default)\n\
       --settings FILE     use another settings file\n\
       -v, --verbose       print every log message\n\
       -h, --help          this help"
}

/// Console output that also works from a windowless Windows build.
fn console_writer() -> Box<dyn Write> {
    #[cfg(windows)]
    {
        extern "system" {
            fn AttachConsole(pid: u32) -> i32;
        }
        // SAFETY: plain Win32 call, no pointers involved.
        unsafe {
            AttachConsole(u32::MAX);
        }
        if let Ok(f) = std::fs::OpenOptions::new().write(true).open("CONOUT$") {
            return Box::new(f);
        }
    }
    Box::new(std::io::stdout())
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let mut items: Vec<String> = Vec::new();
    let (mut no_gui, mut report, mut per_match, mut verbose) = (false, false, None, false);
    let mut green_rows: Option<bool> = None;
    let mut settings_file: Option<PathBuf> = None;
    let mut it = args.into_iter();
    while let Some(a) = it.next() {
        match a.as_str() {
            "--no-gui" => no_gui = true,
            "--report" => report = true,
            "--per-match" => per_match = Some(true),
            "--green-rows" => green_rows = Some(true),
            "--winner-green" => green_rows = Some(false),
            "-v" | "--verbose" => verbose = true,
            "--settings" => settings_file = it.next().map(PathBuf::from),
            "-h" | "--help" => {
                let mut out = console_writer();
                let _ = writeln!(out, "{}", usage());
                return;
            }
            _ => items.push(a),
        }
    }
    let mut settings = Settings::load(settings_file.as_deref());
    if let Some(p) = per_match {
        settings.per_match = p;
    }
    if let Some(g) = green_rows {
        settings.green_team_is_game_winner = !g;
    }
    let paths = pipeline::expand_paths(&items);

    #[cfg(feature = "gui")]
    if !no_gui {
        if let Err(e) = kd_ocr::gui::run(settings, paths.clone()) {
            let _ = std::fs::write(kd_ocr::config::base_dir().join("kd_ocr_crash.log"), format!("{e}"));
        }
        return;
    }
    let _ = no_gui;

    let mut out = console_writer();
    if paths.is_empty() {
        let _ = writeln!(out, "No image to analyze.\n\n{}", usage());
        std::process::exit(1);
    }
    let (analysis, output) = pipeline::process_images(&paths, &settings, &|_, _, _| {});
    let mut entries = analysis.log.clone();
    entries.extend(output.log.iter().filter(|e| !analysis.log.contains(e)).cloned());
    log::write_file(&entries);
    for (lvl, msg) in &entries {
        if *lvl == log::Level::Warn || verbose {
            eprintln!("[{lvl:?}] {msg}");
        }
    }
    if !output.ok() {
        let _ = writeln!(out, "No player found: nothing to display.");
        std::process::exit(1);
    }
    let _ = writeln!(out, "{}", output.standings);
    if report {
        let _ = writeln!(out, "\n{}", output.report);
    }
}
