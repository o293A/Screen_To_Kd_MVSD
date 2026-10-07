//! Minimal logging: everything is kept in memory (shown by the GUI / CLI) and written to kd_ocr.log.

use std::sync::Mutex;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Level {
    Debug,
    Info,
    Warn,
}

static LOG: Mutex<Vec<(Level, String)>> = Mutex::new(Vec::new());

pub fn push(level: Level, msg: String) {
    if let Ok(mut l) = LOG.lock() {
        l.push((level, msg));
    }
}

/// Takes (and clears) every message logged so far.
pub fn drain() -> Vec<(Level, String)> {
    LOG.lock().map(|mut l| std::mem::take(&mut *l)).unwrap_or_default()
}

#[macro_export]
macro_rules! warn_log { ($($a:tt)*) => { $crate::log::push($crate::log::Level::Warn, format!($($a)*)) } }
#[macro_export]
macro_rules! info_log { ($($a:tt)*) => { $crate::log::push($crate::log::Level::Info, format!($($a)*)) } }
#[macro_export]
macro_rules! debug_log { ($($a:tt)*) => { $crate::log::push($crate::log::Level::Debug, format!($($a)*)) } }

/// Writes messages to `kd_ocr.log` next to the program (best effort).
pub fn write_file(entries: &[(Level, String)]) {
    use std::io::Write;
    let path = crate::config::base_dir().join("kd_ocr.log");
    if let Ok(mut f) = std::fs::File::create(path) {
        for (lvl, msg) in entries {
            let tag = match lvl {
                Level::Debug => "DEBUG",
                Level::Info => "INFO",
                Level::Warn => "WARNING",
            };
            let _ = writeln!(f, "{tag} {msg}");
        }
    }
}
