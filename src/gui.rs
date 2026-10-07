//! Graphical interface: dark "terminal" theme (egui, native window, instant start-up).
//!
//! The window opens immediately; the OCR network is warmed up on a background thread and the
//! analysis runs on another one, so the interface never freezes. Changing a display option
//! rebuilds the results text instantly without reading the screenshots again.

use crate::config::{self as cfg, Settings, APP_VERSION};
use crate::log::{self, Level};
use crate::model::ScreenshotResult;
use crate::pipeline::{self, Analysis, RunOutput};
use eframe::egui::{self, Color32, FontId, RichText, Stroke};
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{channel, Receiver};
use std::sync::Arc;
use std::time::Instant;

// ---- palette ----------------------------------------------------------------------------------
const BG: Color32 = Color32::from_rgb(0x0a, 0x0e, 0x14);
const PANEL: Color32 = Color32::from_rgb(0x0f, 0x14, 0x1b);
const PANEL2: Color32 = Color32::from_rgb(0x15, 0x1c, 0x26);
const BORDER: Color32 = Color32::from_rgb(0x1f, 0x2a, 0x37);
const TEXT: Color32 = Color32::from_rgb(0xd5, 0xdd, 0xe8);
const MUTED: Color32 = Color32::from_rgb(0x66, 0x76, 0x8c);
const DIM: Color32 = Color32::from_rgb(0x3b, 0x47, 0x57);
const GREEN: Color32 = Color32::from_rgb(0x3d, 0xdc, 0x84);
const CYAN: Color32 = Color32::from_rgb(0x56, 0xb6, 0xf2);
const YELLOW: Color32 = Color32::from_rgb(0xe5, 0xc0, 0x7b);
const RED: Color32 = Color32::from_rgb(0xff, 0x6b, 0x6b);
const SELECT: Color32 = Color32::from_rgb(0x17, 0x3a, 0x2b);

enum Msg {
    Progress(usize, usize, String),
    Done(Analysis),
}

#[derive(Clone, Copy, PartialEq)]
enum LogKind {
    Info,
    Ok,
    Warn,
    Err,
    Head,
    Faint,
}

struct App {
    settings: Settings,
    files: Vec<PathBuf>,
    selected: HashSet<usize>,
    analysis: Option<Analysis>,
    output: Option<RunOutput>,
    console: Vec<(String, LogKind, String)>,
    rx: Option<Receiver<Msg>>,
    progress: (usize, usize),
    running: Option<Instant>,
    engine_ready: Arc<AtomicBool>,
    status: (String, LogKind),
    start: Instant,
    copied_at: Option<Instant>,
    pending_start: bool,
    file_status: HashMap<String, &'static str>,
}

pub fn run(settings: Settings, initial: Vec<PathBuf>) -> Result<(), eframe::Error> {
    let options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1060.0, 760.0])
            .with_min_inner_size([820.0, 600.0])
            .with_title("Screen To Kd")
            .with_icon(load_icon())
            .with_drag_and_drop(true),
        ..Default::default()
    };
    eframe::run_native(
        "Screen To Kd",
        options,
        Box::new(move |cc| {
            setup_style(&cc.egui_ctx);
            Ok(Box::new(App::new(settings, initial)))
        }),
    )
}

/// Window / taskbar icon (embedded in the exe, so it works wherever the .exe is placed).
fn load_icon() -> egui::IconData {
    match image::load_from_memory(include_bytes!("../assets/icon.png")) {
        Ok(img) => {
            let img = img.into_rgba8();
            let (width, height) = img.dimensions();
            egui::IconData { rgba: img.into_raw(), width, height }
        }
        Err(_) => egui::IconData::default(),
    }
}

fn setup_style(ctx: &egui::Context) {
    let mut style = (*ctx.style()).clone();
    style.visuals = egui::Visuals::dark();
    let v = &mut style.visuals;
    v.override_text_color = Some(TEXT);
    v.panel_fill = BG;
    v.window_fill = PANEL;
    v.extreme_bg_color = PANEL;
    v.faint_bg_color = PANEL2;
    v.selection.bg_fill = SELECT;
    v.selection.stroke = Stroke::new(1.0_f32, GREEN);
    v.widgets.noninteractive.bg_stroke = Stroke::new(1.0_f32, BORDER);
    v.widgets.inactive.weak_bg_fill = PANEL2;
    v.widgets.inactive.bg_fill = PANEL2;
    v.widgets.inactive.bg_stroke = Stroke::new(1.0_f32, BORDER);
    v.widgets.hovered.weak_bg_fill = Color32::from_rgb(0x1b, 0x25, 0x33);
    v.widgets.hovered.bg_stroke = Stroke::new(1.0_f32, GREEN);
    v.widgets.hovered.fg_stroke = Stroke::new(1.0_f32, GREEN);
    v.widgets.active.weak_bg_fill = Color32::from_rgb(0x22, 0x30, 0x3f);
    v.widgets.active.bg_stroke = Stroke::new(1.0_f32, GREEN);
    for r in [&mut v.widgets.inactive.corner_radius, &mut v.widgets.hovered.corner_radius, &mut v.widgets.active.corner_radius] {
        *r = egui::CornerRadius::same(3);
    }
    style.spacing.item_spacing = egui::vec2(8.0, 6.0);
    style.spacing.button_padding = egui::vec2(10.0, 5.0);
    use egui::TextStyle::*;
    style.text_styles = [
        (Heading, FontId::monospace(22.0)),
        (Body, FontId::monospace(13.0)),
        (Monospace, FontId::monospace(13.0)),
        (Button, FontId::monospace(13.0)),
        (Small, FontId::monospace(11.0)),
    ]
    .into();
    ctx.set_style(style);
}

fn card() -> egui::Frame {
    egui::Frame::new().fill(PANEL).stroke(Stroke::new(1.0_f32, BORDER)).inner_margin(egui::Margin::same(10))
}

fn section_title(ui: &mut egui::Ui, text: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new("//").color(DIM));
        ui.label(RichText::new(format!(" {text} ")).color(CYAN).strong());
    });
}

fn kind_color(k: LogKind) -> Color32 {
    match k {
        LogKind::Info => MUTED,
        LogKind::Ok => GREEN,
        LogKind::Warn => YELLOW,
        LogKind::Err => RED,
        LogKind::Head => CYAN,
        LogKind::Faint => DIM,
    }
}

fn open_in_default_app(path: &std::path::Path) {
    #[cfg(windows)]
    let _ = std::process::Command::new("cmd").args(["/C", "start", ""]).arg(path).spawn();
    #[cfg(target_os = "macos")]
    let _ = std::process::Command::new("open").arg(path).spawn();
    #[cfg(all(unix, not(target_os = "macos")))]
    let _ = std::process::Command::new("xdg-open").arg(path).spawn();
}

impl App {
    fn new(settings: Settings, initial: Vec<PathBuf>) -> App {
        let ready = Arc::new(AtomicBool::new(false));
        let flag = ready.clone();
        std::thread::spawn(move || {
            crate::nn::Crnn::global();
            flag.store(true, Ordering::SeqCst);
        });
        let mut app = App {
            settings,
            files: Vec::new(),
            selected: HashSet::new(),
            analysis: None,
            output: None,
            console: Vec::new(),
            rx: None,
            progress: (0, 0),
            running: None,
            engine_ready: ready,
            status: ("ready.".into(), LogKind::Info),
            start: Instant::now(),
            copied_at: None,
            pending_start: false,
            file_status: HashMap::new(),
        };
        app.log(&format!("kd_ocr v{APP_VERSION} - ready."), LogKind::Ok);
        if !initial.is_empty() {
            app.add_files(initial);
            app.pending_start = true;
        }
        app
    }

    fn log(&mut self, msg: &str, kind: LogKind) {
        let mark = match kind {
            LogKind::Ok => "ok",
            LogKind::Warn => "!!",
            LogKind::Err => "xx",
            _ => "..",
        };
        let ts = chrono::Local::now().format("[%H:%M:%S]").to_string();
        self.console.push((ts, kind, format!("[{mark}] {msg}")));
        if self.console.len() > 2500 {
            self.console.drain(..500);
        }
    }

    fn add_files(&mut self, items: Vec<PathBuf>) {
        let strings: Vec<String> = items.iter().map(|p| p.to_string_lossy().to_string()).collect();
        let mut added = 0;
        let mut all = std::mem::take(&mut self.files);
        for p in pipeline::expand_paths(&strings) {
            if !all.contains(&p) {
                all.push(p);
                added += 1;
            }
        }
        // chronological order for dated names (undated files keep their position)
        self.files = crate::datesort::sort_paths(&all);
        self.selected.clear();
        if added > 0 {
            self.log(&format!("{added} file(s) added ({} total).", self.files.len()), LogKind::Info);
        }
    }

    fn start_analysis(&mut self, ctx: &egui::Context) {
        if self.running.is_some() {
            return;
        }
        if self.files.is_empty() {
            self.set_status("add screenshots first.", LogKind::Warn);
            return;
        }
        let (tx, rx) = channel();
        self.rx = Some(rx);
        self.running = Some(Instant::now());
        self.progress = (0, self.files.len());
        self.file_status.clear();
        self.set_status("analyzing...", LogKind::Info);
        self.log(&format!("analyzing {} screenshot(s)...", self.files.len()), LogKind::Head);
        let files = self.files.clone();
        let dedupe = self.settings.deduplicate_images;
        let ctx = ctx.clone();
        std::thread::spawn(move || {
            let tx2 = std::sync::Mutex::new(tx.clone());
            let ctx2 = ctx.clone();
            let progress = move |done: usize, total: usize, name: &str| {
                if let Ok(t) = tx2.lock() {
                    let _ = t.send(Msg::Progress(done, total, name.to_string()));
                }
                ctx2.request_repaint();
            };
            let mut analysis = pipeline::analyze(&files, dedupe, &progress);
            analysis.log = log::drain();
            let _ = tx.send(Msg::Done(analysis));
            ctx.request_repaint();
        });
    }

    fn set_status(&mut self, text: &str, kind: LogKind) {
        self.status = (text.to_string(), kind);
    }

    fn rebuild_output(&mut self) {
        if let Some(a) = &self.analysis {
            let out = pipeline::build_output(a, &self.settings);
            self.output = Some(out);
        }
    }

    fn finish(&mut self, analysis: Analysis, ctx: &egui::Context) {
        let secs = self.running.take().map(|t| t.elapsed().as_secs_f32()).unwrap_or(0.0);
        self.rx = None;
        // per-file markers
        let mut status: HashMap<String, &'static str> = HashMap::new();
        for s in &analysis.screens {
            status.insert(s.source.clone(), if s.players.is_empty() { "??" } else { "ok" });
            for m in &s.merged_sources {
                status.insert(m.clone(), "++");
            }
        }
        for s in &analysis.skipped {
            status.insert(s.clone(), "--");
        }
        self.file_status = status;
        for (lvl, msg) in &analysis.log {
            match lvl {
                Level::Warn => self.log(msg, LogKind::Warn),
                Level::Info => self.log(msg, LogKind::Info),
                Level::Debug => {}
            }
        }
        log::write_file(&analysis.log);
        self.analysis = Some(analysis);
        self.rebuild_output();
        let ok = self.output.as_ref().map(|o| o.ok()).unwrap_or(false);
        if ok {
            let games = self.output.as_ref().map(|o| o.assignment.rounds).unwrap_or(0);
            self.log(&format!("done in {secs:.1}s - {games} game(s)."), LogKind::Ok);
            self.set_status("done.", LogKind::Ok);
            if self.settings.auto_copy {
                self.copy_results(ctx);
            }
        } else {
            self.log("no player found: nothing to display.", LogKind::Err);
            self.set_status("no player found.", LogKind::Err);
        }
    }

    fn copy_results(&mut self, ctx: &egui::Context) {
        if let Some(o) = &self.output {
            if o.ok() {
                ctx.copy_text(o.standings.clone());
                self.copied_at = Some(Instant::now());
                self.log("results copied to the clipboard.", LogKind::Ok);
            }
        }
    }

    fn poll(&mut self, ctx: &egui::Context) {
        let mut done: Option<Analysis> = None;
        if let Some(rx) = &self.rx {
            while let Ok(msg) = rx.try_recv() {
                match msg {
                    Msg::Progress(d, t, _name) => self.progress = (d, t),
                    Msg::Done(a) => done = Some(a),
                }
            }
        }
        if let Some(a) = done {
            self.finish(a, ctx);
        }
    }

    fn marker(&self, p: &std::path::Path) -> &'static str {
        let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
        self.file_status.get(&name).copied().unwrap_or("  ")
    }
}

// ---- rich results text -----------------------------------------------------------------------------
fn standings_job(text: &str) -> egui::text::LayoutJob {
    let mut job = egui::text::LayoutJob::default();
    let font = FontId::monospace(13.0);
    let add = |job: &mut egui::text::LayoutJob, s: &str, color: Color32| {
        job.append(s, 0.0, egui::TextFormat { font_id: font.clone(), color, ..Default::default() });
    };
    for line in text.lines() {
        let t = line.trim_end();
        if t.is_empty() {
        } else if t.starts_with("Match Results") || (t.starts_with("Game ") && t.ends_with(" Results")) {
            add(&mut job, t, CYAN);
        } else if t.starts_with("Green Team:") {
            add(&mut job, t, GREEN);
        } else if t.starts_with("Red Team:") {
            add(&mut job, t, RED);
        } else if t.len() > 10 && t.as_bytes()[2] == b'/' && t.as_bytes()[5] == b'/' && t.contains(" at ") {
            add(&mut job, t, DIM);
        } else if let Some((name, rest)) = split_player_line(t) {
            add(&mut job, name, Color32::WHITE);
            add(&mut job, ": ", DIM);
            // rest = "k/d (r KD)[ (played a/b)][ medal]"
            let kd_end = rest.find(' ').unwrap_or(rest.len());
            add(&mut job, &rest[..kd_end], YELLOW);
            // the copied text keeps the real emoji; the window shows readable tags instead
            // (the bundled font has no colour emoji)
            let tail = &rest[kd_end..];
            let (body, medal) = if let Some(b) = tail.strip_suffix(crate::formatter::CROWN) {
                (b, Some(("CROWN", Color32::from_rgb(0xff, 0xd7, 0x00))))
            } else if let Some(b) = tail.strip_suffix(crate::formatter::SILVER) {
                (b, Some(("SILVER", Color32::from_rgb(0xc0, 0xc8, 0xd4))))
            } else {
                (tail, None)
            };
            add(&mut job, body, MUTED);
            if let Some((tag, c)) = medal {
                add(&mut job, &format!("[{tag}]"), c);
            }
        } else {
            add(&mut job, t, TEXT);
        }
        add(&mut job, "\n", TEXT);
    }
    job
}

/// Splits "name: 12/5 (2.40 KD) ..." into (name, "12/5 (2.40 KD) ...").
fn split_player_line(line: &str) -> Option<(&str, &str)> {
    let mut from = 0;
    while let Some(i) = line[from..].find(": ") {
        let at = from + i;
        let rest = &line[at + 2..];
        let digits_a = rest.chars().take_while(|c| c.is_ascii_digit()).count();
        if digits_a > 0 && rest[digits_a..].starts_with('/') {
            let after = &rest[digits_a + 1..];
            let digits_b = after.chars().take_while(|c| c.is_ascii_digit()).count();
            if digits_b > 0 && after[digits_b..].starts_with(" (") {
                return Some((&line[..at], rest));
            }
        }
        from = at + 2;
    }
    None
}

// ---- UI ------------------------------------------------------------------------------------------------
fn button(ui: &mut egui::Ui, text: &str, color: Color32) -> egui::Response {
    ui.add(egui::Button::new(RichText::new(text).color(color)).min_size(egui::vec2(0.0, 26.0)))
}

fn toggle(ui: &mut egui::Ui, value: &mut bool, text: &str) -> bool {
    let mark = if *value { "[x]" } else { "[ ]" };
    let color = if *value { GREEN } else { MUTED };
    let r = ui.add(
        egui::Label::new(RichText::new(format!("{mark} {text}")).color(if *value { TEXT } else { MUTED }).background_color(Color32::TRANSPARENT))
            .sense(egui::Sense::click()),
    );
    let _ = color;
    if r.clicked() {
        *value = !*value;
        return true;
    }
    r.on_hover_cursor(egui::CursorIcon::PointingHand);
    false
}

impl eframe::App for App {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.poll(ctx);
        if self.pending_start {
            self.pending_start = false;
            self.start_analysis(ctx);
        }
        // drag and drop
        let dropped: Vec<PathBuf> = ctx.input(|i| i.raw.dropped_files.iter().filter_map(|f| f.path.clone()).collect());
        if !dropped.is_empty() {
            self.add_files(dropped);
        }
        // shortcuts
        let (open, delete, analyze, clear_log) = ctx.input(|i| {
            (
                i.modifiers.command && i.key_pressed(egui::Key::O),
                i.key_pressed(egui::Key::Delete),
                i.key_pressed(egui::Key::F5) || (i.modifiers.command && i.key_pressed(egui::Key::Enter)),
                i.modifiers.command && i.key_pressed(egui::Key::L),
            )
        });
        if open {
            self.pick_files();
        }
        if delete {
            self.remove_selected();
        }
        if analyze {
            self.start_analysis(ctx);
        }
        if clear_log {
            self.console.clear();
        }
        if self.running.is_some() || !self.engine_ready.load(Ordering::SeqCst) || self.copied_at.is_some() {
            ctx.request_repaint_after(std::time::Duration::from_millis(100));
        }

        // ---- header -------------------------------------------------------------------------
        egui::TopBottomPanel::top("header").frame(egui::Frame::new().fill(BG).inner_margin(egui::Margin::symmetric(18, 12))).show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.vertical(|ui| {
                    ui.horizontal(|ui| {
                        ui.spacing_mut().item_spacing.x = 0.0;
                        ui.label(RichText::new(">").size(22.0).color(GREEN));
                        ui.label(RichText::new(" kd_ocr").size(22.0).color(TEXT));
                        let blink = (self.start.elapsed().as_millis() / 530) % 2 == 0;
                        ui.label(RichText::new("_").size(22.0).color(if blink { GREEN } else { BG }));
                    });
                    ui.label(RichText::new(format!("v{APP_VERSION} made by o293A")).size(11.0).color(DIM));
                });
                ui.add_space(18.0);
                ui.label(RichText::new("# RESULTS screenshots  \u{2192}  Scoreboards").size(11.0).color(MUTED));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    let ready = self.engine_ready.load(Ordering::SeqCst);
                    let (color, label) = if ready { (GREEN, "OCR engine: ready") } else { (YELLOW, "OCR engine: loading\u{2026}") };
                    ui.label(RichText::new(label).size(11.0).color(if ready { GREEN } else { MUTED }));
                    let pulse = if ready { 1.0 } else { 0.5 + 0.5 * ((self.start.elapsed().as_secs_f32() * 4.0).sin()) };
                    let (rect, _) = ui.allocate_exact_size(egui::vec2(12.0, 12.0), egui::Sense::hover());
                    ui.painter().circle_filled(rect.center(), 5.0, color.gamma_multiply(pulse));
                });
            });
        });

        // ---- status bar ---------------------------------------------------------------------
        egui::TopBottomPanel::bottom("status")
            .frame(egui::Frame::new().fill(PANEL).stroke(Stroke::new(1.0_f32, BORDER)).inner_margin(egui::Margin::symmetric(12, 5)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.label(RichText::new(&self.status.0).size(11.0).color(kind_color(self.status.1)));
                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if let Some(t) = self.running {
                            let s = t.elapsed().as_secs();
                            ui.label(RichText::new(format!("{}:{:02}", s / 60, s % 60)).size(11.0).color(DIM));
                        }
                        ui.label(RichText::new("Ctrl+O add \u{b7} Del remove \u{b7} F5 analyze").size(11.0).color(DIM));
                    });
                });
            });

        // ---- progress bar -------------------------------------------------------------------
        egui::TopBottomPanel::top("progress").frame(egui::Frame::new().fill(BG)).show_separator_line(false).show(ctx, |ui| {
            let frac = if self.running.is_some() && self.progress.1 > 0 { self.progress.0 as f32 / self.progress.1 as f32 } else { 0.0 };
            let (rect, _) = ui.allocate_exact_size(egui::vec2(ui.available_width(), 3.0), egui::Sense::hover());
            ui.painter().rect_filled(rect, 0.0, PANEL2);
            if self.running.is_some() {
                let mut r = rect;
                r.set_width(rect.width() * frac.max(0.03));
                ui.painter().rect_filled(r, 0.0, GREEN);
            }
        });

        // ---- left column ----------------------------------------------------------------------
        egui::SidePanel::left("left")
            .resizable(true)
            .default_width(400.0)
            .min_width(320.0)
            .frame(egui::Frame::new().fill(BG).inner_margin(egui::Margin { left: 18, right: 9, top: 12, bottom: 12 }))
            .show(ctx, |ui| self.left_ui(ui, ctx));

        // ---- right column ---------------------------------------------------------------------
        egui::CentralPanel::default().frame(egui::Frame::new().fill(BG).inner_margin(egui::Margin { left: 9, right: 18, top: 12, bottom: 12 })).show(ctx, |ui| {
            self.right_ui(ui, ctx);
        });
    }
}

impl App {
    fn pick_files(&mut self) {
        if let Some(files) = rfd::FileDialog::new().set_title("Add screenshots").add_filter("images", &cfg::SUPPORTED_EXTENSIONS).pick_files() {
            self.add_files(files);
        }
    }

    fn remove_selected(&mut self) {
        if self.selected.is_empty() {
            return;
        }
        let keep: Vec<PathBuf> =
            self.files.iter().enumerate().filter(|(i, _)| !self.selected.contains(i)).map(|(_, p)| p.clone()).collect();
        let removed = self.files.len() - keep.len();
        self.files = keep;
        self.selected.clear();
        self.log(&format!("{removed} file(s) removed."), LogKind::Info);
    }

    fn left_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.horizontal(|ui| {
            section_title(ui, "files");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                ui.label(RichText::new(self.files.len().to_string()).size(11.0).color(MUTED));
            });
        });
        let list_height = (ui.available_height() - 305.0).max(100.0);
        card().show(ui, |ui| {
            ui.set_min_height(list_height);
            ui.set_max_height(list_height);
            ui.set_width(ui.available_width());
            if self.files.is_empty() {
                ui.centered_and_justified(|ui| {
                    ui.label(RichText::new("no files\n\ndrop screenshots here\nor Ctrl+O to add").size(11.0).color(DIM));
                });
            } else {
                egui::ScrollArea::vertical().auto_shrink([false, false]).show(ui, |ui| {
                    let mut clicked: Option<(usize, bool)> = None;
                    for (i, p) in self.files.iter().enumerate() {
                        let name = p.file_name().map(|n| n.to_string_lossy().to_string()).unwrap_or_default();
                        let mark = self.marker(p);
                        let color = match mark {
                            "ok" => GREEN,
                            "++" => CYAN,
                            "??" => YELLOW,
                            "--" => RED,
                            _ => TEXT,
                        };
                        let sel = self.selected.contains(&i);
                        let text = if mark.trim().is_empty() { format!("    {name}") } else { format!("[{mark}] {name}") };
                        let r = ui.selectable_label(sel, RichText::new(text).color(if sel { GREEN } else { color }));
                        if r.clicked() {
                            clicked = Some((i, ui.input(|inp| inp.modifiers.command || inp.modifiers.shift)));
                        }
                    }
                    if let Some((i, multi)) = clicked {
                        if !multi {
                            self.selected.clear();
                        }
                        if !self.selected.remove(&i) || !multi {
                            self.selected.insert(i);
                        }
                    }
                });
            }
        });
        ui.add_space(6.0);
        ui.columns(2, |cols| {
            if button(&mut cols[0], "+ images", TEXT).clicked() {
                self.pick_files();
            }
            if button(&mut cols[1], "+ folder", TEXT).clicked() {
                if let Some(dir) = rfd::FileDialog::new().set_title("Add a folder of screenshots").pick_folder() {
                    self.add_files(vec![dir]);
                }
            }
        });
        ui.columns(2, |cols| {
            if button(&mut cols[0], "- remove", TEXT).clicked() {
                self.remove_selected();
            }
            if button(&mut cols[1], "x clear", RED).clicked() {
                self.files.clear();
                self.selected.clear();
                self.file_status.clear();
                self.log("file list cleared.", LogKind::Info);
            }
        });
        ui.add_space(8.0);
        section_title(ui, "options");
        card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            let mut changed = false;
            changed |= toggle(ui, &mut self.settings.per_match, "one game per image (stacked)");
            changed |= toggle(ui, &mut self.settings.green_team_is_game_winner, "green team = game winner");
            if !self.settings.green_team_is_game_winner {
                ui.label(RichText::new("   green = small green bands, red = small red bands").size(11.0).color(DIM));
            }
            toggle(ui, &mut self.settings.auto_copy, "auto-copy results");
            if changed {
                self.rebuild_output();
            }
            ui.add_space(4.0);
            ui.horizontal(|ui| {
                ui.label(RichText::new(format!("startup settings: {}", cfg::settings_path().file_name().unwrap().to_string_lossy())).size(11.0).color(MUTED));
                ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                    if ui.add(egui::Button::new(RichText::new("edit").size(11.0))).clicked() {
                        let path = cfg::settings_path();
                        if !path.exists() {
                            let _ = Settings::load(None);
                        }
                        open_in_default_app(&path);
                    }
                });
            });
        });
        ui.add_space(10.0);
        let busy = self.running.is_some();
        let label = if busy { "...  A N A L Y Z I N G" } else { ">  A N A L Y Z E" };
        let b = egui::Button::new(RichText::new(label).size(15.0).strong().color(Color32::from_rgb(0x04, 0x14, 0x0b)))
            .fill(if busy { Color32::from_rgb(0x17, 0x66, 0x40) } else { Color32::from_rgb(0x1c, 0x7a, 0x4c) })
            .min_size(egui::vec2(ui.available_width(), 38.0));
        if ui.add_enabled(!busy, b).clicked() {
            self.start_analysis(ctx);
        }
    }

    fn right_ui(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        ui.horizontal(|ui| {
            section_title(ui, "results");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                let can = self.output.as_ref().map(|o| o.ok()).unwrap_or(false);
                let flash = self.copied_at.map(|t| t.elapsed().as_secs_f32() < 1.2).unwrap_or(false);
                if !flash {
                    self.copied_at = None;
                }
                let label = if flash { "COPIED" } else { "COPY" };
                if ui.add_enabled(can, egui::Button::new(RichText::new(label).size(11.0).color(GREEN))).clicked() {
                    self.copy_results(ctx);
                }
            });
        });
        let total = ui.available_height();
        let results_h = (total * 0.55 - 30.0).max(120.0);
        card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.set_min_height(results_h);
            ui.set_max_height(results_h);
            egui::ScrollArea::vertical().id_salt("results").auto_shrink([false, false]).show(ui, |ui| match &self.output {
                Some(o) if o.ok() => {
                    ui.add(egui::Label::new(standings_job(&o.standings)).selectable(true));
                }
                _ => {
                    ui.label(RichText::new("# results appear here after analysis").size(12.0).color(DIM));
                }
            });
        });
        ui.add_space(10.0);
        ui.horizontal(|ui| {
            section_title(ui, "console");
            ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                if ui.add(egui::Button::new(RichText::new("clear").size(11.0))).clicked() {
                    self.console.clear();
                }
                if ui.add_enabled(self.output.is_some(), egui::Button::new(RichText::new("report").size(11.0))).clicked() {
                    if let Some(o) = &self.output {
                        let report = o.report.clone();
                        self.log("report:", LogKind::Head);
                        for l in report.lines() {
                            self.console.push((String::new(), LogKind::Faint, l.to_string()));
                        }
                    }
                }
            });
        });
        card().show(ui, |ui| {
            ui.set_width(ui.available_width());
            ui.set_min_height(ui.available_height() - 4.0);
            egui::ScrollArea::vertical().id_salt("console").auto_shrink([false, false]).stick_to_bottom(true).show(ui, |ui| {
                for (ts, kind, msg) in &self.console {
                    ui.horizontal_wrapped(|ui| {
                        ui.spacing_mut().item_spacing.x = 4.0;
                        if !ts.is_empty() {
                            ui.label(RichText::new(ts).size(11.0).color(DIM));
                        }
                        ui.label(RichText::new(msg).size(11.0).color(kind_color(*kind)));
                    });
                }
            });
        });
    }
}

#[allow(dead_code)]
fn _unused(_: &ScreenshotResult) {}
