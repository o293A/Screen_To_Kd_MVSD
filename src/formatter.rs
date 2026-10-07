//! Builds the paste-ready match-results text and a plain-text diagnostic report.
//!
//! ```text
//! Match Results (from 10 rounds)
//!
//! Red Team: (5/10 Game Win)
//! iloveamber: 57/59 (0.97 KD) 🥈
//! folde_r: 56/49 (1.14 KD)
//!
//! Green Team: (5/10 Game Win)
//! DNSSREVENGE: 67/49 (1.37 KD) 👑
//!
//! 06/10/2026 at 00:29
//! ```
//!
//! * the RED team is listed first, the GREEN team second;
//! * "(5/10 Game Win)" = games won by the team / number of games; "(played 2/10)" for a player
//!   who is not on every game of his team;
//! * 👑 goes to the player with the MOST KILLS (then the FEWEST DEATHS); players still tied
//!   all get it, but only WITHIN a team. If the perfect tie spans both teams, the tied players
//!   of the winning team get 👑 and those of the losing team get 🥈;
//! * 🥈 follows the same rule inside the team opposed to the crown (and only there);
//! * inside a team: most kills first, then fewest deaths, then alphabetical order;
//! * with PER_MATCH, one block per game ("Game 1 Results"), teams marked (Win) / (Loss).

use crate::teams::{display_ratio, Entry, Team, TeamAssignment};
use std::collections::HashMap;

pub const CROWN: &str = "\u{1F451}";
pub const SILVER: &str = "\u{1F948}";

pub fn format_kd_ratio(kills: u32, deaths: u32) -> String {
    format!("{:.2}", display_ratio(kills, deaths))
}

pub fn format_timestamp() -> String {
    chrono::Local::now().format("%d/%m/%Y at %H:%M").to_string()
}

fn display_order(entries: &[Entry]) -> Vec<Entry> {
    let mut v = entries.to_vec();
    v.sort_by(|a, b| {
        b.kills
            .cmp(&a.kills)
            .then(a.deaths.cmp(&b.deaths))
            .then(a.name.to_lowercase().cmp(&b.name.to_lowercase()))
            .then(a.name.cmp(&b.name))
    });
    v
}

/// Names sharing the best (kills, then deaths) of these entries; nobody when the best has no kill.
fn top(entries: &[Entry]) -> Vec<String> {
    let Some(best) = entries.iter().min_by_key(|e| (std::cmp::Reverse(e.kills), e.deaths)) else { return vec![] };
    if best.kills == 0 {
        return vec![];
    }
    entries.iter().filter(|e| (e.kills, e.deaths) == (best.kills, best.deaths)).map(|e| e.name.clone()).collect()
}

/// {(team, name): medal}. `winner` = the team that won (None = unknown).
pub fn medals(green: &[Entry], red: &[Entry], winner: Option<Team>) -> HashMap<(Team, String), &'static str> {
    let everyone: Vec<(Team, &Entry)> =
        green.iter().map(|e| (Team::Green, e)).chain(red.iter().map(|e| (Team::Red, e))).collect();
    let mut out = HashMap::new();
    let Some(best) = everyone.iter().map(|(_, e)| e).min_by_key(|e| (std::cmp::Reverse(e.kills), e.deaths)) else { return out };
    if best.kills == 0 {
        return out;
    }
    let key = (best.kills, best.deaths);
    let crowned: Vec<(Team, String)> =
        everyone.iter().filter(|(_, e)| (e.kills, e.deaths) == key).map(|(t, e)| (*t, e.name.clone())).collect();
    let has = |team: Team| crowned.iter().any(|(t, _)| *t == team);
    if has(Team::Green) && has(Team::Red) {
        // Perfect tie spanning both teams: crown for the winning team, silver for the losing one.
        match winner {
            Some(w) => {
                for (t, n) in &crowned {
                    out.insert((*t, n.clone()), if *t == w { CROWN } else { SILVER });
                }
            }
            None => {
                for (t, n) in &crowned {
                    out.insert((*t, n.clone()), CROWN);
                }
            }
        }
        return out;
    }
    let crown_team = crowned[0].0;
    for (t, n) in &crowned {
        out.insert((*t, n.clone()), CROWN);
    }
    let (opposed_team, opposed) = if crown_team == Team::Green { (Team::Red, red) } else { (Team::Green, green) };
    for name in top(opposed) {
        out.insert((opposed_team, name), SILVER);
    }
    out
}

fn player_line(e: &Entry, rounds: Option<usize>, medal: &str) -> String {
    let mut line = format!("{}: {}/{} ({} KD)", e.name, e.kills, e.deaths, format_kd_ratio(e.kills, e.deaths));
    if let Some(r) = rounds {
        if r > 0 && (e.played as usize) < r {
            line.push_str(&format!(" (played {}/{})", e.played, r));
        }
    }
    if !medal.is_empty() {
        line.push(' ');
        line.push_str(medal);
    }
    line
}

fn team_block(
    label: &str,
    detail: &str,
    team: Team,
    entries: &[Entry],
    med: &HashMap<(Team, String), &'static str>,
    rounds: Option<usize>,
) -> String {
    if entries.is_empty() {
        return String::new();
    }
    let mut lines = vec![format!("{label}: ({detail})")];
    for e in display_order(entries) {
        let m = med.get(&(team, e.name.clone())).copied().unwrap_or("");
        lines.push(player_line(&e, rounds, m));
    }
    lines.join("\n")
}

pub fn format_combined(a: &TeamAssignment) -> String {
    if a.entries.is_empty() {
        return String::new();
    }
    let rounds = a.rounds;
    let (green, red) = (a.team_entries(Team::Green), a.team_entries(Team::Red));
    let med = medals(&green, &red, a.winning_team());
    let blocks = [
        team_block("Red Team", &format!("{}/{} Game Win", a.red_wins(), rounds), Team::Red, &red, &med, Some(rounds)),
        team_block("Green Team", &format!("{}/{} Game Win", a.green_wins(), rounds), Team::Green, &green, &med, Some(rounds)),
    ];
    let body: Vec<&str> = blocks.iter().filter(|b| !b.is_empty()).map(|s| s.as_str()).collect();
    if body.is_empty() {
        return String::new();
    }
    let word = if rounds == 1 { "round" } else { "rounds" };
    format!("Match Results (from {rounds} {word})\n\n{}\n\n{}", body.join("\n\n"), format_timestamp())
}

pub fn format_per_match(a: &TeamAssignment) -> String {
    let mut blocks = Vec::new();
    for (i, screen) in a.per_screen.iter().enumerate() {
        let mk = |ps: &[crate::model::PlayerScore]| -> Vec<Entry> {
            ps.iter().map(|p| Entry { name: p.name.clone(), kills: p.kills, deaths: p.deaths, played: 1 }).collect()
        };
        let (green, red) = (mk(&screen.green), mk(&screen.red));
        let (gt, rt, winner) = match screen.green_won {
            None => ("Result unknown", "Result unknown", None),
            Some(true) => ("Win", "Loss", Some(Team::Green)),
            Some(false) => ("Loss", "Win", Some(Team::Red)),
        };
        let med = medals(&green, &red, winner);
        let parts = [
            team_block("Red Team", rt, Team::Red, &red, &med, None),
            team_block("Green Team", gt, Team::Green, &green, &med, None),
        ];
        let body: Vec<&str> = parts.iter().filter(|p| !p.is_empty()).map(|s| s.as_str()).collect();
        if !body.is_empty() {
            blocks.push(format!("Game {} Results\n\n{}", i + 1, body.join("\n\n")));
        }
    }
    if blocks.is_empty() {
        return String::new();
    }
    format!("{}\n\n{}", blocks.join("\n\n"), format_timestamp())
}

pub fn format_standings(a: &TeamAssignment, per_match: bool) -> String {
    if per_match {
        format_per_match(a)
    } else {
        format_combined(a)
    }
}

/// Plain-text diagnostic report (totals, decision reasoning).
pub fn format_report(a: &TeamAssignment) -> String {
    if a.entries.is_empty() {
        return "No player was detected.".to_string();
    }
    let mut lines = vec!["Combined totals:".to_string()];
    for (team, label) in [(Team::Green, "green"), (Team::Red, "red")] {
        for e in a.team_entries(team) {
            lines.push(format!("  {label:<5} {:<20} {}/{}  played {}/{}", e.name, e.kills, e.deaths, e.played, a.rounds));
        }
    }
    for (i, g) in a.groups.iter().enumerate() {
        let title = if a.groups.len() == 1 { "Teams".to_string() } else { format!("Group {}", i + 1) };
        lines.push(format!("{title} ({} game(s)):", g.screens));
        lines.push(format!("  green: {} - {} win(s)", g.green.join(", "), g.green_wins));
        lines.push(format!("  red:   {} - {} win(s)", g.red.join(", "), g.red_wins));
        lines.push(format!("  green team chosen: {}", g.reason.describe()));
    }
    lines.join("\n")
}
