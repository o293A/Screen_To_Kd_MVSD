//! Medal rules, team naming option, settings file.

use kd_ocr::config::Settings;
use kd_ocr::formatter::{medals, CROWN, SILVER};
use kd_ocr::model::{Color, PlayerScore, ScreenshotResult};
use kd_ocr::teams::{Entry, Team, TeamBuilder};

fn e(name: &str, k: u32, d: u32) -> Entry {
    Entry { name: name.into(), kills: k, deaths: d, played: 1 }
}

#[test]
fn crown_goes_to_most_kills_then_fewest_deaths() {
    let m = medals(&[e("Bob", 15, 5), e("Jean", 15, 2)], &[e("Zed", 9, 1)], Some(Team::Green));
    assert_eq!(m.get(&(Team::Green, "Jean".into())), Some(&CROWN));
    assert_eq!(m.get(&(Team::Green, "Bob".into())), None);
    assert_eq!(m.get(&(Team::Red, "Zed".into())), Some(&SILVER));
}

#[test]
fn perfect_tie_inside_one_team_crowns_everybody() {
    let m = medals(&[e("A", 10, 2), e("B", 10, 2), e("C", 3, 3)], &[e("Z", 5, 5)], Some(Team::Red));
    assert_eq!(m.get(&(Team::Green, "A".into())), Some(&CROWN));
    assert_eq!(m.get(&(Team::Green, "B".into())), Some(&CROWN));
    assert_eq!(m.get(&(Team::Red, "Z".into())), Some(&SILVER));
}

#[test]
fn perfect_tie_across_both_teams_crown_to_winner_silver_to_loser() {
    let green = [e("G1", 12, 3), e("G2", 1, 1)];
    let red = [e("R1", 12, 3), e("R2", 12, 3)];
    let m = medals(&green, &red, Some(Team::Green));
    assert_eq!(m.get(&(Team::Green, "G1".into())), Some(&CROWN));
    assert_eq!(m.get(&(Team::Red, "R1".into())), Some(&SILVER));
    assert_eq!(m.get(&(Team::Red, "R2".into())), Some(&SILVER));
    let m = medals(&green, &red, Some(Team::Red));
    assert_eq!(m.get(&(Team::Green, "G1".into())), Some(&SILVER));
    assert_eq!(m.get(&(Team::Red, "R1".into())), Some(&CROWN));
}

#[test]
fn nobody_is_crowned_without_a_kill() {
    assert!(medals(&[e("A", 0, 3)], &[e("B", 0, 1)], None).is_empty());
}

fn p(name: &str, color: Color, k: u32, d: u32) -> PlayerScore {
    PlayerScore { name: name.into(), color, kills: k, deaths: d, row_index: 0, name_score: 100.0 }
}

fn game(players: Vec<PlayerScore>, winner: Color) -> ScreenshotResult {
    ScreenshotResult { source: "x".into(), players, winner_color: Some(winner), ..Default::default() }
}

#[test]
fn green_team_option() {
    // the capturing player's team (Nemo, green rows) LOSES the only game
    let g = game(vec![p("Nemo", Color::Green, 5, 6), p("cata", Color::Red, 8, 2)], Color::Red);
    for (winner_option, nemo_team, cata_team) in [(true, Team::Red, Team::Green), (false, Team::Green, Team::Red)] {
        let mut b = TeamBuilder::new(winner_option);
        b.add_screenshot(&g);
        let a = b.resolve();
        let team_of = |n: &str| a.entries.iter().find(|(_, e)| e.name == n).unwrap().0;
        assert_eq!(team_of("Nemo"), nemo_team);
        assert_eq!(team_of("cata"), cata_team);
    }
}

#[test]
fn settings_defaults_and_legacy_keys() {
    let d = Settings::default();
    assert!(d.green_team_is_game_winner && !d.per_match && d.auto_copy && !d.deduplicate_images);
    let s = Settings::parse("USE_TRACKED_LIST = false\nWINNER_IS_GREEN = true\nGREEN_TEAM_IS_GAME_WINNER = off\nPER_MATCH = yes\n");
    assert!(!s.green_team_is_game_winner && s.per_match);
    assert_eq!(Settings::parse(&d.file_text()), d);
}
