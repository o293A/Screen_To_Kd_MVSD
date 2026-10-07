//! Team reconstruction, score merging, and green/red team resolution.
//!
//! The unit of reasoning is the SCREENSHOT. Every screenshot shows two groups of rows: the
//! GREEN rows (always the capturing player's own team) and the RED rows (his opponents). Which
//! real team a group is changes from one screenshot to the next, so the groups of different
//! screenshots are linked together: a player shown in the same colour on two screenshots says
//! "same team", in opposite colours "opposite teams". Every screenshot gets one bit ("is my
//! green-row group side 0 or side 1?"), chosen to satisfy as many observations as possible.
//!
//! A player is identified by (team, name): somebody seen in the red team on 2 screenshots and
//! in the green team on 1 is TWO players of the output.
//!
//! Which side is the GREEN team depends on `green_team_is_game_winner`:
//! * true  (default): the side that WON the most games (banner colour only, never its text);
//!   on a tie the side holding the crown, then the better KD ratio, then the first side seen.
//! * false: the side shown with the most small GREEN row bands; the result of the games plays
//!   no role (same tie-breaks as above if the green bands are equally frequent).

use crate::model::{Color, PlayerScore, ScreenshotResult};
use crate::warn_log;
use std::collections::BTreeMap;

#[derive(Clone, Copy, PartialEq, Eq, Debug, Hash)]
pub enum Team {
    Green,
    Red,
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum Reason {
    Wins,
    GreenRows,
    Crown,
    BestKd,
    PerfectTie,
}

impl Reason {
    pub fn describe(self) -> &'static str {
        match self {
            Reason::Wins => "won the most games",
            Reason::GreenRows => "shown with the green row bands",
            Reason::Crown => "tie: the crown holder's team",
            Reason::BestKd => "tie: crown holders on both sides, best KD ratio",
            Reason::PerfectTie => "perfect tie: first side encountered",
        }
    }
}

#[derive(Clone, Debug)]
pub struct GroupDecision {
    pub green: Vec<String>,
    pub red: Vec<String>,
    pub green_wins: u32,
    pub red_wins: u32,
    pub screens: usize,
    pub reason: Reason,
}

#[derive(Clone, Debug, Default)]
pub struct ScreenAssignment {
    pub source: String,
    pub green: Vec<PlayerScore>,
    pub red: Vec<PlayerScore>,
    /// None = unknown
    pub green_won: Option<bool>,
}

#[derive(Clone, Debug)]
pub struct Entry {
    pub name: String,
    pub kills: u32,
    pub deaths: u32,
    pub played: u32,
}

#[derive(Clone, Debug, Default)]
pub struct TeamAssignment {
    /// (team, name) -> entry, in order of first appearance.
    pub entries: Vec<(Team, Entry)>,
    pub per_screen: Vec<ScreenAssignment>,
    pub groups: Vec<GroupDecision>,
    /// Number of games analyzed (the "10" of "from 10 rounds").
    pub rounds: usize,
}

impl TeamAssignment {
    pub fn team_entries(&self, team: Team) -> Vec<Entry> {
        self.entries.iter().filter(|(t, _)| *t == team).map(|(_, e)| e.clone()).collect()
    }
    pub fn green_wins(&self) -> u32 {
        self.groups.iter().map(|g| g.green_wins).sum()
    }
    pub fn red_wins(&self) -> u32 {
        self.groups.iter().map(|g| g.red_wins).sum()
    }
    /// The team that won the match overall: most games won, then best total KD ratio.
    pub fn winning_team(&self) -> Option<Team> {
        let (gw, rw) = (self.green_wins(), self.red_wins());
        if gw != rw {
            return Some(if gw > rw { Team::Green } else { Team::Red });
        }
        let ratio = |t: Team| {
            let (k, d) = self.entries.iter().filter(|(x, _)| *x == t).fold((0u32, 0u32), |a, (_, e)| (a.0 + e.kills, a.1 + e.deaths));
            display_ratio(k, d)
        };
        let (rg, rr) = (ratio(Team::Green), ratio(Team::Red));
        if rg > rr {
            Some(Team::Green)
        } else if rr > rg {
            Some(Team::Red)
        } else {
            None
        }
    }
}

pub fn display_ratio(kills: u32, deaths: u32) -> f64 {
    if deaths > 0 {
        kills as f64 / deaths as f64
    } else {
        kills as f64
    }
}

struct Screen {
    source: String,
    greens: Vec<PlayerScore>,
    reds: Vec<PlayerScore>,
    winner: Option<Color>,
}

impl Screen {
    fn colors(&self) -> BTreeMap<&str, Color> {
        let mut m = BTreeMap::new();
        for p in &self.greens {
            m.insert(p.name.as_str(), Color::Green);
        }
        for p in &self.reds {
            m.insert(p.name.as_str(), Color::Red);
        }
        m
    }
}

pub struct TeamBuilder {
    screens: Vec<Screen>,
    green_team_is_game_winner: bool,
}

type Edges = BTreeMap<(usize, usize), (i64, i64)>;

struct XorShift(u64);
impl XorShift {
    fn next_bit(&mut self) -> u8 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 & 1) as u8
    }
}

impl TeamBuilder {
    pub fn new(green_team_is_game_winner: bool) -> Self {
        TeamBuilder { screens: Vec::new(), green_team_is_game_winner }
    }

    /// Records one screenshot (ignored when it has no player).
    pub fn add_screenshot(&mut self, result: &ScreenshotResult) {
        let mut seen: Vec<&str> = Vec::new();
        let (mut greens, mut reds) = (Vec::new(), Vec::new());
        for p in &result.players {
            if seen.contains(&p.name.as_str()) {
                warn_log!("{}: \"{}\" appears twice on the same image: duplicate skipped.", result.source, p.name);
                continue;
            }
            seen.push(&p.name);
            if p.color == Color::Green {
                greens.push(p.clone());
            } else {
                reds.push(p.clone());
            }
        }
        if !greens.is_empty() || !reds.is_empty() {
            self.screens.push(Screen { source: result.source.clone(), greens, reds, winner: result.winner_color });
        }
    }

    fn edges(&self) -> Edges {
        let colors: Vec<_> = self.screens.iter().map(|s| s.colors()).collect();
        let mut edges = Edges::new();
        for i in 0..colors.len() {
            for j in i + 1..colors.len() {
                let (mut same, mut opposed) = (0, 0);
                for (name, color) in &colors[i] {
                    match colors[j].get(name) {
                        Some(o) if o == color => same += 1,
                        Some(_) => opposed += 1,
                        None => {}
                    }
                }
                if same > 0 || opposed > 0 {
                    edges.insert((i, j), (same, opposed));
                }
            }
        }
        edges
    }

    fn components(count: usize, edges: &Edges) -> Vec<Vec<usize>> {
        let mut adjacent = vec![Vec::new(); count];
        for &(i, j) in edges.keys() {
            adjacent[i].push(j);
            adjacent[j].push(i);
        }
        let mut seen = vec![false; count];
        let mut comps = Vec::new();
        for start in 0..count {
            if seen[start] {
                continue;
            }
            let (mut stack, mut members) = (vec![start], Vec::new());
            seen[start] = true;
            while let Some(node) = stack.pop() {
                members.push(node);
                for &o in &adjacent[node] {
                    if !seen[o] {
                        seen[o] = true;
                        stack.push(o);
                    }
                }
            }
            members.sort();
            comps.push(members);
        }
        comps
    }

    /// One bit per screenshot maximising the number of satisfied observations.
    fn solve_bits(members: &[usize], edges: &Edges) -> BTreeMap<usize, u8> {
        if members.len() == 1 {
            return BTreeMap::from([(members[0], 0)]);
        }
        let mut weights: BTreeMap<usize, Vec<(usize, i64, i64)>> = members.iter().map(|&m| (m, vec![])).collect();
        for (&(i, j), &(same, opposed)) in edges {
            if weights.contains_key(&i) && weights.contains_key(&j) {
                weights.get_mut(&i).unwrap().push((j, same, opposed));
                weights.get_mut(&j).unwrap().push((i, same, opposed));
            }
        }
        let satisfied = |bits: &BTreeMap<usize, u8>, node: usize, value: u8| -> i64 {
            weights[&node].iter().map(|&(o, s, p)| if bits[&o] == value { s } else { p }).sum()
        };
        let score = |bits: &BTreeMap<usize, u8>| -> i64 { members.iter().map(|&n| satisfied(bits, n, bits[&n])).sum::<i64>() / 2 };
        let climb = |mut bits: BTreeMap<usize, u8>| -> BTreeMap<usize, u8> {
            for _ in 0..1000 {
                let mut improved = false;
                for &n in members {
                    if satisfied(&bits, n, 1 - bits[&n]) > satisfied(&bits, n, bits[&n]) {
                        let v = 1 - bits[&n];
                        bits.insert(n, v);
                        improved = true;
                    }
                }
                if !improved {
                    break;
                }
            }
            bits
        };
        // start: spanning tree following the strongest relation of each link
        let mut tree = BTreeMap::from([(members[0], 0u8)]);
        let mut queue = std::collections::VecDeque::from([members[0]]);
        while let Some(node) = queue.pop_front() {
            let mut neigh = weights[&node].clone();
            neigh.sort();
            for (o, s, p) in neigh {
                if !tree.contains_key(&o) {
                    tree.insert(o, if s >= p { tree[&node] } else { 1 - tree[&node] });
                    queue.push_back(o);
                }
            }
        }
        let mut best = climb(tree);
        let mut best_score = score(&best);
        let mut rng = XorShift(0x9E3779B97F4A7C15 ^ 12345);
        if members.len() > 2 {
            for _ in 0..60 {
                let cand = climb(members.iter().map(|&m| (m, rng.next_bit())).collect());
                let sc = score(&cand);
                if sc > best_score {
                    best = cand;
                    best_score = sc;
                }
            }
        }
        let first = best[&members[0]];
        members.iter().map(|&m| (m, best[&m] ^ first)).collect()
    }

    pub fn resolve(&self) -> TeamAssignment {
        let mut assignment = TeamAssignment { rounds: self.screens.len(), ..Default::default() };
        if self.screens.is_empty() {
            return assignment;
        }
        let edges = self.edges();
        let comps = Self::components(self.screens.len(), &edges);
        if comps.len() > 1 {
            warn_log!("{} groups of screenshots share no player: their teams were resolved independently.", comps.len());
        }
        let mut per_screen: BTreeMap<usize, ScreenAssignment> = BTreeMap::new();
        for members in &comps {
            let bits = Self::solve_bits(members, &edges);
            let (decision, screens) = self.decide(members, &bits);
            assignment.groups.push(decision);
            per_screen.extend(screens);
        }
        for index in 0..self.screens.len() {
            let screen = per_screen.remove(&index).unwrap();
            for (team, players) in [(Team::Green, &screen.green), (Team::Red, &screen.red)] {
                for p in players {
                    match assignment.entries.iter_mut().find(|(t, e)| *t == team && e.name == p.name) {
                        Some((_, e)) => {
                            e.kills += p.kills;
                            e.deaths += p.deaths;
                            e.played += 1;
                        }
                        None => assignment.entries.push((team, Entry { name: p.name.clone(), kills: p.kills, deaths: p.deaths, played: 1 })),
                    }
                }
            }
            assignment.per_screen.push(screen);
        }
        assignment
    }

    fn decide(&self, members: &[usize], bits: &BTreeMap<usize, u8>) -> (GroupDecision, BTreeMap<usize, ScreenAssignment>) {
        let mut wins = [0u32; 2];
        let mut green_rows = [0usize; 2];
        let mut side_players: [Vec<(String, u32, u32)>; 2] = [vec![], vec![]];
        for &s in members {
            let screen = &self.screens[s];
            for (color, players) in [(Color::Green, &screen.greens), (Color::Red, &screen.reds)] {
                let side = if color == Color::Green { bits[&s] } else { 1 - bits[&s] } as usize;
                if color == Color::Green {
                    green_rows[side] += players.len();
                }
                for p in players {
                    match side_players[side].iter_mut().find(|x| x.0 == p.name) {
                        Some(x) => {
                            x.1 += p.kills;
                            x.2 += p.deaths;
                        }
                        None => side_players[side].push((p.name.clone(), p.kills, p.deaths)),
                    }
                }
            }
            if let Some(w) = screen.winner {
                let side = if w == Color::Green { bits[&s] } else { 1 - bits[&s] };
                wins[side as usize] += 1;
            }
        }

        let tie_break = |reason_out: &mut Reason| -> usize {
            let crowns: Vec<Option<((u32, i64), f64)>> = (0..2).map(|s| Self::best_of(&side_players[s])).collect();
            match (&crowns[0], &crowns[1]) {
                (Some(a), Some(b)) if a.0 != b.0 => {
                    *reason_out = Reason::Crown;
                    if a.0 > b.0 { 0 } else { 1 }
                }
                (Some(a), Some(b)) => {
                    if a.1 != b.1 {
                        *reason_out = Reason::BestKd;
                        if a.1 > b.1 { 0 } else { 1 }
                    } else {
                        *reason_out = Reason::PerfectTie;
                        0
                    }
                }
                (Some(_), None) => {
                    *reason_out = Reason::Crown;
                    0
                }
                (None, Some(_)) => {
                    *reason_out = Reason::Crown;
                    1
                }
                (None, None) => {
                    *reason_out = Reason::PerfectTie;
                    0
                }
            }
        };

        let mut reason = Reason::Wins;
        let green_side;
        if !self.green_team_is_game_winner && green_rows[0] != green_rows[1] {
            green_side = if green_rows[0] > green_rows[1] { 0 } else { 1 };
            reason = Reason::GreenRows;
        } else if self.green_team_is_game_winner && wins[0] != wins[1] {
            green_side = if wins[0] > wins[1] { 0 } else { 1 };
        } else {
            green_side = tie_break(&mut reason);
            if reason == Reason::PerfectTie {
                warn_log!(
                    "Perfect tie between {} and {}: the first side encountered was declared the green team.",
                    side_players[0].iter().map(|p| p.0.as_str()).collect::<Vec<_>>().join(", "),
                    side_players[1].iter().map(|p| p.0.as_str()).collect::<Vec<_>>().join(", ")
                );
            }
        }
        let red_side = 1 - green_side;

        let mut screens = BTreeMap::new();
        for &s in members {
            let screen = &self.screens[s];
            let mut sa = ScreenAssignment { source: screen.source.clone(), ..Default::default() };
            for (color, players) in [(Color::Green, &screen.greens), (Color::Red, &screen.reds)] {
                let side = if color == Color::Green { bits[&s] } else { 1 - bits[&s] } as usize;
                if side == green_side {
                    sa.green.extend(players.iter().cloned());
                } else {
                    sa.red.extend(players.iter().cloned());
                }
            }
            if let Some(w) = screen.winner {
                let winning_side = if w == Color::Green { bits[&s] } else { 1 - bits[&s] } as usize;
                sa.green_won = Some(winning_side == green_side);
            }
            screens.insert(s, sa);
        }
        let names = |side: usize| side_players[side].iter().map(|p| p.0.clone()).collect::<Vec<_>>();
        (
            GroupDecision {
                green: names(green_side),
                red: names(red_side),
                green_wins: wins[green_side],
                red_wins: wins[red_side],
                screens: members.len(),
                reason,
            },
            screens,
        )
    }

    /// (crown rank, best KD ratio among the crown holders) of one side.
    fn best_of(stats: &[(String, u32, u32)]) -> Option<((u32, i64), f64)> {
        let best = stats.iter().map(|(_, k, d)| (*k, -(*d as i64))).max()?;
        let ratio = stats
            .iter()
            .filter(|(_, k, d)| (*k, -(*d as i64)) == best)
            .map(|(_, k, d)| display_ratio(*k, *d))
            .fold(f64::MIN, f64::max);
        Some((best, ratio))
    }
}
