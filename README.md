# Screen To Kd (Rust)

Reads game "RESULTS / Stats" screenshots and prints a ready-to-paste scoreboard:

```
Match Results (from 2 rounds)

Red Team: (0/2 Game Win)
Nemo: 12/12 (1.00 KD) 🥈
bluffar: 8/12 (0.67 KD)

Green Team: (2/2 Game Win)
cata: 19/7 (2.71 KD) 👑
18ms: 12/9 (1.33 KD)

07/10/2026 at 01:15
```

Rust rewrite of the Python/EasyOCR tool: **one self-contained `.exe`** (the EasyOCR `english_g2`
recognizer is embedded and runs in pure Rust: no Python, no PyTorch, no model download), instant
window, ~0.3 s per screenshot. Reads every player (there is no tracked-players list any more).

## Build

```
build_windows.bat          (or: cargo build --release)
```
Needs only Rust (https://rustup.rs). No other dependency. Output: `target/release/kd_ocr.exe`.
`cargo build --release --no-default-features` builds the command-line-only version.
`cargo test --release` runs the tests (real screenshots in `tests/samples`).

## Use

* Double-click `kd_ocr.exe`, or **drop screenshots / a folder on the window or on the .exe**
  (analysis starts at once). `F5` analyze, `Ctrl+O` add, `Del` remove, `Ctrl+L` clear console.
* Results are copied to the clipboard automatically (option), `COPY` copies again.
* Command line: `kd_ocr 1.png 2.png --no-gui [--report] [--per-match] [--green-rows]`.
  Warnings go to stderr and to `kd_ocr.log`.

## Options (`settings.txt`, created next to the exe; also checkboxes in the window)

| key | default | meaning |
|---|---|---|
| `PER_MATCH` | false | one block per game instead of totals |
| `GREEN_TEAM_IS_GAME_WINNER` | **true** | the team that **wins** is the Green team (banner colour). `false`: Green team = players shown with the small **green** row bands, Red team = small **red** bands; victory plays no role, only the team names change |
| `AUTO_COPY_RESULT` | true | copy results after analysis |
| `DEDUPLICATE_IMAGES` | false | skip pixel-identical screenshots |

`USE_TRACKED_LIST` / `WINNER_IS_GREEN` of older versions are ignored if still present.

## Rules

* **Teams**: screenshots are linked through the players they share (same colour = same team,
  opposite colour = opposite teams). Winner of a game = colour of the big banner (never its text,
  so any game language works).
* **👑** = most kills, then fewest deaths; players still tied all get it, but **only inside one
  team**. If the perfect tie spans both teams, the tied players of the **winning team get 👑 and
  those of the losing team get 🥈**.
* **🥈** = same rule inside the team opposed to the crown.
* Order inside a team: kills desc, deaths asc, then alphabetical.
* **Screenshot order**: files named `<any text> YYYY-MM-DD HHMMSS` (`Screenshot 2026-10-06 070359`,
  `Capture d'écran 2026-10-06 070359`, `Capture_decran_2026-10-04_212418`, `Captura de pantalla…`,
  `Bildschirmfoto…`, `Cattura schermata…`) are processed oldest first (date, then time of day:
  000000 earliest, 235959 latest). Files without such a date keep their position.
* **Hidden players (4v4)**: a screenshot with no banner and at most 4 rows (the scrolled-down end
  of the table) is not a game: its players are added, with their row colour, to the game of the
  screenshot dated just before it. It does not count as a round.
* A player with `- / -` (disconnected) is ignored. The same player read slightly differently on
  several screenshots (`0293A` / `o293A`) is unified to the most frequent reading.

## Layout

```
src/nn.rs         embedded CRNN (VGG + BiLSTM + CTC), im2col + SIMD matrix products
src/imgproc.rs    rows/colours from the HSV background, crops, KD split around the "/"
src/ocr.rs        contrast variants + voting, KD reconstruction
src/geometry.rs   pixel-geometry fix of o/0, z/Z/2, 1/l/I
src/detector.rs   one screenshot -> players       src/teams.rs     team linking & naming
src/formatter.rs  text + medals                    src/datesort.rs  chronological order
src/pipeline.rs   analyze / build_output          src/gui.rs       dark egui interface
assets/crnn_weights.bin (from tools/convert_weights.py and EasyOCR's english_g2.pth)
```
