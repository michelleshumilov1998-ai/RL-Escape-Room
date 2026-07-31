# PROJECT R-5 — Escape from the Learning Lab

A reinforcement learning teaching project built as an escape game. R-5, an
experimental navigation robot, wakes up on sub-level 4 of a sealed research
institute and has to cross a series of test chambers to reach the surface. Each
chamber is solved with a different method, so the same story is used to compare
how those methods behave.

No reinforcement learning library is used anywhere. Every algorithm is written
out by hand so the update rules can be read directly in the source.

---

## Three chambers, three methods

| Room | Chamber | Method | The point of it |
|---|---|---|---|
| 1 | Laser Security Chamber | Value Iteration and Policy Iteration | planning when the model is *known* |
| 2 | Broken Bridge Sector | SARSA, Expected SARSA, Q-Learning, Double Q-Learning | learning on-policy when it is not |
| 3 | Reactor Control Chamber | Q-Learning and the three above | a reward a very long way from the first move towards it |

Rooms 4 and 5 — a continuous-state drone hall solved with tile coding, and a
procedurally generated warehouse solved with local features — **do not exist in
this repository any more.** They were only ever implemented in an earlier
Streamlit version of this project, which has been deleted; see *Known gaps*.

## Running it

Nothing to install:

```bash
python3 serve.py
```

Then open <http://localhost:8000>. Start screen → chamber select → Enter.

`serve.py` uses the standard library alone and the browser half is hand-written
JavaScript with no build step, so there is no dependency to manage. `pytest` is
the only entry in `requirements.txt`, and only for the tests.

## How it is put together

The simulation lives entirely in Python, in `game/`. The browser owns rendering
and controls and never sees an environment, a policy or a value table — only
snapshots. That boundary is the reason every claim in this file could be
measured without opening a browser: `game/` is plain Python with no display
dependency, so it can be driven from a script.

Two directories and two files, and that is the whole project.

```
serve.py                    the whole backend: hands out web/, holds sessions
game/
  config.py                 every tunable value, served to the browser as JSON
  grid.py                   one GridWorld for every grid room: tiles, dynamics
  reactor.py                room 3's world — a guard, doors on a cycle, keys
  session.py               one run: the state machine, batching, metrics
  comparison.py             several methods on one room, for the graphs
  recorder.py               whole episodes kept, so a run can be replayed
  definition.py             a room in the shape the browser's contract wants
  algorithms/
    base.py                 the interface, and the planners' shared parts
    value_iteration.py  policy_iteration.py
    tabular.py              the Q-table, ε-greedy, and what the four share
    sarsa.py  expected_sarsa.py  q_learning.py  double_q_learning.py
web/
  index.html                start screen
  levels/                   chamber select
  level/                    room 1's screen — a planner, measured in sweeps
  room/                     rooms 2 and 3 — learners, measured in episodes
  assets/
    sim.js                  the API client: the whole page/simulation boundary
    shapes.js               every drawing recipe, shared by both screens
    room/contract.js        the data contract, and the authority on it
    room/producer.js        turns the API into that contract
    room/shell.js           the room screen: DOM, the live loop, the panels
    room/playback.js        walking a recorded batch, for replays
    room/charts.js          the training graphs
    room/compare.js         several methods on one set of axes
```

Six ideas the code is organised around, each of which is load-bearing rather
than decorative:

**One config, served.** `game/config.py` is handed to the browser verbatim at
`/api/config`, so a speed tier or a parameter range is written down exactly once
and the two halves cannot disagree. Colours are the deliberate exception: they
live in `web/assets/base.css` as custom properties, and the config only names
which property an entity uses. Nothing writes a hex value twice.

**One abstraction over both families.** "One unit of learning" is a sweep for a
planner and a single environment step for a learner (`session.py`). The state
machine, the batching and the time budget above it never branch on which.

**Turbo is time-budgeted, not counted.** 12 ms of simulation per request, so
thousands of episodes are reachable without the interface ever blocking.

**Parameters have scope.** A `live` parameter applies at once; a `reset` one
marks the run *stale* rather than quietly changing the model under a value
table that was computed against the old one.

**Training is watched, not hidden.** Pressing Play advances the run a slice at a
time and draws the world that comes back, so the agent is seen learning. The
recorded episodes are a separate thing, for looking back at afterwards.

**Recording is passive.** The recorder is handed transitions the agent was
taking anyway and draws no randomness, so a run trains identically with
recording on or off. Verified by comparing Q-tables.

### The state, and why it is five numbers

`GridWorld`'s state is `(row, col, battery, last_direction, collapsed)`. A cell
alone is not a Markov state in these rooms:

* A **battery** can only be collected once, so whether it is already in hand
  changes what entering its cell is worth. Without it the model says the battery
  pays out every time it is stepped on, and the best plan is to stand there
  collecting it forever.
* **Oil** carries the agent on in the direction it was already going, so what
  happens next depends on what happened last.
* A **collapsing plank** is a bridge early in an episode and a hole later.
  Without knowing which have gone, the value of stepping onto one is an average
  of "fine" and "fatal" — and an average is not a fact about the state.

Each dimension collapses to a single value in a room that does not contain the
thing, so a room pays only for what it has. Room 3 adds two more of its own; see
its section.

### The tile alphabet

Shared by every grid room, in `game/grid.py`. Room 3 adds to it.

| Char | Tile | Char | Tile |
|---|---|---|---|
| `S` | start | `L` | laser beam — back to the start, run continues |
| `E` | exit | `H` | shaft — **ends the run** |
| `#` | wall | `B` | battery |
| `.` | floor | `T` | teleporter pad |
| `~` | ice | `=` | cracked ice (2 × slip) |
| `o` | oil (lowercase) | `D` | one-way door, down only |
| `G` | sound bridge | `C` | collapsing plank |
| `1` `2` `3` | generators | `a` `b` `c` | their keys — `a` starts `1` |
| `d` | sliding door, on a cycle | `R` | reactor door |

One trap worth naming: `_find` requires at least one `S` and one `E` but
silently takes the **first** of any duplicates, so a stray second exit is not an
error — it is simply ignored, and every `E` still ends a run.

If you are pasting in a map from anywhere older than this file, note that an
earlier version of the project used a different alphabet — `%` for hard ice,
capital `O` for oil, `>` `<` `^` `v` for one-way doors and `G` for the goal.
Those characters are not valid here: an unknown one raises `KeyError`, and a
goal written as `G` gives `ValueError: the layout has no 'E' tile`.

### Comparing methods

Every chamber's sidebar has a **Compare methods** panel. It runs each method the
room offers against the same layout, the same parameters and the same seed —
one `Session` each, advanced together — and draws their curves on shared axes
(`game/comparison.py`, `web/assets/room/compare.js`).

Each curve carries its own x values, which matters more than it sounds. One call
of Value Iteration is one sweep; one call of Policy Iteration is a whole policy
evaluation — a couple of hundred sweeps — followed by an improvement. Plotting
both against point index would put 15 sweeps and 389 sweeps on the same tick and
imply they were the same amount of effort.

---

# Room 1 — The Laser Security Chamber

*Lab sector A-01 · Security protocol active · Dynamic Programming*

The room whose model is **known**. Every rule below is in the model before the
agent has moved, which is what lets Dynamic Programming route around a beam it
has never touched. Nothing is learned from experience here.

## The map

```
      c0 c1 c2 c3 c4 c5 c6 c7 c8 c9
r0     S  .  .  #  .  .  B  .  .  .
r1     .  ~  .  #  .  #  #  #  .  .
r2     .  ~  .  T  .  .  .  #  .  .
r3     .  L  .  L  L  .  .  #  .  .
r4     .  .  .  #  .  .  o  .  .  .
r5     #  #  .  L  #  #  #  .  .  .
r6     .  .  T  .  .  =  .  .  .  .
r7     .  #  #  #  .  D  .  #  #  .
r8     .  .  .  o  .  .  .  .  .  .
r9     .  .  .  .  #  E  .  .  .  .
```

Start `(0,0)`, control panel `(9,5)`, battery `(0,6)`, teleport pads `(2,3)` and
`(6,2)`, one-way door `(7,5)`.

**Column 3 seals the chamber in two.** It is `#`, `#`, `T`, `L`, `#` — walls, a
pad and a beam. The pad is the only gap, so with the teleporter disabled there
is **no route to the exit at all**. Measured: 11 steps via the pad, `None`
without it.

## The state and the actions

`(row, col, battery, last_direction, collapsed)` — 810 states, of which the
collapsed dimension is one value because there are no planks here. Four actions:
up, down, left, right. The action is what the agent *tries*; the floor it is
standing on decides what happens.

## The rewards

| Event | Reward |
|---|---|
| Each step | −1 |
| Walking into a wall | −1 + (−3) = −4 |
| Walking into a beam | −1 + (−30) = −31, **and back to the start** |
| Collecting the battery | −1 + the battery bonus (a slider) |
| Taking a teleport pad | −1, like any other step |
| Reaching the panel | −1 + 100 = **+99** |

A laser is **not fatal**: it throws the agent back to the door it came in by,
which costs it everything it had walked. That is harder to plan around than
simply dying, and it means a laser is effectively a *wall* — no optimal plan
ever steps into one on purpose. Genuine probabilistic risk exists only where a
loose surface sits next to a beam, so a sideways slip can push R-5 into a beam
it never aimed at.

**The pad pays nothing, and that is a fix rather than an oversight.** It used to
pay +5 on entry. Because a pad is entered unconditionally and both pads lead to
each other, that made a cycle the agent could ride: two steps apart via `(5,2)`,
netting +6 every four steps. Above about γ 0.98 riding it beat leaving — the
plan shuttled between the pads **198 times in 400 steps and never reached the
panel**, and V(start) ballooned to 1491 at γ 0.999. Setting the pad's reward to
zero removes it completely; the reward for a shortcut is the steps it saves.

| γ | Before (pad +5) | After (pad 0) |
|---|---|---|
| 0.90 | 11 steps, exit | 11 steps, exit |
| 0.95 | 11 steps, exit | 11 steps, exit |
| **0.99** | **400 steps, lost, 198 pad uses** | 11 steps, exit |
| **0.999** | **400 steps, lost, 2744 sweeps** | 11 steps, exit, 16 sweeps |

## The parameters

| Parameter | Range | Default | Scope |
|---|---|---|---|
| γ discount factor | 0.50 – 0.999 | 0.95 | reset |
| θ stopping threshold | 1e−6 – 1e−1 | 1e−4 | live |
| Ice slipperiness | 0.0 – 0.5 | 0.20 | reset |
| Battery bonus | 0 – 80 | 10 | reset |

## The measured result

At the defaults, both methods converge to the same plan: **11 steps, +89**, via
`(0,0) → (0,1) → (0,2) → (1,2) → (2,2) →` pad `→ (6,2) → (6,3) → (6,4) → (7,4)
→ (8,4) → (8,5) → (9,5)`. V(start) = 51.25.

**Value Iteration takes 15 sweeps. Policy Iteration takes 389.** Same plan, 26×
the work — which is the comparison this room exists to show, and it is what the
Compare panel draws.

The **battery bonus** changes the plan:

| Bonus | Steps | Reward | Fetches the battery? |
|---|---|---|---|
| 0 – 40 | 11 | +89 | no |
| 60 | 23 | +137 | **yes** |
| 80 | 23 | +157 | **yes** |

### Two things this room does not currently show

Both are consequences of the map as it stands, and both are measured rather
than suspected.

**The slipperiness slider does nothing.** V(start) is 51.25 at slip 0.00, 0.10,
0.20, 0.35 and 0.50, and the route is identical at every one. The optimal path
is entirely firm floor, so no slip can ever occur — the ice, the oil and the
cracked ice are decoration the plan routes around. Only one loose cell in the
map is adjacent to a beam (the `~` at `(2,1)`, beside the laser at `(3,1)`) and
the route does not go through it. Making `(6,3)` and `(6,4)` ice would fix it in
one edit: the path already crosses them and the beam at `(5,3)` is directly
above `(6,3)`.

**There is no route choice, and the one-way door is decorative.** The teleporter
is mandatory, so there is nothing for the parameters to flip between. Blocking
the door at `(7,5)` entirely leaves the shortest route at 11 steps.

---

# Room 2 — The Broken Bridge Sector

*Lab sector B-04 · Structural failure · SARSA, and three others for contrast*

The model is not given. The only way to find out what a step does is to take it.

## The sector

```
      c0 c1 c2 c3 c4 c5 c6 c7 c8 c9
r0     #  #  #  #  #  #  #  #  #  #
r1     #  .  .  .  .  .  .  .  .  #     the safe upper corridor
r2     #  .  H  H  H  H  H  H  .  #
r3     #  .  H  H  H  H  H  H  .  #
r4     #  .  H  H  H  H  H  H  .  #     the shaft
r5     #  .  H  H  H  H  H  H  .  #
r6     #  .  H  H  H  H  H  H  .  #
r7     #  .  H  H  H  H  H  H  .  #
r8     #  S  G  C  C  C  C  G  E  #     the span
r9     #  #  #  #  #  #  #  #  #  #
```

Two ways across, measured by breadth-first search:

* **the span** — 7 steps along the bottom. Two sound bridge sections (`G`) with
  four collapsing planks (`C`) between them. Crossed cleanly it is worth **+93**.
* **the lap** — 21 steps: up the west wall, along the corridor at the top, down
  the east wall. Worth **+79**.

## The collapsing planks

A plank can give way **under the step that lands on it**, and that is a fall
into the shaft. Survive it and the plank is gone behind you, so the span cannot
be walked back — a step returning onto a gap is the same fall. The risk
compounds: four planks at chance *p* get across with probability (1−p)⁴.

The model exposes both stages, which is what Dynamic Programming would read and
what the model-free methods here are deliberately not given:

```
stepping onto a sound plank      p=0.90 → cross,  −1
                                 p=0.10 → fall,   −101, run ends
stepping onto a gap              p=1.00 → fall,   −101, run ends
stepping up into the shaft       p=1.00 → fall,   −101, run ends
```

At the default 0.10, walking straight across 2000 times got out **66%** of the
time — exactly 0.9⁴.

## The state

`(row, col, battery, last_direction, collapsed)`, **5,120 states**. The
`collapsed` field is a bitmask, one bit per plank in map order, and it is what
keeps the room Markovian. Without it, stepping toward the span would be worth an
average of "fine" and "fatal".

## The rewards

| Event | Reward |
|---|---|
| Each step | −1 |
| Walking into a wall | −1 + (−2) = −3 |
| Falling into the shaft | −1 + (−100) = −101 |
| A plank giving way underfoot | −1 + (−100) = −101 |
| Reaching the exit | −1 + 100 = **+99** |

## The parameters

α, γ, ε, ε floor, ε decay, initial Q (default 90, optimistic), episodes
(default 1500), and **plank failure chance** (0 – 0.5, default 0.10, reset
scope). There is no slipperiness control: nothing in this sector is loose.

## The measured result, and the problem with it

All four methods train and reach the exit. At 1500 episodes they land within
half a point of each other: SARSA 76.3, Expected SARSA 76.7, Q-Learning 76.8,
Double Q 76.7 mean reward.

**The textbook contrast does not appear on this map.** It was found in none of
sixteen parameter combinations swept over γ, ε floor, initial Q and plank
chance; where two methods differed, it was SARSA on the risky span and
Q-Learning on the safe lap — the reverse of the classic result.

The cause is geometric, not a matter of tuning:

```
span: 8 cells,  6 next to something fatal
lap : 22 cells, 18 next to something fatal
```

**The "safe" route is the more dangerous one.** The shaft is a 6 × 6 block
touching columns 1 and 8, so both vertical legs of the lap run directly
alongside it and one random sideways step anywhere along them is a fall. The lap
has three times the exposure and three times the length, so SARSA is *correctly*
preferring the span. There is no setting that fixes that.

A second finding is worth recording because it is easy to get wrong: **plank
failure cannot separate SARSA from Q-Learning.** It is environment
stochasticity, which both methods price identically — only exploration-induced
risk separates on-policy from off-policy. At plank chance 0 both take the span;
at 0.10 and above the plank risk swamps the +14 margin and both take the lap.

The fix is two rows — wall the vertical corridors off from the shaft, so that
nothing on the lap has a drop beside it:

```
r2     #  .  #  #  #  #  #  #  .  #
r3-r7  #  .  #  H  H  H  H  #  .  #
```

That keeps the shaft directly above the four planks, so the span stays risky
while the lap becomes genuinely safe. It has **not** been applied: the map is as
specified.

---

# Room 3 — The Reactor Control Chamber

*Lab sector C-07 · Power grid offline · Q-Learning*

The chamber with things that move. Rooms 1 and 2 are static — a tile means the
same thing on every step of every episode. Here a guard walks a patrol and the
shaft doors open and shut on a cycle, so the same cell is safe at one moment and
fatal at the next.

## The chamber

```
      c0 c1 c2 c3 c4 c5 c6 c7 c8 c9
r0     #  #  #  #  #  #  #  #  #  #
r1     #  S  .  b  .  .  .  1  #  #     key b, generator 1
r2     #  .  #  #  d  #  #  .  #  #     the shaft's north door
r3     #  .  #  #  .  #  #  .  #  #
r4     #  .  #  #  c  #  #  .  #  #     key c, inside the shaft
r5     #  .  #  #  .  #  #  .  #  #
r6     #  .  #  #  d  #  #  .  #  #     the shaft's south door
r7     #  3  .  a  .  .  .  2  #  #     generator 3, key a, generator 2
r8     #  #  #  #  R  #  #  #  #  #     the reactor door
r9     #  #  #  #  E  #  #  #  #  #     the exit, behind it
```

31 walkable cells. Start `(1,1)`, exit `(9,4)`.

## The task: six errands, in order

Three generators must be brought up **in sequence** — 1, then 2, then 3 — and
each needs **its own key** fetched first. Each key is kept well away from the
generator it belongs to, so every errand is a journey:

| Key | Where | Starts | Where |
|---|---|---|---|
| `a` | `(7,3)` south run | `1` | `(1,7)` north-east |
| `b` | `(1,3)` north run | `2` | `(7,7)` south-east |
| `c` | `(4,4)` in the shaft | `3` | `(7,1)` south-west |

Verified behaviour: on generator 1 with **no key**, nothing happens (−1, stage
unchanged); with **key `a`**, it fires (+19, stage advances); with **key `b`**,
nothing — a key works only on its own generator. Standing on generator 2 first
while holding all three keys leaves the stage at 0.

A generator is drawn **white** while it is off and **green** once it is running,
so how far through the sequence the reactor is can be read off the chamber
itself. The reactor door bumps like a wall until all three are up:

| Stage | Step onto the reactor door |
|---|---|
| 0, 1, 2 | blocked, −3 |
| 3 | opens, −1 |

The shortest complete mission is **37 steps**, ignoring door timing.

## What gets in the way

* **The guard** walks the 24-cell service ring anticlockwise, one cell per step,
  against the direction the errands run — so the two meet head-on once a lap.
  Being caught ends the run. The check counts the **swap** as well as the
  collision: two things stepping one cell towards each other would otherwise
  pass straight through, and a guard you can walk through is not a guard.
* **The doors** at both ends of the shaft are open two steps in every four.
  The shaft is 8 steps to the north run against 12 round the ring, and the guard
  never enters it, so it is both a shortcut and the one reliable refuge.
* **Waiting** is an action here and nowhere else. A shut door opens again two
  steps later, so holding is a real move rather than a wasted step.

## The state

`(row, col, battery, last_direction, collapsed, stage, guard, keys)` —
**23,808 states**. Three fields inherited from the base grid are dead weight
(no battery, no planks, nothing loose), but they cost one value each and keeping
the layout identical means every algorithm, the cell projection and the renderer
work unchanged. `last_direction` is *pinned* to "none" for that reason:
enumerating it would multiply the table by six to describe something no rule in
this room reads.

The three that matter: **stage** (how far through the sequence), **guard**
(patrol index), **keys** (a bitmask, one bit each). Keys cannot be folded into
stage — the errands and the sequence advance independently.

**The doors are derived from the guard, not from a clock.** A door on a timer
would make the room non-Markovian: the same cell would mean different things at
different moments with nothing in the state saying which. The patrol is 24 cells
and the door cycle is 4, so the cycle divides in exactly, the pattern repeats,
and one state dimension covers both moving things. `ReactorWorld` raises if a
room ever breaks that divisibility.

## The rewards

| Event | Reward |
|---|---|
| Each step | −1 |
| Into a wall or a shut door | −1 + (−2) = −3 |
| Picking up a key | −1 + 15 = **+14** |
| A generator, in order and keyed | −1 + 20 = **+19** |
| Being caught by the robot | −1 + (−100) = **−101** |
| Leaving through the reactor door | −1 + 100 = **+99** |

## The parameters

α, γ (default **0.99** — the reward is a long way from the first move towards
it), ε, ε floor, ε decay, initial Q (0), episodes (default 4000).

## The measured result

Compared at the defaults, 4000 episodes each, same seed:

| Method | Mean reward |
|---|---|
| Expected SARSA | **25.18** |
| SARSA | **20.98** |
| Q-Learning | −39.26 |
| Double Q-Learning | −90.76 |

The on-policy methods are ahead, which is the opposite of what room 2 sets out
to show. That is plausible with a patrolling guard punishing optimism, but it
has **not been investigated** and should not be presented as a finding.

Two honest caveats. The 4000-episode default is a guess, not a measurement, and
it is expensive: comparing all four methods takes **245 seconds** over 173
requests. And whether any method converges to a reliable escape at that setting
has not been established.

---
# The interface

Both screens are hand-written canvas and JavaScript — no 3D engine, no charting
library, no framework. `shapes.js` holds one drawing recipe per kind of thing, so
a wall is the same wall in every chamber and the legend is drawn by the very code
that draws the world; a thing cannot appear on the grid without appearing in the
key.

**Room 1** uses `web/level/` — a planner's screen, measured in sweeps, with the
value table as a heatmap and the greedy policy as arrows.

**Rooms 2 and 3** use `web/room/`, built around episodes:

* **Play trains, and you watch it.** Each frame asks the server for a slice of
  work and draws the world that came back. Speeds are rates per *second* — the
  loop carries the fractional remainder between frames, because asking for a
  whole step every frame would make the slowest speed twenty times too fast at
  60 fps. Turbo is not paced at all.
* **Once it is trained, Play shows the route it learned** — and that is a
  different thing from any episode it trained on. ε stops at `epsilon_min`,
  0.05 by default, rather than decaying to zero, so even the last episode of a
  finished run still takes a random step about one time in twenty and visibly
  doubles back on itself. What Play shows instead is one run of the greedy
  policy with the exploration taken out. Measured in room 2 at the defaults:
  the last recorded episode that reached the exit wanders through 22 steps for
  +76, with a step into the west wall in the middle of it, and the policy
  behind that same episode walks the route cleanly in 21 for +79. It is
  recorded once, on the server, and animated by the page, so leaving it looping
  cannot touch what was learned. When the policy does *not* get out — an
  undertrained run loops until the 400-step limit — the panel says so rather
  than presenting the wandering as the answer.
* **Graphs fill in as it learns**, pulled from the recording every two seconds.
* **Episode replay** — 40 whole episodes are kept per run, spread across it with
  the first and last three always included, since that is where behaviour differs
  most. Early ε-random episodes routinely run to the 400-step limit, which is why
  the budget is a few dozen and not all of them.
* **Step replay** — any recorded episode, any single step, held still, with
  everything that was true at that moment: the guard's position, which doors were
  open, which planks had gone, which keys were in hand. It reads the recording
  and changes nothing, so it can be scrubbed mid-run.
* **Compare methods** — described above.

The sidebar opens by default on both screens, since that is where a chamber
explains itself.

Every frame describes the **whole world** rather than what changed since the
last one. That is what lets the same code feed a recorded replay, which is walked
in order, and the live view, which can be joined at any moment.

---

# Verification

None of this is required by the brief; it is self-imposed, and it is how every
number in this file was arrived at.

`game/` is plain Python with no display dependency, so it is
driven directly from scripts: route lengths by breadth-first search over the real
rules, the SARSA-vs-Q-Learning question by parameter sweeps, the plank odds by
2000 crossings, the guard patrol by checking it is a closed loop of adjacent
non-wall cells, and the reward-cycle bug by solving at six values of γ.

The browser half is checked without a browser: `node --check` on every script,
and the page booted under a stubbed DOM and canvas against a running `serve.py`
— which is what caught the metric reading `NaN`, the step inspector resolving
episode numbers as array positions, and two animation loops painting the same
canvas.

**There is almost no automated suite, and that is the biggest hole in the
project.** `tests/` holds one file, `test_game_room1.py`, and it is **broken**:
it imports a name `game.grid` does not have, and its assertions describe an
earlier version of room 1's map. It needs rewriting against the current code.

Ten further test files existed for the deleted Streamlit app. They covered its
rooms, not this one's, so they went with it. Everything verified in this file was
verified by hand — repeatably, but by hand.

```bash
python3 -m pytest tests -q      # one file, currently failing
```

---

# Known gaps

Stated plainly, because a README that documents intentions rather than the code
is worse than none.

**There are three chambers, and the brief asks for at least four.** Rooms 4 and
5 were deleted along with the Streamlit app that was their only implementation.
Rebuilding them here means real work that `game/` is not set up for: room 4 needs
a continuous environment, a tile coder and semi-gradient SARSA — everything in
`game/algorithms/` today is tabular or a planner — and room 5 needs procedural
layouts and local features. The deleted code is recoverable from git
(`git show da150e3 --stat`) if it is worth porting rather than rewriting.

**The chamber select does not know what is built.** `web/assets/levels.js`
hardcodes all five chambers as available and never consults `/api/rooms`, so
entering 4 or 5 shows a failure message rather than being locked.

**The game is not published.** The brief asks for it to be reachable online, and
`serve.py` binds `127.0.0.1`. The option that keeps the Python as the thing that
actually runs is Pyodide: `game/` is pure standard library, so it should load in
the browser as WebAssembly, the whole HTTP layer would disappear and the result
would be a static site deployable anywhere. Untried.

**Room 1's slipperiness control has no effect**, and **room 2 does not reproduce
the on-policy/off-policy contrast.** Both are measured, both are explained in
their sections, and both are properties of the current maps rather than of the
algorithms.

**Room 3 is untuned.** Its 4000-episode default is a guess, its comparison takes
four minutes, and the on-policy methods beating the off-policy ones is
unexplained.

---

# What each algorithm actually does

Each is one file, and the four model-free ones differ in a single method —
`target`, the expression the update moves towards. If implementing the second had
required touching a room, the layering would be wrong.

| Method | Target |
|---|---|
| SARSA | `r + γ Q(s′, a′)` — the action actually taken next |
| Expected SARSA | `r + γ Σ π(a\|s′) Q(s′, a)` — averaged over the policy |
| Q-Learning | `r + γ max_a Q(s′, a)` — the best action available |
| Double Q-Learning | `r + γ Q_b(s′, argmax Q_a)` — chosen by one table, valued by the other |

The two planners share `q_value(env, s, a, V, γ) = Σ P(s′\|s,a)[R + γV(s′)]` and
are built from that one line. They are the only methods handed `env.transitions()`;
`needs_model` says which, and the model-free ones are never given the environment's
model at all — the only way they find out what an action does is to take it.

Two details in the tabular code that are not incidental:

**Ties are broken randomly.** A fresh table is uniform, so every action ties at
the start; always taking the first would send the agent the same way on every
early episode and whole parts of a room would never be seen.

**Optimistic initialisation** (`q_init`, 90 in room 2) makes an untried action
look better than a tried one, so a route that fails early gets reconsidered
instead of abandoned.
