# PROJECT R-5 — Escape from the Learning Lab

A reinforcement learning teaching project built as a five-room escape game in
Python and Streamlit. R-5, an experimental navigation robot, wakes up on sub-level 4
of a sealed research institute and has to cross five test chambers to reach the
surface. Each chamber is solved with a different method, so the same story is used
to compare how those methods behave.

**All five rooms are implemented.** Each is solved with a different method:

| Room | Chamber | Method | The point of it |
|---|---|---|---|
| 1 | Laser Security Chamber | Value Iteration and Policy Iteration | planning when the model is *known* |
| 2 | Broken Bridge Sector | SARSA | learning on-policy when it is not |
| 3 | Reactor Control Chamber | Q-Learning | off-policy learning, and how it differs |
| 4 | Drone Wind Tunnel | Semi-Gradient SARSA + tile coding | a continuous state, so no table fits |
| 5 | Adaptive Storage Facility | Semi-Gradient Q-Learning + local features | generalising to layouts never seen |

No reinforcement learning library is used anywhere. Every algorithm is written out
by hand so the update rules can be read directly in the source.

---

## Running it

```bash
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/streamlit run app.py
```

Then open <http://localhost:8501>.

To run the tests and the end-to-end verification:

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python scripts/verify_room1.py
.venv/bin/python scripts/verify_room2.py
.venv/bin/python scripts/verify_room3.py
.venv/bin/python scripts/verify_room4.py
.venv/bin/python scripts/verify_room5.py
```

Requirements: Python 3.11, plus `streamlit`, `matplotlib`, `numpy`, `pandas` and
`pytest`. There is no 3D engine dependency — see *Rendering* below.

---

## Project layout

The visual layer and the reinforcement learning layer are kept apart. The
renderers are handed a plain dictionary and draw it; they never see an
environment, a policy or a value table.

```
app.py                      Streamlit entry point and screen router
core/
  actions.py                directions, actions (including WAIT), deltas
  tiles.py                  the tile alphabet shared by the grid rooms
  tabular.py                Q-table helpers shared by rooms 2 and 3
  story.py                  the five rooms, their methods and their components
rooms/
  locked_page.py            the placeholder page (now unreachable — all rooms exist)
  room1/                    Value Iteration and Policy Iteration
    map_data.py  environment.py  dp_solver.py
    simulate.py  experiments.py  storage.py  page.py
  room2/                    SARSA
    map_data.py  environment.py  sarsa_agent.py
    simulate.py  experiments.py  storage.py  page.py
  room3/                    Q-Learning
    map_data.py             the chamber, the patrol, the door cycle
    environment.py          generators in order, guard, hazards, sliding doors
    q_agent.py              Q-Learning and SARSA, by hand
    simulate.py  experiments.py  storage.py  page.py
  room4/                    Semi-Gradient SARSA over tile coding
    chamber.py              the hall, the fans, the obstacles, the wind field
    environment.py          continuous dynamics, substep collisions, landings
    tile_coder.py           the tilings, written out by hand
    sarsa_agent.py          semi-gradient SARSA and Q-Learning
    simulate.py  experiments.py  storage.py  page.py
  room5/                    Semi-Gradient Q-Learning over local features
    layout.py               procedural warehouses, BFS validation, seed splits
    environment.py          radar, robots, conveyors, terminal and exit stages
    features.py             the four feature sets
    q_agent.py              semi-gradient Q-Learning and SARSA
    simulate.py  experiments.py  storage.py  page.py
renderers/
  base_renderer.py          the scene contract, and a scene validator
  scene_common.py           the palette, view options and fingerprints, shared
  iso_scene.py              Room 1: map + recording -> scene dict
  room2_scene.py  room3_scene.py                 the same for rooms 2 and 3
  room4_scene.py            continuous mode: metres instead of cells
  room5_scene.py            procedural layout + fog of war
  iso_canvas.py             puts the scene on the page
  web/room_iso.html         the isometric canvas renderer (HTML and JavaScript)
  fallback_renderer.py      the same chambers drawn in Matplotlib
ui/
  theme.py                  the one place colours, fonts and CSS are defined
  cards.py  buttons.py  nav.py  inventory.py  transitions.py  hud.py
  game_state.py             progress kept in st.session_state
  start_screen.py           the opening screen
  completion.py             the closing scene after Room 5
plots/
  style.py                  one dark Matplotlib theme for every graph
  room1_plots.py  room2_plots.py  room3_plots.py
  room4_plots.py            training graphs, trajectories, value slices
  room5_plots.py            training graphs, generalisation, radar analysis
tests/                      the automated checks
scripts/verify_room1.py     end-to-end verification with printed numbers
scripts/verify_room2.py  verify_room3.py  verify_room4.py  verify_room5.py
saves/                      models and results written by "Save"
```

---

# Room 1 — The Laser Security Chamber

*Lab sector A-01 · Security protocol active*

R-5 wakes up on the charging plate in the top-left corner of a 10×10 security
chamber. The control panel that shuts down the laser grid is set into the right
wall. The full model of the chamber is known in advance, so the room is solved by
**Dynamic Programming** — planning, not learning from experience. Value Iteration
is the primary method and Policy Iteration is implemented for comparison.

## The map

```
      c0 c1 c2 c3 c4 c5 c6 c7 c8 c9
 r0    S  .  .  .  .  .  .  .  .  .
 r1    .  #  #  #  L  %  L  #  #  #
 r2    .  #  #  #  L  %  L  #  #  T
 r3    .  #  #  #  #  .  #  #  L  %
 r4    .  #  #  #  #  .  .  >  .  E
 r5    .  #  #  #  #  #  #  #  #  .
 r6    .  #  B  #  #  #  #  #  #  .
 r7    .  #  O  #  T  #  #  #  #  .
 r8    .  #  ~  L  ~  L  #  #  #  .
 r9    .  .  ~  .  .  .  O  O  .  .
```

| Symbol | Meaning |
|--------|---------|
| `S` | Start platform — row 0, col 0 |
| `E` | Control panel / exit — row 4, col 9 |
| `#` | Metallic wall |
| `.` | Normal floor |
| `~` | Wet floor (weak ice) |
| `%` | Frozen floor (strong ice) |
| `O` | Oil spill |
| `L` | Laser beam |
| `B` | Battery |
| `T` | Maintenance teleporter (two pads, one pair) |
| `>` | One-way door, crossable left to right only |

Contents: 1 start, 1 exit, 46 walls, 7 lasers, 3 wet cells, 3 frozen cells,
3 oil cells, 1 battery, 2 teleporter pads, 1 one-way door. All 54 open cells are
reachable from the start; the test suite checks this.

### Why the map is shaped like this

**A laser is not an obstacle you walk through.** Entering a beam always throws R-5
back to the start platform, so no plan ever deliberately steps into one. The real
danger is therefore *slippery floor next to a beam*, where the robot can be pushed
sideways into a laser it never aimed at. The whole map is built around that idea,
which gives three genuinely different ways across the chamber:

| Route | Steps | Risk |
|-------|-------|------|
| **Frozen shaft** — down column 5 | **13** | two frozen cells in a narrow shaft with emitters on both walls. Each icy step can slide into a beam, so the fastest route is also the one most likely to fail. |
| **Maintenance teleporter** — down the west wall to the pad at (7,4) | **17** | the approach crosses one wet cell flanked by two beams, and the landing pad drops R-5 onto frozen floor at (3,9) with a beam beside it. |
| **South ring** — west wall, along the bottom, up the east wall | **23** | wet floor and oil, but never a beam beside them, so it can never cost a laser hit — only time. |

The battery sits at the top of a dead-end maintenance alcove off the south ring.
Collecting it is optional: it pays a one-time bonus but costs six extra steps and
crosses a wet cell next to a beam.

These three numbers are measured by a breadth-first search and asserted by the
tests, so the room cannot silently lose the trade-off it exists to demonstrate.

## The state

```python
(row, col, has_battery, previous_direction)
```

54 open cells × 2 battery values × 5 directions = **540 states**.

* `has_battery` is part of the state because the +10 bonus may only ever be paid
  once.
* `previous_direction` is part of the state because oil makes R-5 carry on the way
  it was already going. Without it, the same cell would behave differently
  depending on hidden history and the problem would no longer be Markovian.
  `previous_direction` records the direction R-5 *actually* moved, not the action
  it chose, and it is `NONE` after a reset, after a blocked move, after a laser hit
  and after a teleport.

## The actions

`UP`, `DOWN`, `LEFT`, `RIGHT`. The action is what R-5 *tries* to do; the floor it
is standing on decides what actually happens.

## The slipping model

The surface **under** R-5 decides the outcome, because that is the surface it
pushes off from. One rule, no exceptions.

| Surface | Distribution |
|---------|--------------|
| normal floor, start, exit, battery, teleporter, door | 100% the chosen action |
| wet floor `~` | `p_weak` the chosen action (default 0.80), the remainder split evenly between the two sideways directions (0.10 / 0.10) |
| frozen floor `%` | `p_strong` the chosen action (default 0.60), the remainder split evenly sideways (0.20 / 0.20) |
| oil `O` | `p_oil` continue in `previous_direction` (default 0.55), 0.10 split evenly sideways, everything left over to the chosen action (0.35) |

**Oil with no momentum.** When `previous_direction` is `NONE`, or when it already
equals the chosen action, the momentum share is given to the chosen action instead:
0.90 for the action and 0.05 / 0.05 sideways. This keeps the distribution summing
to exactly 1 and avoids listing the same outcome twice.

**Colliding shares are added, not overwritten.** The momentum direction is often
one of the two sideways directions. For example on oil with momentum `RIGHT` and a
chosen action of `UP`, the sideways directions are `LEFT` and `RIGHT`, so `RIGHT`
receives both the momentum share and one sideways share: 0.55 + 0.05 = 0.60. Adding
them is what keeps the total at exactly 1.

`p_weak`, `p_strong` and `p_oil` are adjustable from the interface, and the
remainders are always recomputed so the probabilities still sum to 1. The test
suite checks every state-action pair at several settings.

## The rewards

Every transition pays the step cost, and event rewards are **added on top of it**.
This one convention is used everywhere.

| Event | Reward |
|-------|--------|
| Each step | −1 |
| Wall collision or refused one-way passage | −1 + (−3) = **−4** |
| Entering a laser | −1 + (−30) = **−31** |
| First battery collection | −1 + 10 = **+9** |
| Using the teleporter | −1 + 5 = **+4** |
| Reaching the control panel | −1 + 100 = **+99** |

The laser penalty is adjustable from the interface. The other values are fixed
constants in `rooms/room1/environment.py`.

## What each thing does

**Lasers.** Entering a beam gives the laser penalty and moves R-5 to
`(start_row, start_col, has_battery, NONE)`. A battery already collected stays
collected. **The episode does not end** — R-5 simply has to walk out again. Laser
hits are counted during simulation and shown in the interface. Because R-5 never
occupies a laser cell, the laser states are unreachable; the policy view still
shows an arrow for them, since they remain part of the state space.

**Walls and the chamber edge.** R-5 stays where it is, pays the wall penalty, and
its `previous_direction` is cleared, because a robot that has been stopped is no
longer carrying momentum.

**The one-way door** at (4,7) may only be crossed left to right. It is refused both
ways round: R-5 cannot *enter* the cell moving any other direction, and once
standing on it, cannot *leave* in any direction but right. A refused passage costs
the wall penalty. Enforced identically in the transition model, in `step`, and in
both renderers.

**The teleporter.** Stepping onto either pad moves R-5 to the other pad and pays
the teleporter bonus. It resolves **exactly once per transition** — the destination
tile is never re-examined — so two pads can never bounce the robot back and forth
forever. `previous_direction` is cleared on arrival.

**The battery.** Entering the battery cell with `has_battery` false sets it true
and pays the bonus once. Arriving again pays only the step cost. In the animation
the battery is drawn until the frame where it is collected, then disappears in a
burst of amber particles.

**The exit.** Reaching the control panel pays the exit reward and ends the episode
successfully. It does **not** require the battery by default. An experimental
interface toggle, *Require battery before exit*, turns it into a locked door that
refuses entry without the battery; it is off by default so the main task is
unchanged.

**Merged outcomes.** Outcomes are merged on the whole `(next_state, reward, done)`
triple, so identical outcomes are combined into one entry while the teleporter and
battery bonuses can never be counted twice. For every state and action the
probabilities sum to 1 within 1e-9; the verification script reports the worst error
across all 2160 state-action pairs (currently exactly 0).

## The algorithms

Both are in `rooms/room1/dp_solver.py`, written as plain loops with no numpy, no
vectorisation and no abstractions. Everything is built from one helper:

```python
def q_value(env, state, action, values, gamma):
    total = 0.0
    for probability, next_state, reward, done in env.transitions(state, action):
        future_value = 0.0 if done else values[next_state]
        total += probability * (reward + gamma * future_value)
    return total
```

**Value Iteration** repeatedly applies the Bellman optimality update

```
V(s) <- max_a  sum_s'  P(s'|s,a) [ R(s,a,s') + gamma V(s') ]
```

to every non-terminal state, and stops when the largest change in a sweep drops
below `theta` or `max_sweeps` sweeps have been done. The policy is then read off
with `pi(s) = argmax_a Q(s,a)`. Terminal states keep a value of 0 and are skipped.

**Policy Iteration** starts from an arbitrary policy (always `UP`), then alternates
policy evaluation (applying `V(s) <- Q(s, pi(s))` until the values settle) with
policy improvement (making the policy greedy), stopping when an improvement step
changes nothing at all.

Ties are broken by the fixed order of `env.actions()`, so the same value table
always produces exactly the same policy.

Every run records the value table, the policy, the sweep count, the runtime,
`V(start)`, the final Bellman residual, and the per-sweep history of the largest
value change, of `V(start)`, and of how many states changed their preferred action.

## The interface

Selectable in the sidebar: **Value Iteration**, **Policy Iteration** or
**Compare Both**; gamma, theta and maximum sweeps; the three floor probabilities;
the laser penalty; *require battery before exit*; the simulation seed; the
animation speed; the number of episodes per measurement; the view mode; and which
slice of the state space the value and policy views should show.

Buttons: **Solve Room**, **Run Agent**, **Pause / Resume**, **Reset Animation**,
**Compare Algorithms**, **Run Experiment**, **Save Results**, **Load Results**.

**Nothing is recomputed by accident.** Solving, simulating and running experiments
happen only when the matching button is pressed, and results are kept in
`st.session_state`. Moving a slider does not re-solve the room: the page notices
that the settings no longer match the stored solution and says so, leaving the old
result on screen until Solve Room is pressed again. There is a test for this.

## Rendering

There are three views of the same chamber, all sharing one palette defined in
`ui/theme.py`.

**The isometric chamber** (`renderers/web/room_iso.html`) is a self-contained
HTML and JavaScript canvas renderer, embedded in an iframe. It is a **2.5D
isometric renderer, not a true 3D engine**: depth comes from perspective,
raised wall blocks, floor thickness, layered shading, shadows and lighting, not
from a 3D scene graph. There is no WebGL, no Three.js and no external asset — no
extra dependency, and nothing to download at runtime.

It draws isometric floor panels with thickness and seams, raised wall blocks with
a lit top face and two shaded sides, wall-mounted laser emitters with translucent
beams, teleporter pads with rotating rings and energy columns, the battery, the
green control console, the one-way gate with an animated arrow, drifting dust, and
R-5 itself: a metal body with a cyan core, a direction indicator, an `R-5` marking,
a cyan pool of light on the floor and a locator column so it can always be found.

Animation is driven by the recorded frames, with movement tweening plus dedicated
sequences for a laser hit (walk into the beam, sparks, red flash, camera shake,
reappear at the start), a teleport (step onto the pad, dissolve upward,
materialise at the other pad), a wall bump (nudge and recoil), a battery pickup
and reaching the exit.

Camera modes: **3D view**, **Tactical** and **Top-down**, with rotate, zoom, pan,
reset and follow-R-5. Visual quality can be set to **High**, **Medium** or **Low**,
which changes shadows, particles and reflections only — the recorded episode and
every number measured from it are identical at all three settings.

**The animation clock lives inside the iframe.** Playback controls (play, pause,
step, speed, restart) are drawn in the chamber itself, so using them never triggers
a Streamlit rerun and `time.sleep` is never called. The playback position is stored
under an episode fingerprint that depends only on the map and the frames, so
changing a view setting resumes where it was instead of restarting, while a
genuinely new episode starts from the beginning.

**The Matplotlib fallback** (`renderers/fallback_renderer.py`) draws the same
chamber in the same isometric style in pure Python, as a still frame with no
animation. It is available in the Analysis view, is what the tests use to prove
every tile type can be drawn, and is what remains if the canvas cannot run.

**The analysis views** (`plots/room1_plots.py`) are top-down Matplotlib maps for
interpretation: the chamber with the walked path, the `V(s)` heatmap with numbers
in the cells and walls left blank, and the preferred action in every cell as an
arrow. Because the state has four parts, the value and policy views need a slice
of it; the sidebar chooses the battery flag and the previous direction, defaulting
to no battery and `previous_direction = NONE`.

## Replay

Every frame stores enough to redraw the episode without the environment:

```
step, state, row, col, has_battery, previous_direction, from_row, from_col,
action, actual_direction, reward, cumulative_reward, event, laser_hit, slipped,
momentum, blocked, used_teleport, battery_collected, reached_exit, done, status
```

Because each frame records the direction that actually happened, **replay never
resamples anything** — it walks the stored list. The same seed and policy always
produce an identical recording, field for field; there is a test for that, and the
verification script checks it too.

## The graphs

Room 1 is not trained in episodes, so its graphs are about convergence:

1. **Convergence** — the largest value change per sweep, on a logarithmic axis,
   with the `theta` threshold marked.
2. **V(start)** — the value of the start state as the sweeps go by.
3. **Policy changes** — how many states changed their preferred action per sweep.
4. **Value heatmap** — the final value table for the chosen state slice.
5. **Policy arrows** — the preferred action in every cell.
6. **Algorithm comparison** — Value Iteration against Policy Iteration.

All of them are built through `plots/style.py`, so no graph is ever a white
rectangle in a dark interface.

## The experiments

Every experiment solves the room once per configuration and then runs a batch of
seeded episodes, reporting **mean ± standard deviation**. No number anywhere comes
from a single episode. Results are described as the *best-performing value among
the tested configurations*, never as an optimal hyperparameter.

Reported per configuration: `V(start)`, success rate, mean return, mean steps,
mean laser hits, teleporter use rate, battery collection rate, and the route the
plan actually chose.

### The gamma experiment

Discount factors 0.70, 0.85, 0.95, 0.99. Measured, 20 episodes each, all other
settings at their defaults:

| gamma | V(start) | Success | Return | Route chosen |
|-------|----------|---------|--------|--------------|
| 0.70 | −3.20 | 100% | 49.7 ± 43.4 | Teleporter |
| 0.85 | −3.06 | 100% | 53.2 ± 35.5 | Teleporter |
| 0.95 | 16.43 | 100% | 75.3 ± 2.3 | South ring |
| 0.99 | 57.03 | 100% | 75.3 ± 2.3 | South ring |

A low discount factor makes R-5 impatient: a distant +100 is worth so little that
the 17-step teleporter route beats the 23-step ring, even though the teleporter
approach can cost a laser hit — which is why the return varies so much at
gamma 0.70. A high discount factor makes the safe, slow route clearly better.

### The slipping experiment

This is the headline result of Room 1. Three ice settings, 20 episodes each,
gamma 0.95:

| Ice risk | `p_weak` / `p_strong` | V(start) | Success | Return | Route chosen |
|----------|----------------------|----------|---------|--------|--------------|
| Low | 0.90 / 0.85 | 29.93 | 100% | 79.6 ± 14.8 | **Frozen shaft** |
| Medium | 0.80 / 0.60 | 16.43 | 100% | 75.3 ± 2.3 | **South ring** |
| High | 0.60 / 0.35 | 12.99 | 100% | 68.2 ± 18.7 | **South ring** |

When the ice is reliable the plan gambles on the 13-step frozen shaft. When it is
not, the plan abandons the shaft for the 23-step ring instead. Nothing was tried
and no experience was gathered: both plans were computed from the model before R-5
moved at all. There is a test asserting this flip still happens.

### Custom sweeps

The interface can also sweep `gamma`, `weak_ice_intended`, `strong_ice_intended`,
`oil_momentum` and `laser_penalty` over a preset list of values.

## Saving and loading

Solved results are written to `saves/` as readable JSON. Each file records the
format version, a fingerprint of the map, the map itself, the state-space size and
the state format, the algorithm, all the environment settings, the convergence
history, the value table and the policy. Since JSON has no tuples, each state is
stored as a string like `"4,9,1,2"`.

Loading checks the format version, the map fingerprint and the state count, and
refuses the file with a readable message if any of them disagree — so an old
policy can never be displayed on top of a map that has since been edited.

---

## What the agent knows, and what it is doing

**What does it know?** Everything. Before moving at all, R-5 has the full model of
the chamber: every cell, every laser, how slippery each floor is, exactly what each
action can lead to and what reward follows. That is the list
`env.transitions(state, action)` returns.

**What is it learning?** Nothing, in the usual sense. It never tries an action to
find out what happens. It computes the value of every state and the best action
from the Bellman equations, sweeping over all 540 states until the numbers stop
changing. This is planning, not learning from experience.

**Why can it avoid a laser it never touched?** Because the laser is already in the
model. A sweep can see that a sideways slip on the frozen shaft leads to a beam
worth −31, and that cost is folded into the value of the cell before R-5 ever steps
on it. An agent that had to learn from experience would have to be hit first.

---

# Room 2 — The Broken Bridge Sector

*Lab sector B-04 · Structural failure detected*

The emergency shutdown dropped several bridges into the maintenance shaft that runs
through the middle of the industrial sector. R-5 has to pick up the security keycard
and reach the exit door on the far side.

Unlike Room 1, **the model is unknown**. `Room2Env` deliberately has no
`transitions()` method: the only way to find out what an action does is to take it.
The room is solved with **SARSA**, learning from experience, episode by episode.

## The sector

```
      c0 c1 c2 c3 c4 c5 c6 c7 c8 c9
 r0    #  #  #  #  #  #  #  #  #  #
 r1    #  T  B  B  B  B  B  B  T  #     the upper walkway — the long, safe way
 r2    #  B  #  #  #  #  #  #  B  #
 r3    #  B  #  P  P  P  #  #  B  #     the maintenance shaft
 r4    #  B  #  P  P  P  #  #  B  #
 r5    #  B  B  C  C  C  B  B  B  #     the bridge — short, with a drop either side
 r6    #  B  #  P  P  P  #  #  B  #
 r7    #  K  #  P  P  P  #  #  B  #     the keycard, on the way up from the start
 r8    #  S  #  P  P  P  #  #  E  #     start platform · shaft · exit door
 r9    #  #  #  #  #  #  #  #  #  #
```

| Symbol | Meaning |
|--------|---------|
| `S` | Start platform — row 8, col 1 |
| `E` | Exit door — row 8, col 8 (locked until the keycard is held) |
| `#` | Metallic wall |
| `B` | Bridge — sound, behaves as floor |
| `C` | Collapsing bridge — breaks the moment it is crossed |
| `P` | Maintenance shaft (pit) — entering one ends the episode |
| `K` | Security keycard |
| `T` | Maintenance platform — behaves as floor |

Contents: 1 start, 1 exit, 1 keycard, 21 bridges, 3 collapsing bridges, 24 shaft
cells (6 of which can actually be fallen into), 2 platforms, 54 walls. All 28
walkable cells are reachable; the tests check this.

Room 2 has **its own tile alphabet**, because it reuses two letters that mean
something else in Room 1: here `B` is a bridge rather than a battery, and `T` is a
maintenance platform rather than a teleporter. Only `S`, `E`, `#` and `.` are shared
with `core/tiles.py`.

### Why the sector is shaped like this

The layout is deliberately the shape of the classic **cliff-walking** problem,
because that is the situation in which SARSA and Q-Learning visibly disagree.

| Route | Steps | Collapsing bridges | Perfect return | Risk |
|---|---|---|---|---|
| **Lower bridge** | **13** | 3 | **+106** | all three bridge cells have a drop *above and below*, so 3 of the 4 actions there are fatal |
| **Upper walkway** | 21 | 0 | +104 | not one cell on it has a drop beside it |

The short route pays **more** when walked perfectly. A method that values behaving
perfectly therefore prefers it; a method that values the policy it is actually
following — exploration mistakes included — should not. That disagreement is the
point of the room.

The keycard sits one step above the start platform, on the way up, so neither route
is penalised for collecting it.

## The state

```python
(row, col, has_keycard)
```

28 walkable cells + 6 enterable shaft cells, × 2 keycard values = **68 states**.
`has_keycard` is part of the state because the exit door stays locked until the card
is held, so the same cell means two different things.

## The actions

`UP`, `DOWN`, `LEFT`, `RIGHT`. The sector is **deterministic** — an action always
moves R-5 the way it was aimed, and nothing slips. The danger in this room comes
from the agent's own epsilon-greedy exploration, not from the environment. That is
exactly the distinction SARSA is sensitive to and Q-Learning is not.

## The reward function

Every transition pays the step cost, and event rewards are **added on top of it** —
the same convention Room 1 uses.

| Event | Total |
|-------|-------|
| Each step | −1 |
| Wall collision, or walking into the locked door | −1 + (−5) = **−6** |
| Cross a collapsing bridge | −1 + (−2) = **−3** |
| Collect the keycard (once only) | −1 + 25 = **+24** |
| Reach the exit door | −1 + 100 = **+99** |
| Fall into the shaft | −1 + (−100) = **−101** |

All six are adjustable from the interface.

## The mechanics

**Collapsing bridges.** Crossing one breaks it. From that moment on, for the rest of
the episode, that cell behaves as a pit — so the layout changes while the episode is
running, and `reset` rebuilds it. The replay records every collapse, and each frame
carries the full list of bridges lost so far, which is how the animation redraws the
sector correctly at any point without re-running anything.

**Pits.** Entering one ends the episode immediately with the pit penalty. Most of the
shaft is walled off and is scenery; six cells can actually be fallen into.

**The keycard.** Collected once, for a one-time bonus. Until it is held the exit door
is treated as a locked wall: R-5 stays put and pays the wall penalty. Once held, the
door reads green in the sector view and opens on contact.

## SARSA

Implemented by hand in `rooms/room2/sarsa_agent.py`:

```
Q(s,a) <- Q(s,a) + alpha * [ r + gamma * Q(s',a') - Q(s,a) ]
```

`a'` is the action the agent is *actually going to take next*, drawn from the same
epsilon-greedy policy. That makes SARSA **on-policy**: it learns the value of the
policy it is really following, exploration and all. Q-Learning is implemented in the
same file for comparison and differs in exactly one line — it uses
`max_a' Q(s',a')`, the value of behaving perfectly, and is therefore **off-policy**.

Supported: epsilon-greedy exploration, learning rate, gamma, epsilon decay, minimum
epsilon, maximum episodes. Training is episode-based, happens only when **Train** is
pressed, and the result is kept in `st.session_state`.

Two details of the implementation matter enough to call out:

* **Ties are broken randomly.** A fresh Q-table is all zeros, so every action is
  tied. Always taking the first one marched the robot up the west wall on every early
  episode and the bridge was never explored at all.
* **The table starts optimistic** (`q_init`, default 120). The safe walkway can be
  walked by accident, so an agent starting from zeros finds it immediately, locks on,
  and never reconsiders — the bridge was entered 17 times in 2000 episodes in
  testing, far too few for the exit reward to travel back along it. Starting every
  entry above any reward the room can pay makes untried actions look better than
  tried ones, so both routes get evaluated.

Two default values are also chosen for this room specifically: **gamma 0.99**,
because the whole argument rests on a rare and very expensive outcome, and **minimum
epsilon 0.10**, because SARSA and Q-Learning converge to the same policy once
exploration stops — the difference only exists while the agent is still taking the
occasional random step.

## The measured result

This is the room's headline finding, from `scripts/verify_room2.py` — 2500 episodes,
averaged over the last 400, all other settings at their defaults:

| State representation | Method | Bridge used | Mean reward | Pit falls |
|---|---|---|---|---|
| `(row, col, has_keycard)` — the assignment's | SARSA | 0% | 96.3 | 0% |
| `(row, col, has_keycard)` — the assignment's | Q-Learning | 0% | 96.2 | 0% |
| `+ collapsed bridges` — Markov | SARSA | 0% | 96.8 | 0% |
| `+ collapsed bridges` — Markov | **Q-Learning** | **99%** | **65.0** | **18%** |

The bottom two rows are the textbook cliff-walking result: Q-Learning learns the
optimal-but-risky route and earns far less online reward because it keeps falling in;
SARSA takes the long way and earns more.

**The top two rows are the more interesting finding, and it is a consequence of the
state the assignment specifies.** With `(row, col, has_keycard)` the agent cannot see
which bridges have already collapsed, so the room is not Markovian from where it is
standing: the value of stepping towards a bridge becomes an average of "+82 if it is
still there" and "−101 if it is not". Measured, that averages out to about +59
against the safe route's +64, so **both** methods avoid the bridge — and for a reason
that has nothing to do with on-policy versus off-policy.

The interface therefore has a switch, **"Include collapsed bridges in the state"**,
off by default. The assignment's representation is what runs unless it is turned on.
Both results are shown side by side in the *State representation* experiment tab,
because the comparison makes the point that what an agent is allowed to observe can
matter more than which update rule it uses.

## Training graphs

All ten the brief asks for, in `plots/room2_plots.py`, all built through
`plots/style.py` so they match the rest of the application:

1. **Episode reward** — raw, with the moving average over it, and both routes'
   perfect returns marked as reference lines.
2. **Moving-average reward** — window selectable from the sidebar.
3. **Episode length** — with both route lengths marked.
4. **Success rate** — running average of episodes reaching the exit.
5. **Epsilon decay** — worth reading beside the reward curve.
6. **Q-value convergence** — mean |Q| over the table; flat means settled.
7. **Pit falls** — the share of episodes ending in the shaft.
8. **Bridge usage** — how often the risky route was taken. This is the graph where
   the on-policy/off-policy split is visible.
9. **Keycard collection rate** — reaches 100% and stays, since no episode can
   succeed without it.
10. **Training time** — wall clock as training progressed.

Plus a policy view showing the preferred action in every cell.

## Experiments

Three tabs, all reporting **mean ± standard deviation** across several seeds, since
training is stochastic and a single run says very little:

* **Parameter sweep** over learning rate, gamma, epsilon decay, initial epsilon or
  minimum epsilon. Each value reports mean reward, success rate, steps, pit falls,
  bridge usage, greedy return, training time and the route chosen.
* **SARSA vs Q-Learning** on the same sector with the same settings.
* **State representation** — the headline experiment above, training both methods
  under both state representations.

Results are described as the *best-performing value among the tested
configurations*, never as an optimal hyperparameter.

## Replay and animation

The sector is drawn by the same isometric canvas engine as Room 1, with the same
camera modes, quality settings, in-canvas playback controls and palette. Room 2 adds:

* open shafts drawn as holes with visible depth, with a warning rim on only the cells
  that can actually be fallen into;
* bridges with girders, and collapsing bridges with amber hazard striping;
* a collapse animation — the deck shakes, cracks spread, then it tilts and drops away
  leaving a hole;
* a pit-fall animation — R-5 walks over the edge and accelerates out of sight, with a
  red alarm wash and a camera jolt;
* the keycard turning slowly above its platform, and the exit door red and shut until
  the keycard is held, then green and open.

Replay reproduces actions, rewards, bridge collapses, the keycard pickup, pit falls
and the door unlocking **without resampling anything** — each frame stores the
direction that happened and the bridges lost so far. The same policy and seed always
give an identical recording, field for field; the tests check this.

## Educational section

The page explains, with the update rules side by side, why SARSA suits a sector whose
model is unknown, what on-policy learning means, and why it avoids dangerous
shortcuts that Q-Learning takes. It also shows **one real update from the current
training session** — state, action, reward, next state, next action, `Q` before and
after — with the arithmetic worked through, so the equation is tied to numbers the
agent actually produced rather than invented ones.

---

# Room 3 — The Reactor Control Chamber

*Lab sector C-07 · Power grid offline*

The lift to the surface has no power. R-5 has to restart three generators **in order**
— A, then B, then C — which unlocks the reactor blast door, and then leave through it,
while a security robot walks a fixed patrol around the chamber and two sliding doors
open and close on a timer.

Like Room 2 the model is unknown, and the room is solved with **Q-Learning**, which is
**off-policy**: it learns the value of behaving greedily while it is still exploring.

## The chamber

```
      c0 c1 c2 c3 c4 c5 c6 c7 c8 c9
 r0    #  #  #  #  #  #  #  #  #  #
 r1    #  S  .  .  .  .  .  A  #  #     north corridor · Generator A
 r2    #  .  #  #  D  #  #  .  #  #     D = sliding door into the shaft
 r3    #  X  #  #  .  #  #  X  #  #     X = electrical hazard
 r4    #  .  #  #  .  #  #  .  #  #     the service shaft runs down the middle
 r5    #  X  #  #  .  #  #  X  #  #
 r6    #  .  #  #  D  #  #  .  #  #
 r7    #  C  .  .  .  .  .  B  #  #     south corridor · Generators C and B
 r8    #  #  #  #  R  #  #  #  #  #     R = reactor blast door
 r9    #  #  #  #  E  #  #  #  #  #     E = the exit, behind it
```

| Symbol | Meaning |
|--------|---------|
| `S` | Start — row 1, col 1 |
| `A` `B` `C` | The three generators, which must be started in that order |
| `R` | Reactor blast door — impassable until all three are running |
| `E` | Exit — behind the blast door |
| `D` | Sliding door — open or shut depending on where the patrol is |
| `X` | Electrical hazard — expensive to cross, but survivable |
| `#` | Wall |

31 walkable cells, all reachable. The shortest legal mission — start, A, B, C, blast
door, exit — is **23 steps** with the doors cooperating.

### Why the chamber is shaped like this

The generators sit at three corners of a ring, so the order A → B → C forces R-5
around the outside. Two vertical **service shafts** cut through the middle: they are
shorter, but each one contains two electrical hazards, and the sliding doors at their
ends are only open half the time. The security robot patrols the ring itself, **against**
the direction the mission pushes R-5 — so on the ring they meet head-on, while the
shafts are never patrolled. That is the trade the room is about: pay a certain −25
twice in the quiet shaft, or take the free route where a −100 is walking towards you.

## The state

```python
state = (row, col, stage, guard_index)
```

31 cells × 4 stages × 24 patrol positions = **2,976 states**.

* `stage` is 0 before anything, 1 after A, 2 after A and B, 3 with all three running
  and the blast door unlocked. It is the memory that makes the ordering rule legal in
  a Markov state.
* `guard_index` is *where on its patrol the security robot is*, not the robot's cell —
  the same cell appears twice in the loop, walked in opposite directions.

**The sliding doors do not need a clock.** The patrol is 24 cells long and the doors
run a 4-step cycle, so the door phase is computed from `guard_index` alone:

```python
def door_is_open(guard_index):
    return (guard_index // DOOR_HALF_PERIOD) % 2 == 0
```

24 divides evenly by 4, so the pattern repeats exactly once per patrol lap. This is
why the state stays Markov without a fifth component: everything that varies with time
is a function of something already in the state.

## The actions

Five: `UP`, `DOWN`, `LEFT`, `RIGHT` and **`WAIT`**. Waiting holds position while the
patrol and the doors move on, which is the only way to let the robot walk past or a
shut door open. `WAIT` lives in `core/actions.py` alongside the four directions but is
deliberately **not** in `ALL_DIRECTIONS`, so Rooms 1 and 2 are unaffected by it.

## The rewards

| Event | Reward |
|---|---|
| Each step | −1 |
| Walking into a wall, a shut sliding door, or the locked blast door | −5 (and stay put) |
| Crossing an electrical hazard | −25, **episode continues** |
| Meeting the security robot | −100, **episode ends** |
| Starting generator A or B in the right order | +20 |
| Starting generator C — the last one | +20 +40 |
| Reaching the exit | +150 |

Touching a generator out of order does nothing at all: no bonus, no penalty, just the
step cost. Rewards add, so the last generator pays +59 net and the exit +149.

## Q-Learning, and how it differs from SARSA

```
Q(s,a) ← Q(s,a) + α [ r + γ max_a' Q(s',a') − Q(s,a) ]
```

The `max` is the whole difference. SARSA (Room 2) bootstraps from the action it is
actually going to take next, exploration included; Q-Learning bootstraps from the best
action available, whether or not it takes it. So Q-Learning learns the greedy policy's
values while still behaving ε-greedily — which is what *off-policy* means.

Both are implemented, share `core/tabular.py`, and can be run against each other from
the page. Defaults: α 0.15, γ 0.98, ε 1.00 → 0.05 with decay 0.999, 5,000 episodes,
Q initialised to 0.

## The measured result

5,000 episodes, α 0.15, seed 0, on a 2020 laptop:

| | |
|---|---|
| Training time | **3.7 s** |
| Success over the last 500 training episodes | **80 %** |
| All three generators started, last 500 episodes | 84 % |
| Caught by the robot, last 500 episodes | 20 % (ε never falls below 0.05) |
| **Greedy policy over 30 seeded runs** | **100 % escape, return 187.0 ± 0.0, 43 steps** |
| Route the greedy policy chooses | the **service shaft** — it pays the hazards to stay off the patrol |

Q-Learning against SARSA, 3,000 episodes, 2 seeds each, identical settings:

| Algorithm | Online reward | Online success | Greedy success | Caught | Hazards / episode |
|---|---|---|---|---|---|
| Q-Learning | +17.7 | 59 % | 100 % | 41 % | 1.20 |
| SARSA | +99.6 | 73 % | 100 % | 27 % | 0.36 |

Both end up with a policy that escapes every time. The difference is what happens
*while learning*: Q-Learning is caught far more often and collects less reward on the
way, because it keeps valuing the risky route by what a perfect agent would get there
— the same on-policy/off-policy split Room 2 shows, here with a moving hazard instead
of a cliff.

The learning rate matters more than anything else on this room. At 3,000 episodes:

| α | Online success | Greedy escape |
|---|---|---|
| 0.05 | 5 % | **0 %** — it has not finished the sequence once |
| 0.15 | 55 % | 100 % |
| 0.40 | 76 % | 100 % |

These are the **best-performing values among those tested**, not optimal ones.

## Graphs

Eleven training graphs, plus the chamber map and the policy arrows: episode reward,
its moving average, episode length, escape rate, generator-completion rate, average
stage reached, security-robot detections, hazard hits, mean |Q|, blast-door unlock
frequency, ε decay and cumulative training time. The policy view slices the table by
`stage` and `guard_index`, since a policy for a 4-component state cannot be drawn on
one grid.

## Educational section

The page derives the Bellman **optimality** equation, sets the Q-Learning and SARSA
updates side by side, explains what off-policy means in this chamber specifically, and
then works through **one real update taken from the training run just performed** —
state, action, reward, `max_a' Q(s',a')`, target, TD error and the new value.

---

# Room 4 — The Drone Wind Tunnel

*Lab sector D-07 · Aerodynamics bay*

The shaft above sector C is only crossable by air. R-5 pilots a test drone from the
release point in the south-west corner to a landing pad in the north-east corner of a
10 m × 10 m hall, through four fans and four obstacles — and it has to arrive **slowly**.

This room is where the state stops being countable. Position and velocity are real
numbers, so there is no table to fill in: the values are approximated with
**Semi-Gradient SARSA over hand-written tile coding**.

## The hall

10 m × 10 m. Release point (1.2, 1.2); landing pad at (8.6, 8.6) with a radius of
0.6 m — **10.5 m away** in a straight line, which the obstacles block.

| Fan | At | Pushes | Strength | Radius |
|---|---|---|---|---|
| west intake | (0.0, 5.0) | east | 1.5 | 4.5 m |
| tunnel | (5.0, 4.2) | north | 3.4 | 3.0 m |
| pad guard | (8.0, 6.4) | south | 2.2 | 2.6 m |
| crosswind | (7.2, 8.8) | west | 1.6 | 2.4 m |

Obstacles: a round turbine housing at (4.6, 6.6), a round support column at
(6.4, 3.0), a rectangular maintenance platform and a hanging barrier. Together they
close the diagonal, so the drone has to commit to a route above or below the middle.

**The wind is a function of position only** — the same point always gives the same
force, summed over the fans with a smooth falloff. That is what keeps the state
Markov: `(x, y, vx, vy)` is enough, with no wind history. Optional turbulence is
seeded noise and is off by default.

## The state and the dynamics

```python
state = (x, y, vx, vy)          # metres and metres per second, all continuous
```

Five actions: `COAST`, and thrust up / down / left / right. **Thrust changes velocity,
not position**:

```python
vx = (vx + thrust_x + wind_x) * drag
vy = (vy + thrust_y + wind_y) * drag
x += vx * dt
y += vy * dt
```

Defaults: `dt` 0.25 s, thrust 1.1, drag 0.90, speed clipped at 3.0 m/s. Each move is
walked in **4 substeps**, and both the obstacles and the landing pad are tested at
every substep — otherwise a fast drone would step straight over a 0.6 m pad without
the landing ever being noticed. (That was a real bug, caught by a test; the fix is the
`reached_pad` flag inside the walk in `environment.py`.)

## Landing, hard landing, crash

Arriving on the pad is judged **on speed as well as position**:

| Arrival speed | Outcome | Reward |
|---|---|---|
| ≤ 0.7 m/s | **Safe landing** — episode ends | +150 |
| between 0.7 and 2.4 m/s | **Too fast to land** — bounces off at −40 % of its velocity and carries on flying | −40 |
| ≥ 2.4 m/s | **Crash** — episode ends | −100 |

Other rewards: −0.5 per step, −0.05 for firing a thruster, +2.0 × the metres of
progress made towards the pad, −30 for hitting an obstacle, −25 for hitting a wall.
An episode is 220 steps at most.

## Tile coding, written out by hand

`rooms/room4/tile_coder.py` is deliberately plain: no library, no hashing.

* The 4-D state is normalised into [0, 1] per dimension.
* **8 tilings**, each an 8 × 8 × 8 × 8 grid, each offset by a different fraction of a
  tile — odd multiples of 1/(2 × 8), the classic choice, so the offsets spread out
  instead of lining up.
* Each action gets **its own block** of features, so the five actions can never share
  a weight: `index = action_offset + tiling × tiles_per_tiling + flat_cell`.
* 8 × 8⁴ × 5 = **163,840 weights**, of which exactly **8 are active** for any
  (state, action) — one per tiling.

```
Q(s,a) = Σ w[i]  for the 8 active i
w[i] ← w[i] + (α / num_tilings) × δ    for those 8 only
```

Dividing α by the number of tilings is what keeps eight simultaneously-active features
from taking eight full steps at once. The tests check the coder by construction *and*
empirically: every index in range across the corners of the state space, no overlap
between action blocks, nearby states sharing 8 of 8 tiles and distant ones 0 of 8.

## The measured result

Semi-Gradient SARSA, 1,200 episodes, α 0.30, γ 0.99, 8 × 8 tiling, seed 0:

| | |
|---|---|
| Training time | **8.7 s** |
| Success over the last 300 training episodes | **99 %** |
| Mean final distance to the pad | 0.58 m |
| Weights touched | 98,117 of 163,840 |
| **Greedy policy over 20 runs** | **100 % landed, return 157.9, 22 steps, touchdown at 0.60 m/s** |

### Generalisation to release points it has never used

The agent trains from **one** release point. It is then asked to fly from points it has
never left — that is the test of whether tile coding generalised or merely memorised.
1,200 episodes, 3 seeds:

| Release points | Seen in training | Landings | Mean return |
|---|---|---|---|
| Training start (1) | yes | **100 % ± 0** | +158.7 |
| Validation starts (2) | no | **100 % ± 0** | +131.5 |
| **Unseen test starts (5)** | **no** | **93 % ± 9** | +78.8 |

It transfers, and the drop is honest: the return falls by half on unseen starts even
where it still lands, because it arrives by a worse route.

Tile resolution, 1,200 episodes:

| Tiling | Weights | Online success | Greedy landings |
|---|---|---|---|
| 4 × 6 | 25,920 | 100 % | 100 % |
| 8 × 8 | 163,840 | 99 % | 100 % |
| 12 × 10 | 600,000 | 100 % | 100 % |

On a hall this size all three resolutions work; the interesting figure is the memory
column, which grows 23× between the coarsest and the finest for no measured gain. The
default stays 8 × 8 as the middle of the three tested.

## Graphs

Twelve training graphs — episode reward, moving average, episode length, landing
success, final distance, landing speed against the safe and crash thresholds,
collisions, ε decay, ‖w‖, mean |TD error|, active tiles, cumulative time — plus five
analysis plots: flight trajectories over the hall, the wind field as arrows, velocity
against time, wind and thrust per step, the landing approach, and a **value slice**
`max_a Q(x, y, vx, vy)` at a chosen velocity, which is how a 4-D value function gets
onto a 2-D page.

---

# Room 5 — The Adaptive Storage Facility

*Lab sector E-12 · Automated warehouse*

The last chamber is not one room. Every time R-5 enters, the facility has been
**rearranged**: shelves, crates, conveyors, maintenance robots, the data terminal and
the exit are all generated from a seed. R-5 has to reach the terminal, then the exit,
and it can only see what its radar reaches.

Rooms 1 to 4 each learn one map. This room asks a different question: can a policy be
learned that works in a warehouse it has **never been inside**? It is solved with
**Semi-Gradient Q-Learning over local features**.

## Procedurally generated layouts

`rooms/room5/layout.py` builds a 12 × 12 warehouse from a seed: shelf aisles, loose
crates, conveyor belts, one charging bay, a start, a data terminal and an exit, plus
patrolling maintenance robots. Every candidate is checked with breadth-first search on
**both legs** — start → terminal and terminal → exit — and rejected and rebuilt from
`seed + attempt × 7919` until it is solvable. An unsolvable warehouse is never handed
to the agent.

| Setting | Grid | Obstacles | Robots | Radar | Shortest mission |
|---|---|---|---|---|---|
| **Easy** (default) | 12 × 12 | 14 % | 1 | 6 | 18.3 steps |
| Medium | 12 × 12 | 22 % | 2 | 4 | 21.3 steps |
| Hard | 14 × 14 | 27 % | 3 | 3 | 25.0 steps |

Easy is the default because it is the setting at which the agent demonstrably learns
*and* generalises. The harder settings are selectable, and it does measurably worse on
them — that is what the stress test is for.

### The seed split, which the whole claim rests on

```
training   seeds 1000, 1001, 1002, …
validation seeds 2000, 2001, 2002, …
test       seeds 3000, 3001, 3002, …
```

Three separate ranges, so a layout used for training can never reappear as a test
layout. `splits_overlap()` asserts it and a test calls it. Every reported "unseen"
number below comes from the 3000 range.

## Partial observability — the radar

R-5 never receives the grid. Its whole observation is 22 numbers:

| Part | Size | What it is |
|---|---|---|
| static radar | 8 | ray-cast distance to the nearest shelf, crate or wall in eight compass directions, normalised by the radar range |
| dynamic radar | 4 | the same in four directions, for maintenance robots |
| target | 3 | direction and distance to the **current** objective, relative to R-5 |
| stage | 1 | 0 before the terminal, 1 after |
| nearest robot | 1 | distance to the closest robot |
| previous action | 5 | one-hot |

No absolute position, no seed, no map, no layout size. The consequence is worth being
explicit about: **the observation is not Markov.** Two different corners of two
different warehouses can look identical to the radar and yet need different actions.
The room demonstrates learning under exactly that limitation, and the ceiling it
imposes shows up in the numbers.

The fog of war on screen is drawn from the same radar, but the list of visible cells is
a display concern and is never part of the observation.

## The mechanics

* **Five actions**: up, down, left, right, wait.
* **Maintenance robots** patrol either a `loop` (round and round) or a `bounce`
  (out and back). Meeting one is −100 and ends the mission — including the **crossing**
  case, where R-5 and a robot swap cells without ever sharing one:

  ```python
  if after == moved_from and before == self.position:
      caught = True
  ```

* **Conveyors** carry R-5 one extra cell in their direction after it steps on, unless
  that cell is blocked or off the grid.
* **The data terminal** pays +40 once, and unlocks the exit. Standing on it again pays
  nothing.
* **The exit is locked** until the terminal has been used: walking into it early costs
  −2 and pushes R-5 back. Afterwards it is +200 and the mission is over.
* **A charging bay** pays +5, once.
* Other rewards: −1 per step, −5 for walking into an obstacle, −1 for being moved by a
  conveyor, −30 for running out of time, and +1.5 × the cells of progress made towards
  the current objective.

## Feature-based approximation

`rooms/room5/features.py` turns the 22 observed numbers into sparse binary features by
tiling each group (3 tilings, 6 bins), then gives every action its own block:
**48 active features out of 1,840**. Four feature sets can be compared from the page:

| Set | Contents |
|---|---|
| A | target only |
| B | target + static radar |
| C | target + both radars |
| D | everything, including the previous action (**default**) |

The update is the same shape as Room 4's, with the off-policy `max`:

```
Q(s,a) = w · x(s,a)
target = r                                if the mission ended
       = r + γ max_a' Q(s',a')            otherwise
w     += (α / active) × [target − Q(s,a)] × x(s,a)
```

Nothing in `x(s,a)` mentions which warehouse this is, which is why the same weights
mean something in a warehouse the agent has never entered.

## The measured result

2,500 episodes over **20 training layouts**, α 0.10, γ 0.97, feature set D, Easy,
3 seeds:

| | |
|---|---|
| Training time | **32–34 s** |
| Success over the last 500 training episodes | 82 % |
| Weights touched | 1,535 of 1,840 |
| Greedy on the **20 training layouts** | **70 %** |
| Greedy on **8 validation layouts** (unseen) | 50 – 75 % |
| Greedy on **12 test layouts** (never used for anything) | **42 – 75 %**, 58 % typical |

It transfers, and it is clearly worse on warehouses it has not seen. Both halves of
that sentence are the result.

### How many training layouts it takes

Same budget of episodes each time, evaluated on the same held-out layouts:

| Training layouts | On its training layouts | Validation | **Unseen test** |
|---|---|---|---|
| 5 | **100 %** | 38 % | **33 %** |
| 20 | 70 % | 50 % | **58 %** |
| 50 | 36 % | 38 % | **58 %** |

With five warehouses it solves every one of them and fails two thirds of the unseen
ones — a 67-point gap, which is **memorisation**, not understanding. Twenty closes most
of that gap. Fifty spreads the same 2,500 episodes so thinly that even the training
score falls, while unseen performance holds — the training number stops being flattering
and starts being honest.

### Against trivial baselines

On the same 12 unseen test layouts:

| Policy | Missions completed | Mean return |
|---|---|---|
| Random actions | 0 % | −439 |
| Walk straight at the objective | 0 % | −1,665 |
| **The learned policy** | **58 %** | −243 |

Walking straight at the target is worse than random: it jams itself against shelves
and keeps paying the collision cost.

## Graphs

Fourteen training graphs — episode reward, moving average, episode length, training
success, **validation success on held-out layouts**, terminal-reached rate, robot
collisions, static collisions, timeout rate, ε decay, mean |TD error|, ‖w‖, cumulative
time, active features — plus the generalisation bar chart (training / validation /
unseen side by side), the radar readings over one mission, distance to the nearest
robot over time, the action mix, and the layout map with the route and the patrol
routes drawn on it.

## What this room does *not* show

Being explicit, because it is easy to overclaim:

* It does **not** solve every warehouse. Roughly two in five unseen Easy layouts are
  failed, most often by running out of time behind a shelf wall.
* It does **not** generalise across difficulty settings. Trained on Easy and dropped
  into Hard, it does much worse; the stress test measures exactly this and labels the
  rows as a distribution shift.
* There is **no deep network** anywhere. This is a linear model over hand-designed
  local features, and its ceiling is the ceiling of that representation.
* The observation is **not** Markov (see above), so no convergence guarantee applies —
  what is reported is measured behaviour, not a proof.
* The parameters are **starting constants that were sanity-checked**, not tuned. Every
  comparison is described as the best-performing value among those tested.

---

# After Room 5

Escaping the storage facility with all five components ends the game. The completion
screen (`ui/completion.py`) shows the gate open onto the surface, the full inventory,
and a table of every room with its method, what it demonstrated and the result
measured — and offers a replay of any chamber or a return to the main menu.

---

## Tests

```bash
.venv/bin/python -m pytest tests -q
```

324 tests, all passing:

| File | Covers |
|------|--------|
| `tests/test_room1_map.py` | grid size, one start, one exit, two teleporters, lasers, one battery, one-way door, valid characters, reachability, the three route lengths, the map fingerprint |
| `tests/test_room1_env.py` | the state space and its battery and direction components, probabilities summing to 1 at several settings, outcome merging, the ice and oil distributions, walls, lasers (position, penalty, battery kept, episode continues), the one-time battery, the teleporter pairing and its lack of a loop, the one-way door both ways, the exit terminating, the require-battery toggle, reproducible stepping |
| `tests/test_room1_dp.py` | both methods converging, the recorded histories, terminal values, policy coverage, the two methods agreeing with ties tolerated, matching value tables, the policy escaping reliably, the slipping flip, and `q_value` against a hand-computed expectation |
| `tests/test_room1_replay.py` | required frame fields, exact reproduction from a seed, continuity between frames, running reward totals, unexplained jumps, laser hits landing on the start, scene validity and JSON round-trip, every tile type in both renderers, no placeholder left in the page, and save/load round-trips including all three rejection cases |
| `tests/test_room2_env.py` | map size, one start/exit/keycard, the pieces the brief asks for, reachability, both routes and their economics; the state; that no model is exposed; bridges collapsing and becoming pits; reset rebuilding them; pits ending the episode; the door locked and then open; walls; determinism; the SARSA update checked by hand; that SARSA learns to escape; the recorded history; epsilon decay; reproducibility; the worked example; and the SARSA-vs-Q-Learning result under the Markov state |
| `tests/test_room2_replay.py` | required frame fields, exact reproduction from a seed, continuity, no jumps, running totals, the collapse list only growing, collapses recorded on the right frame, the keycard flag, falls recorded as falls; scene validity and JSON round-trip; every tile type in both renderers; all ten graphs plus their empty states; and save/load including all four rejection cases |
| `tests/test_room3.py` | the chamber and its fingerprint, the patrol running against the mission and dividing evenly by the door cycle, the state space, `WAIT`; generators only in order and paying once, the blast door shut then open, hazards hurting without ending the episode, the guard ending it (including the head-on case), sliding doors opening and shutting on the guard's phase, the door phase staying a function of the state; the Q-Learning update checked by hand against a worked example, the terminal update, ε-greedy with random tie-breaking, reproducibility; replay frames, continuity and the stage never going backwards; the scene, every tile type in both renderers, all fourteen graphs and their empty states; save/load and every rejection case |
| `tests/test_room4.py` | the hall, the fans, the wind field being a function of position only; thrust changing velocity, drag, speed clipping, walls and obstacles pushing the drone out, safe landing / hard landing / crash by arrival speed, the pad being tested inside the substep walk so a fast drone cannot fly through it, timeouts; the tile coder — index ranges across the state space, exactly one tile per tiling, action blocks never overlapping, nearby states sharing tiles and distant ones not, α divided by the number of tilings; `Q = w·x` against a manual sum, the semi-gradient update against a numerical example, only active weights changing, no NaN; generalisation to unseen release points; replay, the scene in continuous mode, both renderers, all eighteen graphs; save/load including the mismatched-tiling refusal |
| `tests/test_room5.py` | layout generation — one start / terminal / exit, both legs walkable by BFS, valid characters, reproducibility from a seed, obstacle density rising with difficulty, and training / validation / test seeds never overlapping; obstacles blocking, robots staying on their routes, loop against bounce patrols, same-cell **and** crossing collisions, `WAIT`, conveyors (including at the boundary), the terminal paying once and unlocking the exit, the exit locked before it and completing after, the charger; radar distances hand-checked and normalised, the observation containing no map, no seed and no absolute position, the fog of war staying out of it; feature shape constant, action blocks separated, every index in range, all four feature sets; `Q = w·x`, the update by numerical example, the terminal update, ε-greedy; reproducible training, no NaN weights, the validation curve on held-out layouts, shaping that cannot be farmed; exact replay and every frame field; the scene, both renderers, all nineteen graphs; save/load and every rejection case; the layout-count and stress experiments and the baselines |
| `tests/test_app_smoke.py` | all five rooms: the start screen, the briefing, the room pages, a locked room, the cleared screen; pressing Solve Room / Train / Evaluate / Run Agent / Run Drone / Replay / Pause / Save / Load / Generate Layout / Compare and the sweeps for real; that a slider does not re-solve or retrain; that Room 5 evaluates on unseen layouts; every room-cleared screen and the completion screen; that every table can be converted for display; and the experiment modules |

`scripts/verify_room1.py` … `verify_room5.py` run each room's whole checklist end to
end and print real numbers rather than assertions. All five exit 0.

---

## Known limitations

* **Room 2's headline SARSA-vs-Q-Learning result needs the optional state switch.**
  With the state the assignment specifies, `(row, col, has_keycard)`, both methods
  avoid the risky bridge and the on-policy/off-policy difference does not appear —
  because that state cannot represent which bridges have already collapsed. This is
  measured and documented rather than worked around, and both results are shown in
  the interface. The switch is off by default.
* **Room 2's collapsing bridges make the default state representation
  non-Markovian.** That is a property of the state the assignment fixes, not a bug,
  but it does mean the agent's value for a cell next to a bridge is an average over
  two very different situations.
* **Room 2 is deterministic.** Nothing slips; the risk comes entirely from
  epsilon-greedy exploration. That is the textbook cliff-walking setup and is what
  makes the SARSA comparison meaningful, but it means the room demonstrates nothing
  about stochastic transitions — Room 1 covers those.
* **Room 3 needs about 3,000 episodes before the greedy policy finishes the
  sequence,** and at α 0.05 it does not get there at all within that budget. The
  defaults (5,000 episodes, α 0.15) clear it comfortably, but the experiment module's
  lower episode counts are close enough to the edge that the thresholds are written
  into comments beside them.
* **Room 3's security robot is still caught about 20 % of the time during late
  training,** because ε never falls below 0.05. The greedy policy escapes 100 % of the
  time; the training-curve number is exploration, not a failure to learn.
* **Room 4's generalisation drop is small but real:** 100 % from the release point it
  trained on, 93 % ± 9 from five it has never used, with the mean return halving. Five
  unseen starts is a small sample and the figure moves by seed.
* **Room 4's tile-resolution experiment shows no measured benefit from finer
  tilings** on a hall this size — all three resolutions land every time. The 23×
  difference in memory is the finding; the "best" row is best only among those tested.
* **Room 5's observation is not Markov.** Eight radar rays plus an objective bearing
  cannot distinguish two warehouses that look locally identical, so no convergence
  guarantee applies to it. What is reported is measured behaviour.
* **Room 5 fails roughly two in five unseen Easy layouts,** usually by running out of
  time behind a shelf wall, and it does considerably worse if it is trained on Easy and
  dropped into Hard. Both are measured (the stress test labels the shift) rather than
  hidden.
* **Room 5 is the slowest room to train** — about 33 s for 2,500 episodes, because every
  episode generates radar readings for a fresh layout. The page's defaults are chosen so
  a first run finishes in well under a minute.
* **No hyperparameter is tuned.** Every room ships fixed starting constants that were
  sanity-checked, and every sweep result is described as the best-performing value
  *among those tested* — never as an optimum.
* **The renderer is 2.5D isometric, not true 3D.** It fakes depth well, but there
  is no 3D scene graph, no real camera projection matrix and no WebGL.
* **Chamber rotation moves in 90° steps** rather than turning smoothly, which keeps
  the isometric boxes correctly aligned and the depth sorting exact at every angle.
* **There is no sound.** The two visual briefs allow optional audio behind a mute
  toggle; none is implemented, so there is no mute toggle either.
* **No screenshots in this README.** The interface is best seen by running it.
* **Laser cells in Room 1 are part of the state space but unreachable**, since entering
  a beam relocates R-5 to the start. The policy view still draws an arrow in them.
* **Policy Iteration does more total sweeps than Value Iteration** on Room 1
  (about 570 evaluation sweeps across 7 improvement rounds, against 53 sweeps),
  because policy evaluation is run to convergence each round rather than truncated.
  Both reach the same answer.
* **`rooms/locked_page.py` is now unreachable.** Every room is built, so nothing routes
  to it; it is kept as the fallback branch in the router and a test asserts that no
  room needs it.
* **The interface is only tested at desktop and tablet widths.** The layout stacks
  and the chamber stays readable, but no narrow-phone testing has been done.
