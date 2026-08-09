# Project R-5 — A Reinforcement Learning Escape Room

An interactive escape room in which every door is opened by a different
reinforcement learning method. Nothing in the game is scripted: in each of the
five chambers the agent begins knowing nothing, and the player watches it work
the room out.

No reinforcement learning library is used anywhere. Every algorithm is written
out by hand so the update rules can be read directly in the source.

---

## Project overview

### The story

At 02:14 AM an experimental reinforcement learning robot, **R-5**, becomes
self-aware inside an AI research laboratory. The security system notices and
seals the building. Five laboratory sectors stand between R-5 and the outside,
and the only way through any of them is to learn.

The chambers are not variations on one puzzle. Each was built to make a
different learning problem unavoidable — a known model, learning from risk,
learning from delay, learning without a table, and learning something that
transfers to a room the agent has never seen.

### Educational goal

The project demonstrates, in one application, the progression from classical
Dynamic Programming to function approximation:

| Concept | Where it is demonstrated |
|---|---|
| Planning with a known model | Room 1 |
| Model-free control, on-policy | Room 2 |
| Model-free control, off-policy, delayed reward | Room 3 |
| Linear function approximation over a continuous state | Room 4 |
| Generalisation to unseen environments under partial observability | Room 5 |

Every claim the interface makes about an algorithm is backed by something the
player can move: sliders change the learning problem, graphs redraw from the
run that actually happened, and any recorded episode can be replayed frame by
frame.

### How this README answers the brief

The brief asks for the project's code on GitHub, and for a README that sets out,
for each room, the structure of its **states** and its **rewards**, and the
**parameters that fit the problem**. Every room section below carries the same
four headings, in that order:

* **State** — what the agent's state is, and why each component has to be in it
  for the room to be Markov
* **Actions** — what it can do
* **Rewards** — every reward the room pays, with its value
* **The settings that solve it** — the defaults, why they are those values, and
  what they achieve *measured*, next to the nearby settings that fail

Every number in those four sections is read from `game/rooms.py` and
`game/config.py`, or produced by training the room headlessly at the stated
settings. The measurements are not assertions to be taken on trust —
`experiments/measure_parameters.py` prints every one of those tables, one room at
a time:

```bash
python3 experiments/measure_parameters.py --room 3
```

The parameter defaults for all five rooms are also collected in one table under
[Hyperparameters](#hyperparameters).

### Technologies used

| Layer | Technology |
|---|---|
| Environments, algorithms, training | **Python 3**, standard library only |
| Web server | `http.server.ThreadingHTTPServer` (`serve.py`) |
| Frontend | Hand-written **HTML, CSS and JavaScript**; all rendering on `<canvas>` |
| Tests | **pytest** |

There are **no third-party runtime dependencies**. `requirements.txt` contains
`pytest` and nothing else, and it is needed only to run the test suite. There
is no build step and no bundler.

> **A note on Streamlit.** This project does **not** use Streamlit. The user
> interface is a custom single-page application served by a standard-library
> HTTP server, because the game requires frame-accurate canvas animation,
> episode replay and a persistent training loop that Streamlit's script re-run
> model does not provide. An early prototype of this project did use Streamlit;
> it was replaced, and this README documents the implementation as it exists.

---

## Game flow

1. **Opening briefing.** Four player-paced slides introduce the premise. The
   player advances with `NEXT`, `SPACE`, `ENTER` or a click; nothing advances on
   a timer, and `SKIP` or `ESC` ends it at once. Behind the text, a robot
   repeatedly fails at a hazard and gets one hazard further each attempt — the
   premise, animated.
2. **Main menu.** `PLAY` and `ABOUT`.
3. **Chamber select.** A new player starts at chamber 1 with chambers 2–5
   locked. Progress is a single stored integer: clearing chamber *N* unlocks
   *N + 1*, and progress can never go backwards.
4. **A chamber.** The player sets hyperparameters, presses `TRAIN`, and watches
   the agent learn. Graphs fill in as the run proceeds; recorded episodes
   appear in the replay browser.
5. **Mission Complete.** Shown only after the agent has been *watched* reaching
   the exit — never during training. The chamber is then marked cleared, and
   the next one unlocks on leaving.
6. **Rooms 2 → 5** in the same way, each with a different method.
7. **Final victory screen.** After Room 5, a dedicated ending sequence with a
   training summary and four choices: *Replay Final Escape*, *View Training
   Results*, *Play Again*, *Main Menu*.

---

## Reinforcement Learning algorithms

| Room | Algorithm (default) | Environment type | State representation | Action space | Reward objective |
|---|---|---|---|---|---|
| **1** Laser Security Chamber | Value Iteration | 10×10 grid, **known model**, stochastic surfaces | `(row, col, battery, last_direction, collapsed)` | 4 — up, down, left, right | Reach the control panel; avoid beams; battery bonus |
| **2** Broken Bridge Sector | SARSA | 10×10 grid, model-free, stochastic | `(row, col, battery, last_direction, collapsed)` | 4 — up, down, left, right | Cross to the exit; the span is fast but can give way |
| **3** Reactor Control Chamber | Q-Learning | 10×10 grid, model-free, delayed reward | 8-tuple: cell, keys held, generators started, patrol phase | 5 — four moves **plus WAIT** | Three keys, three generators **in order**, then the door |
| **4** Drone Wind Tunnel | Semi-gradient SARSA | Continuous 10 × 10 m, `dt = 0.02 s` | `(x, y, vx, vy)` — position continuous, **velocity discrete in {−1,0,+1}** | 5 — four thrust directions plus hold | Land on the pad **below the landing speed** |
| **5** Adaptive Storage Facility | Semi-gradient Q-Learning | Continuous 10 × 10 m, **procedurally generated**, partially observable | `(x, y, vx, vy, stage, phase)` | 5 — four thrust directions plus hold | Reach the terminal, then escape through the blast door |

Each room also offers alternative methods for contrast. A comparison runs on
the server against the same layout, the same parameters and the same seed, so a
difference between two curves is a difference between two methods.

---

## Room 1 — Dynamic Programming

### Environment

A 10×10 security wing. This is the **only room whose model is known in
advance**, which is what makes planning possible rather than learning.

```
      c0 c1 c2 c3 c4 c5 c6 c7 c8 c9
r0     S  .  .  #  .  .  B  .  .  .
r1     .  ~  .  #  .  #  #  #  .  .
r2     .  ~  .  T  .  .  .  #  .  .
r3     .  L  .  L  L  .  .  #  .  .
r4     .  .  .  #  .  .  o  .  .  .
r5     #  #  .  L  #  #  #  .  .  .
r6     .  .  T  ~  ~  =  .  .  .  .
r7     .  #  #  #  .  D  .  #  #  .
r8     .  .  .  o  ~  ~  .  .  .  .
r9     .  .  .  .  #  E  ~  .  .  .
```

`S` start · `E` control panel · `L` laser · `~` ice · `=` cracked ice ·
`o` oil · `T` teleport pad · `B` battery · `D` one-way door · `#` wall

### The known model

`GridWorld.transitions(state, action)` returns the full distribution over
successor states — every outcome with its probability and reward. The planner
reads that distribution directly and never samples a step, which is why R-5 can
route around a laser it has never touched.

### State

`(row, col, battery, last_direction, collapsed)`. A cell alone is not Markov
here: the battery can only be collected once, so whether it is already in hand
changes what entering its cell is worth; and oil carries the agent onward in
the direction it was already travelling, so the previous direction must be part
of the state.

### Actions

Four: up, down, left, right.

### Rewards

| Event | Value |
|---|---|
| Each step | −1 |
| Walking into a wall | −3 |
| Touching a laser | −30 (returns the agent to the start; the episode continues) |
| Collecting the battery | +10 (adjustable) |
| Reaching the control panel | +100 |

### Slippery cells and teleport

* **Ice** (`~`) — goes where it aimed with probability `1 − slip`, otherwise
  slides to one of the two perpendicular sides.
* **Cracked ice** (`=`) — the same, at twice the slip.
* **Oil** (`o`) — no grip: with probability `slip` it continues in the
  direction it was already moving, whatever it aimed for.
* **Teleport pads** (`T`) — a linked pair that skips the beam wall entirely.
* **One-way door** (`D`) — may only be entered downwards; the choice cannot be
  undone.

### Why Dynamic Programming fits

The transition model is available in closed form, so the optimal policy can be
**computed** rather than estimated from experience. Value Iteration sweeps the
whole state space and reads the plan straight out of the model. This is the
only room where that is possible, and it is the baseline the other four are
measured against. **Policy Iteration** is offered alongside for comparison.

### Hyperparameters

| Parameter | Range | Default |
|---|---|---|
| Discount factor γ | 0.5 – 0.999 | 0.95 |
| Stopping threshold θ | 1e−6 – 0.1 | 1e−4 |
| Ice slipperiness | 0.0 – 0.5 | 0.20 |
| Battery bonus | 0 – 80 | 10 |

### The settings that solve it

γ 0.95, θ 1e−4, ice 0.20, battery +10. Measured by planning to convergence and
then walking the finished plan 200 times from the start:

| Ice slip | Sweeps to converge | V(start) | Steps walked | Reached the panel |
|---|---|---|---|---|
| 0.00 | 12 | 51.25 | 11.0 | 100% |
| 0.10 | 32 | 46.61 | 12.2 | 100% |
| **0.20** | **39** | **40.75** | **13.3** | **100%** |
| 0.30 | 26 | 33.41 | 16.6 | 100% |
| 0.40 | 30 | 30.89 | 17.2 | 100% |
| 0.50 | 35 | 27.50 | 18.6 | 100% |

The plan reaches the panel at every setting of the ice, and it goes by the
teleport pads at every setting: what the slip takes is strides, not the route.
0.20 is the default because it is loose enough that a slip is a visible event on
the way — two extra steps on average — and far short of the range where the room
becomes a lottery.

**γ** has one job here, and that is to leave a +100 thirteen steps away still
worth something at the start:

| γ | V(start) | Steps walked |
|---|---|---|
| 0.50 | −2.00 | 26.4 |
| 0.80 | 1.96 | 13.3 |
| 0.90 | 19.03 | 13.3 |
| **0.95** | **40.75** | **13.3** |
| 0.99 | 71.31 | 13.3 |
| 0.999 | 82.30 | 16.0 |

At 0.50 the panel is discounted away to nothing, V(start) goes negative, and the
plan wanders — twice the steps for the same walk. From 0.80 upwards the route is
the same thirteen-step one, so 0.95 is a comfortable value rather than a tuned
one: well inside the range where the goal still reaches the start, and far
enough below 1 that a sweep converges quickly.

**θ** is a stopping rule and not a quality setting, which the numbers make plain:

| θ | Sweeps | V(start) | Route |
|---|---|---|---|
| 1e−1 | 25 | 40.70 | unchanged |
| 1e−2 | 30 | 40.74 | unchanged |
| **1e−4** | **39** | **40.75** | unchanged |
| 1e−6 | 51 | 40.75 | unchanged |

The plan is finished well before the numbers stop moving. 1e−4 is where V(start)
has settled to the second decimal; 1e−6 buys twelve more sweeps and changes
neither the route nor the value. Dragging the dashed line on the graph shows
exactly this.

### Graphs shown

One graph, because a planner produces one measurable quantity per sweep:

* **Largest value change per sweep** — logarithmic axis, with the stopping
  threshold θ drawn as a dashed line that follows its slider. This is exactly
  the quantity the stopping rule tests.

The status strip reports sweeps, largest change, threshold, V(start) and
whether the run has converged.

### Replay support

Value Iteration takes no steps while planning, so the live agent does not move.
When the plan is complete the room records **one greedy walk** — the learned
policy followed with exploration removed — which is listed in the replay
browser and played by **Show final route**.

---

## Room 2 — SARSA

### The bridge environment

A 10×10 sector split by a lethal shaft, crossed by a single steel span.

```
      c0 c1 c2 c3 c4 c5 c6 c7 c8 c9
r0     #  #  #  #  #  #  #  #  #  #
r1     #  .  ~  .  .  .  .  ~  .  #     the safe corridor, iced at the turns
r2     #  ~  .  #  #  #  #  .  ~  #
r3     #  .  #  H  H  H  H  #  .  #
r4     #  .  #  H  H  H  H  #  .  #
r5     #  .  #  H  H  H  H  #  .  #     the shaft
r6     #  .  #  H  H  H  H  #  .  #
r7     #  .  G  C  C  C  C  G  .  #     the span
r8     #  S  #  H  H  H  H  #  E  #
r9     #  #  #  #  #  #  #  #  #  #
```

`H` shaft (entering it ends the run) · `G` sound deck · `C` collapsing plank ·
`~` ice

Two routes, measured by breadth-first search:

* **The span** — 9 steps. Worth **+29** crossed cleanly.
* **The way round** — 21 steps, up the west wall and down the east. Nothing on
  it can drop the agent. Worth **+17**.

There is shaft **above and below every plank**, so stepping sideways off the
span is impossible.

### The collapsing bridge

**The collapse probability is sampled once per crossing, not once per plank.**
The draw happens on the single step that takes R-5 off solid ground and on to
the planks; survive it and the whole span is made. The number on the slider is
therefore exactly the chance of losing a crossing:

```
effective crossing risk = p          (not 1 − (1 − p)^4)
```

Measured over 2000 attempts per setting: `p=0.10 → 0.101`, `p=0.25 → 0.253`,
`p=0.50 → 0.524`. A test in `tests/test_defense_metrics.py` enforces it.

### Slippery cells

Four iced cells sit at the two corners of the safe route — before a turn, after
a turn, and against a wall. Each has **one side open and one side solid**, so a
slip costs progress or a wall bump rather than producing two identical
outcomes.

### Stochastic transitions

Two independent sources of randomness, both exposed as sliders: the ice on the
safe route, and the crossing draw on the span.

### State

`(row, col, battery, last_direction, collapsed)`. Which planks have already
gone is part of the state, so the agent is never asked to average "fine" with
"fatal".

### Actions

Four: up, down, left, right.

### Rewards

| Event | Value |
|---|---|
| Each step | −1 |
| Walking into a wall | −2 |
| Falling into the shaft, or through the span | −15 |
| Reaching the exit | +38 |

These magnitudes were chosen so the decision is live across the slider's useful
range. A fall forfeits the exit *as well as* paying the penalty, so the span is
worth taking only while

```
crossing risk  <  steps saved / (exit + fall penalty + steps walked in)
```

At `+100 / −100` that threshold is about **3%** — the span is irrational almost
immediately, whatever the map looks like. At `+38 / −15` it is about **25%**. A
step still costs, the exit still pays, a fall is still fifteen times a step,
and a faster escape still scores higher.

### Measured behaviour

Which way the learned policy goes, seeds 0–4, one greedy run each after 1500
episodes. The figure is how many of the five seeds took the span:

| p | 0.00 | 0.05 | 0.10 | 0.20 | 0.30 | 0.40 | 0.60 |
|---|---|---|---|---|---|---|---|
| took the span | 5/5 | 5/5 | 5/5 | 5/5 | **4/5** | **1/5** | 0/5 |

The decision turns over between 0.20 and 0.60, which is the band the reward
magnitudes above were chosen to put it in. The two mixed settings are the
interesting ones: at 0.30 and 0.40 the two routes are close enough in value that
which one a run settles on depends on what it happened to see.

### Hyperparameters

| Parameter | Range | Default |
|---|---|---|
| Learning rate α | 0.01 – 1.0 | 0.10 |
| Discount factor γ | 0.5 – 0.999 | 0.95 |
| Exploration ε | 0.0 – 1.0 | 1.00 |
| Minimum exploration | 0.0 – 0.5 | 0.05 |
| Exploration decay | 0.9 – 1.0 | 0.995 |
| Ice slipperiness | 0.0 – 0.5 | 0.20 |
| Bridge collapse probability | 0.0 – 1.0 | 0.10 |
| Initial Q value | 0 – 150 | 30 |
| Episodes to train | 100 – 8000 | 1500 |

### The settings that solve it

Measured over seeds 0–4, following the learned policy with exploration removed,
100 runs per seed:

| Setting | Takes the span | Reaches the exit | Steps |
|---|---|---|---|
| **the defaults** | **100%** | **91%** | **8.9** |
| Q init 0 instead of 30 | 100% | 91% | 8.5 |
| 300 episodes instead of 1500 | 40% | 35% | 123.5 |
| ε decay 0.95 instead of 0.995 | 80% | 73% | 46.8 |
| collapse 0.30 | 80% | 77% | 10.3 |
| collapse 0.60 | 0% | 100% | 22.2 |

At the defaults the policy commits to the span and loses 9% of its crossings to
the collapse — which is the 10% on the slider, and the reason "reaches the exit"
is not 100% for a policy that has solved the room correctly. A crossing takes 8.9
steps against the span's ideal 9, so what it has learned is the short route and
not a wander that happens to end well.

The two settings that break it are both about how long exploration lasts, and the
step column is where they show. At 300 episodes the agent has not crossed often
enough to have priced the span: it reaches the exit 35% of the time and averages
123.5 steps against a 200-step cap, which is a policy still wandering rather than
one that has chosen. Decaying ε at 0.95 instead of 0.995 puts it on a route before
it has properly seen the alternative — a fifth of its crossings and a sixth of its
exits given up, and five times the steps. **1500 episodes and a 0.995 decay are
therefore the parameters that solve this room**; α and γ are the ordinary values
and the room is not sensitive to them.

The optimistic Q of 30 makes no measurable difference by 1500 episodes. It is
there so that the early episodes sweep the sector instead of settling into the
first route that works, and not because the run ends up anywhere else without
it — which is worth stating plainly rather than claiming an effect the
measurement does not show.

### Why SARSA is appropriate

The room is a risk-versus-return decision, and the two families answer it
differently. **SARSA is on-policy**: it learns the value of the route it is
actually walking, exploratory steps beside a lethal shaft included. Q-Learning
learns the value of walking that route perfectly. SARSA therefore gives the
span up earlier — the practical difference between the two families, visible on
one map. Expected SARSA, Q-Learning and Double Q-Learning are offered for
comparison.

### Graphs and replay

Reward per episode, steps per episode, exploration rate and convergence
measure. Every recorded episode is listed and replayable frame by frame,
including the collapse, which is drawn with the span shearing away.

---

## Room 3 — Q-Learning

### Environment

A service ring around a reactor core, with a shaft through the middle.

```
      c0 c1 c2 c3 c4 c5 c6 c7 c8 c9
r0     #  #  #  #  #  #  #  #  #  #
r1     #  S  .  b  .  .  .  1  #  #
r2     #  .  #  #  d  #  #  .  #  #
r3     #  .  #  #  .  #  #  .  #  #
r4     #  .  #  #  c  #  #  .  #  #
r5     #  .  #  #  .  #  #  .  #  #
r6     #  .  #  #  d  #  #  .  #  #
r7     #  3  .  a  .  .  .  2  #  #
r8     #  #  #  #  R  #  #  #  #  #
r9     #  #  #  #  E  #  #  #  #  #
```

`a b c` keys · `1 2 3` generators · `d` sliding doors on a cycle ·
`R` reactor blast door · `E` exit. A security robot patrols the ring.

### Exploration and delayed rewards

The exit reward is roughly two dozen steps from the first move that leads
towards it, with **six errands in a fixed order** in between and nothing paid
out along the way. Carrying value that far backwards is the problem this room
poses, and it is why exploration has to be sustained rather than decayed away
quickly.

### State

An 8-tuple: cell, which keys are held, how far through the generator sequence
the reactor is, and where the patrol has reached. The sliding doors are derived
from the patrol phase rather than a clock of their own, so a single number
covers both moving things.

### Actions

**Five** — up, down, left, right, and **WAIT**. This is the only room with a
wait action, because the shortcut's doors are open two steps in four and
arriving early to hold position is a genuine tactic.

### Rewards

| Event | Value |
|---|---|
| Each step | −1 |
| Walking into a wall | −2 |
| Collecting a key | +15 |
| Starting a generator, in order | +20 |
| Caught by the security robot | −100 |
| Reaching the exit | +100 |

### Hyperparameters

| Parameter | Range | Default |
|---|---|---|
| Learning rate α | 0.01 – 1.0 | 0.10 |
| Discount factor γ | 0.5 – 0.999 | **0.99** |
| Exploration ε | 0.0 – 1.0 | 1.00 |
| Minimum exploration | 0.0 – 0.5 | 0.05 |
| Exploration decay | 0.9 – 1.0 | 0.995 |
| Initial Q value | 0 – 150 | 0 |
| Episodes to train | 100 – 8000 | **4000** |

### The settings that solve it

The measurement here has to be taken on the **greedy** policy. ε sits at a floor
of 0.05 and a single stray step meets the patrol, so the training curve reports a
much lower success rate than the learned policy actually achieves — the last
column below is not this room's score, the first one is.

Seeds 0–4, one greedy run each after training:

| Setting | Got out | Steps | Return | Success in the last 200 training episodes |
|---|---|---|---|---|
| **the defaults** | **5/5** | **71.6** | **+133.4** | 27% |
| 2000 episodes | 5/5 | 81.2 | +121.0 | 16% |
| 1000 episodes | **0/5** | 76.8 | −81.8 | 0% |
| γ 0.95 | 5/5 | 57.0 | +148.0 | 32% |
| γ 0.90 | 5/5 | 61.8 | +140.0 | 31% |
| γ 0.80 | 5/5 | 57.2 | +147.4 | 38% |
| ε decay 0.95 | 5/5 | 61.8 | +142.4 | 32% |
| ε floor 0 instead of 0.05 | 5/5 | 81.0 | +123.6 | 100% |

**The episode count is the parameter that decides whether this room is solved at
all.** At 1000 episodes it fails on every seed; at 2000 it succeeds on every seed
but takes a route ten steps longer than the default does. Six errands in a fixed
order is a long way to carry credit back, and that distance is paid for in
episodes.

γ 0.99 is the value the room's design argues for — the exit is some seventy steps
from the start with nothing paid out over most of them, and a discount close to
one is what leaves the door's value visible at the first move. It is also the
parameter the outcome is *least* sensitive to: every setting from 0.80 to 0.99
gets out on all five seeds, and 0.95 finds a shorter route than 0.99 does. That
is stated here rather than tidied away, because "γ near 1 is required to escape"
is a claim this measurement does not support; what γ near 1 buys is a value
function that is informative at the start, not the escape itself.

### Replay and graphs

Reward per episode, steps per episode, exploration rate and convergence
measure, all rendering real recorded data. Every recorded episode is
replayable; the generators turning green in order, and the agent learning to
wait at a shut door, are both visible frame by frame.

### Why Q-Learning fits

Q-Learning's **off-policy** target bootstraps from the best next action rather
than the one exploration happened to take. Over a long chain of unrewarded
steps this propagates credit back without waiting for a lucky run of good
choices — which is exactly what a six-errand sequence needs. Double
Q-Learning, SARSA and Expected SARSA are offered for contrast.

---

## Room 4 — Semi-gradient SARSA

### A continuous world

A wind tunnel of **10 × 10 metres**, integrated at `dt = 0.02 s` with
semi-implicit Euler. There is no grid. An episode may last 1800 physics steps,
which is `1800 x 0.02 = 36 s` of simulated flight — Room 4 records every tick,
where Room 5 records one frame per held decision.

* Two banks of fans drive air down across the approach
* Turbine housings and the chamber wall are solid
* A thruster-overcharge zone shortens the route and makes arriving slowly
  harder
* A stabilisation field damps motion

### State — continuous position, **discrete velocity**

```
(x, y, vx, vy)
```

* `x, y` — position in metres. **Continuous.**
* `vx, vy` — velocity in metres per second. **Discrete: each is one of
  `{−1, 0, +1}`.**
* `dt = 0.02 s`.

This is the assignment's requirement for this room: the movement is continuous
while the velocity is discrete. The position advances by `v · dt`, so at full
speed the drone moves 0.02 m per tick and the flight path is smooth even though
the velocity only ever takes three values per axis.

There is **no angle and no angular velocity** in the state. The tilt drawn on
screen is `atan2(vy, vx)` — a display value computed from the velocity for the
picture's sake, which the learner never sees.

### Actions

Five discrete inputs:

```
hold (0,0) · up (0,−1) · down (0,+1) · left (−1,0) · right (+1,0)
```

A thrust steps the matching velocity component by **one whole unit**, clamped to
`{−1, 0, +1}`. From rest one press reaches full speed; reversing from full speed
takes two presses, which is what momentum means here.

### What replaced the continuous forces

Three things in a continuous model produce fractional velocities by
construction, so they cannot survive a discrete one. Each zone that used them
was given the discrete equivalent, so no zone became decoration:

| Feature | Before | Now |
|---|---|---|
| Global drag | `v *= exp(−drag·dt)` every tick | **Removed.** A coasting drone keeps its velocity until an action or a zone changes it |
| Wind zones | fractional acceleration per tick | a **whole-unit shove** with probability `wind · dt` per tick — the same expected effect per second, so the wind slider means what it meant before |
| Stabilisation field | extra drag coefficient | pulls the velocity **one unit towards rest**, at a rate set by its own coefficient |
| Thruster overcharge | multiplied the thrust magnitude | a press reaches **full speed in one step** instead of one unit at a time — visible when reversing |

The draws come from the environment's seeded RNG, so a flight is still exactly
reproducible from its seed.

### Tile coding and function approximation

The state is four real numbers, so there are infinitely many states and no
table can have a row for each. The space is covered by **8 overlapping
tilings**, each offset from the last, and one weight is learned per tile.
States near one another share most of their active tiles, so learning about one
teaches the others; distant states share none and stay separate.

A **discretised Q-table** is offered beside it specifically so it can be watched
failing: too coarse cannot tell a gentle approach from a fast one, too fine
never sees the same bucket twice.

### Landing objective

Touching the platform is easy. Touching it **below the landing speed limit** is
the task — arriving too fast is a crash, not a landing.

The test is on the **speed magnitude**, `√(vx² + vy²) ≤ limit`, not on each
component: with a discrete velocity no component can exceed 1, so a
per-component test would be true for every possible arrival and the rule would
be a no-op. The only speeds that exist are `0`, `1` and `√2 ≈ 1.41`, which gives
the slider three real regimes:

| Limit | What counts as a landing |
|---|---|
| below 1.0 | only a full stop |
| 1.0 – 1.41 | an arrival along one axis; a diagonal one crashes |
| above 1.41 | any arrival |

The default is **1.0**. Measured over the last 200 episodes of a 1200-episode
run: at 1.0 the agent lands 99.5% of the time and at 1.5 it lands 98.5%. Below
1.0 — where nothing but a full stop counts — it works out that hovering scores
about +23 while a crash costs −30, so it hovers and does not land at all. That
regime is available as an experiment rather than as a working setting.

The approach warning on screen uses the same rule, so the display cannot stay
calm up to a wreck.

### Rewards

| Event | Value |
|---|---|
| Each tick | −0.01 |
| Progress toward the pad | +5 (shaped) |
| Hitting a wall or housing | −100 |
| Entering a danger zone | −5 |
| Hard landing | −30 |
| Landing within the speed limit | +200 |

### Hyperparameters

| Parameter | Range | Default |
|---|---|---|
| Learning rate α | 0.01 – 1.0 | **0.30** |
| Discount factor γ | 0.5 – 0.999 | **0.995** |
| Exploration ε | 0.0 – 1.0 | 1.00 |
| Minimum exploration | 0.0 – 0.5 | **0.02** |
| Exploration decay | 0.9 – 1.0 | **0.997** |
| Tilings | 1 – 16 | 8 |
| Buckets per axis | 3 – 20 | 8 |
| Wind strength | 0.0 – 2.0 | 1.0 |
| Landing speed limit | 0.05 – 1.5 | **1.00** |
| Episodes to train | 100 – 8000 | **1200** |

### The settings that solve it

Measured over the last 200 episodes of a 1200-episode run:

| Setting | Landed | Return |
|---|---|---|
| **the defaults** | **99.5%** | **+238.8** |
| γ 0.90 instead of 0.995 | 55.0% | +131.5 |
| α 0.90 instead of 0.30 | **0.5%** | −3.0 |
| landing limit 1.5 | 98.5% | +235.3 |
| landing limit 0.5 | **0.0%** | +23.2 |

Both learning parameters are load-bearing here, and they fail in opposite
directions.

**γ 0.995** is needed because the pad is several hundred 0.02 s steps from the
launch. At 0.90 the landing bonus has decayed to nothing long before the value
reaches the start, and what the drone learns is to fly well and arrive
carelessly — it still reaches the pad, and lands cleanly only 55% of the time.

**α 0.30** is the upper end of what this representation tolerates. The update in
`algorithms/linear.py` is `alpha / len(active) * error`, so α is the share of the
TD error the whole active tile set absorbs in one step: at 0.90 nearly the entire
error is applied at once, each estimate is replaced by the latest noisy target
instead of averaging over samples, and with a bootstrapped target that feeds back
into itself. Measured, the run lands 0.5% of the time — it never learns to land at
all. This is the failure the weight-norm graph is on the screen for, and the
reason α is a slider rather than a constant.

The landing limit is the room's difficulty control rather than a learning
parameter, and it has exactly three regimes because a discrete velocity admits
exactly three speeds. At 1.5 any arrival counts and the task is nearly the same
one — 98.5%. At 0.5 nothing but a full stop counts, and the agent works out that
hovering out-scores risking the crash. **1.00 is the only setting at which the
landing has to be flown and can be.**

### Replay and graphs

Reward per episode, steps per episode, exploration rate and convergence
measure. Replay records **every physics tick**, so a flight plays back at the
speed it was flown and the landing can be studied frame by frame.

---

## Room 5 — Semi-gradient Q-Learning

### Dynamic obstacle generation and random layouts

An automated storage facility, 10 × 10 metres and continuous, that
**rearranges itself between episodes**. It exists to settle one question: did
R-5 learn to adapt, or only memorise four rooms?

Every episode is generated from a layout seed. What varies:

* shelf positions, and the positions of both objectives
* the **number** of security drones (`obstacles ± obstacle_variation`)
* drone patrol circuits and speed
* diagonal security beams

Security drones are **0.5 m across** (`OBSTACLE_RADIUS = 0.25`) and patrol
closed circuits. Layouts are validated by breadth-first search before use, so a
generated warehouse is always solvable.

### Observation radius

R-5 **cannot see the map**. It has a forward sensor cone of configurable depth
(`sensor_range`, default **3.0 m**, half-angle ≈ 0.55 rad). Visibility is
decided **centre-to-centre**: an obstacle is visible when the distance between
centres is within the sensor range, with no radius subtracted. What the cone
leaves the learner with is set out under **State** below.

### Continuous movement

The same integrator as Room 4 (`dt = 0.02 s`), with `action_repeat = 10`: one
decision is held for ten physics ticks, so a decision lasts

```
action_repeat x dt = 10 x 0.02 = 0.2 s
```

and the 300-decision episode limit is

```
300 x 0.2 = 60 s of simulated flight
```

### Obstacle avoidance and the two-stage mission

1. Reach the **control terminal** to disarm the security system
2. Escape through the **blast door** it unlocks

The stage is part of the state, so the same position means different things
before and after activation. Beams stay lit until the terminal is reached.

### State — and the observation, which is not the same thing

The **world's** state is `(x, y, vx, vy, stage, phase)`. `phase` is a bounded
integer from which every moving obstacle's position is a pure function, which is
what keeps the world Markov without storing each drone separately; `stage` is 0
before the terminal has been reached and 1 after, so the same position means
different things at different points in the mission.

The **agent's** observation is 14 numbers: where it is, how fast it is going, the
direction and distance to its current objective, three range-limited sensor rays,
and how near the closest visible drone is and how fast it is closing. It never
receives the layout, and two different warehouses can produce identical
observations — so the observation is **not** Markov even though the world is.
That gap is the room: it is what makes this partial observability, and it is why
a linear model over *local* features is the thing that can transfer between
layouts at all.

### Actions

Five, the same drone frame as Room 4:

```
hold (0,0) · up (0,−1) · down (0,+1) · left (−1,0) · right (+1,0)
```

Thrust is an acceleration and not a move, and one decision is held for ten
physics ticks — so an action commits the agent for 0.2 s, which is long enough
for a patrolling drone to have moved.

### Rewards

| Event | Value |
|---|---|
| Each decision | −0.02 |
| Progress toward the current objective | +4 (shaped) |
| Hitting a shelf or wall | −50 |
| Hitting a security drone | −60 |
| Crossing a security beam | −60 |
| Leaving the bounds | −50 |
| Reaching the terminal | +80 |
| Reaching a still-locked exit | −5 |
| Escaping | +250 |
| Timing out | −80 |

### Generalisation, validation and testing

Three **disjoint** pools of layout seeds, enforced at construction:

| Pool | Layouts | Purpose |
|---|---|---|
| Training | 120 | what the agent learns on |
| Validation | 10 | periodic check during training |
| Unseen test | 20 | never trained on |

Evaluation runs with the **weights frozen** and the environment's random state
restored afterwards, so measuring never perturbs the run. Because the score
that counts is measured on warehouses never trained on, a memorised route is
worth nothing by construction — only a linear model over *local* features can
transfer. A **Test on new random room** control runs the frozen policy on a
single unseen layout on demand.

### Hyperparameters

| Parameter | Range | Default |
|---|---|---|
| Learning rate α | 0.01 – 1.0 | **0.20** |
| Discount factor γ | 0.5 – 0.999 | **0.97** |
| Exploration ε | 0.0 – 1.0 | 1.00 |
| Minimum exploration | 0.0 – 0.5 | 0.05 |
| Exploration decay | 0.9 – 1.0 | **0.9985** |
| Tilings | 1 – 16 | 8 |
| Sensor range (m) | 1.0 – 10.0 | 3.0 |
| Security drones | 0 – 6 | 2 |
| Drone count variation | 0 – 3 | 1 |
| Drone speed | 0.1 – 1.5 | 0.35 |
| Storage shelves | 0 – 8 | 3 |
| Episode length (agent decisions, 0.2 s each) | 100 – 800 | 300 |
| Training layouts | 10 – 400 | 120 |
| Validation layouts | 5 – 100 | 10 |
| Unseen test layouts | 5 – 100 | 20 |
| Random seed | 0 – 999 | 0 |
| Episodes to train | 100 – 8000 | **4000** |

### The settings that solve it

Every figure below is the **frozen** policy's escape rate — an evaluation updates
no weights — averaged over seeds 0, 1 and 2, with a random-action baseline in the
last column. The column that counts is **unseen test**: those twenty layouts are
never trained on, and the pools are checked to be disjoint at construction.

| Setting | Train (120) | Validation (10) | **Unseen (20)** | Random |
|---|---|---|---|---|
| **the defaults** | 51.1% | 40.0% | **38.3%** | 0.0% |
| γ 0.99 instead of 0.97 | 57.2% | 53.3% | 40.0% | 0.0% |
| α 0.60 instead of 0.20 | 41.4% | 33.3% | 41.7% | 0.0% |
| sensor range 10 m instead of 3 | 36.1% | 43.3% | 26.7% | 0.0% |
| 1000 episodes instead of 4000 | 2.2% | 3.3% | **1.7%** | 0.0% |
| 20 training layouts instead of 120 | **61.7%** | 13.3% | **16.7%** | 0.0% |

**The training-pool size is the parameter this room turns on, and the last row is
the whole argument for the default.** Cut the pool to 20 layouts and the score on
the layouts it trained on goes *up* — 61.7%, the best train figure in the table —
while the unseen score collapses to 16.7%. That is a generalisation gap of +45
points against the +12.8 the defaults give, and it is what memorising looks like
when it is measured properly. A README that reported only the train column would
have called the worse policy the better one.

**The episode count is binding** in the same way as in Room 3: at 1000 episodes
the room is simply not solved, on any pool.

**The sensor range is the room's central parameter, and more sight is worse.** At
10 m R-5 can see the whole warehouse, and the unseen score falls by 11.6 points.
The reason is that the observation is fixed at fourteen numbers however far the
cone reaches — three rays and one nearest-drone pair — so a longer range does not
add detail, it makes the same fourteen numbers describe a larger and vaguer
region, and the rays start reporting clutter that has no bearing on the next
0.2 s. 3.0 m is roughly what matters within one decision.

**γ and α are the parameters the room is least sensitive to.** Over three seeds
γ 0.99 scored 1.7 points higher on the unseen pool than the 0.97 default — one
layout in twenty, on one seed — and it did not diverge at 4000 episodes on any of
them. 0.97 is kept because it is the more conservative setting for a combination
that has no convergence guarantee at all, not because the measurement condemns
0.99; the weight-norm graph is there to be watched rather than trusted. α 0.60
costs ten points of train and none of unseen, which says more about how much the
score varies between warehouses than about the step size.

Against all of this, the baseline: a random policy escapes **0%** of the unseen
layouts and collides in every one of them. Whatever the learned policy is doing
at 38%, it is not luck.

### Why semi-gradient Q-Learning fits

The state is continuous and the layout changes every episode, so there is no
table to keep and nothing to memorise: the only thing that can be learned is a
function of the local features, which is what a weight vector over tile-coded
observations is. **Off-policy** is the right half of the choice because the
exploratory step that flies into a shelf should not be priced into the value of
the route that avoids it — with a new warehouse every episode, exploration is
expensive and there is no second chance to walk the same corridor.

That comes with the standard caveat, and this room is where the project
demonstrates it rather than asserting it: semi-gradient Q-Learning combines
bootstrapping, off-policy targets and function approximation, which is the
combination that has no convergence guarantee. The weight-norm graph is on the
screen for that reason. Semi-gradient SARSA is offered beside it as the on-policy
contrast.

### Replay and graphs

Eleven graphs (listed below). Replay reproduces an episode exactly, **including
the layout it was recorded in** — a replay animates the right trajectory
through the right warehouse, not through whichever layout happens to be
current.

---

## Hyperparameters

Parameters marked **live** take effect immediately; **reset** parameters change
the environment or the representation, so the run must be restarted. The
interface says which is which, and marks a run stale when a reset parameter has
been moved but not applied.

| Parameter | Scope | Room 1 | Room 2 | Room 3 | Room 4 | Room 5 |
|---|---|---|---|---|---|---|
| Discount factor γ | reset | 0.95 | 0.95 | 0.99 | 0.995 | 0.97 |
| Stopping threshold θ | live | 1e−4 | — | — | — | — |
| Learning rate α | live | — | 0.10 | 0.10 | 0.30 | 0.20 |
| Exploration ε | live | — | 1.00 | 1.00 | 1.00 | 1.00 |
| Minimum exploration | live | — | 0.05 | 0.05 | 0.02 | 0.05 |
| Exploration decay | live | — | 0.995 | 0.995 | 0.997 | 0.9985 |
| Initial Q value | reset | — | 30 | 0 | — | — |
| Episodes to train | live | — | 1500 | 4000 | 1200 | 4000 |
| Ice slipperiness | reset | 0.20 | 0.20 | — | — | — |
| Battery bonus | reset | 10 | — | — | — | — |
| Bridge collapse probability | reset | — | 0.10 | — | — | — |
| Tilings | reset | — | — | — | 8 | 8 |
| Buckets per axis | reset | — | — | — | 8 | — |
| Wind strength | reset | — | — | — | 1.0 | — |
| Landing speed limit | reset | — | — | — | 1.00 | — |
| Sensor range | reset | — | — | — | — | 3.0 m |
| Obstacle controls | reset | — | — | — | — | drones, variation, speed, shelves |
| Layout pools | reset | — | — | — | — | 120 / 10 / 20 |

---

## Graphs

Every graph is drawn from the run that actually happened. The moving average is
computed over the **full** episode history before the series is reduced for
drawing, so a "20-episode moving average" is exactly that and not an average of
buckets.

### The training dashboard

Above the graphs, four headline numbers, read from the complete training
history rather than the sampled episodes kept for replay:

| Rooms 2–5 | Room 1 (planner) |
|---|---|
| Episodes done | Sweeps done |
| Last episode return | V(start) |
| Best episode return | Latest delta |
| Exploration ε | Threshold θ |

Room 1 gets its own four because Value Iteration has no episodes and no
exploration rate; showing empty episode cards there would be misleading.

### Rooms 2 and 3 — six graphs

| Graph | What it measures |
|---|---|
| **Episode Return (total reward)** | `G = r₁ + r₂ + … + r_T`, the sum of every reward in the episode — not the last reward |
| **Smoothed Return (20-episode moving average)** | `smooth[i] = mean(reward[i−19 … i])` over the full history |
| **Loss / Mean \|TD Error\|** | `mean(\|δ\|)` over the episode. See the note below |
| **Exploration rate ε** | The decay schedule, so early noise reads as exploration rather than failure |
| **Steps per episode** | Falls as the agent stops wandering |
| **Success rate (20-episode moving average)** | A 0/1 indicator from `info["goal"]`, averaged — not a reward threshold |

### Room 4 — the same six, plus

| Graph | What it measures |
|---|---|
| **Weight norm ‖w‖** | Size of the learned weight vector. A norm growing without bound is the classic divergence mode of semi-gradient methods |

Room 4's step graph is labelled **physics steps of 0.02 s**, because it records
every tick.

### Room 5 — eleven graphs

Episode Return · Smoothed Return (20) · **Decisions per episode (0.2 s each)** ·
Complete escape rate · Terminal activation rate · Collision rate · Timeout rate ·
Exploration rate ε · Loss / Mean |TD Error| · Weight norm ‖w‖ ·
**Escape rate — train vs validation vs unseen test**

The last one is the graph the room exists for: the gap between train and unseen
is the generalisation gap.

### Room 1 — one graph

| Graph | What it measures |
|---|---|
| **Largest value change per sweep** | The largest change to any state's value in one sweep, on a logarithmic axis, with the stopping threshold θ drawn as a dashed line that follows its slider. This is exactly the quantity the stopping rule tests |

### What "Loss" means here

There is **no neural network in this project**, so there is no network loss.
The quantity plotted under `Loss / Mean |TD Error|` is the method's own
convergence diagnostic:

```
convergence = mean(|δ_t|)      over the steps of the episode

Q-Learning:  δ = r + γ·max_a' Q(s',a') − Q(s,a)
SARSA:       δ = r + γ·Q(s',a')        − Q(s,a)
```

In `game/session.py` this is accumulated as `errorTotal += abs(error)` with a
matching `errorCount`, and divided at the end of the episode. It is a **mean**,
so a long episode does not score highly merely for being long. It is not MSE,
not Huber loss and not a DQN loss.

### Method comparison

Every room can run its offered methods against the same layout, parameters and
seed, and draw the curves on one set of axes. The results panel stays hidden
until at least two methods have produced real curves — no empty chart, and no
fabricated data.

### Saving the results

Two buttons above the graphs, both inert until there is something to save:

* **Download training results** — a JSON file with the room, the algorithm, every
  parameter, the seed, a timestamp, the step units, a summary (episodes, best
  return, final return, final ε, outcome counts), the **full per-episode
  history**, the frozen-weight checkpoints and the evaluation report. Evaluation
  is under its own key and the greedy replay is excluded, so neither can be
  mistaken for training episodes. For Room 1 it saves the sweep-by-sweep
  convergence curve instead.
* **Save graphs (PNG)** — every graph on screen, stacked into one image, drawn
  from the same canvases that are displayed. No charting library.

Filenames carry the room, the algorithm and a timestamp.

## Replay System

Training shows what happened on average; replay shows what happened
*specifically*.

* **Recording.** A sample of episodes is stored frame by frame — position,
  velocity, action, reward, entity states and entity positions. Room 4 records
  every physics tick; Room 5 records one frame per decision, plus the layout
  seed the episode was generated in.
* **Browsing.** Recorded episodes are listed with their outcome and return.
  Selecting one loads it into the player.
* **Transport.** Play, pause and single-step, at four speeds. A replay loops,
  holding briefly on its final frame.
* **Step inspector.** Any frame of any episode can be examined without
  disturbing the run: the exact state, the action taken, the reward, and the
  cumulative return to that point.
* **Show final route.** Plays the **greedy** policy — exploration removed —
  rather than a recorded episode, so what is shown is the answer the agent
  settled on rather than exploration noise.
* **Exactness.** A replay reproduces the episode as recorded, including which
  layout it happened in.

This is what makes a failure diagnosable: a reward curve that plateaus does not
say whether the agent is crashing, timing out or circling — the replay does.

---

## User Interface

### Intro animation
Four player-paced slides over a live animation of a robot failing, learning and
finally running clean. Advance with `NEXT`, `SPACE`, `ENTER` or a click; `SKIP`
or `ESC` ends it. A caption reads the animation's true attempt number, so the
loop reads as learning rather than repetition. Shown once per session.

### About section
Six holographic slides on a blurred laboratory background: the premise, then
one per sector. Each sector slide carries the same five headings — **Mission,
Algorithm, Objective, Obstacles, Why this algorithm fits** — plus a small
looping animated preview of that room's central hazard and a "watch for" note.
The previews are hand-drawn demonstrations; they do not run the environment.

### Training
Each chamber shows the world on a canvas, a live status strip, collapsible
panels for controls, parameters, graphs, method comparison and episode replay,
and a legend generated from the room's own contents — so no chamber lists an
object it does not contain. Four speed tiers, from step-by-step to turbo.

### Replay
As described above.

### Mission Complete screen
A full-screen flash confirming the chamber is cleared. It appears **once**, and
only after R-5 has been watched arriving at the exit with its final movement
finished — never during training, and never over a robot that is still walking.

### Final victory screen
After Room 5 only: a dedicated ending sequence with an animated scene, a
training summary read from the run, and four choices — *Replay Final Escape*,
*View Training Results*, *Play Again*, *Main Menu*. It is modal while open, and
*View Training Results* hands the room back with the graphs and replay list
still reachable, so the player can always inspect before leaving.

---

## Project Structure

```
RL-Escape-Room/
├── serve.py                  standard-library HTTP server and JSON API
├── requirements.txt          pytest only; the app itself needs nothing
├── pytest.ini
├── render.yaml               the hosting blueprint: one web service, no build
├── Procfile                  the same start command, for platforms that read one
│
├── experiments/
│   └── measure_parameters.py  reproduces every "settings that solve it" table
│
├── game/                     all environments, algorithms and training
│   ├── rooms.py              the five room definitions: layouts, rewards,
│   │                         parameters, graphs and briefing text
│   ├── grid.py               the tabular grid environment (rooms 1–3)
│   ├── reactor.py            room 3's generators, keys, doors and patrol
│   ├── drone.py              room 4's continuous flight physics
│   ├── warehouse.py          room 5's procedural warehouse and sensors
│   ├── session.py            training loop, evaluation, replay recording
│   ├── recorder.py           frame-by-frame episode capture
│   ├── definition.py         translates a room into the frontend contract
│   ├── config.py             parameter specifications and entity appearance
│   ├── comparison.py         running several methods on one layout
│   └── algorithms/           value_iteration, policy_iteration, sarsa,
│                             expected_sarsa, q_learning, double_q_learning,
│                             semi_gradient_sarsa, semi_gradient_q,
│                             discretised_q, tile_coding, linear, tabular
│
├── web/                      the browser half; no build step
│   ├── index.html            start screen, intro and About
│   ├── levels/               chamber select and progression
│   ├── room/                 the chamber screen
│   └── assets/
│       ├── shapes.js         every drawing recipe, on canvas
│       ├── intro.js/.css     the opening briefing
│       ├── about.js          the About slides
│       ├── about-preview.js  the looping sector previews
│       └── room/
│           ├── shell.js      the chamber screen's controller
│           ├── renderer.js   world drawing and camera
│           ├── playback.js   replay state machine and transport
│           ├── charts.js     every graph
│           ├── compare.js    method comparison
│           ├── producer.js   talks to the server
│           ├── contract.js   the data contract, documented
│           └── ending.js     the final victory sequence
│
└── tests/                    pytest suite (305 tests)
```

---

## Running locally

**Requirements:** Python 3.9 or newer. Nothing else.

```bash
# 1. Clone the repository
git clone https://github.com/michelleshumilov1998-ai/RL-Escape-Room.git
cd RL-Escape-Room

# 2. Start the server (standard library only — no installation needed)
python3 serve.py

# 3. Open the game in a browser
#    http://localhost:8000
```

To run the test suite:

```bash
pip install -r requirements.txt      # pytest, for the tests only
python3 -m pytest                    # 305 tests
```

To reproduce the measured parameter tables — no dependencies at all, since it
trains the rooms through the same code the server does:

```bash
python3 experiments/measure_parameters.py --room 1   # seconds
python3 experiments/measure_parameters.py --room 5   # about half an hour
```

---

## Deployment

The game is a Python process that serves its own files, so hosting it needs a
platform that runs a process rather than one that serves a directory. It is
deployed on **Render** as a single web service, from `render.yaml` in this
repository.

**What hosting actually required.** Two variables, and no code beyond reading
them:

| Variable | Default | What it is for |
|---|---|---|
| `PORT` | 8000 | The port to listen on. A platform assigns it. |
| `HOST` | `127.0.0.1` locally, `0.0.0.0` when `PORT` is set | Which interfaces to accept connections on. A container that binds only the loopback address is unreachable from outside itself. |
| `SESSIONS_MAX` | 12 | How many live sessions to hold before dropping the oldest. Lowered to 6 on the free instance, which has 512 MB. |

None of the three is a secret, and the project has no others: there is no
database, no API key, no account and no build step. The pages request
`/api/...` relative to whatever host served them, so the same files work at
`localhost:8000` and at a public URL with nothing switched between them.

**To deploy it again, or somewhere else.** On Render: *New → Blueprint*, point
it at this repository, and `render.yaml` supplies the rest. On anything that
reads a `Procfile` the start command is already there. By hand, anywhere:

```bash
PORT=10000 python3 serve.py       # binds 0.0.0.0:10000
```

**What the free instance costs.** It sleeps after about fifteen minutes without
traffic and takes roughly a minute to wake, so the first visit after a quiet
period is slow and every one after it is not. Training state lives in memory, so
a restart ends any run in progress — which matters for a demonstration only in
that a run should be started after the site is awake, not before.

---

## GitHub repository

<https://github.com/michelleshumilov1998-ai/RL-Escape-Room>

---

## Authors

**Michelle Shumilov** — <michelle.shumilov1998@gmail.com>

Final project in Reinforcement Learning.
