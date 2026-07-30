"""One run of one room: the state machine, the batching, and the metrics.

This is the layer the page talks to.  It owns the environment and the
algorithm, and it is the only thing that knows about run states — the
environment does not, and neither does the algorithm.

    IDLE       built, nothing done yet
    TRAINING   learning, and allowed to advance
    TRAINED    the criterion is met; a greedy replay is available
    REPLAYING  showing the finished policy, learning nothing

Exactly one state at a time, and every transition goes through `_enter`, so
an illegal one is impossible rather than merely discouraged.

Nothing here renders, sleeps or touches a clock other than to measure its
own time budget.  The page decides when to ask for more work; this decides
how much work that turns into.
"""

import random
import time

from game import algorithms, config, definition, rooms
from game.algorithms.base import Transition
from game.grid import ACTION_ARROWS, ACTION_NAMES, GridWorld
from game.reactor import ReactorWorld
from game.recorder import Recorder

# Which environment a room is built on. Most rooms are the plain grid; a room
# with things that move on their own says so by naming its own class here,
# which is a change to this table and to nothing else.
WORLDS = {"reactor": ReactorWorld}

IDLE = "IDLE"
TRAINING = "TRAINING"
TRAINED = "TRAINED"
REPLAYING = "REPLAYING"

# Which states each state may become. Anything absent cannot happen.
ALLOWED = {
    IDLE: (TRAINING, IDLE),
    TRAINING: (TRAINING, TRAINED, IDLE),
    TRAINED: (REPLAYING, TRAINING, IDLE),
    REPLAYING: (TRAINED, TRAINING, IDLE),
}


class IllegalTransition(Exception):
    """Raised when something asks for a transition the machine forbids."""


class Session:
    """A single run. Not thread-safe; one request touches it at a time."""

    def __init__(self, room_number, algorithm_key=None, parameters=None,
                 seed=None):
        self.room = rooms.room(room_number)
        if not self.room.get("built"):
            raise ValueError("room %s is not built yet" % room_number)

        self.seed = config.SESSION["seed_default"] if seed is None else seed
        self.algorithm_key = algorithm_key or self.room["algorithm_default"]
        self.parameters = self._defaults()
        if parameters:
            self.parameters.update(
                {name: value for name, value in parameters.items()
                 if name in self.parameters})

        self.state = IDLE
        self.stale = False
        self.stale_reason = ""
        self.playing = False
        self.history = _empty_history()
        self.replay = None
        self.episode = None

        self._build()

    # ------------------------------------------------------------------
    # Building and rebuilding
    # ------------------------------------------------------------------

    def _defaults(self):
        """The room's parameters, at their documented defaults.

        A room may override any of them — room 2's ice figure is measured
        for that room specifically and is not the global default.
        """
        values = {name: config.PARAMETERS[name]["default"]
                  for name in self.room["parameters"]}
        values.update(self.room.get("parameter_defaults") or {})
        return values

    # Which parameters drive an entry of the room's reward table. A room that
    # does not expose one keeps whatever its own table says.
    REWARD_PARAMETERS = {"battery_reward": "battery"}

    def _reward_overrides(self):
        return {reward: self.parameters[name]
                for name, reward in self.REWARD_PARAMETERS.items()
                if name in self.parameters}

    def _build(self):
        """A fresh environment and a fresh algorithm from the parameters."""
        slip = self.parameters.get("slip", 0.0)
        world = WORLDS.get(self.room.get("world"), GridWorld)
        self.env = world(self.room, slip=slip, seed=self.seed,
                         rewards=self._reward_overrides(),
                         collapse=self.parameters.get("collapse_chance", 0.0))
        # The algorithm's own generator, separate from the environment's, so
        # exploration and slipping cannot consume each other's randomness.
        self.algorithm = algorithms.build(self.algorithm_key, self.env,
                                          self.parameters,
                                          random.Random(self.seed + 1))
        self.env.reset(seed=self.seed)
        self.episode = None
        # Only the learners have episodes to record; a planner never takes a
        # step in the environment at all.
        self.recorder = Recorder(
            self.env,
            config.SESSION["episodes_recorded"],
            int(self.parameters.get("episodes", 0)),
            ACTION_NAMES) if self.learns else None

    @property
    def learns(self):
        """True for the methods that learn from experience rather than plan."""
        return not self.algorithm.needs_model

    def _enter(self, state):
        if state not in ALLOWED[self.state]:
            raise IllegalTransition("%s cannot become %s" % (self.state, state))
        self.state = state

    # ------------------------------------------------------------------
    # Controls
    # ------------------------------------------------------------------

    def play(self):
        if self.state == TRAINED:
            # Continuing after convergence is allowed: it is how a player
            # gets back to training from a replay.
            self._enter(TRAINING)
        elif self.state == REPLAYING:
            self._enter(TRAINING)
        elif self.state == IDLE:
            self._enter(TRAINING)
        self.playing = True
        self.replay = None
        return self.snapshot()

    def pause(self):
        self.playing = False
        return self.snapshot()

    def step_once(self):
        """Exactly one unit of learning, then paused."""
        if self.state in (IDLE, TRAINED, REPLAYING):
            self._enter(TRAINING)
        self.playing = False
        self._advance_one()
        self._check_finished()
        return self.snapshot()

    def reset(self):
        """Back to episode zero with the parameters intact."""
        self._enter(IDLE)
        self.playing = False
        self.stale = False
        self.stale_reason = ""
        self.history = _empty_history()
        self.replay = None
        self._build()
        return self.snapshot()

    def set_algorithm(self, key):
        """Changing the algorithm always invalidates the run."""
        if key not in self.room["algorithms"]:
            raise ValueError("room %s does not offer %r"
                             % (self.room["number"], key))
        if key == self.algorithm_key:
            return self.snapshot()
        self.algorithm_key = key
        return self._mark_stale("the algorithm changed")

    def set_parameters(self, incoming):
        """Apply live parameters at once; mark reset-scope ones as stale.

        A reset-scope parameter is never quietly applied to a run that has
        already learned something — the value is remembered, the run is
        marked stale, and it takes effect on the next Reset.
        """
        applied = []
        deferred = []

        for name, value in incoming.items():
            if name not in self.parameters:
                continue
            specification = config.PARAMETERS[name]
            value = _clamp(value, specification)
            if value == self.parameters[name]:
                continue

            self.parameters[name] = value
            if specification["scope"] == "live":
                self.algorithm.set_parameter(name, value)
                # Moving the finish line moves which episodes are worth
                # keeping; the ones already recorded stay, since they happened.
                if name == "episodes" and self.recorder is not None:
                    self.recorder.retarget(int(value))
                applied.append(name)
                continue
            deferred.append(specification["label"])

        if deferred and self.state != IDLE:
            return self._mark_stale(", ".join(deferred) + " changed")

        if deferred:
            # Nothing has been learned yet, so it can simply be rebuilt.
            self._build()

        return self.snapshot()

    def reset_parameter(self, name):
        if name not in self.parameters:
            return self.snapshot()
        return self.set_parameters({name: config.PARAMETERS[name]["default"]})

    def _mark_stale(self, reason):
        self.stale = True
        self.stale_reason = reason
        self.playing = False
        return self.snapshot()

    # ------------------------------------------------------------------
    # Advancing
    # ------------------------------------------------------------------

    def _advance_one(self):
        """One unit of learning, and the numbers that came out of it.

        The two families mean different things by it, deliberately: a sweep
        over the state space for a planner, a single step through the
        environment for a learner. Everything above this method — the state
        machine, the batching, the budget — is written in units and does not
        care which.
        """
        if self.learns:
            return self._learn_one_step()

        report = self.algorithm.update()
        self.history["delta"].append(report.get("delta", 0.0))
        self.history["startValue"].append(
            self.algorithm.values[self.env.start_state()])
        self.history["work"].append(self.algorithm.sweeps)
        return report

    def _learn_one_step(self):
        """One step taken, one update applied, and the episode kept in place."""
        limit = config.SESSION["max_steps_per_episode"]

        if self.episode is None:
            state = self.env.reset()
            self.episode = {"state": state, "action": self.algorithm.act(state),
                            "reward": 0.0, "steps": 0}
            self.recorder.begin(self.algorithm.episodes, state,
                                self.algorithm.epsilon)

        state = self.episode["state"]
        action = self.episode["action"]
        next_state, reward, done, info = self.env.step(action)

        # The next action is drawn before the update, because on-policy
        # methods need it and off-policy ones must be handed it anyway so
        # that the two differ in the update rule alone.
        next_action = None if done else self.algorithm.act(next_state)
        report = self.algorithm.update(
            Transition(state, action, reward, next_state, next_action, done))

        # Passive: the recorder is handed the transition that was taken
        # anyway and draws no randomness, so recording cannot change what is
        # learned.
        self.recorder.step(next_state, action, reward, report, info)

        self.episode["reward"] += reward
        self.episode["steps"] += 1
        out_of_time = self.episode["steps"] >= limit

        if not done and not out_of_time:
            self.episode["state"] = next_state
            self.episode["action"] = next_action
            return {"done": False}

        self.algorithm.end_episode()
        self.history["reward"].append(self.episode["reward"])
        self.history["length"].append(self.episode["steps"])
        self.history["success"].append(1.0 if info["goal"] else 0.0)
        self.history["fell"].append(1.0 if info["hazard"] else 0.0)
        self.history["work"].append(self.algorithm.episodes)
        # A run that neither got out nor fell in was stopped by the step
        # limit, which is a third outcome and worth naming as one.
        if info["goal"]:
            outcome = "success"
        elif done:
            outcome = "failure"
        else:
            outcome = "timeout"
        self.recorder.end(outcome, self.episode["reward"])
        self.episode = None
        return {"done": True, "goal": info["goal"], "hazard": info["hazard"]}

    def _mean_recent(self, key):
        values = self.history.get(key) or []
        if not values:
            return 0.0
        window = values[-config.SESSION["metric_window"]:]
        return sum(window) / len(window)

    def _check_finished(self):
        """Move to TRAINED the moment the criterion is met."""
        if self.state != TRAINING:
            return
        if not self.algorithm.finished:
            return
        self._enter(TRAINED)
        self.playing = False

    def advance(self, steps=None, budget_ms=None):
        """Do some work and report back.

        `steps` is the animated tiers: a fixed, small number of units.
        `budget_ms` is Turbo: as many as fit in the time allowed, which is
        what lets a method that needs thousands of them get there without
        the interface ever waiting on a long call.
        """
        if self.state != TRAINING or not self.playing:
            return self.snapshot()

        done = 0
        if budget_ms is not None:
            deadline = time.perf_counter() + budget_ms / 1000.0
            ceiling = config.TURBO_MAX_STEPS
            while done < ceiling:
                self._advance_one()
                done += 1
                if self.algorithm.finished:
                    break
                if time.perf_counter() >= deadline:
                    break
        else:
            for _ in range(max(1, steps or 1)):
                self._advance_one()
                done += 1
                if self.algorithm.finished:
                    break

        self._check_finished()
        snapshot = self.snapshot()
        snapshot["advanced"] = done
        return snapshot

    # ------------------------------------------------------------------
    # Replay
    # ------------------------------------------------------------------

    def start_replay(self, limit=None):
        """Record one greedy run and hand it over.

        The trajectory is recorded once and animated by the page, so replay
        cannot touch the policy however long it is left looping.
        """
        self._enter(REPLAYING)
        self.playing = False
        self.replay = self._record_greedy_run(limit)
        return self.snapshot()

    def _record_greedy_run(self, limit=None):
        limit = limit or config.SESSION["max_steps_per_episode"]
        state = self.env.reset(seed=self.seed)
        # A state may carry more than a cell; a frame only ever wants the
        # two numbers the grid is drawn on.
        cell = self.env.cell_of if hasattr(self.env, "cell_of") else tuple
        frames = [{"cell": list(cell(state)), "reward": 0.0, "total": 0.0,
                   "action": None, "slipped": False, "battery": False}]
        total = 0.0
        outcome = "lost"
        # Read the policy once, and follow it. `act` would still be
        # ε-greedy for a learner, and a replay has to show what was learned
        # rather than what it was still trying out.
        policy = self.algorithm.greedy_policy()

        for _ in range(limit):
            action = policy.get(state)
            if action is None:
                break
            state, reward, done, info = self.env.step(action)
            total += reward
            frames.append({
                "cell": list(cell(state)),
                "reward": reward,
                "total": total,
                "action": action,
                "slipped": info["slipped"],
                # Carried so a replay can show the battery being picked up.
                "battery": bool(state[2]) if len(state) > 2 else False,
            })
            if not done:
                continue
            outcome = "goal" if info["goal"] else "hazard"
            break

        # Leave the environment where a replay found it.
        self.env.reset(seed=self.seed)
        return {"frames": frames, "steps": len(frames) - 1,
                "totalReward": total, "outcome": outcome}

    # ------------------------------------------------------------------
    # What the page is told
    # ------------------------------------------------------------------

    def solved(self):
        """Whether this run counts as having beaten the room."""
        if self.state not in (TRAINED, REPLAYING):
            return False
        if self.replay is not None:
            return self.replay["outcome"] == "goal"
        return self.algorithm.finished

    def curve(self):
        """The learning curve, described so the page does not have to guess.

        A planner's interesting series is the largest value change per sweep,
        which spans orders of magnitude and wants a log axis. A learner's is
        the reward per episode, which does not. Saying which here keeps the
        renderer from carrying a rule per room.
        """
        # `x` and `points` are downsampled by the same stride, so a pair
        # always belongs together.
        if self.learns:
            return {
                "label": "Reward per episode",
                "xLabel": "Episodes",
                "scale": "linear",
                "x": _downsample(self.history["work"]),
                "points": _downsample(self.history["reward"]),
                "threshold": None,
            }
        return {
            "label": "Largest value change",
            "xLabel": "Sweeps",
            "scale": "log",
            "x": _downsample(self.history["work"]),
            "points": _downsample(self.history["delta"]),
            "threshold": self.parameters.get("theta"),
        }

    def readout(self):
        """The numbers under the curve, as label/value pairs."""
        learned = self.algorithm.snapshot()

        if self.learns:
            return [
                ["Episodes", "%d / %d" % (learned["episodes"],
                                          learned["episodeTarget"])],
                ["Exploration ε", "%.3f" % learned["epsilon"]],
                ["Mean reward", "%.1f" % self._mean_recent("reward")],
                ["Reached the exit", "%.0f%%" % (100 * self._mean_recent("success"))],
                ["Fell in", "%.0f%%" % (100 * self._mean_recent("fell"))],
                ["Mean steps", "%.1f" % self._mean_recent("length")],
                ["Mean |Q|", "%.2f" % learned["meanAbsQ"]],
            ]

        rows = [["Sweeps", str(learned["sweeps"])]]
        if learned.get("rounds") is not None:
            rows.append(["Improvement rounds", str(learned["rounds"])])
        delta = learned.get("delta")
        rows.extend([
            ["Largest change", "—" if delta is None else "%.2e" % delta],
            ["Threshold", "%.0e" % self.parameters.get("theta", 0.0)],
            ["V(start)", "%.3f" % learned["startValue"]],
            ["Converged", "yes" if learned["converged"] else "no"],
        ])
        return rows

    def _scene(self):
        """Where the agent is now, and whatever else the room has moving."""
        state = self.env.state
        entity_states, positions = self.env.frame_extras(state)
        return {
            "position": self.env.world_position(state),
            "entityStates": entity_states,
            "entityPositions": positions,
        }

    def snapshot(self):
        """Everything the page needs, and nothing it should not have."""
        algorithm = self.algorithm.snapshot()
        if self.learns:
            # Session-level measurements, folded in beside the algorithm's own
            # so a room can name any of them as its headline metric.
            algorithm["meanReward"] = self._mean_recent("reward")
            algorithm["successRate"] = self._mean_recent("success")
            algorithm["fallRate"] = self._mean_recent("fell")
        metric = self.room["metric"]
        return {
            "room": self.room["number"],
            "state": self.state,
            "playing": self.playing,
            "stale": self.stale,
            "staleReason": self.stale_reason,
            "algorithm": self.algorithm_key,
            "parameters": dict(self.parameters),
            "seed": self.seed,
            "progress": {
                # A planner counts sweeps, not episodes: it never takes a
                # step in the environment. The page labels it accordingly.
                "unit": "episodes" if self.learns else "sweeps",
                "count": algorithm.get("episodes" if self.learns else "sweeps", 0),
                "converged": bool(algorithm.get("converged")),
            },
            "metric": {
                "label": metric["label"],
                "value": algorithm.get(metric["key"]),
                "format": metric["format"],
            },
            "grid": self.env.snapshot(),
            # The world as it stands, in the shape the room screen draws
            # frames in. This is what lets that screen show training as it
            # happens rather than only replaying it afterwards: the same
            # fields a recorded frame carries, for the current moment.
            "scene": self._scene(),
            "learned": algorithm,
            "curve": self.curve(),
            "readout": self.readout(),
            "replay": self.replay,
            "solved": self.solved(),
        }

    def batch(self):
        """The episodes kept step by step, for the screen that replays them.

        A planner has none — it never takes a step in the environment — and
        an empty batch is legal, so this needs no special case.
        """
        if self.recorder is None:
            return {"episodes": [], "metrics": []}
        return self.recorder.batch()

    def describe(self):
        """The static half: layout, entities, words. Sent once."""
        return {
            # The room screen's contract, built from the same room data the
            # rest of this payload comes from. The two screens want different
            # shapes of the same facts; neither is derived from the other.
            "definition": definition.build(self.room, self.env,
                                           self.parameters),
            "room": {
                "number": self.room["number"],
                "name": self.room["name"],
                "sector": self.room["sector"],
                "info": self.room["info"],
                "metric": self.room["metric"],
            },
            "layout": self.env.layout_snapshot(),
            "algorithms": algorithms.describe_all(self.room["algorithms"]),
            "algorithmDefault": self.room["algorithm_default"],
            "parameters": [
                dict(config.PARAMETERS[name], name=name)
                for name in self.room["parameters"]
            ],
            "actions": [{"id": action, "name": ACTION_NAMES[action],
                         "arrow": ACTION_ARROWS[action]}
                        for action in self.env.actions()],
        }


def _empty_history():
    """Both families' series. Each only ever fills its own.

    `work` is how much the method had actually done by the time each point
    was taken, and it exists because that is not the same as how many points
    there are. One call of Value Iteration is one sweep; one call of Policy
    Iteration is a whole policy evaluation — dozens of sweeps — followed by an
    improvement. Plotting the two against point index would put 15 sweeps and
    389 sweeps on the same tick and quietly claim they were comparable.
    """
    return {"delta": [], "startValue": [], "work": [],
            "reward": [], "length": [], "success": [], "fell": []}


def _downsample(values):
    """At most `curve_points` points, so a long run does not ship every one."""
    points = config.SESSION["curve_points"]
    if len(values) <= points:
        return list(values)
    stride = len(values) / float(points)
    return [values[int(index * stride)] for index in range(points)]


def _clamp(value, specification):
    """Keep an incoming parameter inside its documented range."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return specification["default"]
    number = max(specification["minimum"], min(specification["maximum"], number))
    choices = specification.get("choices")
    if not choices:
        return number
    # A parameter with a fixed list of values snaps to the nearest of them.
    return min(choices, key=lambda choice: abs(choice - number))
