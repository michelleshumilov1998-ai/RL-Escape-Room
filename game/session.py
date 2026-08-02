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

import math
import random
import time

from game import algorithms, config, definition, recorder, rooms
from game.algorithms.base import Transition
from game.drone import ACTION_NAMES as DRONE_ACTION_NAMES
from game.drone import DroneWorld
from game.grid import ACTION_ARROWS, ACTION_NAMES, GridWorld
from game.reactor import ReactorWorld
from game.warehouse import WarehouseWorld

# Which environment a room is built on. Most rooms are the plain grid; a room
# with things that move on their own, or with no grid at all, says so by naming
# its own class here, which is a change to this table and to nothing else.
WORLDS = {"reactor": ReactorWorld, "drone": DroneWorld,
          "warehouse": WarehouseWorld}

# What an action is called, per world. Room 4's actions are thrusts and are
# neither named nor numbered like a grid's four directions, so the recorder is
# handed the right table rather than the grid's one regardless of the room.
# Room 5 flies the same drone frame as room 4, so it shares its action names.
ACTION_NAMES_FOR = {"drone": DRONE_ACTION_NAMES,
                    "warehouse": DRONE_ACTION_NAMES}

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

        # A room may also expose the seed as a *control*, which room 5 does.
        # The two then have to be kept in step, and this is the only place that
        # can do it.
        #
        # THE CONSTRUCTOR'S SEED WAS SILENTLY DEAD
        # `_build` reads `parameters["seed"]` and assigns it to `self.seed`, so
        # a seed passed to the constructor — which is what `api_create` does
        # with the request's `seed` field, and what every experiment script
        # does — was overwritten by the control's default of 0 before it was
        # ever used. Measured: four sessions built with seeds 0 to 3 trained
        # identically and reported the same escape rate to the digit. The
        # control worked, so the tests that used it passed, and the argument
        # beside it did nothing at all.
        #
        # An explicit `parameters={"seed": n}` still wins, because that is the
        # control being set deliberately; otherwise the argument populates it.
        if "seed" in self.parameters:
            if seed is not None and not (parameters and "seed" in parameters):
                self.parameters["seed"] = seed
            self.seed = int(round(self.parameters["seed"]))

        self.state = IDLE
        self.stale = False
        self.stale_reason = ""
        self.playing = False
        self.history = _empty_history()
        self.replay = None
        self.episode = None

        # One entry per episode ever run, for the graphs. Distinct from the
        # recorder's `metrics`, which covers only the few dozen episodes kept
        # step by step: a curve over 40 sampled points is not a picture of a
        # 4000-episode run, and the assignment asks for graphs that track
        # learning *throughout* training.
        self.episode_log = []
        # Periodic held-out measurements taken during training, with the
        # weights frozen. See `_checkpoint`.
        self.checkpoints = []
        # The last full evaluation and the last unseen-room run.
        self.evaluation = None
        self.test_episode = None
        self._test_seeds_used = []

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

    # Parameters that shape the *world* rather than the learning, and which
    # the world reads off its own room dict. Laid over the room below so a
    # world never has to know that a slider exists.
    WORLD_PARAMETERS = ("wind", "landing_speed", "sensor_range", "obstacles",
                        "obstacle_variation", "obstacle_speed", "shelves",
                        "max_steps", "train_layouts", "validation_layouts",
                        "test_layouts")

    def _integer_world_parameters(self):
        """Which of the world parameters are counts rather than measurements.

        The slider layer works in floats throughout, and a world that is handed
        `shelves = 3.0` where it expected 3 either fails a range check or draws
        a different warehouse from the one the same integer would have drawn.
        Rounded once, here, rather than defensively at each use.
        """
        return ("obstacles", "obstacle_variation", "shelves", "max_steps",
                "train_layouts", "validation_layouts", "test_layouts")

    def _room_for_world(self):
        """The room, with any world-shaping parameter laid over the top.

        Room 4's wind strength and landing speed limit are part of the model,
        so they belong to the environment rather than to the algorithm. Passing
        them this way keeps `DroneWorld` reading plain numbers off its room and
        keeps `GridWorld`'s signature untouched. Every one of them is
        reset-scope for the usual reason: they change the model, and anything
        already learned was learned against the old one.
        """
        integers = self._integer_world_parameters()
        overrides = {name: (int(round(self.parameters[name]))
                            if name in integers else self.parameters[name])
                     for name in self.WORLD_PARAMETERS
                     if name in self.parameters}
        if not overrides:
            return self.room
        return dict(self.room, **overrides)

    @property
    def action_names(self):
        """What this room's actions are called. Not every room has four."""
        return ACTION_NAMES_FOR.get(self.room.get("world"), ACTION_NAMES)

    @property
    def step_limit(self):
        """How long one episode may run for.

        A room may set its own. Room 4 integrates a fiftieth of a second per
        step and is ten metres across, so the shared limit would end every
        flight in mid-air several hundred steps short of the pad.

        A room may also expose it as a control, which room 5 does — and the
        control has to win over the room's own figure, or the slider would move
        the world's internal limit while this one stayed put and the two would
        end episodes at different times.
        """
        if "max_steps" in self.parameters:
            return int(round(self.parameters["max_steps"]))
        return int(self.room.get("max_steps",
                                 config.SESSION["max_steps_per_episode"]))

    def _build(self):
        """A fresh environment and a fresh algorithm from the parameters."""
        # A room may expose the seed as a control rather than leaving it to
        # whoever created the session. Room 5 does, because reproducing a run
        # exactly is part of what it is demonstrating, and a seed that can only
        # be set at construction cannot be changed from the screen at all.
        if "seed" in self.parameters:
            self.seed = int(round(self.parameters["seed"]))
        slip = self.parameters.get("slip", 0.0)
        world = WORLDS.get(self.room.get("world"), GridWorld)
        self.env = world(self._room_for_world(), slip=slip, seed=self.seed,
                         rewards=self._reward_overrides(),
                         collapse=self.parameters.get("collapse_chance", 0.0))
        # The algorithm's own generator, separate from the environment's, so
        # exploration and slipping cannot consume each other's randomness.
        self.algorithm = algorithms.build(self.algorithm_key, self.env,
                                          self.parameters,
                                          random.Random(self.seed + 1))
        # `evaluating` where the world offers it, so the room *opens* at the
        # real start of its mission.
        #
        # Room 5 begins a share of training episodes part-way through, at the
        # terminal in stage 1, so the second half of the mission gets practised
        # — see `WarehouseWorld.reset`. That is right for training and quite
        # wrong for the first thing a player sees: two times in five, opening
        # the chamber showed the terminal already spent, the blast door already
        # open and the beams already down, which is the end of the mission
        # presented as its beginning. Training is unaffected because the first
        # episode resets again in `_learn_one_step`.
        try:
            self.env.reset(seed=self.seed, evaluating=True)
        except TypeError:
            self.env.reset(seed=self.seed)
        self.episode = None
        # Only the learners have episodes to record; a planner never takes a
        # step in the environment at all.
        # How many episodes are kept frame by frame. A room may ask for fewer:
        # room 5's frames carry a whole warehouse of moving parts and its
        # episodes run to 300 steps, so the shared budget of 40 produced a 3 MB
        # batch — and the screen refetches the batch every couple of seconds
        # while training is being watched.
        self.recorder = recorder.Recorder(
            self.env,
            int(self.room.get("episodes_recorded",
                              config.SESSION["episodes_recorded"])),
            int(self.parameters.get("episodes", 0)),
            self.action_names) if self.learns else None

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
        self.episode_log = []
        self.checkpoints = []
        self.evaluation = None
        self.test_episode = None
        self._test_seeds_used = []
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
        limit = self.step_limit

        if self.episode is None:
            state = self.env.reset()
            self.episode = {"state": state, "action": self.algorithm.act(state),
                            "reward": 0.0, "steps": 0,
                            # For the per-episode mean |TD error| curve. The
                            # algorithm's own counters run for the whole session,
                            # so a per-episode figure has to be gathered here.
                            "errorTotal": 0.0, "errorCount": 0,
                            # Which warehouse this episode was flown in, taken
                            # at the start because the layout is fixed for the
                            # episode and the world moves on afterwards.
                            "layoutSeed": _layout_seed(self.env),
                            "epsilon": self.algorithm.epsilon}
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

        error = (report or {}).get("tdError")
        if error is not None:
            self.episode["errorTotal"] += abs(error)
            self.episode["errorCount"] += 1

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

        # A run that neither got out nor fell in was stopped by the step limit,
        # which is a third outcome and worth naming as one.
        #
        # A room may end the episode itself when time runs out — room 5 does,
        # because running out of time carries a penalty and only the thing that
        # applies rewards can apply that. So `done` is true and yet nothing was
        # hit, and the old test called it a failure. The environment's own event
        # is believed first where there is one.
        timed_out = info.get("event") == "timeout" or (not done and out_of_time)
        if info["goal"]:
            outcome = "success"
        elif timed_out:
            outcome = "timeout"
        elif done:
            outcome = "failure"
        else:
            outcome = "timeout"

        self.recorder.end(outcome, self.episode["reward"])
        self._log_episode(outcome, info, timed_out)
        self.episode = None
        self._checkpoint()
        return {"done": True, "goal": info["goal"], "hazard": info["hazard"]}

    # Which environment events count as a collision. Named once so the graphs,
    # the evaluation table and the episode list cannot disagree about it.
    COLLISION_EVENTS = ("dynamic", "static", "boundary", "laser")

    def _log_episode(self, outcome, info, timed_out):
        """One row of the training history, for the graphs.

        Every episode, not the sampled few the recorder keeps in full — a curve
        over 40 points out of 4000 is a picture of the sampling, not of the run.
        The 0/1 indicators are rates once the charts take a rolling average over
        them, which is how every rate on screen is produced.
        """
        episode = self.episode
        event = info.get("event")
        errors = episode["errorCount"]
        entry = {
            "episode": self.algorithm.episodes - 1,
            "collapseChance": self.environment_collapse_chance(),
            "reward": episode["reward"],
            "steps": episode["steps"],
            "epsilon": episode["epsilon"],
            # The contract's "something that ought to fall towards zero".
            "convergence": (episode["errorTotal"] / errors) if errors else 0.0,
            "outcome": outcome,
            "success": 1.0 if info.get("goal") else 0.0,
            "collision": 1.0 if event in self.COLLISION_EVENTS else 0.0,
            "timeout": 1.0 if timed_out else 0.0,
            "boundary": 1.0 if event == "boundary" else 0.0,
            "weightNorm": self._weight_norm(),
        }
        # Room 5's own columns. Absent in every other room, and the charts that
        # read them are declared by room 5 alone.
        if "stage" in info:
            entry["terminalReached"] = 1.0 if info.get("stage") else 0.0
        if "layoutSeed" in info:
            entry["layoutSeed"] = info["layoutSeed"]
        self.episode_log.append(entry)

    # ------------------------------------------------------------------
    # Held-out measurement
    # ------------------------------------------------------------------

    def _checkpoint(self):
        """Measure the frozen policy on held-out layouts, now and then.

        WHY THIS IS NOT CHEATING
        Nothing here updates a weight. `evaluate` runs the greedy policy and
        calls `env.step`; it never calls `algorithm.update`, so the test pool
        stays unseen *by the learner* however many times it is measured. What
        it buys is the one graph the assignment asks for that cannot be drawn
        from training episodes alone: whether the gap between the layouts it
        trains on and the layouts it has never met is opening or closing.
        """
        if not self.supports_splits:
            return
        every = int(self.room.get("checkpoint_every", 0))
        if not every or self.algorithm.episodes % every:
            return
        layouts = int(self.room.get("checkpoint_layouts", 8))
        self.checkpoints.append({
            "episode": self.algorithm.episodes,
            "train": self.evaluate("train", layouts)["escapeRate"],
            "validation": self.evaluate("validation", layouts)["escapeRate"],
            "test": self.evaluate("test", layouts)["escapeRate"],
        })

    @property
    def supports_splits(self):
        """Whether this room has train / validation / test layout pools."""
        return bool(getattr(self.env, "pools", None))

    def evaluate(self, split, layouts=None, policy="greedy"):
        """Run the policy over a pool of layouts and report what happened.

        THE WEIGHTS ARE NOT TOUCHED. `algorithm.update` is never called from
        here — only `greedy_action`, which reads them. That is the whole
        contract of an evaluation and the reason it can be run against the
        unseen pool without spending it.

        The environment is left where it was found, so measuring mid-training
        does not disturb the episode in progress.
        """
        if not self.supports_splits:
            raise ValueError("room %s has no layout pools to evaluate over"
                             % self.room["number"])
        pool = self.env.pools[split]
        layouts = len(pool) if layouts is None else min(int(layouts), len(pool))

        # Everything an evaluation disturbs, kept so it can be put back
        # exactly — including the two caches that belong to a layout. Those
        # matter: `_shelf_bounds` is what the ray sensor reads, so a restore
        # that left it describing a different warehouse would have the trainer
        # sensing shelves that are not there.
        keep = {
            "state": self.env.state,
            "steps": self.env.steps,
            "split": self.env.split,
            "layout": self.env.layout,
            "episode": self.episode,
            "shelf_bounds": getattr(self.env, "_shelf_bounds", None),
            "movers_at": getattr(self.env, "_movers_at", None),
            "observations": getattr(self.env, "_observations", None),
            # The learner's own generator. `greedy_action` breaks a tie with
            # it, so an evaluation would otherwise consume the learner's
            # randomness and shift its exploration — the same failure as the
            # environment's stream above, one layer up. Rare enough to pass a
            # test by luck, which is exactly why it is restored rather than
            # hoped about.
            "algorithm_rng": (self.algorithm.rng.getstate()
                              if getattr(self.algorithm, "rng", None) else None),
        }
        chooser = self._policy(policy)

        counts = {"escape": 0, "terminal": 0, "collision": 0, "timeout": 0}
        rewards, lengths, seeds = [], [], []
        try:
            for index in range(layouts):
                state = self.env.reset(split=split, index=index,
                                       evaluating=True)
                seeds.append(self.env.layout["seed"])
                total, reached, steps, event = 0.0, False, 0, None
                for steps in range(1, self.step_limit + 1):
                    state, reward, done, info = self.env.step(chooser(state))
                    total += reward
                    if info.get("stage"):
                        reached = True
                    if done:
                        event = info.get("event")
                        break
                if event == "escaped":
                    counts["escape"] += 1
                elif event in self.COLLISION_EVENTS:
                    counts["collision"] += 1
                else:
                    counts["timeout"] += 1
                if reached:
                    counts["terminal"] += 1
                rewards.append(total)
                lengths.append(steps)
        finally:
            # Exactly as it was, including the episode the trainer was in the
            # middle of. An evaluation that moved the world would change what
            # was learned by measuring it.
            #
            # THE FIELDS ARE PUT BACK DIRECTLY, NOT BY RE-PICKING A LAYOUT
            # This used to end with `choose_layout(split=...)` and no index,
            # to rebuild the caches. That call draws a layout *at random*, so
            # measuring the agent quietly consumed a number from the
            # environment's generator and shifted every layout the trainer saw
            # afterwards — an evaluation changing the run it was measuring,
            # which is the one thing it must never do. Worse, it then left
            # `_shelf_bounds` describing that random warehouse while
            # `self.layout` was the real one, so the ray sensor read shelves
            # from a different building until the next reset.
            self.env.state = keep["state"]
            self.env.steps = keep["steps"]
            self.env.split = keep["split"]
            self.env.layout = keep["layout"]
            self.episode = keep["episode"]
            if keep["shelf_bounds"] is not None:
                self.env._shelf_bounds = keep["shelf_bounds"]
            if keep["movers_at"] is not None:
                self.env._movers_at = keep["movers_at"]
            if keep["observations"] is not None:
                self.env._observations = keep["observations"]
            if keep["algorithm_rng"] is not None:
                self.algorithm.rng.setstate(keep["algorithm_rng"])

        return _report(split, policy, layouts, counts, rewards, lengths, seeds)

    def _policy(self, policy):
        """How an evaluation picks its actions.

        "greedy" is the learned policy with the exploration taken out. "random"
        is the baseline the learned one has to beat, and it draws from a
        generator of its own so that measuring it cannot shift the training
        run's random stream.
        """
        if policy == "random":
            chooser = random.Random(self.seed + 977)
            actions = self.env.actions()
            return lambda _state: chooser.choice(actions)
        greedy = getattr(self.algorithm, "greedy_action", None)
        if greedy is None:
            table = self.algorithm.greedy_policy()
            return lambda state: table.get(state, self.env.actions()[0])
        return greedy

    def run_evaluation(self, layouts=None):
        """The full table: every split, and the random baseline beside it."""
        report = {
            "train": self.evaluate("train", layouts),
            "validation": self.evaluate("validation", layouts),
            "test": self.evaluate("test", layouts),
            "randomTest": self.evaluate("test", layouts, policy="random"),
            "episodes": self.algorithm.episodes,
        }
        self.evaluation = report
        return report

    # ------------------------------------------------------------------
    # A new random room, after training
    # ------------------------------------------------------------------

    def run_test_room(self, seed=None):
        """Build an unseen warehouse and fly the learned policy in it.

        The assignment's "generate a new random room and evaluate the learned
        policy there". Three properties make it worth anything:

          * the seed comes from the test pool, which no episode has trained on
          * the weights are frozen — `greedy_action` reads them, nothing writes
          * the whole episode is recorded frame by frame, so it can be replayed

        Calling it again picks a *different* unseen seed, so "another one" does
        not quietly show the same warehouse twice.
        """
        if not self.supports_splits:
            raise ValueError("room %s has no unseen layouts"
                             % self.room["number"])
        pool = self.env.pools["test"]
        if seed is None:
            unused = [candidate for candidate in pool
                      if candidate not in self._test_seeds_used]
            if not unused:
                # Every one has been shown; start round again rather than
                # refusing, and say so in the payload.
                self._test_seeds_used = []
                unused = list(pool)
            seed = self.rng_for_tests().choice(unused)
        if seed not in pool:
            raise ValueError("seed %r is not one of the unseen test layouts"
                             % seed)
        self._test_seeds_used.append(seed)

        index = pool.index(seed)
        episode = self._record_policy_run(split="test", index=index)
        episode["unseen"] = True
        episode["split"] = "test"
        episode["layoutSeed"] = seed
        self.test_episode = episode
        # Shown through the same path a replay is, so the page needs no second
        # renderer and no second set of transport controls.
        self.replay = episode
        # Already showing one is the ordinary case — "another unseen room" is
        # pressed from a replay — and REPLAYING is not a transition to itself.
        if self.state == TRAINED:
            self._enter(REPLAYING)
        return episode

    def rng_for_tests(self):
        """The generator that picks which unseen room to show.

        Separate from both the environment's and the algorithm's, so asking for
        another test room cannot shift what either of them would have done.
        """
        if not hasattr(self, "_test_rng"):
            self._test_rng = random.Random(self.seed + 4211)
        return self._test_rng

    def environment_collapse_chance(self):
        """The collapse probability the environment is really running with.

        `None` for a room that has no collapsing sections, which is every room
        but room 2. Read off the world rather than off `self.parameters`
        because the two can legitimately disagree: `collapse_chance` is
        reset-scope, so moving the slider stores the new value and marks the
        run stale while the environment carries on with the old one until
        Reset. Reporting the stored value during that window would label a
        run with a probability it was not trained at.
        """
        if "collapse_chance" not in self.parameters:
            return None
        return float(getattr(self.env, "collapse", 0.0))

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
        """Record one run of the learned policy and hand it over.

        The trajectory is recorded once and animated by the page, so replay
        cannot touch the policy however long it is left looping.
        """
        self._enter(REPLAYING)
        self.playing = False
        self.replay = self._record_greedy_run(limit)
        return self.snapshot()

    def _record_greedy_run(self, limit=None):
        """The learned route in whichever pool training is using."""
        return self._record_policy_run(limit=limit)

    def _record_policy_run(self, limit=None, split=None, index=None,
                           policy="greedy"):
        """One run with the exploration taken out, shaped as an Episode.

        THE DISTINCTION THIS EXISTS TO DRAW
        A recorded episode is what the agent *did* while it was still
        exploring. ε does not decay to zero — it stops at `epsilon_min`, 0.05
        by default — so even the last episode of a finished run takes a random
        step roughly one time in twenty, and the route visibly wanders. That
        is the truth about training and it is worth being able to watch, but
        it is not the route the agent learned and nothing should present it as
        one. This is what the agent would do if it stopped exploring, which is
        the only run that can honestly be called the route it settled on.

        It is shaped as a contract Episode rather than as anything of its own
        so the page replays it through exactly the path it replays a recorded
        episode through — same frames, same renderer, same step inspector.
        It used to carry a shape of its own that predated the contract, which
        no screen could draw; that is most of why nothing ever showed it.
        """
        limit = limit or self.step_limit
        # `evaluating` where the world offers it: room 5 starts a share of
        # *training* episodes part-way through its mission so the second stage
        # gets practised, and a replay must never be one of those — it is
        # supposed to show the whole mission from the real start.
        #
        # `split` and `index` name a particular warehouse out of a particular
        # pool, which is how the unseen-room test asks for a layout the agent
        # has never trained on. Omitted, the world picks from whichever pool
        # training was using, exactly as before.
        try:
            state = self.env.reset(seed=self.seed, split=split, index=index,
                                   evaluating=True)
        except TypeError:
            state = self.env.reset(seed=self.seed)

        # How the policy is asked for the action at one state.
        #
        # A grid room can hand over its whole policy as a table and be read
        # from it, which is the cheaper thing to do for a room that has one.
        # Room 4 cannot: its state is four real numbers, so there is no table
        # to build and `greedy_policy` returns empty. `greedy_action` answers
        # for one state at a time, which is all this loop ever needed —
        # `act` would not do, because for a learner it is still ε-greedy and
        # this has to show what was learned rather than what it was trying out.
        table = self.algorithm.greedy_policy()
        if policy == "greedy" and table:
            chosen = table.get
        elif policy == "greedy":
            chosen = getattr(self.algorithm, "greedy_action", None)
            if chosen is None:
                chosen = lambda _state: None       # noqa: E731 - nothing to run
        else:
            chosen = self._policy(policy)

        # The warehouse this run happens in, captured before a step is taken.
        # A replay that carried only the trajectory would animate the right
        # flight through the wrong building — see `Recorder.begin`.
        entities = (self.env.entities()
                    if getattr(self.env, "layout_varies", False) else None)
        layout_seed = _layout_seed(self.env)

        names = self.action_names
        steps = [recorder.frame(self.env, names, state, 0.0, None)]
        total = 0.0
        # A policy can also simply fail to arrive: it may walk into a hazard,
        # or loop until the step limit. Both are outcomes worth showing rather
        # than hiding, so they are named the same way a recorded episode names
        # them and the page needs no special case for either.
        outcome = "timeout"
        event = None

        for _ in range(limit):
            action = chosen(state)
            if action is None:
                break
            state, reward, done, info = self.env.step(action)
            total += reward
            steps.append(recorder.frame(self.env, names, state, reward, action,
                                        bool(info.get("slipped"))))
            if done:
                event = info.get("event")
                if info["goal"]:
                    outcome = "success"
                elif event == "timeout":
                    outcome = "timeout"
                else:
                    outcome = "failure"
                break

        # Leave the environment at the START of the mission, not wherever a
        # curriculum reset happens to drop it.
        #
        # THIS IS WHAT MADE THE ROOM LOOK AS THOUGH IT WAS WON AT THE TERMINAL
        # This used to be a plain `reset`, which is a *training* reset — and a
        # training reset begins two episodes in five at the control terminal in
        # stage 1, so that the second half of the mission gets practised. The
        # replay itself was fine, but the environment it left behind was not:
        # half the time the idle scene afterwards was R-5 parked on the control
        # terminal, motionless, with the terminal green and spent, the blast
        # door open and the beams down. Measured over twenty seeds: ten of them
        # ended stage 1 with the door open.
        #
        # That is exactly the report — R-5 stops at the terminal and the room
        # appears complete — and it was never a mission-flow bug at all. The
        # episode does continue; it was the still frame afterwards that lied.
        try:
            self.env.reset(seed=self.seed, evaluating=True)
        except TypeError:
            self.env.reset(seed=self.seed)

        return {
            # One past the last episode there was, so it cannot collide with
            # a recorded episode's number and the page can address it by
            # number like any other.
            "index": getattr(self.algorithm, "episodes", 0),
            "steps": steps,
            "entities": entities,
            "layoutSeed": layout_seed,
            "totalReward": total,
            # No exploration in it at all, which is the whole point of it.
            "epsilon": 0.0,
            "outcome": outcome,
            "event": event,
            # What the world was set to for this run, so a route can never be
            # looked at without knowing the risk it was flown against.
            "collapseChance": self.environment_collapse_chance(),
            # What tells the page this is the learned route and not one of the
            # training episodes it sits beside in the batch.
            "greedy": policy == "greedy",
        }

    # ------------------------------------------------------------------
    # What the page is told
    # ------------------------------------------------------------------

    def solved(self):
        """Whether this run counts as having beaten the room.

        FINISHING TRAINING IS NOT BEATING THE ROOM
        This used to answer `self.algorithm.finished` for a learner, which is
        only "the episode counter reached its target". Measured on room 5: a
        twenty-episode run that escaped exactly zero times reported solved =
        True the moment it stopped training. For a room whose whole point is
        that reaching the first objective is not success, a criterion that does
        not even look at the outcomes is the wrong shape entirely.

        A learner has to have actually reached the goal at least once. In room
        5 that means the blast door — `info["goal"]` is set on `escaped` alone,
        never on terminal activation — so no amount of terminal-reaching can
        make this true.

        A planner is a different case and keeps the old test: it never takes a
        step in the environment, so converging on a plan is the only thing it
        can be asked to do.
        """
        if self.state not in (TRAINED, REPLAYING):
            return False
        if self.replay is not None:
            # The greedy run names its outcomes the way a recorded episode
            # does, so this is "success" and not a word of its own.
            return self.replay["outcome"] == "success"
        if not self.learns:
            return self.algorithm.finished
        return any(entry["success"] for entry in self.episode_log)

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

    def _continuous_readout(self, learned):
        """The extra rows a continuous room wants, beyond a learner's usual.

        A grid room's position is a square and the picture says everything;
        four real numbers do not read off a canvas, so they are stated. Every
        one of these is measured from the environment as it stands rather than
        recomputed, and the zone row is the one place the screen is told which
        fields the drone is inside — the same test `step` uses.
        """
        env = self.env
        state = env.state
        # Read through the world's own accessors rather than by unpacking the
        # state. Room 4's is four numbers and room 5's is six, and a readout
        # that takes the tuple apart itself breaks the moment a room carries
        # one more thing — which is exactly how it did break.
        position = env.world_position(state)
        velocity = env.world_velocity(state)
        rows = [
            ["Position", "%.2f, %.2f m" % (position["x"], position["y"])],
            ["Velocity", "%+.3f, %+.3f m/s" % (velocity["x"], velocity["y"])],
            ["Speed", "%.3f m/s" % math.hypot(velocity["x"], velocity["y"])],
        ]

        # Whatever this particular room has to say about the moment. Each room
        # owns its own block; there is no shared field they both happen to have.
        rows.extend(self._room_readout(state, position))

        rows.extend([
            ["Mean reward", "%.1f" % self._mean_recent("reward")],
            ["Mean steps", "%.1f" % self._mean_recent("length")],
            ["Episodes", "%d / %d" % (learned["episodes"],
                                      learned["episodeTarget"])],
            ["Exploration ε", "%.3f" % learned["epsilon"]],
            ["Mean |TD error|", "%.3f" % learned["meanAbsQ"]],
            ["Weight norm", "%.1f" % self._weight_norm()],
        ])
        return rows

    def _room_readout(self, state, position):
        """The rows that only one continuous room has."""
        env = self.env

        # ---- room 4: the fields, and the landing rule -------------------
        if hasattr(env, "zones_at"):
            names = {"wind": "airflow", "slow": "stabiliser",
                     "boost": "overcharge"}
            inside = [zone["type"]
                      for zone in env.zones_at(position["x"], position["y"])]
            return [
                ["Landing limit", "%.2f m/s on each axis" % env.landing_speed],
                ["Distance to pad", "%.2f m"
                 % env.distance_to_pad(position["x"], position["y"])],
                ["In field", ", ".join(names.get(kind, kind)
                                       for kind in inside) or "open air"],
                ["Landed", "%.0f%%" % (100 * self._mean_recent("success"))],
                ["Crashed", "%.0f%%" % (100 * self._mean_recent("fell"))],
            ]

        # ---- room 5: which warehouse, and what the sensors see -----------
        if hasattr(env, "sense"):
            rays = env.sense(state)
            nearest = env.nearest_mover(state)
            target = env.target_of(state)
            stage = int(state[4])
            return [
                ["Layout seed", "%d (%s)" % (env.layout["seed"], env.split)],
                ["Mission stage", "%d — %s" % (
                    stage, "find the terminal" if stage == 0
                    else "leave through the door")],
                ["Current target", "%.2f, %.2f m" % (target[0], target[1])],
                ["Target distance", "%.2f m" % math.hypot(
                    position["x"] - target[0], position["y"] - target[1])],
                ["Sensor range", "%.1f m" % env.sensor_range],
                ["Sensors L/C/R", "%.2f  %.2f  %.2f"
                 % (rays["left"], rays["centre"], rays["right"])],
                ["Nearest cart", "out of range" if nearest is None
                 else "%.2f m" % nearest[0]],
                ["Escaped", "%.0f%%" % (100 * self._mean_recent("success"))],
                ["Collided", "%.0f%%" % (100 * self._mean_recent("fell"))],
            ]

        return []

    def _weight_norm(self):
        """The size of what has been learned, for the readout.

        Only meaningful for the approximating methods, and the number that
        makes a diverging run obvious at a glance rather than after reading a
        curve: it climbs without limit when the weights blow up.
        """
        weights = getattr(self.algorithm, "weights", None)
        if not weights:
            return 0.0
        total = 0.0
        for vector in weights.values():
            for weight in vector:
                total += weight * weight
        return math.sqrt(total)

    def readout(self):
        """The numbers under the curve, as label/value pairs."""
        learned = self.algorithm.snapshot()

        # A room whose state is four real numbers needs different rows from a
        # room whose state is a square, and asking the environment whether it
        # has a velocity is how every other branch in this file is written.
        if self.learns and hasattr(self.env, "world_velocity"):
            return self._continuous_readout(learned)

        if self.learns:
            rows = [
                ["Episodes", "%d / %d" % (learned["episodes"],
                                          learned["episodeTarget"])],
                ["Exploration ε", "%.3f" % learned["epsilon"]],
                ["Mean reward", "%.1f" % self._mean_recent("reward")],
                ["Reached the exit", "%.0f%%" % (100 * self._mean_recent("success"))],
                ["Fell in", "%.0f%%" % (100 * self._mean_recent("fell"))],
                ["Mean steps", "%.1f" % self._mean_recent("length")],
                ["Mean |Q|", "%.2f" % learned["meanAbsQ"]],
            ]
            # The room's own environment settings, stated live.
            #
            # Room 2's whole subject is a risk/return trade, and the risk is a
            # slider — so a mean reward with no indication of which collapse
            # probability produced it is a number that cannot be compared with
            # anything. Read from the ENVIRONMENT rather than from the
            # parameter dictionary, so what is shown is the value actually in
            # force: a reset-scope parameter that has been moved but not yet
            # applied would otherwise be reported as though it were live.
            collapse = self.environment_collapse_chance()
            if collapse is not None:
                rows.insert(0, ["Bridge collapse probability",
                                "%.2f" % collapse])
            return rows

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
        """Where the agent is now, and whatever else the room has moving.

        The same fields a recorded frame carries, for the current moment. That
        correspondence is what lets one renderer draw both the live view and a
        replay, so anything added to `recorder.frame` belongs here too.
        """
        state = self.env.state
        entity_states, positions = self.env.frame_extras(state)
        scene = {
            "position": self.env.world_position(state),
            "entityStates": entity_states,
            "entityPositions": positions,
            # How many steps into the episode, which the screen uses as the
            # clock for the decorative parts of a continuous room's drawing.
            "steps": getattr(self.env, "steps", 0),
        }
        # A room whose layout changes between episodes has to send the layout
        # with the moment, because the definition the page was opened with
        # describes a warehouse that is no longer there.
        #
        # This was a real bug and a bad one: room 5 picks a new warehouse on
        # every reset, the browser cached the entity list from the very first
        # one, and so the drone was drawn flying through shelves that had moved
        # and dying on shelves that were not drawn at all. The room looked
        # static while the physics varied — the exact opposite of what it is for.
        if getattr(self.env, "layout_varies", False):
            scene["entities"] = self.env.entities()
        # The same extra fields a recorded frame carries. Sent live too, so the
        # sensor fan drawn while training is watched is built from exactly the
        # numbers a replay would draw it from.
        if hasattr(self.env, "frame_detail"):
            scene["detail"] = self.env.frame_detail(state)

        if hasattr(self.env, "world_velocity"):
            velocity = self.env.world_velocity(state)
            scene["velocity"] = velocity
            # Which way it is travelling, or None when it is barely moving and
            # an angle would be meaningless. Worked out here rather than in the
            # browser so the live view and a replay agree on the rule.
            if velocity and (abs(velocity["x"]) > 0.04
                             or abs(velocity["y"]) > 0.04):
                scene["facing"] = math.atan2(velocity["y"], velocity["x"])
            else:
                scene["facing"] = None
        return scene

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
            return {"episodes": [], "metrics": [], "history": [],
                    "checkpoints": [], "evaluation": None}
        batch = self.recorder.batch()
        # The full per-episode history, beside the sampled episodes. The charts
        # read this; the episode browser reads `episodes`. They are different
        # lengths on purpose — every episode is measured, a few dozen are kept
        # frame by frame.
        batch["history"] = list(self.episode_log)
        batch["checkpoints"] = list(self.checkpoints)
        batch["evaluation"] = self.evaluation
        # The environment settings this recording was made under. Without it
        # two batches of room 2 episodes are not comparable — the same mean
        # reward means very different things at 0.10 and at 0.80 — and nothing
        # on the replay screen could say which was which.
        collapse = self.environment_collapse_chance()
        if collapse is not None:
            batch["collapseChance"] = collapse
        if self.test_episode is not None:
            batch["testEpisode"] = self.test_episode
        return batch

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
            # Room 4's thrusts have names but no arrow: an arrow means "it
            # went that way", and a thrust is an acceleration rather than a
            # move. The field is kept for every room so the page needs no
            # branch, and left empty where it would be a lie.
            "actions": [{"id": action,
                         "name": self.action_names[action],
                         "arrow": ACTION_ARROWS.get(action, "")
                                  if self.action_names is ACTION_NAMES else ""}
                        for action in self.env.actions()],
        }


def _layout_seed(env):
    """Which generated layout the world is currently in, if it has any.

    Rooms 1 to 4 have one layout for the whole session and answer None; a
    recorded room-5 episode carries the seed so that the exact warehouse it
    happened in can be regenerated and checked against the recording.
    """
    layout = getattr(env, "layout", None)
    if isinstance(layout, dict):
        return layout.get("seed")
    return None


def _report(split, policy, layouts, counts, rewards, lengths, seeds):
    """One evaluation, as rates and means rather than as raw counts.

    The standard deviation is in here because a mean reward on its own hides
    the shape of the thing: a policy that escapes half the time and dies half
    the time has the same mean as one that always half-finishes, and they are
    not the same policy.
    """
    total = max(1, layouts)
    mean = sum(rewards) / total if rewards else 0.0
    if len(rewards) > 1:
        variance = sum((value - mean) ** 2 for value in rewards) / len(rewards)
        deviation = math.sqrt(variance)
    else:
        deviation = 0.0
    return {
        "split": split,
        "policy": policy,
        "layouts": layouts,
        "escapeRate": counts["escape"] / total,
        "terminalRate": counts["terminal"] / total,
        "collisionRate": counts["collision"] / total,
        "timeoutRate": counts["timeout"] / total,
        "meanReward": mean,
        "rewardDeviation": deviation,
        "meanLength": (sum(lengths) / total) if lengths else 0.0,
        # Which layouts these numbers came from, so "unseen" is checkable
        # rather than merely asserted.
        "seeds": list(seeds),
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
