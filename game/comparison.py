"""Several methods against the same room, so their graphs can be read together.

A single run tells you what one method did.  It does not tell you whether that
was any good, and the interesting claims in this project are all comparative:
Value Iteration and Policy Iteration reach the same plan but take a different
number of sweeps to get there; Q-Learning takes room 2's span and SARSA goes
round.  Neither is visible from one curve.

So this runs a room's variants side by side and reports each one's series, and
it is careful about the two things that would make the comparison worthless:

  * **The same seed.**  Every variant gets the identical environment stream and
    its own algorithm stream, drawn the same way a single run draws them, so a
    difference between two curves is a difference between two methods and not
    between two dice.
  * **The same parameters.**  One set, applied to all of them. A variant that
    does not expose a parameter simply ignores it.

Nothing here reimplements training.  Each variant *is* an ordinary `Session`,
driven by the same `advance` the single-run screens use, which is what stops
the comparison and the individual rooms from ever drifting apart.
"""

from game import config, rooms
from game.algorithms import describe_all
from game.session import Session


class Comparison:
    """One room, several methods, advanced together."""

    def __init__(self, room_number, algorithm_keys=None, parameters=None,
                 seed=None):
        self.room = rooms.room(room_number)
        if not self.room.get("built"):
            raise ValueError("room %s is not built yet" % room_number)

        keys = list(algorithm_keys or self.room["algorithms"])
        unknown = [key for key in keys if key not in self.room["algorithms"]]
        if unknown:
            raise ValueError("room %s does not offer %s"
                             % (room_number, ", ".join(unknown)))
        if len(keys) < 2:
            raise ValueError("a comparison needs at least two methods")

        self.seed = config.SESSION["seed_default"] if seed is None else seed
        self.keys = keys
        # One session per method, all with the same seed and the same
        # parameters. Built through the ordinary constructor, so a variant
        # here behaves exactly as it would on its own.
        self.runs = [Session(room_number, algorithm_key=key,
                             parameters=parameters, seed=self.seed)
                     for key in keys]
        for run in self.runs:
            run.play()

    # ------------------------------------------------------------------

    @property
    def finished(self):
        return all(run.state != "TRAINING" for run in self.runs)

    def advance(self, budget_ms=None):
        """Move every unfinished variant along by one request's worth of work.

        Each gets the *full* budget rather than a share of it. Splitting it
        was the obvious thing and it was wrong: the total work per request
        stayed the same while the number of requests multiplied by the number
        of variants, so four methods took four times the round trips to reach
        the same place. A comparison is a deliberate, blocking thing with its
        own "comparing…" state, so a request that takes four frames' worth of
        compute is a fair trade for a quarter of the traffic.

        All of them are advanced together rather than one to completion, so
        the curves fill in side by side — which is the whole point of the
        screen.
        """
        budget = config.TURBO_BUDGET_MS if budget_ms is None else budget_ms
        for run in self.runs:
            if run.state == "TRAINING":
                run.advance(budget_ms=budget)
        return self.snapshot()

    # ------------------------------------------------------------------

    def _variant(self, key, run):
        snapshot = run.snapshot()
        return {
            "algorithm": key,
            "state": snapshot["state"],
            "progress": snapshot["progress"],
            "metric": snapshot["metric"],
            "curve": snapshot["curve"],
            "readout": snapshot["readout"],
            "solved": snapshot["solved"],
        }

    def snapshot(self):
        return {
            "room": self.room["number"],
            "seed": self.seed,
            "finished": self.finished,
            # The parameters are one set shared by all of them, so they are
            # reported once rather than per variant.
            "parameters": dict(self.runs[0].parameters),
            "variants": [self._variant(key, run)
                         for key, run in zip(self.keys, self.runs)],
        }

    def describe(self):
        """The static half: which methods, and what they are. Sent once."""
        return {
            "room": {"number": self.room["number"],
                     "name": self.room["name"],
                     "sector": self.room["sector"]},
            "algorithms": describe_all(self.keys),
            "unit": "episodes" if self.runs[0].learns else "sweeps",
        }
