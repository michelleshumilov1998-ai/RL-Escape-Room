"""Keeping whole episodes, so a finished run can be watched rather than read.

A learning curve says the agent improved.  It does not say what it was
*doing* at episode 40 and what it was doing differently at episode 1400,
and that is the thing this project exists to show — so a sample of episodes
is kept step by step and handed to the page when the run finishes.

WHY A SAMPLE
Every episode contributes to the graphs; only a few dozen are kept in full.
An episode early in a run is ε-random and routinely walks until the step
limit stops it, so its cost is closer to `max_steps_per_episode` than to the
length of a good route, and the whole batch goes to the page in one
response.  `wanted_episodes` therefore spends the budget where the
behaviour actually changes.

WHY THE TRAINING EPISODES THEMSELVES, AND NOT PROBE RUNS
The alternative is to interrupt training every so often and run one extra
episode purely to be watched.  That costs a second code path, and — worse —
a probe drawing from the agent's own generator would shift every random
number after it, so switching recording on would change what was learned.
Recording is passive: it is handed the transitions the agent was taking
anyway and consumes no randomness at all, so a run with recording on and a
run with it off learn identically.

WHAT COMES OUT
The shape the room screen's data contract asks for — see
`web/assets/room/contract.js`, which is the authority on it:

    episodes  [ { index, steps[], totalReward, outcome, epsilon } ]
    metrics   [ { episode, reward, steps, epsilon, convergence } ]

The two arrays are parallel and the page indexes them by position, so they
are appended to together and never separately.
"""


def wanted_episodes(target, budget):
    """Which episode indices to keep, given how many are going to be run.

    Evenly spaced, but with the first and last few always included: the ends
    are where the behaviour is most obviously different, and an even spacing
    on its own can miss both. Returns a set of 0-based indices.
    """
    target = max(1, int(target))
    budget = max(0, int(budget))

    # A budget of nothing means record nothing, which is worth being able to
    # ask for: it is how a run is measured with the recording taken out.
    if budget == 0:
        return set()

    if target <= budget:
        return set(range(target))

    # The ends first, then the spacing fills whatever budget is left.
    edge = min(3, budget // 4)
    chosen = set(range(edge)) | set(range(target - edge, target))

    remaining = budget - len(chosen)
    if remaining > 0:
        stride = (target - 1) / float(remaining + 1)
        for index in range(1, remaining + 1):
            chosen.add(int(round(index * stride)))

    # Rounding can collide with an edge index and leave the budget short. It
    # is a sample either way, so the count is allowed to come out a little
    # under rather than being padded with near-duplicates.
    return {index for index in chosen if 0 <= index < target}


def frame(env, action_names, state, reward, action, slipped=False):
    """One frame of one episode: a picture of the whole world at a moment.

    A module function rather than a method, because a frame is a fact about
    the environment and not about recording. Two callers build them: this
    recorder, passively, while the agent trains; and the greedy replay in
    `session.py`, which is not a recording at all. Sharing this is what
    stops a replayed frame from being able to describe the world in a
    different shape from a recorded one.
    """
    # Whatever else this room has that moves or changes appearance. The room
    # owns that knowledge; a frame only forwards it.
    entity_states, positions = env.frame_extras(state)

    built = {
        "position": env.world_position(state),
        # A grid room has no velocity and answers None; room 4's state carries
        # one, and the renderer uses it to point the drone the way it is going.
        "velocity": (env.world_velocity(state)
                     if hasattr(env, "world_velocity") else None),
        "reward": reward,
        "action": None if action is None else action_names[action],
        "entityStates": entity_states,
        "entityPositions": positions,
        # Whether the floor overruled the action on this step.
        #
        # WHY A FRAME HAS TO CARRY THIS
        # Room 1's policy arrow is the action the plan *tried*; on ice the
        # agent slides sideways instead, which is the room's whole model. With
        # nothing on the frame to say so, the arrow and the movement simply
        # disagreed and it read as a rendering bug. The environment has always
        # reported it in `info`; it just never reached the screen.
        "slipped": bool(slipped),
    }

    # A room that has more to say about the moment says it here. Room 5 carries
    # the mission stage, the objective, what the sensors returned and which
    # obstacles were visible — the things that make a replay inspectable rather
    # than merely watchable. Every other room offers no hook and is unchanged.
    if hasattr(env, "frame_detail"):
        built["detail"] = env.frame_detail(state)
    return built


class Recorder:
    """Accumulates the episodes it was told to keep, and nothing else."""

    def __init__(self, env, budget, episode_target, action_names):
        self.env = env
        self.action_names = action_names
        self.budget = budget
        self.wanted = wanted_episodes(episode_target, budget)
        self.episodes = []
        self.metrics = []
        self.current = None

    def retarget(self, episode_target):
        """The episode count is a live parameter, so the schedule can move.

        Episodes already recorded are kept — they happened. Only which of the
        *remaining* ones get kept changes.
        """
        self.wanted |= wanted_episodes(episode_target, self.budget)

    # ------------------------------------------------------------------

    def begin(self, index, state, epsilon):
        """Called as an episode starts. Cheap when it is not being kept."""
        if index not in self.wanted:
            self.current = None
            return

        self.current = {
            "index": index,
            "epsilon": epsilon,
            # Which generated layout this episode was flown in. The assignment
            # requires a replay to reproduce the original layout, and the seed
            # is what makes that checkable: regenerate from it and the geometry
            # must match the `entities` recorded beside it.
            "layoutSeed": (self.env.layout.get("seed")
                           if isinstance(getattr(self.env, "layout", None), dict)
                           else None),
            "split": getattr(self.env, "split", None),
            # Room 2's bridge risk, straight off the world. An episode that
            # does not say what it was recorded under cannot be compared with
            # one that was recorded under something else.
            "collapseChance": (float(self.env.collapse)
                               if hasattr(self.env, "collapse") else None),
            # The warehouse this episode happened in. Rooms whose furniture
            # never moves send nothing and the page keeps using the room's own
            # entity list; room 5's changes every episode, so a replay that did
            # not carry it would animate the right trajectory through the wrong
            # building.
            "entities": (self.env.entities()
                         if getattr(self.env, "layout_varies", False) else None),
            # The starting position is a step with nothing having happened
            # yet, so the first frame draws the agent where it began.
            "steps": [self._frame(state, 0.0, None)],
            "errors": [],
        }

    def step(self, state, action, reward, report, info=None):
        """One transition, after it has been applied.

        `state` is where the agent ended up, because a frame is a picture of
        the world at a moment rather than a description of a move — and every
        frame describes the whole world, not what changed, so an episode can
        be joined at any step. `frame_extras` is what makes that true.
        """
        if self.current is None:
            return
        self.current["steps"].append(
            self._frame(state, reward, action,
                        bool((info or {}).get("slipped"))))
        error = (report or {}).get("tdError")
        if error is not None:
            self.current["errors"].append(abs(error))

    def end(self, outcome, total_reward):
        if self.current is None:
            return

        errors = self.current["errors"]
        episode = {
            "index": self.current["index"],
            "entities": self.current["entities"],
            "layoutSeed": self.current["layoutSeed"],
            "split": self.current["split"],
            "collapseChance": self.current["collapseChance"],
            "steps": self.current["steps"],
            "totalReward": total_reward,
            "outcome": outcome,
            "epsilon": self.current["epsilon"],
        }
        self.episodes.append(episode)
        self.metrics.append({
            "episode": self.current["index"],
            "reward": total_reward,
            # The starting frame is not a step taken.
            "steps": len(self.current["steps"]) - 1,
            # The rate it *ran* at, not the decayed one it leaves behind, so
            # the number beside an episode describes that episode.
            "epsilon": self.current["epsilon"],
            # "Something that ought to fall towards zero", which the contract
            # leaves to the room. The mean size of the corrections the method
            # made this episode is the honest one for a learner: it is large
            # while the estimates are wrong and small once they are not.
            "convergence": sum(errors) / len(errors) if errors else 0.0,
        })
        self.current = None

    # ------------------------------------------------------------------

    def _frame(self, state, reward, action, slipped=False):
        return frame(self.env, self.action_names, state, reward, action,
                     slipped)

    def batch(self):
        """The whole recording, in the order it happened."""
        return {"episodes": list(self.episodes), "metrics": list(self.metrics)}
