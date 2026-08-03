"""Reproduce the "settings that solve it" tables in the README.

Every number under those headings came out of this file. It trains each room
headlessly at its own defaults and then at the nearby settings that fail, and
prints the result as the table the README carries.

    python3 experiments/measure_parameters.py --room 1     # seconds
    python3 experiments/measure_parameters.py --room 2     # a minute or two
    python3 experiments/measure_parameters.py --room 3     # a minute or two
    python3 experiments/measure_parameters.py --room 4     # a few minutes
    python3 experiments/measure_parameters.py --room 5     # half an hour
    python3 experiments/measure_parameters.py              # all five

WHY THIS IS NOT A TEST
A test has to pass on every machine on every run, so the only thing it can
assert about a learned policy is a loose bound — which is what
`tests/test_defense_metrics.py` does. These are measurements: they report what
happened rather than asserting anything, and their whole value is the figures
next to each other. Numbers will move a little with the Python version's
`random`; the orderings the README argues from are the stable part.

Nothing here is imported by the game. It is a reader's tool.
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game.session import Session                        # noqa: E402  (after path)

SEEDS = (0, 1, 2, 3, 4)


# ----------------------------------------------------------------------
# The one thing every measurement needs: a finished run
# ----------------------------------------------------------------------

def train(room, parameters=None, seed=0):
    """Train to completion with no interface attached."""
    session = Session(room, parameters=parameters or {}, seed=seed)
    session.play()
    while session.state == "TRAINING":
        # The same call the server makes, without the two-second poll between.
        session.advance(budget_ms=2000.0)
    return session


def tail(session, field, window=200):
    """The mean of one logged field over the last `window` episodes."""
    rows = session.episode_log[-window:]
    if not rows:
        return float("nan")
    return sum(row.get(field) or 0 for row in rows) / len(rows)


def greedy_run(session, limit=400):
    """One run of the learned policy with exploration removed.

    Returns (reached the goal, steps, return, the cells or points visited).
    """
    env = session.env
    chooser = getattr(session.algorithm, "greedy_action", None)
    if chooser is None:
        table = session.algorithm.greedy_policy()
        default = env.actions()[0]
        chooser = lambda state: table.get(state, default)   # noqa: E731

    state = env.reset()
    seen = [state[:2]]
    total = 0.0
    for count in range(limit):
        state, reward, done, info = env.step(chooser(state))
        total += reward
        seen.append(state[:2])
        if done:
            return bool(info.get("goal")), count + 1, total, seen
    return False, limit, total, seen


def rule(title):
    print()
    print(title)
    print("-" * len(title))


# ----------------------------------------------------------------------
# Room 1 — a planner, so the measurement is the plan and the walk
# ----------------------------------------------------------------------

def room1():
    def plan(**overrides):
        session = train(1, overrides)
        learned = session.snapshot()["learned"]
        # 200 walks, because the floor is stochastic and one walk is a sample.
        steps = reached = 0
        for _ in range(200):
            got, count, _total, _seen = greedy_run(session)
            steps += count
            reached += got
        return session, learned, steps / 200.0, reached / 200.0

    rule("Room 1 — the ice slider (gamma 0.95, theta 1e-4)")
    print("slip  sweeps  V(start)  steps  reached the panel")
    for slip in (0.00, 0.10, 0.20, 0.30, 0.40, 0.50):
        _s, learned, steps, reached = plan(slip=slip)
        print("%.2f  %6d  %8.2f  %5.1f  %.0f%%"
              % (slip, learned["sweeps"], learned["startValue"], steps,
                 100 * reached))

    rule("Room 1 — the discount (slip 0.20, theta 1e-4)")
    print("gamma  sweeps  V(start)  steps")
    for gamma in (0.50, 0.80, 0.90, 0.95, 0.99, 0.999):
        _s, learned, steps, _reached = plan(gamma=gamma)
        print("%.3f  %6d  %8.2f  %5.1f"
              % (gamma, learned["sweeps"], learned["startValue"], steps))

    rule("Room 1 — the stopping threshold (slip 0.20)")
    print("theta   sweeps  V(start)  steps")
    for theta in (1e-1, 1e-2, 1e-4, 1e-6):
        _s, learned, steps, _reached = plan(theta=theta)
        print("%.0e  %6d  %8.2f  %5.1f"
              % (theta, learned["sweeps"], learned["startValue"], steps))


# ----------------------------------------------------------------------
# Room 2 — which of the two routes the policy settles on
# ----------------------------------------------------------------------

SPAN_CELLS = {(7, column) for column in range(2, 8)}


def room2():
    def measure(parameters, runs=100):
        spans = exits = steps = 0.0
        for seed in SEEDS:
            session = train(2, parameters, seed=seed)
            for _ in range(runs):
                got, count, _total, seen = greedy_run(session, limit=200)
                spans += any(cell in SPAN_CELLS for cell in seen)
                exits += got
                steps += count
        total = len(SEEDS) * runs
        return 100 * spans / total, 100 * exits / total, steps / total

    rule("Room 2 — SARSA, 1500 episodes, seeds 0-4, 100 greedy runs each")
    print("%-34s %6s %6s %6s" % ("setting", "span", "exit", "steps"))
    for label, parameters in (
        ("the defaults", {}),
        ("Q init 0 instead of 30", {"q_init": 0.0}),
        ("300 episodes instead of 1500", {"episodes": 300}),
        ("epsilon decay 0.95", {"epsilon_decay": 0.95}),
        ("collapse 0.30", {"collapse_chance": 0.30}),
        ("collapse 0.60", {"collapse_chance": 0.60}),
    ):
        span, exit_rate, steps = measure(parameters)
        print("%-34s %5.0f%% %5.0f%% %6.1f" % (label, span, exit_rate, steps))

    rule("Room 2 — the route against the collapse probability")
    print("p     the five seeds                    took the span")
    for probability in (0.00, 0.05, 0.10, 0.20, 0.30, 0.40, 0.60):
        routes = []
        for seed in SEEDS:
            session = train(2, {"collapse_chance": probability}, seed=seed)
            _got, _count, _total, seen = greedy_run(session, limit=200)
            routes.append("span" if any(cell in SPAN_CELLS for cell in seen)
                          else "round")
        print("%.2f  %-33s %d/5"
              % (probability, " ".join(routes), routes.count("span")))


# ----------------------------------------------------------------------
# Room 3 — the greedy policy, which is what the room is solved by
# ----------------------------------------------------------------------

def room3():
    rule("Room 3 — Q-Learning, seeds 0-4, one greedy run each after training")
    print("%-34s %7s %6s %8s %s"
          % ("setting", "got out", "steps", "return", "training success"))
    for label, parameters in (
        ("the defaults", {}),
        ("2000 episodes", {"episodes": 2000}),
        ("1000 episodes", {"episodes": 1000}),
        ("gamma 0.95", {"gamma": 0.95}),
        ("gamma 0.90", {"gamma": 0.90}),
        ("gamma 0.80", {"gamma": 0.80}),
        ("epsilon decay 0.95", {"epsilon_decay": 0.95}),
        ("epsilon floor 0 instead of 0.05", {"epsilon_min": 0.0}),
    ):
        out = steps = total = training = 0.0
        for seed in SEEDS:
            session = train(3, parameters, seed=seed)
            got, count, ret, _seen = greedy_run(session)
            out += got
            steps += count
            total += ret
            training += tail(session, "success")
        n = len(SEEDS)
        print("%-34s %5d/%d %6.1f %8.1f %15.0f%%"
              % (label, out, n, steps / n, total / n, 100 * training / n))


# ----------------------------------------------------------------------
# Room 4 — did it land, and within the speed limit
# ----------------------------------------------------------------------

def room4():
    rule("Room 4 — semi-gradient SARSA, the last 200 of 1200 episodes")
    print("%-34s %7s %9s" % ("setting", "landed", "return"))
    for label, parameters in (
        ("the defaults", {}),
        ("gamma 0.90 instead of 0.995", {"gamma": 0.90}),
        ("alpha 0.90 instead of 0.30", {"alpha": 0.90}),
        ("landing limit 1.5", {"landing_speed": 1.5}),
        ("landing limit 0.5", {"landing_speed": 0.5}),
    ):
        session = train(4, parameters)
        print("%-34s %6.1f%% %9.1f"
              % (label, 100 * tail(session, "success"),
                 tail(session, "reward")))


# ----------------------------------------------------------------------
# Room 5 — the frozen policy on layouts it never trained on
# ----------------------------------------------------------------------

def room5(seeds=(0, 1, 2)):
    rule("Room 5 — semi-gradient Q-Learning, frozen-policy escape rate")
    print("(seeds %s; an evaluation updates no weights)"
          % ", ".join(str(seed) for seed in seeds))
    print("%-36s %6s %6s %7s %7s"
          % ("setting", "train", "valid", "unseen", "random"))
    for label, parameters in (
        ("the defaults", {}),
        ("gamma 0.99 instead of 0.97", {"gamma": 0.99}),
        ("alpha 0.60 instead of 0.20", {"alpha": 0.60}),
        ("sensor range 10 m instead of 3", {"sensor_range": 10.0}),
        ("1000 episodes instead of 4000", {"episodes": 1000}),
        ("20 training layouts instead of 120", {"train_layouts": 20}),
    ):
        reports = []
        for seed in seeds:
            session = train(5, parameters, seed=seed)
            reports.append(session.run_evaluation())
        rate = lambda split: 100 * sum(               # noqa: E731
            report[split]["escapeRate"] for report in reports) / len(reports)
        print("%-36s %5.1f%% %5.1f%% %6.1f%% %6.1f%%"
              % (label, rate("train"), rate("validation"), rate("test"),
                 rate("randomTest")))


ROOMS = {1: room1, 2: room2, 3: room3, 4: room4, 5: room5}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--room", type=int, choices=sorted(ROOMS),
                        help="one room; every room if left out")
    arguments = parser.parse_args()

    wanted = [arguments.room] if arguments.room else sorted(ROOMS)
    started = time.time()
    for number in wanted:
        ROOMS[number]()
    print()
    print("%.0f seconds" % (time.time() - started))


if __name__ == "__main__":
    main()
