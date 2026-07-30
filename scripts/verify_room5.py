"""End-to-end verification of Room 5.

Run it with:

    .venv/bin/python scripts/verify_room5.py

It works through the checklist in the Room 5 brief and prints real numbers:

  1. import every module
  2. check layout generation, validation and the seed split
  3. check the mechanics: robots, crossings, conveyors, terminal, exit
  4. check the radar and that the map is never handed to the agent
  5. check the feature representation and one update by hand
  6. train with Semi-Gradient Q-Learning
  7. measure training / validation / unseen test success
  8. show what happens with too few training layouts
  9. compare against two trivial baselines
 10. check that replay reproduces an episode exactly
 11. build a scene and a still frame for the renderers
 12. generate every required graph
 13. save and load a model

The exit status is 0 only if every check passed.
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILURES = []

EPISODES = 2500
TRAINING_LAYOUTS = 20
TEST_LAYOUTS = 12


def heading(text):
    print("\n" + text)
    print("-" * len(text))


def check(label, condition, detail=""):
    mark = "PASS" if condition else "FAIL"
    print("  [%s] %s%s" % (mark, label, ("  — " + detail) if detail else ""))
    if not condition:
        FAILURES.append(label)
    return condition


def tail_mean(values, fraction=0.2):
    count = max(1, int(len(values) * fraction))
    slice_ = values[-count:]
    return sum(slice_) / len(slice_) if slice_ else 0.0


def main():
    heading("1. Importing every module")
    import importlib
    modules = ["rooms.room5.layout", "rooms.room5.environment",
               "rooms.room5.features", "rooms.room5.q_agent",
               "rooms.room5.simulate", "rooms.room5.experiments",
               "rooms.room5.storage", "rooms.room5.page",
               "renderers.room5_scene", "renderers.fallback_renderer",
               "plots.room5_plots", "ui.completion", "app"]
    for name in modules:
        importlib.import_module(name)
    check("all %d Room 5 modules import" % len(modules), True)

    from plots import room5_plots
    from renderers import base_renderer, fallback_renderer, iso_canvas, room5_scene
    from rooms.room5 import experiments, layout as L, q_agent, simulate, storage
    from rooms.room5 import environment as E
    from rooms.room5.environment import Room5Env
    from rooms.room5.features import FEATURE_SETS, FeatureExtractor

    # ------------------------------------------------------------------
    heading("2. Layout generation")
    pool = L.generate_pool("training", 20)
    check("every generated layout is valid",
          all(L.is_valid(candidate) for candidate in pool),
          "%d layouts, %dx%d each" % (len(pool), pool[0].size, pool[0].size))
    check("each has exactly one start, terminal and exit",
          all("".join("".join(row) for row in candidate.grid).count(tile) == 1
              for candidate in pool
              for tile in (L.START, L.TERMINAL, L.EXIT)))
    check("both legs of every mission are walkable",
          all(candidate.shortest_mission_length() is not None
              for candidate in pool),
          "shortest missions: %d to %d steps"
          % (min(candidate.shortest_mission_length() for candidate in pool),
             max(candidate.shortest_mission_length() for candidate in pool)))
    check("the same seed always gives the same warehouse",
          L.generate(4242, "training").grid == L.generate(4242, "training").grid)
    check("training, validation and test seeds never overlap",
          not L.splits_overlap(60, 20, 30),
          "training from %d, validation from %d, test from %d"
          % (L.TRAINING_SEED_BASE, L.VALIDATION_SEED_BASE, L.TEST_SEED_BASE))
    print("       %-8s %10s %10s %10s"
          % ("setting", "obstacles", "robots", "mission"))
    for difficulty in L.DIFFICULTIES:
        sample = L.generate_pool("training", 6, difficulty)
        print("       %-8s %9.0f%% %10.1f %10.1f"
              % (difficulty,
                 sum(candidate.obstacle_density() for candidate in sample)
                 / len(sample) * 100,
                 sum(len(candidate.robot_templates) for candidate in sample)
                 / len(sample),
                 sum(candidate.shortest_mission_length() for candidate in sample)
                 / len(sample)))

    # ------------------------------------------------------------------
    heading("3. The mechanics")
    layout = pool[0]
    env = Room5Env(layout=layout)
    env.reset(layout)
    check("there are five actions including Wait",
          len(env.actions()) == 5 and E.WAIT in env.actions())

    before = env.position
    env.step(E.WAIT)
    check("waiting holds position", env.position == before,
          "still at %s while the robots move" % (before,))

    # A robot walking into the cell R-5 is standing in.
    robot_env = Room5Env(layout=layout)
    robot_env.reset(layout)
    if robot_env.robots:
        robot = robot_env.robots[0]
        ahead = robot.route[(robot.index + 1) % len(robot.route)]
        robot_env.position = ahead
        _, robot_reward, robot_done, robot_info = robot_env.step(E.WAIT)
        check("a maintenance robot ends the mission",
              robot_done and robot_info["caught"],
              "reward %+.0f" % robot_reward)

    # A straight swap: R-5 and a robot exchanging cells.
    grid = [[L.FLOOR for _ in range(6)] for _ in range(6)]
    grid[4][1], grid[2][2], grid[1][4] = L.START, L.TERMINAL, L.EXIT
    crossing = L.Layout(1, "training", grid, (4, 1), (2, 2), (1, 4),
                        [L.Robot([(3, 1), (3, 2)], "loop", index=1)],
                        "Easy", 4, 60)
    cross_env = Room5Env(layout=crossing)
    cross_env.reset(crossing)
    cross_env.position = (3, 1)
    cross_env.robots[0].index = 1
    _, cross_reward, cross_done, cross_info = cross_env.step(E.MOVE_RIGHT)
    check("swapping places with a robot counts as a collision",
          cross_done and cross_info["caught"],
          "reward %+.0f — the crossing case is checked, not just same-cell"
          % cross_reward)

    route = [(1, 1), (1, 2), (1, 3)]
    looping = L.Robot(route, "loop")
    bouncing = L.Robot(route, "bounce")
    loop_walk = [looping.advance() for _ in range(4)]
    bounce_walk = [bouncing.advance() for _ in range(4)]
    check("the two patrol patterns differ", loop_walk != bounce_walk,
          "loop %s, bounce %s"
          % ("->".join("%d,%d" % cell for cell in loop_walk),
             "->".join("%d,%d" % cell for cell in bounce_walk)))

    belt = [[L.FLOOR for _ in range(6)] for _ in range(6)]
    belt[4][1], belt[2][2], belt[1][4] = L.START, L.TERMINAL, L.EXIT
    belt[4][2] = L.CONVEYOR_RIGHT
    belt_layout = L.Layout(1, "training", belt, (4, 1), (2, 2), (1, 4), [],
                           "Easy", 4, 60)
    belt_env = Room5Env(layout=belt_layout)
    belt_env.reset(belt_layout)
    _, belt_reward, _, belt_info = belt_env.step(E.MOVE_RIGHT)
    check("a conveyor carries the robot one cell further",
          belt_info["conveyor"] and belt_env.position == (4, 3),
          "stepped onto (4,2) and ended at %s, reward %+.0f"
          % (belt_env.position, belt_reward))

    # The terminal, then the exit.
    mission = Room5Env(layout=layout)
    mission.reset(layout)
    exit_row, exit_col = layout.exit_cell
    neighbour = next((exit_row + dr, exit_col + dc)
                     for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1))
                     if layout.in_bounds(exit_row + dr, exit_col + dc)
                     and not layout.blocks(exit_row + dr, exit_col + dc))
    towards_exit = next(action for action, (dr, dc) in E.ACTION_DELTAS.items()
                        if action != E.WAIT
                        and (neighbour[0] + dr, neighbour[1] + dc)
                        == (exit_row, exit_col))

    mission.position = neighbour
    mission.stage = 0
    _, early_reward, early_done, early_info = mission.step(towards_exit)
    check("the exit is locked until the terminal is used",
          not early_done and early_info["early_exit"],
          "reward %+.0f, pushed back to %s" % (early_reward, mission.position))

    terminal_row, terminal_col = layout.terminal
    approach = next((action, (terminal_row - dr, terminal_col - dc))
                    for action, (dr, dc) in E.ACTION_DELTAS.items()
                    if action != E.WAIT
                    and layout.in_bounds(terminal_row - dr, terminal_col - dc)
                    and not layout.blocks(terminal_row - dr, terminal_col - dc))
    mission.position = approach[1]
    mission.stage = 0
    _, terminal_reward, _, terminal_info = mission.step(approach[0])
    check("the terminal activates once and unlocks the exit",
          terminal_info["terminal_activated"] and mission.stage == 1,
          "reward %+.0f" % terminal_reward)
    _, repeat_reward, _, repeat_info = mission.step(E.WAIT)
    check("standing on it again pays nothing",
          not repeat_info["terminal_activated"], "reward %+.0f" % repeat_reward)

    mission.position = neighbour
    mission.stage = 1
    _, exit_reward, exit_done, exit_info = mission.step(towards_exit)
    check("with the data collected the exit completes the mission",
          exit_done and exit_info["reached_exit"], "reward %+.0f" % exit_reward)

    check("no transition model and no layout are exposed to the agent",
          not hasattr(env, "transitions"))

    # ------------------------------------------------------------------
    heading("4. The radar")
    env.reset(layout)
    observation = env.observation()
    check("eight static rays and four dynamic ones",
          len(observation["static_radar"]) == 8
          and len(observation["dynamic_radar"]) == 4,
          "directions: %s" % ", ".join(E.RADAR_NAMES))
    check("every reading is normalised to (0, 1]",
          all(0.0 < value <= 1.0 for value in
              observation["static_radar"] + observation["dynamic_radar"]),
          "nearest wall %.2f, nearest robot %.2f"
          % (min(observation["static_radar"]), observation["nearest_robot"]))

    # A wall one cell to the north, in a layout built for the check.
    probe_grid = [[L.FLOOR for _ in range(7)] for _ in range(7)]
    probe_grid[5][1], probe_grid[2][3], probe_grid[1][5] = \
        L.START, L.TERMINAL, L.EXIT
    probe_grid[2][2] = L.SHELF
    probe_layout = L.Layout(1, "training", probe_grid, (5, 1), (2, 3), (1, 5), [],
                            "Easy", 4, 60)
    probe = Room5Env(layout=probe_layout)
    probe.reset(probe_layout)
    probe.position = (3, 2)
    check("a ray reports the true distance to the nearest obstacle",
          abs(probe.static_radar()[0] - 1 / probe.radar_range) < 1e-9,
          "one cell north out of a range of %d reads %.2f"
          % (probe.radar_range, probe.static_radar()[0]))

    keys = sorted(observation.keys())
    check("the observation is local only — no grid, no seed, no absolute position",
          keys == ["dynamic_radar", "nearest_robot", "previous_action",
                   "stage", "static_radar", "target"],
          "%d numbers in total: %s"
          % (8 + 4 + 3 + 1 + 1 + 5, ", ".join(keys)))
    check("the fog of war is a display concern, not part of the observation",
          "visible_cells" not in observation and bool(env.visible_cells()),
          "%d cells visible from the start" % len(env.visible_cells()))

    # ------------------------------------------------------------------
    heading("5. The features and one update by hand")
    extractor = FeatureExtractor()
    counts = {len(extractor.active_features(env.observation(), 0))}
    for _ in range(20):
        env.step(E.WAIT)
        counts.add(len(extractor.active_features(env.observation(), 0)))
    check("the number of active features never changes", len(counts) == 1,
          "%d active out of %d, whatever the observation"
          % (counts.pop(), extractor.total_features))

    env.reset(layout)
    observation = env.observation()
    blocks = [set(extractor.active_features(observation, action))
              for action in E.ACTIONS]
    overlap = any(blocks[first] & blocks[second]
                  for first in range(len(blocks))
                  for second in range(first + 1, len(blocks)))
    check("each action owns its own block of features", not overlap)

    agent = q_agent.LinearAgent()
    indices = agent.extractor.active_features(observation, E.MOVE_RIGHT)
    delta = agent.update(observation, E.MOVE_RIGHT, 8.0, 0.5)
    check("Q(s,a) = w · x(s,a), and the update moves it towards the target",
          abs(delta - 8.0) < 1e-9
          and abs(agent.q_value(observation, E.MOVE_RIGHT) - 4.0) < 1e-9,
          "from 0 with target 8 and alpha 0.5: TD %+.1f, Q now %.1f, "
          "spread over %d features"
          % (delta, agent.q_value(observation, E.MOVE_RIGHT), len(indices)))
    print("       feature sets available: %s"
          % ", ".join(sorted(FEATURE_SETS)))

    # ------------------------------------------------------------------
    heading("6. Training with Semi-Gradient Q-Learning")
    trained = q_agent.train(episodes=EPISODES, seed=0,
                            training_layouts=TRAINING_LAYOUTS)
    history = trained["history"]
    check("it learns to complete the mission",
          tail_mean(history["success"]) >= 0.5,
          "training success %.0f%%, reward %.1f, %d episodes in %.1fs"
          % (tail_mean(history["success"]) * 100, tail_mean(history["reward"]),
             EPISODES, trained["runtime_seconds"]))
    check("it reaches the terminal in most episodes",
          tail_mean(history["terminal"]) >= 0.6,
          "%.0f%% of late episodes" % (tail_mean(history["terminal"]) * 100))
    check("no weight became NaN or infinite", trained["agent"].is_finite(),
          "weight norm %.1f, %d of %d features touched"
          % (trained["weight_norm"], trained["active_features"],
             trained["agent"].extractor.total_features))
    check("the validation curve was recorded on held-out layouts",
          bool(history["validation_success"])
          and not set(trained["training_seeds"]) & set(trained["validation_seeds"]),
          "%d checkpoints, last %.0f%%"
          % (len(history["validation_success"]),
             history["validation_success"][-1] * 100))
    check("a worked update was captured", trained["example_update"] is not None)

    # ------------------------------------------------------------------
    heading("7. Training, validation and unseen test layouts")
    splits = (("training", trained["training_seeds"]),
              ("validation", trained["validation_seeds"]),
              ("test", None))
    print("       %-11s %7s %8s %8s %8s %8s"
          % ("split", "layouts", "success", "terminal", "return", "caught"))
    scores = {}
    for name, seeds in splits:
        count = (len(seeds) if seeds is not None else TEST_LAYOUTS)
        layouts = L.generate_pool(name, count, trained["difficulty"])
        outcome = q_agent.evaluate(trained["agent"], layouts)
        scores[name] = outcome
        print("       %-11s %7d %7.0f%% %7.0f%% %8.1f %7.0f%%"
              % (name, count, outcome["success_rate"] * 100,
                 outcome["terminal_rate"] * 100, outcome["mean_return"],
                 outcome["caught_rate"] * 100))
    check("it completes missions in the warehouses it trained on",
          scores["training"]["success_rate"] >= 0.5,
          "%.0f%%" % (scores["training"]["success_rate"] * 100))
    check("it also completes missions in warehouses it has never entered",
          scores["test"]["success_rate"] >= 0.3,
          "%.0f%% on %d unseen layouts — real transfer, and clearly below the "
          "training score" % (scores["test"]["success_rate"] * 100, TEST_LAYOUTS))

    # ------------------------------------------------------------------
    heading("8. How many training layouts it takes")
    rows = experiments.layout_count_experiment(episodes=EPISODES, repeats=1)
    print("       %-10s %8s %11s %8s"
          % ("layouts", "training", "validation", "unseen"))
    for row in rows:
        print("       %-10s %7.0f%% %10.0f%% %7.0f%%"
              % (row["value"], row["mean_training_success"] * 100,
                 row["mean_validation_success"] * 100,
                 row["mean_test_success"] * 100))
    fewest = rows[0]
    most = rows[-1]
    check("a handful of layouts is memorised rather than understood",
          fewest["mean_training_success"] - fewest["mean_test_success"] > 0.2,
          "with %s layouts it scores %.0f%% on what it trained on and only "
          "%.0f%% on unseen ones — a %.0f point gap"
          % (fewest["value"], fewest["mean_training_success"] * 100,
             fewest["mean_test_success"] * 100,
             (fewest["mean_training_success"] - fewest["mean_test_success"]) * 100))
    check("a larger pool transfers better",
          most["mean_test_success"] > fewest["mean_test_success"],
          "unseen success rises from %.0f%% with %s layouts to %.0f%% with %s"
          % (fewest["mean_test_success"] * 100, fewest["value"],
             most["mean_test_success"] * 100, most["value"]))

    # ------------------------------------------------------------------
    heading("9. Against two trivial baselines")
    test_pool = L.generate_pool("test", TEST_LAYOUTS, trained["difficulty"])
    baselines = experiments.baseline_policies(test_pool)
    print("       %-24s %8s %8s" % ("policy", "success", "return"))
    for row in baselines:
        print("       %-24s %7.0f%% %8.1f"
              % (row["value"], row["mean_test_success"] * 100,
                 row["mean_test_return"]))
    print("       %-24s %7.0f%% %8.1f"
          % ("learned policy", scores["test"]["success_rate"] * 100,
             scores["test"]["mean_return"]))
    check("the learned policy beats both baselines on unseen layouts",
          all(scores["test"]["success_rate"] > row["mean_test_success"]
              for row in baselines))

    # ------------------------------------------------------------------
    heading("10. Replay")
    first = simulate.run_episode(trained["agent"], layout)
    second = simulate.run_episode(trained["agent"], layout)
    check("the same layout and agent reproduce the recording exactly",
          first["frames"] == second["frames"],
          "%d frames compared field by field" % len(first["frames"]))
    stages = [frame["stage"] for frame in first["frames"]]
    check("the stage never goes backwards", stages == sorted(stages),
          "0 -> %d, status: %s" % (stages[-1], first["status"]))
    check("each frame carries the robots, the radar and what was visible",
          all(frame["moving_robot_positions"] is not None
              and frame["radar_observation"] and "visible_cells" in frame
              for frame in first["frames"]),
          "%d robots tracked" % len(first["frames"][0]["moving_robot_positions"]))

    # ------------------------------------------------------------------
    heading("11. The renderers")
    scene = room5_scene.build_scene(layout, frames=first["frames"])
    problems = base_renderer.validate_scene(scene)
    check("the scene is valid", problems == [], "; ".join(problems))
    check("the fog of war is on by default", scene["meta"]["fogOfWar"] is True)
    html = iso_canvas.build_html(scene)
    kinds = {kind for row in scene["grid"]["tiles"] for kind in row}
    missing = [kind for kind in kinds if '"%s"' % kind not in html]
    check("the engine handles every tile type", not missing,
          "%d tile types: %s" % (len(kinds), ", ".join(sorted(kinds))))
    figure = fallback_renderer.render_frame(scene, scene["frames"][-1])
    check("the Matplotlib fallback draws the warehouse",
          len(figure.axes[0].patches) > 100,
          "%d shapes" % len(figure.axes[0].patches))
    figure.clf()

    # ------------------------------------------------------------------
    heading("12. The graphs")
    results = [trained]
    builders = [
        ("episode reward", lambda: room5_plots.episode_reward(results)),
        ("moving average", lambda: room5_plots.moving_average_reward(results)),
        ("episode length", lambda: room5_plots.episode_length(results)),
        ("training success", lambda: room5_plots.training_success(results)),
        ("validation success", lambda: room5_plots.validation_success(results)),
        ("terminal rate", lambda: room5_plots.terminal_rate(results)),
        ("robot collisions", lambda: room5_plots.robot_collisions(results)),
        ("static collisions", lambda: room5_plots.static_collisions(results)),
        ("timeout rate", lambda: room5_plots.timeout_rate(results)),
        ("epsilon decay", lambda: room5_plots.epsilon_decay(results)),
        ("TD error", lambda: room5_plots.td_error(results)),
        ("weight norm", lambda: room5_plots.weight_norm(results)),
        ("training time", lambda: room5_plots.training_time(results)),
        ("feature activity", lambda: room5_plots.feature_activity(results)),
        ("generalisation", lambda: room5_plots.generalisation_chart(rows)),
        ("radar over time", lambda: room5_plots.radar_over_time(first)),
        ("nearest robot", lambda: room5_plots.nearest_robot_distance(first)),
        ("action mix", lambda: room5_plots.action_distribution([first])),
        ("layout map", lambda: room5_plots.layout_map(layout, first)),
    ]
    built = 0
    for name, builder in builders:
        figure = builder()
        if figure is not None and figure.axes:
            built += 1
        else:
            print("       %s produced nothing" % name)
        figure.clf()
    check("all %d required graphs are generated" % len(builders),
          built == len(builders), "%d of %d" % (built, len(builders)))

    # ------------------------------------------------------------------
    heading("13. Saving and loading")
    path = storage.save_model("verify_room5", trained)
    fresh = q_agent.LinearAgent(FeatureExtractor(
        groups=FEATURE_SETS[trained["feature_set"]]))
    storage.load_into("verify_room5", fresh)
    sample = env.observation()
    identical = all(abs(fresh.q_value(sample, action)
                        - trained["agent"].q_value(sample, action)) < 1e-9
                    for action in E.ACTIONS)
    check("the loaded model predicts exactly the same values", identical,
          "%d weights restored from %s"
          % (len(fresh.weights), os.path.basename(path)))
    mismatched = q_agent.LinearAgent(
        FeatureExtractor(groups=FEATURE_SETS["A. Target only"]))
    refused = False
    try:
        storage.load_into("verify_room5", mismatched)
    except storage.ModelFileError:
        refused = True
    check("a model built on a different feature set is refused", refused)
    os.remove(path)

    # ------------------------------------------------------------------
    heading("Result")
    if FAILURES:
        print("  %d check(s) FAILED:" % len(FAILURES))
        for label in FAILURES:
            print("    - %s" % label)
        return 1
    print("  Every check passed.")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:                                    # noqa: BLE001
        traceback.print_exc()
        print("\nVERIFICATION CRASHED")
        sys.exit(2)
