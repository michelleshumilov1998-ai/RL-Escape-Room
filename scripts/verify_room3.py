"""End-to-end verification of Room 3.

Run it with:

    .venv/bin/python scripts/verify_room3.py

It works through the checklist in the Room 3 brief and prints real numbers, so
nothing has to be taken on trust:

  1. import every module
  2. check the map and the patrol
  3. check the mechanics: generators in order, guard, hazards, sliding doors
  4. check the state space and that the model is hidden
  5. train with Q-Learning
  6. compare Q-Learning against SARSA
  7. evaluate the greedy policy
  8. check that replay reproduces an episode exactly
  9. build a scene and a still frame for the renderers
 10. generate every required graph
 11. run a parameter experiment
 12. save and load a model

The exit status is 0 only if every check passed.
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILURES = []

EPISODES = 5000


def heading(text):
    print("\n" + text)
    print("-" * len(text))


def check(label, condition, detail=""):
    mark = "PASS" if condition else "FAIL"
    print("  [%s] %s%s" % (mark, label, ("  — " + detail) if detail else ""))
    if not condition:
        FAILURES.append(label)
    return condition


def tail_mean(values, count=500):
    slice_ = values[-count:]
    return sum(slice_) / len(slice_) if slice_ else 0.0


def main():
    heading("1. Importing every module")
    import importlib
    modules = ["rooms.room3.map_data", "rooms.room3.environment",
               "rooms.room3.q_agent", "rooms.room3.simulate",
               "rooms.room3.experiments", "rooms.room3.storage",
               "rooms.room3.page", "renderers.room3_scene",
               "renderers.fallback_renderer", "plots.room3_plots", "app"]
    for name in modules:
        importlib.import_module(name)
    check("all %d Room 3 modules import" % len(modules), True)

    from core import actions
    from plots import room3_plots
    from renderers import base_renderer, fallback_renderer, iso_canvas, room3_scene
    from rooms.room3 import experiments, map_data, q_agent, simulate, storage
    from rooms.room3 import environment as E
    from rooms.room3.environment import Room3Env

    # ------------------------------------------------------------------
    heading("2. The chamber and the patrol")
    check("10x10 grid", map_data.GRID_ROWS == 10 and map_data.GRID_COLS == 10)
    flat = "".join("".join(row) for row in map_data.ROOM_MAP)
    counts = {tile: flat.count(tile) for tile in
              (map_data.START, map_data.EXIT, map_data.REACTOR_DOOR,
               map_data.GENERATOR_A, map_data.GENERATOR_B, map_data.GENERATOR_C)}
    check("one start, one exit, one reactor door, three generators",
          all(count == 1 for count in counts.values()),
          ", ".join("%s=%d" % pair for pair in counts.items()))
    check("hazards and sliding doors are present",
          len(map_data.hazard_cells()) >= 4
          and len(map_data.sliding_door_cells()) >= 2,
          "%d hazards, %d doors" % (len(map_data.hazard_cells()),
                                    len(map_data.sliding_door_cells())))
    check("every walkable cell is reachable",
          set(map_data.walkable_cells()) == map_data.reachable_cells(),
          "%d walkable cells" % len(map_data.walkable_cells()))
    check("the patrol is a closed loop inside the room",
          all(map_data.in_bounds(row, col) for row, col in map_data.PATROL)
          and abs(map_data.PATROL[0][0] - map_data.PATROL[-1][0])
          + abs(map_data.PATROL[0][1] - map_data.PATROL[-1][1]) == 1,
          "%d cells" % map_data.PATROL_LENGTH)
    check("the patrol divides evenly by the door cycle",
          map_data.PATROL_LENGTH % map_data.DOOR_PERIOD == 0,
          "%d cells / %d step cycle"
          % (map_data.PATROL_LENGTH, map_data.DOOR_PERIOD))
    open_phases = [index for index in range(map_data.DOOR_PERIOD)
                   if map_data.door_is_open(index)]
    check("the doors are open half the time",
          len(open_phases) * 2 == map_data.DOOR_PERIOD,
          "open at guard phases %s" % (open_phases,))
    print("       mission route: %d steps" % map_data.mission_route_length())
    print("       map fingerprint: %s" % map_data.map_hash())

    # ------------------------------------------------------------------
    heading("3. The mechanics")
    env = Room3Env()
    state = env.reset()
    check("the state is (row, col, stage, guard_index)", len(state) == 4,
          "%d states in total" % len(env.all_states()))
    check("Wait is one of the five actions",
          len(env.actions()) == 5 and actions.WAIT in env.actions())

    # Generators must be visited A, then B, then C.
    row, col = map_data.generator_cell(map_data.GENERATOR_B)
    env.state = (row, col - 1, E.STAGE_NONE, 0)
    _, wrong_reward, _, wrong_info = env.step(actions.RIGHT)
    check("generator B does nothing before A",
          wrong_info["generator"] is None and wrong_info["wrong_order"],
          "reward %+d, stage still %d" % (wrong_reward, env.state[2]))

    row, col = map_data.generator_cell(map_data.GENERATOR_A)
    env.state = (row, col - 1, E.STAGE_NONE, 0)
    _, a_reward, _, a_info = env.step(actions.RIGHT)
    check("generator A starts the sequence",
          a_info["generator"] == map_data.GENERATOR_A and env.state[2] == E.STAGE_A,
          "reward %+d" % a_reward)

    row, col = map_data.generator_cell(map_data.GENERATOR_C)
    env.state = (row, col + 1, E.STAGE_AB, 0)
    _, c_reward, _, c_info = env.step(actions.LEFT)
    check("the third generator unlocks the reactor door",
          c_info["unlocked"] and env.state[2] == E.STAGE_UNLOCKED,
          "reward %+d (final bonus included)" % c_reward)

    door_row, door_col = map_data.reactor_door_cell()
    env.state = (door_row - 1, door_col, E.STAGE_NONE, 0)
    _, locked_reward, locked_done, locked_info = env.step(actions.DOWN)
    check("the reactor door is shut before all three generators run",
          locked_info["door_blocked"] and not locked_done
          and env.state[:2] == (door_row - 1, door_col),
          "reward %+d" % locked_reward)

    env.state = (door_row - 1, door_col, E.STAGE_UNLOCKED, 0)
    _, open_reward, open_done, open_info = env.step(actions.DOWN)
    check("and opens once they do", not open_done and not open_info["door_blocked"],
          "reward %+d" % open_reward)

    exit_row, exit_col = map_data.exit_cell()
    env.state = (exit_row - 1, exit_col, E.STAGE_UNLOCKED, 0)
    _, exit_reward, exit_done, exit_info = env.step(actions.DOWN)
    check("reaching the exit ends the episode",
          exit_done and exit_info["reached_exit"], "reward %+d" % exit_reward)

    # The hazards sit in the service shafts, which the guard also walks, so the
    # phase has to be one where the robot is nowhere near.
    hazard_row, hazard_col = map_data.hazard_cells()[0]
    quiet_index = next(index for index in range(map_data.PATROL_LENGTH)
                       if map_data.guard_cell(map_data.next_guard_index(index))
                       != (hazard_row, hazard_col))
    env.state = (hazard_row - 1, hazard_col, E.STAGE_NONE, quiet_index)
    _, hazard_reward, hazard_done, hazard_info = env.step(actions.DOWN)
    check("a hazard hurts but does not end the episode",
          hazard_info["hazard"] and not hazard_done
          and hazard_reward <= env.hazard_penalty,
          "reward %+d" % hazard_reward)

    # Stand where the guard is about to arrive.
    guard_index = 0
    ahead = map_data.guard_cell(map_data.next_guard_index(guard_index))
    env.state = (ahead[0], ahead[1], E.STAGE_NONE, guard_index)
    _, guard_reward, guard_done, guard_info = env.step(actions.WAIT)
    check("the security robot ends the episode",
          guard_done and guard_info["caught"]
          and guard_reward <= env.guard_penalty,
          "reward %+d" % guard_reward)

    closed_index = next(index for index in range(map_data.DOOR_PERIOD)
                        if not map_data.door_is_open(index))
    sliding_row, sliding_col = map_data.sliding_door_cells()[0]
    env.state = (sliding_row - 1, sliding_col, E.STAGE_NONE, closed_index)
    before = env.state[:2]
    _, shut_reward, _, shut_info = env.step(actions.DOWN)
    check("a closed sliding door blocks the way",
          shut_info["blocked"] and env.state[:2] == before,
          "reward %+d at guard phase %d" % (shut_reward, closed_index))

    check("the door phase is derived from the guard, so the state stays Markov",
          env.doors_open((0, 0, 0, 0)) == map_data.door_is_open(0))
    check("no transition model is exposed to the agent",
          not hasattr(env, "transitions"))

    # ------------------------------------------------------------------
    heading("4. Training with Q-Learning")
    trained = q_agent.train(Room3Env(), algorithm=q_agent.Q_LEARNING,
                            episodes=EPISODES, seed=0)
    history = trained["history"]
    check("the agent learns to escape", tail_mean(history["success"]) >= 0.7,
          "success %.0f%%, reward %.1f, %d episodes in %.1fs"
          % (tail_mean(history["success"]) * 100, tail_mean(history["reward"]),
             EPISODES, trained["runtime_seconds"]))
    check("all three generators are started",
          tail_mean(history["generators_done"]) >= 0.7,
          "the full sequence in %.0f%% of late episodes, %.2f of 3 on average"
          % (tail_mean(history["generators_done"]) * 100,
             tail_mean(history["stage"])))
    # Exploration never switches off entirely (epsilon_min = 0.05), so some late
    # training episodes still walk into the robot. The greedy policy is checked
    # separately in step 6.
    check("being caught becomes the exception",
          tail_mean(history["guard_caught"]) <= 0.3,
          "%.0f%% of late episodes" % (tail_mean(history["guard_caught"]) * 100))
    check("a worked update was captured", trained["example_update"] is not None)
    if trained["example_update"]:
        example = trained["example_update"]
        print("       one real update: Q %.2f -> target %.2f (TD %+.2f)"
              % (example["q_before"], example["target"],
                 example["target"] - example["q_before"]))

    # ------------------------------------------------------------------
    heading("5. Q-Learning against SARSA")
    print("       %-11s %8s %8s %8s %8s %8s"
          % ("algorithm", "reward", "online", "greedy", "caught", "hazards"))
    rows = experiments.compare_algorithms(episodes=experiments.EPISODES_DEFAULT,
                                          repeats=2)
    for row in rows:
        print("       %-11s %8.1f %7.0f%% %7.0f%% %7.0f%% %8.2f"
              % (row["value"], row["mean_reward"], row["mean_success"] * 100,
                 row["mean_greedy_success"] * 100, row["mean_caught"] * 100,
                 row["mean_hazards"]))
    # Online success is depressed by the residual exploration; what matters is
    # that the policy each one leaves behind escapes.
    check("both algorithms end up with a policy that escapes",
          all(row["mean_greedy_success"] >= 0.5 for row in rows))
    print("       route taken: %s" % rows[0]["route"])

    # ------------------------------------------------------------------
    heading("6. Evaluating the greedy policy")
    env = Room3Env()
    batch = simulate.run_many(env, trained["policy"], episodes=30)
    check("the greedy policy escapes reliably", batch["success_rate"] >= 0.9,
          "success %.0f%%, return %.1f ± %.1f, steps %.1f, caught %.0f%%"
          % (batch["success_rate"] * 100, batch["mean_return"],
             batch["std_return"], batch["mean_steps"],
             batch["caught_rate"] * 100))
    run = simulate.run_episode(env, trained["policy"], seed=0)
    check("the sequence is completed in order", run["stage"] == E.STAGE_UNLOCKED,
          "final stage: %s" % env.stage_name(run["frames"][-1]["state"]))
    print("       route on seed 0: %s" % simulate.describe_route(run["frames"]))

    # ------------------------------------------------------------------
    heading("7. Replay")
    first = simulate.run_episode(env, trained["policy"], seed=7)
    second = simulate.run_episode(env, trained["policy"], seed=7)
    check("the same seed reproduces the recording exactly",
          first["frames"] == second["frames"],
          "%d frames compared field by field" % len(first["frames"]))
    joined = all((current["from_row"], current["from_col"])
                 == (previous["row"], previous["col"])
                 for previous, current in zip(first["frames"], first["frames"][1:]))
    check("frames join up into a continuous walk", joined)
    stages = [frame["stage"] for frame in first["frames"]]
    check("the stage never goes backwards", stages == sorted(stages),
          "0 -> %d" % stages[-1])

    # ------------------------------------------------------------------
    heading("8. The renderers")
    scene = room3_scene.build_scene(frames=run["frames"])
    problems = base_renderer.validate_scene(scene)
    check("the scene is valid", problems == [], "; ".join(problems))
    html = iso_canvas.build_html(scene)
    kinds = {kind for row in scene["grid"]["tiles"] for kind in row}
    missing = [kind for kind in kinds if '"%s"' % kind not in html]
    check("the engine handles every tile type", not missing,
          "%d tile types: %s" % (len(kinds), ", ".join(sorted(kinds))))
    figure = fallback_renderer.render_frame(scene, scene["frames"][-1])
    check("the Matplotlib fallback draws the chamber",
          len(figure.axes[0].patches) > 100,
          "%d shapes" % len(figure.axes[0].patches))
    figure.clf()

    # ------------------------------------------------------------------
    heading("9. The graphs")
    results = [trained]
    builders = [
        ("episode reward", lambda: room3_plots.episode_reward(results)),
        ("moving average", lambda: room3_plots.moving_average_reward(results)),
        ("episode length", lambda: room3_plots.episode_length(results)),
        ("success rate", lambda: room3_plots.success_rate(results)),
        ("generator completion", lambda: room3_plots.generator_completion_rate(results)),
        ("average stage", lambda: room3_plots.average_stage(results)),
        ("guard detections", lambda: room3_plots.guard_detections(results)),
        ("hazard hits", lambda: room3_plots.hazard_hits(results)),
        ("Q magnitude", lambda: room3_plots.q_value_magnitude(results)),
        ("door unlocks", lambda: room3_plots.door_unlock_frequency(results)),
        ("epsilon decay", lambda: room3_plots.epsilon_decay(results)),
        ("training time", lambda: room3_plots.training_time(results)),
        ("chamber map", lambda: room3_plots.chamber_map(
            [(frame["row"], frame["col"]) for frame in run["frames"]])),
        ("policy arrows", lambda: room3_plots.policy_arrows(trained["policy"])),
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
    heading("10. A parameter experiment")
    sweep = experiments.parameter_experiment(
        "alpha", values=[0.05, 0.15, 0.40],
        episodes=experiments.EPISODES_DEFAULT, repeats=1)
    print("       %-8s %8s %8s %8s" % ("alpha", "reward", "success", "greedy"))
    for row in sweep:
        print("       %-8s %8.1f %7.0f%% %7.0f%%"
              % (row["value"], row["mean_reward"], row["mean_success"] * 100,
                 row["mean_greedy_success"] * 100))
    best = experiments.best_performing_row(sweep)
    check("the sweep ranks the tested values", best is not None,
          "best-performing of those tested: alpha = %s" % best["value"])

    # ------------------------------------------------------------------
    heading("11. Saving and loading")
    path = storage.save_model("verify_room3", env, trained)
    loaded = storage.load_model("verify_room3")
    rebuilt = q_agent.greedy_policy(loaded["q"], env)
    check("the loaded model gives back the same policy",
          rebuilt == trained["policy"],
          "%d states restored from %s"
          % (len(loaded["q"]), os.path.basename(path)))
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
