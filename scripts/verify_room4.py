"""End-to-end verification of Room 4.

Run it with:

    .venv/bin/python scripts/verify_room4.py

It works through the checklist in the Room 4 brief and prints real numbers:

  1. import every module
  2. check the chamber geometry, the fans and the wind field
  3. check the dynamics: thrust, drag, walls, obstacles, landing, crashing
  4. check the tile coder by hand
  5. train with Semi-Gradient SARSA
  6. compare it with Semi-Gradient Q-Learning
  7. evaluate the greedy policy
  8. test generalisation to unseen starting positions
  9. check that replay reproduces an episode exactly
 10. build a scene and a still frame for the renderers
 11. generate every required graph
 12. run the tile-resolution experiment
 13. save and load a model

The exit status is 0 only if every check passed.
"""

import math
import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILURES = []

EPISODES = 1200


def heading(text):
    print("\n" + text)
    print("-" * len(text))


def check(label, condition, detail=""):
    mark = "PASS" if condition else "FAIL"
    print("  [%s] %s%s" % (mark, label, ("  — " + detail) if detail else ""))
    if not condition:
        FAILURES.append(label)
    return condition


def tail_mean(values, fraction=0.25):
    count = max(1, int(len(values) * fraction))
    slice_ = values[-count:]
    return sum(slice_) / len(slice_) if slice_ else 0.0


def main():
    heading("1. Importing every module")
    import importlib
    modules = ["rooms.room4.chamber", "rooms.room4.environment",
               "rooms.room4.tile_coder", "rooms.room4.sarsa_agent",
               "rooms.room4.simulate", "rooms.room4.experiments",
               "rooms.room4.storage", "rooms.room4.page",
               "renderers.room4_scene", "renderers.fallback_renderer",
               "plots.room4_plots", "app"]
    for name in modules:
        importlib.import_module(name)
    check("all %d Room 4 modules import" % len(modules), True)

    from plots import room4_plots
    from renderers import base_renderer, fallback_renderer, iso_canvas, room4_scene
    from rooms.room4 import chamber, experiments, sarsa_agent, simulate, storage
    from rooms.room4 import environment as E
    from rooms.room4.environment import Room4Env
    from rooms.room4.tile_coder import TileCoder

    # ------------------------------------------------------------------
    heading("2. The chamber")
    check("the hall is %.0f x %.0f metres" % (chamber.WIDTH, chamber.HEIGHT),
          chamber.WIDTH > 0 and chamber.HEIGHT > 0)
    check("the start is clear of every obstacle", chamber.start_is_clear(),
          "start %s, pad %s (radius %.1f m)"
          % (chamber.START_POSITION, chamber.GOAL_POSITION, chamber.GOAL_RADIUS))
    check("there are fans and obstacles",
          len(chamber.FANS) >= 2 and len(chamber.OBSTACLES) >= 2,
          "%d fans, %d obstacles" % (len(chamber.FANS), len(chamber.OBSTACLES)))
    calm_x, calm_y = chamber.wind_at(chamber.START_POSITION[0],
                                     chamber.START_POSITION[1])
    strongest = max((math.hypot(*chamber.wind_at(fan.x, fan.y)), fan.name)
                    for fan in chamber.FANS)
    check("the wind depends on where the drone is",
          math.hypot(calm_x, calm_y) < strongest[0],
          "%.2f m/s² at the start, %.2f m/s² at %s"
          % (math.hypot(calm_x, calm_y), strongest[0], strongest[1]))
    check("the wind is the same every time the drone is in the same place",
          chamber.wind_at(5.0, 5.0) == chamber.wind_at(5.0, 5.0),
          "so the state stays Markov")
    check("the goal test agrees with the distance",
          chamber.inside_goal(*chamber.GOAL_POSITION)
          and not chamber.inside_goal(*chamber.START_POSITION),
          "start is %.1f m from the pad"
          % chamber.distance_to_goal(*chamber.START_POSITION))

    # ------------------------------------------------------------------
    heading("3. The dynamics")
    env = Room4Env()
    state = env.reset()
    check("the state is (x, y, vx, vy)", len(state) == 4,
          "start state %s" % (tuple(round(value, 2) for value in state),))
    check("there are five actions including holding the thrusters off",
          len(env.actions()) == 5 and E.NO_THRUST in env.actions())

    env.reset()
    _, _, _, info = env.step(E.THRUST_RIGHT)
    check("thrust changes the velocity, not the position directly",
          env.state[2] > 0 and abs(env.state[3]) < 0.5,
          "vx %.3f m/s after one push" % env.state[2])

    calm = Room4Env(wind_multiplier=0.0)
    calm.reset()
    calm.state = (5.0, 5.0, 3.0, 0.0)
    speed_before = 3.0
    calm.step(E.NO_THRUST)
    check("drag slows a coasting drone", calm.state[2] < speed_before,
          "%.3f -> %.3f m/s with no wind and no thrust"
          % (speed_before, calm.state[2]))

    walled = Room4Env(wind_multiplier=0.0)
    walled.reset()
    walled.state = (0.3, 5.0, -4.0, 0.0)
    _, wall_reward, wall_done, wall_info = walled.step(E.NO_THRUST)
    check("a wall stops the drone rather than letting it leave the hall",
          wall_info["boundary"] and chamber.inside_bounds(walled.state[0],
                                                          walled.state[1]),
          "reward %+.1f, x = %.2f, vx = %.2f"
          % (wall_reward, walled.state[0], walled.state[2]))

    # Aim straight at the middle of a round obstacle from its left.
    column = next(obstacle for obstacle in chamber.OBSTACLES
                  if obstacle.kind == "circle")
    blocked = Room4Env(wind_multiplier=0.0)
    blocked.reset()
    blocked.state = (column.x - column.radius - 0.6, column.y, 3.0, 0.0)
    _, obstacle_reward, _, obstacle_info = blocked.step(E.NO_THRUST)
    still_outside = chamber.blocking_obstacle(blocked.state[0],
                                              blocked.state[1]) is None
    check("an obstacle blocks the drone",
          obstacle_info["collision"] and still_outside,
          "the %s pushed it back out to (%.2f, %.2f), reward %+.1f"
          % (column.name, blocked.state[0], blocked.state[1], obstacle_reward))

    gentle = Room4Env(wind_multiplier=0.0)
    gentle.reset()
    goal_x, goal_y = chamber.GOAL_POSITION
    gentle.state = (goal_x - 0.2, goal_y, 0.1, 0.0)
    _, land_reward, land_done, land_info = gentle.step(E.NO_THRUST)
    check("arriving slowly is a safe landing",
          land_done and land_info["safe_landing"] and not land_info["crashed"],
          "reward %+.1f at %.2f m/s" % (land_reward, land_info["speed"]))

    fast = Room4Env(wind_multiplier=0.0, v_max=10.0)
    fast.reset()
    entry_speed = (fast.crash_speed + 1.0) / fast.drag
    fast.state = (goal_x - 0.3, goal_y, entry_speed, 0.0)
    _, crash_reward, crash_done, crash_info = fast.step(E.NO_THRUST)
    check("arriving too fast is a crash, not a landing",
          crash_done and crash_info["crashed"] and not crash_info["safe_landing"],
          "reward %+.1f at %.2f m/s (crash above %.1f)"
          % (crash_reward, crash_info["speed"], fast.crash_speed))

    middling = Room4Env(wind_multiplier=0.0, v_max=10.0)
    middling.reset()
    middle_speed = (middling.safe_landing_speed + middling.crash_speed) / 2
    middling.state = (goal_x - 0.3, goal_y, middle_speed / middling.drag, 0.0)
    _, hard_reward, hard_done, hard_info = middling.step(E.NO_THRUST)
    check("a hard landing bounces off instead of ending the flight",
          hard_info["hard_landing"] and not hard_done,
          "safe below %.1f m/s, crash above %.1f, this one arrived at %.2f"
          % (middling.safe_landing_speed, middling.crash_speed,
             hard_info["speed"]))

    tunnel = Room4Env(wind_multiplier=0.0, v_max=20.0)
    tunnel.reset()
    tunnel.state = (goal_x - 3.0, goal_y, 18.0, 0.0)
    _, _, tunnel_done, tunnel_info = tunnel.step(E.NO_THRUST)
    check("a very fast drone cannot fly straight through the pad",
          tunnel_info["crashed"] or tunnel_info["hard_landing"]
          or tunnel_info["safe_landing"],
          "the pad is tested inside the substep walk")

    check("no transition model is exposed to the agent",
          not hasattr(env, "transitions"))

    # ------------------------------------------------------------------
    heading("4. The tile coder")
    coder = TileCoder(env.state_bounds(), num_tilings=8, tiles_per_dimension=8,
                      num_actions=len(env.actions()))
    corners = [(low, low, low, low) for low, _ in env.state_bounds()]
    samples = [env.reset(), chamber.GOAL_POSITION + (0.0, 0.0),
               (0.0, 0.0, -5.0, -5.0), (chamber.WIDTH, chamber.HEIGHT, 5.0, 5.0)]
    in_range = True
    for state in samples:
        for action in range(len(env.actions())):
            indices = coder.active_tiles(state, action)
            if len(indices) != 8 or any(not 0 <= index < coder.total_features
                                        for index in indices):
                in_range = False
    check("every state activates exactly one tile per tiling, all in range",
          in_range, "%d tilings, %d features in total"
          % (coder.num_tilings, coder.total_features))

    first = set(coder.active_tiles(samples[0], 0))
    second = set(coder.active_tiles(samples[0], 1))
    check("each action has its own block of features", not first & second,
          "no shared indices between action 0 and action 1")

    nearby = (samples[0][0] + 0.01, samples[0][1], samples[0][2], samples[0][3])
    distant = (chamber.WIDTH - 0.5, chamber.HEIGHT - 0.5, 4.0, 4.0)
    shared_near = len(first & set(coder.active_tiles(nearby, 0)))
    shared_far = len(first & set(coder.active_tiles(distant, 0)))
    check("nearby states share tiles and distant ones do not",
          shared_near >= 6 and shared_far == 0,
          "%d/8 shared nearby, %d/8 shared far away" % (shared_near, shared_far))

    # ------------------------------------------------------------------
    heading("5. Training with Semi-Gradient SARSA")
    sarsa = sarsa_agent.train(algorithm=sarsa_agent.SARSA, episodes=EPISODES,
                              seed=0)
    history = sarsa["history"]
    check("the drone learns to land", tail_mean(history["success"]) >= 0.7,
          "success %.0f%%, reward %.1f, %d episodes in %.1fs"
          % (tail_mean(history["success"]) * 100, tail_mean(history["reward"]),
             EPISODES, sarsa["runtime_seconds"]))
    check("the final distance to the pad shrinks",
          tail_mean(history["final_distance"]) < 2.0,
          "%.2f m on average" % tail_mean(history["final_distance"]))
    check("no weight became NaN or infinite", sarsa["agent"].is_finite(),
          "weight norm %.1f, %d features touched"
          % (sarsa["weight_norm"], sarsa["active_features"]))
    check("a worked update was captured", sarsa["example_update"] is not None)
    if sarsa["example_update"]:
        example = sarsa["example_update"]
        print("       one real update: Q %.2f -> target %.2f (TD %+.2f) over "
              "%d active tiles"
              % (example["q_before"], example["target"], example["td_error"],
                 len(example["active_tiles"])))

    # ------------------------------------------------------------------
    heading("6. SARSA against Q-Learning")
    print("       %-28s %8s %8s %8s %8s"
          % ("algorithm", "reward", "online", "greedy", "crash"))
    rows = experiments.compare_algorithms(episodes=EPISODES, repeats=1)
    for row in rows:
        print("       %-28s %8.1f %7.0f%% %7.0f%% %7.0f%%"
              % (row["value"], row["mean_reward"], row["mean_success"] * 100,
                 row["mean_eval_success"] * 100, row["mean_crash"] * 100))
    check("both methods learn to land",
          all(row["mean_eval_success"] >= 0.5 for row in rows))

    # ------------------------------------------------------------------
    heading("7. Evaluating the greedy policy")
    batch = simulate.run_many(sarsa["agent"], episodes=20)
    check("the greedy policy lands reliably", batch["success_rate"] >= 0.9,
          "success %.0f%%, return %.1f ± %.1f, steps %.1f, landing speed %.2f m/s"
          % (batch["success_rate"] * 100, batch["mean_return"],
             batch["std_return"], batch["mean_steps"],
             batch["mean_landing_speed"]))
    run = simulate.run_episode(sarsa["agent"], seed=0)
    check("the flight is recorded frame by frame", len(run["frames"]) > 5,
          "%d frames, status: %s" % (len(run["frames"]), run["status"]))

    # ------------------------------------------------------------------
    heading("8. Generalisation to unseen starting positions")
    generalisation = experiments.generalisation_test(episodes=EPISODES, repeats=1)
    print("       %-20s %5s %8s %8s %8s"
          % ("start group", "seen", "success", "return", "speed"))
    for row in generalisation:
        print("       %-20s %5s %7.0f%% %8.1f %8.2f"
              % (row["value"], "yes" if row["seen_during_training"] else "no",
                 row["mean_eval_success"] * 100, row["mean_eval_return"],
                 row["mean_eval_landing_speed"]))
    training_row = generalisation[0]
    test_row = generalisation[-1]
    check("it lands from the position it trained on",
          training_row["mean_eval_success"] >= 0.9,
          "%.0f%%" % (training_row["mean_eval_success"] * 100))
    check("and from starting positions it has never seen",
          test_row["mean_eval_success"] >= 0.5,
          "%.0f%% on %d unseen starts the agent never left"
          % (test_row["mean_eval_success"] * 100, len(chamber.TEST_STARTS)))

    # ------------------------------------------------------------------
    heading("9. Replay")
    first_run = simulate.run_episode(sarsa["agent"], seed=5)
    second_run = simulate.run_episode(sarsa["agent"], seed=5)
    check("the same seed reproduces the recording exactly",
          first_run["frames"] == second_run["frames"],
          "%d frames compared field by field" % len(first_run["frames"]))
    joined = all(abs(current["from_x"] - previous["x"]) < 1e-9
                 for previous, current in zip(first_run["frames"],
                                              first_run["frames"][1:]))
    check("frames join up into a continuous flight", joined)

    # ------------------------------------------------------------------
    heading("10. The renderers")
    scene = room4_scene.build_scene(frames=run["frames"])
    problems = base_renderer.validate_scene(scene)
    check("the scene is valid", problems == [], "; ".join(problems))
    check("the scene is in continuous mode",
          scene["meta"]["sceneKind"] == "continuous",
          "%d wind samples, %d fans, %d obstacles"
          % (len(scene["continuous"]["windField"]),
             len(scene["continuous"]["fans"]),
             len(scene["continuous"]["obstacles"])))
    html = iso_canvas.build_html(scene)
    check("the engine has the continuous-flight routines",
          all(name in html for name in ("function drawDrone(",
                                        "function drawWindField(",
                                        "function drawLandingPad(",
                                        "function drawContinuousLayer(")))
    figure = fallback_renderer.render_frame(scene, scene["frames"][-1])
    check("the Matplotlib fallback draws the wind tunnel",
          len(figure.axes[0].patches) > 50,
          "%d shapes" % len(figure.axes[0].patches))
    figure.clf()

    # ------------------------------------------------------------------
    heading("11. The graphs")
    results = [sarsa]
    builders = [
        ("episode reward", lambda: room4_plots.episode_reward(results)),
        ("moving average", lambda: room4_plots.moving_average_reward(results)),
        ("episode length", lambda: room4_plots.episode_length(results)),
        ("success rate", lambda: room4_plots.success_rate(results)),
        ("final distance", lambda: room4_plots.final_distance(results)),
        ("landing speed", lambda: room4_plots.landing_speed(results)),
        ("collisions", lambda: room4_plots.collision_count(results)),
        ("epsilon decay", lambda: room4_plots.epsilon_decay(results)),
        ("weight norm", lambda: room4_plots.weight_norm(results)),
        ("TD error", lambda: room4_plots.td_error(results)),
        ("active tiles", lambda: room4_plots.active_tiles(results)),
        ("training time", lambda: room4_plots.training_time(results)),
        ("trajectories", lambda: room4_plots.trajectories([run], show_wind=True)),
        ("wind field", room4_plots.wind_field),
        ("velocity over time", lambda: room4_plots.velocity_over_time(run)),
        ("wind and thrust", lambda: room4_plots.wind_and_thrust(run)),
        ("landing analysis", lambda: room4_plots.landing_analysis(run)),
        ("value slice", lambda: room4_plots.value_slice(sarsa["agent"],
                                                        samples=14)),
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
    heading("12. The tile-resolution experiment")
    resolution = experiments.tile_resolution_experiment(episodes=EPISODES,
                                                        repeats=1)
    print("       %-12s %8s %8s %8s %10s"
          % ("tiling", "reward", "online", "greedy", "weights"))
    for row in resolution:
        print("       %-12s %8.1f %7.0f%% %7.0f%% %10d"
              % (row["value"], row["mean_reward"], row["mean_success"] * 100,
                 row["mean_eval_success"] * 100, row["weights"]))
    best = experiments.best_performing_row(resolution)
    check("coarser and finer tilings are compared", len(resolution) >= 3,
          "best-performing of those tested: %s" % best["value"])

    # ------------------------------------------------------------------
    heading("13. Saving and loading")
    path = storage.save_model("verify_room4", sarsa)
    fresh = sarsa_agent.SemiGradientAgent(Room4Env())
    storage.load_into("verify_room4", fresh)
    state = env.reset()
    identical = all(abs(fresh.q_value(state, action)
                        - sarsa["agent"].q_value(state, action)) < 1e-9
                    for action in env.actions())
    check("the loaded model predicts exactly the same values", identical,
          "%d weights restored from %s"
          % (len(fresh.weights), os.path.basename(path)))
    mismatched = sarsa_agent.SemiGradientAgent(Room4Env(), tiles_per_dimension=4)
    refused = False
    try:
        storage.load_into("verify_room4", mismatched)
    except storage.ModelFileError:
        refused = True
    check("a model built on a different tiling is refused", refused)
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
