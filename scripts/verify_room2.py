"""End-to-end verification of Room 2.

Run it with:

    .venv/bin/python scripts/verify_room2.py

It works through the checklist in the Room 2 brief and prints real numbers, so
nothing has to be taken on trust:

  1. import every module
  2. check the map
  3. check the mechanics: bridges, pits, keycard, door
  4. train with SARSA
  5. train with Q-Learning
  6. compare the two, in both state representations
  7. run a batch of evaluation episodes
  8. check that replay reproduces an episode exactly
  9. build a scene and a still frame for the renderers
 10. generate every required graph
 11. save and load a model

The exit status is 0 only if every check passed.
"""

import os
import sys
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILURES = []

EPISODES = 2500


def heading(text):
    print("\n" + text)
    print("-" * len(text))


def check(label, condition, detail=""):
    mark = "PASS" if condition else "FAIL"
    print("  [%s] %s%s" % (mark, label, ("  — " + detail) if detail else ""))
    if not condition:
        FAILURES.append(label)
    return condition


def tail_mean(values, count=400):
    slice_ = values[-count:]
    return sum(slice_) / len(slice_) if slice_ else 0.0


def main():
    heading("1. Importing every module")
    import importlib
    modules = ["rooms.room2.map_data", "rooms.room2.environment",
               "rooms.room2.sarsa_agent", "rooms.room2.simulate",
               "rooms.room2.experiments", "rooms.room2.storage",
               "rooms.room2.page", "renderers.room2_scene",
               "renderers.scene_common", "renderers.fallback_renderer",
               "plots.room2_plots", "app"]
    for name in modules:
        importlib.import_module(name)
    check("all %d Room 2 modules import" % len(modules), True)

    from core import actions
    from plots import room2_plots
    from renderers import base_renderer, fallback_renderer, iso_canvas, room2_scene
    from rooms.room2 import map_data, sarsa_agent, simulate, storage
    from rooms.room2.environment import Room2Env

    # ------------------------------------------------------------------
    heading("2. The map")
    check("10x10 grid", map_data.GRID_ROWS == 10 and map_data.GRID_COLS == 10)
    walkable = set(map_data.walkable_cells())
    check("every walkable cell is reachable",
          walkable == map_data.reachable_cells(),
          "%d walkable cells" % len(walkable))
    routes = map_data.route_lengths()
    check("two routes, the risky one shorter",
          routes["span"] < routes["walkway"],
          "bridge %d steps, walkway %d steps" % (routes["span"], routes["walkway"]))
    check("the short route pays more when walked perfectly",
          map_data.perfect_return("span") > map_data.perfect_return("walkway"),
          "bridge %+d, walkway %+d" % (map_data.perfect_return("span"),
                                       map_data.perfect_return("walkway")))
    exposed = map_data.exposed_cells()
    walkway_cells = {(1, col) for col in range(map_data.GRID_COLS)}
    check("only the short route has anything to fall off",
          bool(exposed) and not set(exposed) & walkway_cells,
          "%d exposed cells, all on the bridge" % len(exposed))
    check("at least five pits can be fallen into", len(map_data.enterable_pits()) >= 5,
          "%d of %d shaft cells are enterable"
          % (len(map_data.enterable_pits()), len(map_data.pit_cells())))
    print("       map fingerprint: %s" % map_data.map_hash())

    # ------------------------------------------------------------------
    heading("3. The mechanics")
    env = Room2Env()
    env.reset()
    for _ in range(3):
        env.step(actions.UP)
    env.step(actions.RIGHT)
    _, reward, done, info = env.step(actions.RIGHT)
    check("crossing a collapsing bridge collapses it",
          info["bridge_collapsed"] and env.collapsed,
          "reward %+d, collapsed %s" % (reward, env.collapsed_cells()))
    env.step(actions.RIGHT)
    _, back_reward, back_done, back_info = env.step(actions.LEFT)
    check("a collapsed bridge is now a pit",
          back_done and back_info["pit_fall"], "reward %+d" % back_reward)

    env.reset()
    env.step(actions.UP)
    env.step(actions.UP)
    env.step(actions.UP)
    env.step(actions.RIGHT)
    env.step(actions.RIGHT)
    _, pit_reward, pit_done, pit_info = env.step(actions.DOWN)
    check("a pit ends the episode", pit_done and pit_info["pit_fall"],
          "reward %+d" % pit_reward)

    env.reset()
    env.state = (8, 7, False)
    locked_state, locked_reward, _, locked_info = env.step(actions.RIGHT)
    check("the door stays locked without the keycard",
          locked_info["door_locked"] and (locked_state[0], locked_state[1]) == (8, 7),
          "reward %+d" % locked_reward)
    env.state = (8, 7, True)
    open_state, open_reward, open_done, open_info = env.step(actions.RIGHT)
    check("the door opens with the keycard",
          open_done and open_info["reached_exit"], "reward %+d" % open_reward)

    check("the state is (row, col, has_keycard)", len(env.reset()) == 3,
          "%d states" % len(env.all_states()))
    check("no transition model is exposed to the agent",
          not hasattr(env, "transitions"))

    # ------------------------------------------------------------------
    heading("4. Training with SARSA")
    sarsa = sarsa_agent.train(Room2Env(), algorithm=sarsa_agent.SARSA,
                              episodes=EPISODES, seed=0)
    check("SARSA learns to escape", tail_mean(sarsa["history"]["success"]) >= 0.8,
          "success %.0f%%, reward %.1f, %d episodes in %.1fs"
          % (tail_mean(sarsa["history"]["success"]) * 100,
             tail_mean(sarsa["history"]["reward"]), EPISODES,
             sarsa["runtime_seconds"]))
    check("the keycard is always collected",
          tail_mean(sarsa["history"]["keycard"]) >= 0.9,
          "%.0f%%" % (tail_mean(sarsa["history"]["keycard"]) * 100))
    check("a worked update was captured", sarsa["example_update"] is not None)

    # ------------------------------------------------------------------
    heading("5. Training with Q-Learning")
    q_learning = sarsa_agent.train(Room2Env(), algorithm=sarsa_agent.Q_LEARNING,
                                   episodes=EPISODES, seed=0)
    check("Q-Learning learns to escape",
          tail_mean(q_learning["history"]["success"]) >= 0.8,
          "success %.0f%%, reward %.1f, %.1fs"
          % (tail_mean(q_learning["history"]["success"]) * 100,
             tail_mean(q_learning["history"]["reward"]),
             q_learning["runtime_seconds"]))

    # ------------------------------------------------------------------
    heading("6. SARSA against Q-Learning, in both state representations")
    print("       %-34s %-11s %8s %8s %8s"
          % ("state", "algorithm", "bridge", "reward", "pitfall"))
    outcome = {}
    for include_bridge_state in (False, True):
        label = ("(row, col, keycard)" if not include_bridge_state
                 else "(row, col, keycard, collapsed)")
        for algorithm in (sarsa_agent.SARSA, sarsa_agent.Q_LEARNING):
            result = sarsa_agent.train(
                Room2Env(include_bridge_state=include_bridge_state),
                algorithm=algorithm, episodes=EPISODES, seed=0)
            bridge = tail_mean(result["history"]["used_span"])
            reward = tail_mean(result["history"]["reward"])
            pit = tail_mean(result["history"]["pit_fall"])
            outcome[(include_bridge_state, algorithm)] = (bridge, reward, pit)
            print("       %-34s %-11s %7.0f%% %8.1f %7.0f%%"
                  % (label, algorithm, bridge * 100, reward, pit * 100))

    assignment_sarsa = outcome[(False, sarsa_agent.SARSA)]
    assignment_q = outcome[(False, sarsa_agent.Q_LEARNING)]
    markov_sarsa = outcome[(True, sarsa_agent.SARSA)]
    markov_q = outcome[(True, sarsa_agent.Q_LEARNING)]

    check("with the assignment's state both methods avoid the bridge",
          assignment_sarsa[0] < 0.2 and assignment_q[0] < 0.2,
          "the collapse is invisible in that state, so it prices the bridge down "
          "for both")
    check("with the Markov state Q-Learning takes the bridge", markov_q[0] > 0.5,
          "%.0f%% of episodes" % (markov_q[0] * 100))
    check("with the Markov state SARSA avoids it", markov_sarsa[0] < 0.2,
          "%.0f%% of episodes" % (markov_sarsa[0] * 100))
    check("SARSA collects more online reward than Q-Learning there",
          markov_sarsa[1] > markov_q[1],
          "%.1f against %.1f" % (markov_sarsa[1], markov_q[1]))
    check("and falls in less often", markov_sarsa[2] < markov_q[2],
          "%.0f%% against %.0f%%" % (markov_sarsa[2] * 100, markov_q[2] * 100))

    # ------------------------------------------------------------------
    heading("7. Evaluating the learned policy")
    env = Room2Env()
    policy_result = sarsa_agent.train(env, episodes=EPISODES, seed=1)
    batch = simulate.run_many(env, policy_result["policy"], episodes=30)
    check("the greedy policy escapes reliably", batch["success_rate"] >= 0.9,
          "success %.0f%%, return %.1f ± %.1f, steps %.1f, pit falls %.0f%%"
          % (batch["success_rate"] * 100, batch["mean_return"],
             batch["std_return"], batch["mean_steps"],
             batch["pit_fall_rate"] * 100))
    example = simulate.run_episode(env, policy_result["policy"], seed=0)
    print("       route on seed 0: %s" % simulate.describe_route(example["frames"]))

    # ------------------------------------------------------------------
    heading("8. Replay")
    first = simulate.run_episode(env, policy_result["policy"], seed=99)
    second = simulate.run_episode(env, policy_result["policy"], seed=99)
    check("the same seed reproduces the recording exactly",
          first["frames"] == second["frames"],
          "%d frames compared field by field" % len(first["frames"]))
    joined = all((current["from_row"], current["from_col"])
                 == (previous["row"], previous["col"])
                 for previous, current in zip(first["frames"], first["frames"][1:]))
    check("frames join up into a continuous walk", joined)

    bridge_env = Room2Env(include_bridge_state=True)
    bridge_result = sarsa_agent.train(bridge_env, algorithm=sarsa_agent.Q_LEARNING,
                                      episodes=EPISODES, seed=0)
    bridge_run = simulate.run_episode(bridge_env, bridge_result["policy"], seed=1)
    grows = all(len(later["collapsed"]) >= len(earlier["collapsed"])
                for earlier, later in zip(bridge_run["frames"],
                                          bridge_run["frames"][1:]))
    check("a replay records the bridges collapsing", grows,
          "%d bridges recorded across %d frames"
          % (len(bridge_run["frames"][-1]["collapsed"]), len(bridge_run["frames"])))

    # ------------------------------------------------------------------
    heading("9. The renderers")
    scene = room2_scene.build_scene(frames=bridge_run["frames"])
    problems = base_renderer.validate_scene(scene)
    check("the scene is valid", problems == [], "; ".join(problems))
    html = iso_canvas.build_html(scene)
    kinds = {kind for row in scene["grid"]["tiles"] for kind in row}
    missing = [kind for kind in kinds if '"%s"' % kind not in html]
    check("the engine handles every tile type", not missing,
          "%d tile types: %s" % (len(kinds), ", ".join(sorted(kinds))))
    figure = fallback_renderer.render_frame(scene, scene["frames"][-1])
    check("the Matplotlib fallback draws the sector",
          len(figure.axes[0].patches) > 100,
          "%d shapes" % len(figure.axes[0].patches))
    figure.clf()

    # ------------------------------------------------------------------
    heading("10. The graphs")
    results = [sarsa, q_learning]
    builders = [
        ("episode reward", room2_plots.episode_reward),
        ("moving average", room2_plots.moving_average_reward),
        ("episode length", room2_plots.episode_length),
        ("success rate", room2_plots.success_rate),
        ("epsilon decay", room2_plots.epsilon_decay),
        ("Q convergence", room2_plots.q_value_convergence),
        ("pit falls", room2_plots.pit_falls),
        ("bridge usage", room2_plots.bridge_usage),
        ("keycard rate", room2_plots.keycard_rate),
        ("training time", room2_plots.training_time),
    ]
    built = 0
    for name, builder in builders:
        figure = builder(results)
        if figure is not None and figure.axes:
            built += 1
        figure.clf()
    check("all ten required graphs are generated", built == len(builders),
          "%d of %d" % (built, len(builders)))

    # ------------------------------------------------------------------
    heading("11. Saving and loading")
    path = storage.save_model("verify_room2", env, policy_result)
    loaded = storage.load_model("verify_room2")
    rebuilt = sarsa_agent.greedy_policy(loaded["q"], env)
    check("the loaded model gives back the same policy",
          rebuilt == policy_result["policy"],
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
