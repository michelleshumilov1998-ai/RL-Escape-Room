"""End-to-end verification of Room 1.

Run it with:

    .venv/bin/python scripts/verify_room1.py

It works through the checklist in the project brief and prints real numbers, so
nothing has to be taken on trust:

  1. import every module
  2. check the map
  3. check the transition model
  4. solve with Value Iteration
  5. solve with Policy Iteration
  6. compare the two policies
  7. run 30 simulations of each policy
  8. check that replay reproduces an episode exactly
  9. run the gamma experiment
 10. run the slipping experiment
 11. build a scene and a still frame for the renderers
 12. save and load a result

The exit status is 0 only if every check passed.
"""

import os
import sys
import traceback

# Allow running the file directly from the project root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

FAILURES = []


def heading(text):
    print("\n" + text)
    print("-" * len(text))


def check(label, condition, detail=""):
    """Report one check and remember whether it passed."""
    mark = "PASS" if condition else "FAIL"
    print("  [%s] %s%s" % (mark, label, ("  — " + detail) if detail else ""))
    if not condition:
        FAILURES.append(label)
    return condition


def main():
    heading("1. Importing every module")
    import importlib
    modules = ["core.actions", "core.tiles", "core.story", "ui.theme", "ui.cards",
               "ui.buttons", "ui.nav", "ui.inventory", "ui.transitions",
               "ui.game_state", "ui.start_screen", "plots.style",
               "plots.room1_plots", "renderers.base_renderer",
               "renderers.iso_scene", "renderers.iso_canvas",
               "renderers.fallback_renderer", "rooms.room1.map_data",
               "rooms.room1.environment", "rooms.room1.dp_solver",
               "rooms.room1.simulate", "rooms.room1.experiments",
               "rooms.room1.storage", "rooms.room1.page", "rooms.locked_page",
               "app"]
    for name in modules:
        importlib.import_module(name)
    check("all %d modules import" % len(modules), True)

    from renderers import base_renderer, fallback_renderer, iso_canvas, iso_scene
    from rooms.room1 import dp_solver, experiments, map_data, simulate, storage
    from rooms.room1.environment import Room1Env

    # ------------------------------------------------------------------
    heading("2. The map")
    check("10x10 grid", map_data.GRID_ROWS == 10 and map_data.GRID_COLS == 10)
    walkable = set(map_data.walkable_cells())
    reachable = map_data.reachable_cells()
    check("every open cell is reachable", walkable == reachable,
          "%d open cells" % len(walkable))
    routes = map_data.route_lengths()
    check("three routes, short < middle < long",
          routes["shaft"] < routes["teleport"] < routes["ring"],
          "shaft %s, teleporter %s, ring %s"
          % (routes["shaft"], routes["teleport"], routes["ring"]))
    print("       map fingerprint: %s" % map_data.map_hash())

    # ------------------------------------------------------------------
    heading("3. The transition model")
    env = Room1Env()
    states = env.all_states()
    check("state space size", len(states) == len(walkable) * 2 * 5,
          "%d states = %d cells x 2 battery x 5 directions"
          % (len(states), len(walkable)))
    worst = env.worst_probability_error()
    check("every P(s'|s,a) sums to 1", worst < 1e-9,
          "worst error %.2e over %d state-action pairs" % (worst, len(states) * 4))

    # ------------------------------------------------------------------
    heading("4. Value Iteration")
    value_result = dp_solver.value_iteration(env, gamma=0.95, theta=1e-6,
                                             max_sweeps=2000)
    check("converged", value_result["converged"],
          "%d sweeps in %.0f ms, final delta %.2e"
          % (value_result["iterations"], value_result["runtime_seconds"] * 1000,
             value_result["delta_history"][-1]))
    print("       V(start) = %.4f, Bellman residual = %.2e"
          % (value_result["start_state_value"], value_result["bellman_residual"]))

    # ------------------------------------------------------------------
    heading("5. Policy Iteration")
    policy_result = dp_solver.policy_iteration(env, gamma=0.95, theta=1e-6,
                                               max_iterations=200)
    check("converged", policy_result["converged"],
          "%d improvement rounds (%d evaluation sweeps) in %.0f ms"
          % (policy_result["iterations"], policy_result["evaluation_sweeps"],
             policy_result["runtime_seconds"] * 1000))
    print("       V(start) = %.4f, Bellman residual = %.2e"
          % (policy_result["start_state_value"], policy_result["bellman_residual"]))

    # ------------------------------------------------------------------
    heading("6. Comparing the two policies")
    comparison = dp_solver.compare_policies(env, value_result, policy_result, 0.95)
    check("the two methods agree (ties allowed)", comparison["different"] == 0,
          "%d/%d identical, %d ties, %d genuine differences"
          % (comparison["identical"], comparison["states_compared"],
             comparison["tied"], comparison["different"]))
    check("both find the same V(start)",
          abs(value_result["start_state_value"]
              - policy_result["start_state_value"]) < 1e-4)

    # ------------------------------------------------------------------
    heading("7. Simulating both policies (30 seeded episodes each)")
    for name, result in (("Value Iteration", value_result),
                         ("Policy Iteration", policy_result)):
        batch = simulate.run_many(env, result["policy"], episodes=30)
        check("%s escapes reliably" % name, batch["success_rate"] >= 0.9,
              "success %.0f%%, return %.1f ± %.1f, steps %.1f ± %.1f, "
              "laser hits %.2f"
              % (batch["success_rate"] * 100, batch["mean_return"],
                 batch["std_return"], batch["mean_steps"], batch["std_steps"],
                 batch["mean_laser_hits"]))
        example = simulate.run_episode(env, result["policy"], seed=0)
        print("       route on seed 0: %s"
              % simulate.describe_route(example["frames"]))

    # ------------------------------------------------------------------
    heading("8. Replay")
    first = simulate.run_episode(env, value_result["policy"], seed=99)
    second = simulate.run_episode(env, value_result["policy"], seed=99)
    identical = first["frames"] == second["frames"]
    check("the same seed reproduces the recording exactly", identical,
          "%d frames compared field by field" % len(first["frames"]))
    joined = all((current["from_row"], current["from_col"])
                 == (previous["row"], previous["col"])
                 for previous, current in zip(first["frames"], first["frames"][1:]))
    check("frames join up into a continuous walk", joined)

    distinct = {tuple((frame["row"], frame["col"]) for frame in
                      simulate.run_episode(env, value_result["policy"],
                                           seed=seed)["frames"])
                for seed in range(40)}
    check("different seeds give different episodes", len(distinct) > 1,
          "%d distinct paths over 40 seeds" % len(distinct))

    # ------------------------------------------------------------------
    heading("9. The gamma experiment")
    gamma_rows = experiments.gamma_experiment(episodes=20)
    check("one row per discount factor",
          [row["value"] for row in gamma_rows] == experiments.GAMMA_VALUES)
    print("       %-7s %-10s %-9s %-16s %s"
          % ("gamma", "V(start)", "success", "return", "route"))
    for row in gamma_rows:
        print("       %-7.2f %-10.2f %-9s %-16s %s"
              % (row["value"], row["start_state_value"],
                 "%.0f%%" % (row["success_rate"] * 100),
                 "%.1f ± %.1f" % (row["mean_return"], row["std_return"]),
                 row["route"]))

    # ------------------------------------------------------------------
    heading("10. The slipping experiment")
    ice_rows = experiments.slipping_experiment(episodes=20)
    routes_by_level = {row["value"]: row["route"] for row in ice_rows}
    print("       %-13s %-10s %-9s %-16s %s"
          % ("ice risk", "V(start)", "success", "return", "route"))
    for row in ice_rows:
        print("       %-13s %-10.2f %-9s %-16s %s"
              % (row["value"], row["start_state_value"],
                 "%.0f%%" % (row["success_rate"] * 100),
                 "%.1f ± %.1f" % (row["mean_return"], row["std_return"]),
                 row["route"]))
    check("safe ice makes the plan take the frozen shaft",
          "Frozen shaft" in routes_by_level["Low risk"])
    check("slippery ice makes the plan abandon the shaft",
          "Frozen shaft" not in routes_by_level["High risk"])

    # ------------------------------------------------------------------
    heading("11. The renderers")
    run = simulate.run_episode(env, value_result["policy"], seed=7)
    scene = iso_scene.build_scene(frames=run["frames"])
    problems = base_renderer.validate_scene(scene)
    check("the scene is valid", problems == [], "; ".join(problems))
    html = iso_canvas.build_html(scene)
    check("the page is built with the scene injected",
          "__SCENE__" not in html and scene["episodeId"] in html,
          "%d characters of HTML" % len(html))
    kinds = {kind for row in scene["grid"]["tiles"] for kind in row}
    missing = [kind for kind in kinds if '"%s"' % kind not in html]
    check("the engine handles every tile type on the map", not missing,
          "%d tile types: %s" % (len(kinds), ", ".join(sorted(kinds))))
    figure = fallback_renderer.render_frame(scene, scene["frames"][-1])
    check("the Matplotlib fallback draws the chamber",
          len(figure.axes[0].patches) > 100,
          "%d shapes" % len(figure.axes[0].patches))
    figure.clf()

    # ------------------------------------------------------------------
    heading("12. Saving and loading")
    path = storage.save_solution("verify_room1", env, value_result)
    loaded = storage.load_solution("verify_room1")
    check("the loaded policy is identical",
          loaded["policy"] == value_result["policy"],
          "%d states restored from %s"
          % (len(loaded["policy"]), os.path.basename(path)))
    values_match = all(abs(loaded["values"][state] - value)
                       < 1e-9 for state, value in value_result["values"].items())
    check("the loaded value table is identical", values_match)
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
