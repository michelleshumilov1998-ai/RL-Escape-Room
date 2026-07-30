"""The Streamlit page for Room 1 — the Laser Security Chamber.

Layout, top to bottom:

  * the laboratory nameplate, the room navigation rail and the inventory bar
  * the action buttons
  * the chamber itself, with the information cards beside it
  * the graphs
  * the experiments
  * the explanation of what Dynamic Programming is actually doing here

Nothing is ever recomputed by accident.  Solving, simulating and running
experiments only happen when the matching button is pressed, and the results are
kept in `st.session_state`, so moving a slider or switching a view never throws
away work.
"""

import pandas as pd
import streamlit as st

from core import actions
from plots import room1_plots
from renderers import fallback_renderer, iso_canvas, iso_scene
from rooms.room1 import dp_solver, experiments, map_data, simulate, storage
from rooms.room1.environment import Room1Env
from ui import buttons, cards, game_state, inventory, nav, theme

ROOM_NUMBER = 1

ALGORITHM_CHOICES = ["Value Iteration", "Policy Iteration", "Compare Both"]

BELLMAN = r"V_*(s) = \max_a \sum_{s'} P(s' \mid s, a)\,\big[R(s,a,s') + \gamma V_*(s')\big]"


# ----------------------------------------------------------------------
# The controls
# ----------------------------------------------------------------------

def _sidebar():
    """Draw the sidebar and return the current settings as a dictionary."""
    sidebar = st.sidebar
    sidebar.markdown('<div class="r5-label">Room 1 controls</div>',
                     unsafe_allow_html=True)

    algorithm = sidebar.selectbox("Algorithm", ALGORITHM_CHOICES, index=0,
                                  key="r1_algorithm",
                                  help="Both methods use the same known model of "
                                       "the room. 'Compare Both' runs them side "
                                       "by side.")

    sidebar.markdown("**Planning**")
    gamma = sidebar.slider("Gamma (discount factor)", 0.50, 0.999, 0.95, 0.005,
                           key="r1_gamma",
                           help="How much a future reward is worth now. Low "
                                "values make R-5 impatient.")
    theta = sidebar.select_slider("Theta (stopping threshold)",
                                  options=[1e-3, 1e-4, 1e-5, 1e-6, 1e-8],
                                  value=1e-6, format_func=lambda value: "%g" % value,
                                  key="r1_theta")
    max_sweeps = sidebar.slider("Maximum sweeps", 50, 3000, 1000, 50,
                                key="r1_max_sweeps")

    sidebar.markdown("**Floor conditions**")
    weak_ice = sidebar.slider("Wet floor — intended action", 0.40, 1.00, 0.80, 0.05,
                              key="r1_weak",
                              help="Chance of moving the intended way on a wet "
                                   "cell. The rest is split between the two "
                                   "sideways directions.")
    strong_ice = sidebar.slider("Frozen floor — intended action", 0.20, 1.00, 0.60,
                                0.05, key="r1_strong",
                                help="The frozen shaft is the short route. Raise "
                                     "this and the shaft becomes worth the risk.")
    oil_momentum = sidebar.slider("Oil — momentum", 0.00, 0.85, 0.55, 0.05,
                                  key="r1_oil",
                                  help="Chance of sliding on in the previous "
                                       "direction instead of the chosen one.")
    laser_penalty = sidebar.slider("Laser penalty", -120, -5, -30, 5,
                                   key="r1_laser")
    require_battery = sidebar.checkbox("Require battery before exit", value=False,
                                       key="r1_require_battery",
                                       help="Experimental. Off by default, so the "
                                            "main task stays unchanged.")

    sidebar.markdown("**Simulation and animation**")
    seed = sidebar.number_input("Random seed", min_value=0, max_value=99999,
                                value=7, step=1, key="r1_seed")
    speed = sidebar.select_slider("Animation speed", options=[0.5, 1.0, 2.0, 4.0],
                                  value=1.0, format_func=lambda v: "%.1gx" % v,
                                  key="r1_speed")
    episodes = sidebar.slider("Episodes per measurement", 10, 100, 30, 10,
                              key="r1_episodes",
                              help="Every reported number is averaged over this "
                                   "many seeded episodes.")

    sidebar.markdown("**Views**")
    view_mode = sidebar.radio("View", ["Gameplay", "Analysis"], horizontal=True,
                              key="r1_view_mode",
                              help="Gameplay shows the 3D chamber. Analysis shows "
                                   "the top-down maps used for interpretation.")
    show_values = sidebar.checkbox("Show V(s) values", value=True, key="r1_show_v")
    show_policy = sidebar.checkbox("Show policy arrows", value=True, key="r1_show_pi")
    slice_battery = sidebar.checkbox("State slice: battery collected", value=False,
                                     key="r1_slice_battery")
    slice_direction = sidebar.selectbox(
        "State slice: previous direction", actions.ALL_DIRECTIONS, index=0,
        format_func=actions.direction_name, key="r1_slice_direction")

    nav.sidebar_navigation(ROOM_NUMBER)

    return {
        "algorithm": algorithm, "gamma": gamma, "theta": theta,
        "max_sweeps": max_sweeps, "weak_ice_intended": weak_ice,
        "strong_ice_intended": strong_ice, "oil_momentum": oil_momentum,
        "laser_penalty": laser_penalty, "require_battery": require_battery,
        "seed": int(seed), "speed": speed, "episodes": episodes,
        "view_mode": view_mode, "show_values": show_values,
        "show_policy": show_policy, "slice_battery": slice_battery,
        "slice_direction": slice_direction,
    }


def _environment(settings):
    """Build an environment from the current settings."""
    return Room1Env(weak_ice_intended=settings["weak_ice_intended"],
                    strong_ice_intended=settings["strong_ice_intended"],
                    oil_momentum=settings["oil_momentum"],
                    laser_penalty=settings["laser_penalty"],
                    require_battery=settings["require_battery"])


def _solver_fingerprint(settings):
    """The settings that a solution actually depends on.

    Used to notice when a solution has gone stale, without re-solving anything.
    """
    return (settings["algorithm"], settings["gamma"], settings["theta"],
            settings["max_sweeps"], settings["weak_ice_intended"],
            settings["strong_ice_intended"], settings["oil_momentum"],
            settings["laser_penalty"], settings["require_battery"])


# ----------------------------------------------------------------------
# The actions behind the buttons
# ----------------------------------------------------------------------

def _solve(store, settings):
    env = _environment(settings)
    algorithm = settings["algorithm"]

    if algorithm == "Compare Both":
        with st.spinner("Running Value Iteration and Policy Iteration…"):
            outcome = experiments.compare_algorithms(
                environment_parameters=env.parameter_summary(),
                gamma=settings["gamma"], theta=settings["theta"],
                max_sweeps=settings["max_sweeps"], episodes=settings["episodes"])
        store["comparison"] = outcome
        store["solution"] = outcome["value_iteration"]
        store["second_solution"] = outcome["policy_iteration"]
    else:
        with st.spinner("Solving with %s…" % algorithm):
            store["solution"] = dp_solver.solve(env, algorithm,
                                                gamma=settings["gamma"],
                                                theta=settings["theta"],
                                                max_sweeps=settings["max_sweeps"])
        store["second_solution"] = None
        store["comparison"] = None

    store["fingerprint"] = _solver_fingerprint(settings)
    store["batch"] = None
    store["run"] = None


def _run_robot(store, settings):
    solution = store.get("solution")
    if not solution:
        st.warning("Solve the room first, so R-5 has a policy to follow.")
        return
    env = _environment(settings)
    store["run"] = simulate.run_episode(env, solution["policy"],
                                        seed=settings["seed"])
    store["batch"] = simulate.run_many(env, solution["policy"],
                                       episodes=settings["episodes"])
    store["playing"] = True
    if store["run"]["success"]:
        game_state.mark_solved(ROOM_NUMBER)


# ----------------------------------------------------------------------
# The page
# ----------------------------------------------------------------------

def render():
    """Draw the whole Room 1 page."""
    settings = _sidebar()
    store = game_state.room_store(ROOM_NUMBER)

    solution = store.get("solution")
    run = store.get("run")
    stale = bool(solution) and store.get("fingerprint") != _solver_fingerprint(settings)

    status = "SOLVED" if game_state.is_solved(ROOM_NUMBER) else (
        "ACTIVE" if solution else "LOCKED")
    if run and run["success"]:
        status = "ESCAPED"
    note = ""
    if solution:
        note = "%s · %d sweeps" % (solution["algorithm"], solution["iterations"])

    nav.room_header(ROOM_NUMBER, status, note,
                    algorithm=settings["algorithm"])
    nav.rail(ROOM_NUMBER)
    inventory.bar()
    nav.progress_line()

    if stale:
        st.info("The settings have changed since the room was last solved. "
                "Press **Solve Room** to plan again with the new numbers — "
                "nothing is recomputed until you do.", icon="ℹ️")

    _action_buttons(store, settings)

    # The keyed container gives the stylesheet something to aim at, so this one
    # split can be told to stack on a narrow screen without affecting the rest.
    with st.container(key="room1_main"):
        left, right = st.columns([2.05, 1], gap="medium")
        with left:
            _chamber(store, settings)
        with right:
            _cards(store, settings, status)

    st.divider()
    _graphs(store, settings)
    st.divider()
    _experiments(store, settings)
    st.divider()
    _explanation(settings)


def _action_buttons(store, settings):
    """The standard actions, in two rows of four so no label has to wrap."""
    top = st.columns(4)
    with top[0]:
        if buttons.action("solve", "r1"):
            _solve(store, settings)
            st.rerun()
    with top[1]:
        if buttons.action("run", "r1", disabled=not store.get("solution")):
            _run_robot(store, settings)
            st.rerun()
    with top[2]:
        playing = store.get("playing", True)
        label = "‖ Pause" if playing else "▶ Resume"
        if buttons.action("pause", "r1", label=label,
                          disabled=not store.get("run")):
            store["playing"] = not playing
            st.rerun()
    with top[3]:
        if buttons.action("reset", "r1", disabled=not store.get("run")):
            store["run"] = None
            store["playing"] = True
            st.rerun()

    bottom = st.columns(4)
    with bottom[0]:
        if buttons.action("compare", "r1"):
            # Solve both methods without disturbing the sidebar selection.
            saved = settings["algorithm"]
            settings["algorithm"] = "Compare Both"
            _solve(store, settings)
            settings["algorithm"] = saved
            st.rerun()
    with bottom[1]:
        if buttons.action("experiment", "r1"):
            store["show_experiments"] = True
            st.rerun()
    with bottom[2]:
        if buttons.action("save", "r1", disabled=not store.get("solution")):
            env = _environment(settings)
            path = storage.save_solution("room1_latest", env, store["solution"])
            st.success("Saved to %s" % path.split("/")[-1])
    with bottom[3]:
        if buttons.action("load", "r1"):
            try:
                loaded = storage.load_solution("room1_latest")
            except storage.SolutionFileError as problem:
                st.error(str(problem))
            else:
                store["solution"] = loaded
                store["second_solution"] = None
                store["comparison"] = None
                store["run"] = None
                store["fingerprint"] = None
                st.success("Loaded a %s result from disk." % loaded["algorithm"])
                st.rerun()


def _chamber(store, settings):
    """The chamber itself: the 3D view, or the top-down analysis views."""
    run = store.get("run")
    frames = run["frames"] if run else None

    if settings["view_mode"] == "Analysis":
        st.markdown("#### Analysis view")
        st.caption("Top-down maps used to read the plan. The chamber, the value "
                   "table and the policy all use the same grid orientation.")
        path = [(frame["row"], frame["col"]) for frame in frames] if frames else None
        st.pyplot(room1_plots.room_map(highlight_path=path))
        with st.expander("Isometric still frame (fallback renderer)"):
            st.caption("The same chamber drawn in Python. This is what is shown "
                       "if the interactive canvas cannot run in the browser.")
            still_scene = iso_scene.build_scene(frames=frames)
            st.pyplot(fallback_renderer.render_frame(
                still_scene, still_scene["frames"][-1]))
        return

    solution = store.get("solution")
    # The chamber always starts the recording with the security system armed.
    # The renderer switches the readout to DISABLED at the frame where R-5
    # actually reaches the control panel, so the status matches the animation
    # instead of jumping ahead of it.
    scene = iso_scene.build_scene(
        frames=frames, status="ACTIVE",
        options={"speed": settings["speed"],
                 "autoplay": store.get("playing", True) and bool(frames)})
    iso_canvas.render(scene)

    if not solution:
        st.caption("R-5 is standing by on the charging plate. Press **Solve Room** "
                   "to plan a route, then **Run Agent** to watch it.")
    elif not run:
        st.caption("A plan is ready. Press **Run Agent** to send R-5 into the "
                   "chamber.")
    else:
        st.caption("Playback controls are inside the chamber, so using them never "
                   "reloads the page. Drag to pan, scroll to zoom.")


def _cards(store, settings, status):
    """The information cards and the event log."""
    run = store.get("run")
    solution = store.get("solution")
    frames = run["frames"] if run else None
    frame = frames[-1] if frames else None

    routes = map_data.route_lengths()
    theme.html(cards.mission_card(
        "Control panel at row %d, col %d" % map_data.exit_cell(),
        requirements=[
            ("Control panel reached", bool(run and run["success"])),
            ("Battery collected (optional, +10)",
             bool(run and run["battery_collected"])),
        ],
        complete=bool(run and run["success"])))

    theme.html(cards.agent_state_card(frame, extra_rows=[
        ("Frozen shaft route", "%s steps" % routes["shaft"]),
        ("Teleporter route", "%s steps" % routes["teleport"]),
        ("South ring route", "%s steps" % routes["ring"]),
    ]))

    theme.html(cards.reward_card(
        frame,
        laser_hits=run["laser_hits"] if run else 0,
        wall_collisions=run["wall_collisions"] if run else 0,
        teleports=run["teleports_used"] if run else 0))

    theme.html(cards.algorithm_card(
        name=solution["algorithm"] if solution else settings["algorithm"],
        update_rule="V(s) ← max_a Σ P(s'|s,a)[R + γV(s')]",
        model_known=True,
        exploration="none — the model is planned over, not explored",
        hyperparameters=[
            ("gamma", "%.3f" % settings["gamma"]),
            ("theta", "%g" % settings["theta"]),
            ("sweeps used", str(solution["iterations"]) if solution else "—"),
            ("runtime", "%.0f ms" % (solution["runtime_seconds"] * 1000)
             if solution else "—"),
            ("V(start)", "%.2f" % solution["start_state_value"] if solution else "—"),
            ("Bellman residual", "%.2e" % solution["bellman_residual"]
             if solution else "—"),
        ]))

    status_rows = []
    if run:
        status_rows = [
            ("Steps taken", str(run["steps"])),
            ("Outcome", run["status"]),
            ("Route used", simulate.describe_route(run["frames"])),
            ("Seed", str(run["seed"])),
        ]
    batch = store.get("batch")
    if batch:
        status_rows.append(("Success over %d seeds" % batch["episodes"],
                            "%.0f%%" % (batch["success_rate"] * 100)))
        status_rows.append(("Mean return", "%.1f ± %.1f"
                            % (batch["mean_return"], batch["std_return"])))
    theme.html(cards.room_status_card(status, status_rows))

    st.markdown('<div class="r5-label" style="margin-top:10px">Event log</div>',
                unsafe_allow_html=True)
    theme.html(cards.event_log(frames or []))


def _graphs(store, settings):
    """The Dynamic Programming graphs."""
    st.markdown("### Convergence and results")
    solution = store.get("solution")
    if not solution:
        st.caption("Solve the room to produce the graphs.")
        return

    results = [solution]
    second = store.get("second_solution")
    if second:
        results.append(second)

    if "delta_history" not in solution:
        st.caption("This result was loaded from a file that stores no convergence "
                   "history.")
        return

    first, second_column = st.columns(2)
    with first:
        st.pyplot(room1_plots.convergence(results, theta=settings["theta"]))
    with second_column:
        st.pyplot(room1_plots.start_state_value(results))

    third, fourth = st.columns(2)
    with third:
        st.pyplot(room1_plots.policy_changes(results))
    with fourth:
        comparison = store.get("comparison")
        if comparison:
            st.pyplot(room1_plots.comparison_chart(comparison))
        else:
            st.caption("Choose **Compare Both** or press **Compare Algorithms** "
                       "to see the two methods side by side.")

    if store.get("comparison"):
        _comparison_table(store["comparison"])

    fifth, sixth = st.columns(2)
    with fifth:
        if settings["show_values"]:
            st.pyplot(room1_plots.value_heatmap(
                solution["values"], has_battery=settings["slice_battery"],
                previous_direction=settings["slice_direction"]))
    with sixth:
        if settings["show_policy"]:
            st.pyplot(room1_plots.policy_arrows(
                solution["policy"], has_battery=settings["slice_battery"],
                previous_direction=settings["slice_direction"]))


def _comparison_table(comparison):
    """The Value Iteration against Policy Iteration table."""
    st.markdown("#### Value Iteration vs Policy Iteration")
    value_result = comparison["value_iteration"]
    policy_result = comparison["policy_iteration"]
    value_batch = comparison["value_iteration_batch"]
    policy_batch = comparison["policy_iteration_batch"]
    agreement = comparison["comparison"]

    # Every cell is a string.  The rows measure different things, so a column
    # holds a mixture of counts, times and formatted numbers, and a column of
    # mixed types cannot be converted for display.
    table = pd.DataFrame([
        {"Measure": "Iterations / sweeps",
         "Value Iteration": "%d" % value_result["iterations"],
         "Policy Iteration": "%d" % policy_result["iterations"]},
        {"Measure": "Runtime (ms)",
         "Value Iteration": "%.1f" % (value_result["runtime_seconds"] * 1000),
         "Policy Iteration": "%.1f" % (policy_result["runtime_seconds"] * 1000)},
        {"Measure": "V(start)",
         "Value Iteration": "%.4f" % value_result["start_state_value"],
         "Policy Iteration": "%.4f" % policy_result["start_state_value"]},
        {"Measure": "Final Bellman residual",
         "Value Iteration": "%.2e" % value_result["bellman_residual"],
         "Policy Iteration": "%.2e" % policy_result["bellman_residual"]},
        {"Measure": "Success rate over %d seeds" % value_batch["episodes"],
         "Value Iteration": "%.0f%%" % (value_batch["success_rate"] * 100),
         "Policy Iteration": "%.0f%%" % (policy_batch["success_rate"] * 100)},
        {"Measure": "Mean return",
         "Value Iteration": "%.1f ± %.1f" % (value_batch["mean_return"],
                                             value_batch["std_return"]),
         "Policy Iteration": "%.1f ± %.1f" % (policy_batch["mean_return"],
                                              policy_batch["std_return"])},
        {"Measure": "Mean steps to exit",
         "Value Iteration": "%.1f ± %.1f" % (value_batch["mean_steps"],
                                             value_batch["std_steps"]),
         "Policy Iteration": "%.1f ± %.1f" % (policy_batch["mean_steps"],
                                              policy_batch["std_steps"])},
        {"Measure": "Mean laser hits",
         "Value Iteration": "%.2f" % value_batch["mean_laser_hits"],
         "Policy Iteration": "%.2f" % policy_batch["mean_laser_hits"]},
        {"Measure": "Policy agreement",
         "Value Iteration": "%.1f%%" % agreement["agreement_percent"],
         "Policy Iteration": "%.1f%%" % agreement["agreement_percent"]},
    ])
    st.dataframe(table, hide_index=True)

    if agreement["different"] == 0:
        st.success("Both methods produced the same action in all %d non-terminal "
                   "states (%d of those were exact ties in value)."
                   % (agreement["states_compared"], agreement["tied"]))
    else:
        st.warning("The two policies differ in %d states beyond ties. Those "
                   "differences are listed below with the Q values that caused "
                   "them." % agreement["different"])
        # The state is a tuple, which cannot be shown in a table directly.
        differences = [{
            "State (row, col, battery, prev dir)": "%d, %d, %s, %s" % (
                detail["state"][0], detail["state"][1],
                "yes" if detail["state"][2] else "no",
                actions.direction_name(detail["state"][3])),
            "Value Iteration action": actions.direction_name(detail["first_action"]),
            "Policy Iteration action": actions.direction_name(detail["second_action"]),
            "Q (Value Iteration)": "%.6f" % detail["first_value"],
            "Q (Policy Iteration)": "%.6f" % detail["second_value"],
        } for detail in agreement["different_details"][:20]]
        st.dataframe(pd.DataFrame(differences),
                     hide_index=True)


def _experiments(store, settings):
    """The gamma and slipping experiments."""
    st.markdown("### Parameter experiments")
    st.caption("Every row is averaged over %d seeded episodes and reported as "
               "mean ± standard deviation. A sweep can only identify the "
               "best-performing value among the values that were tested."
               % settings["episodes"])

    gamma_tab, ice_tab, custom_tab = st.tabs(
        ["Gamma experiment", "Slipping experiment", "Custom sweep"])

    base = _environment(settings).parameter_summary()

    with gamma_tab:
        if buttons.action("experiment", "r1_gamma", label="≣ Run gamma sweep"):
            with st.spinner("Solving the room once for each discount factor…"):
                store["gamma_rows"] = experiments.gamma_experiment(
                    base_parameters=base, episodes=settings["episodes"])
        _show_experiment(store.get("gamma_rows"),
                         "Discount factor gamma",
                         "How much a future reward is worth now. A low gamma "
                         "makes R-5 impatient, which changes the route it picks.")

    with ice_tab:
        if buttons.action("experiment", "r1_ice", label="≣ Run slipping sweep"):
            with st.spinner("Solving the room once for each ice setting…"):
                store["ice_rows"] = experiments.slipping_experiment(
                    base_parameters=base, gamma=settings["gamma"],
                    episodes=settings["episodes"])
        _show_experiment(store.get("ice_rows"),
                         "Ice risk",
                         "This is the headline result of Room 1: as the floor "
                         "gets more slippery, the plan abandons the short frozen "
                         "shaft for the long safe ring — worked out from the "
                         "model alone, without ever trying it.")

    with custom_tab:
        parameter = st.selectbox("Setting to sweep",
                                 list(experiments.SWEEPABLE_PARAMETERS.keys()),
                                 key="r1_sweep_parameter")
        if buttons.action("experiment", "r1_custom", label="≣ Run sweep"):
            with st.spinner("Sweeping %s…" % parameter):
                store["custom_rows"] = experiments.parameter_experiment(
                    parameter, experiments.SWEEPABLE_PARAMETERS[parameter],
                    base_parameters=base, gamma=settings["gamma"],
                    episodes=settings["episodes"])
        _show_experiment(store.get("custom_rows"), parameter, "")


def _show_experiment(rows, title, explanation):
    """One experiment's table, chart and best-performing row."""
    if not rows:
        st.caption("Press the button above to run this experiment.")
        return
    if explanation:
        st.caption(explanation)

    table = pd.DataFrame([{
        "Value": row["value"],
        "V(start)": round(row["start_state_value"], 2),
        "Success rate": "%.0f%%" % (row["success_rate"] * 100),
        "Mean return": "%.1f ± %.1f" % (row["mean_return"], row["std_return"]),
        "Mean steps": "%.1f ± %.1f" % (row["mean_steps"], row["std_steps"]),
        "Laser hits": "%.2f ± %.2f" % (row["mean_laser_hits"],
                                       row["std_laser_hits"]),
        "Teleporter use": "%.0f%%" % (row["teleport_use_rate"] * 100),
        "Battery taken": "%.0f%%" % (row["battery_rate"] * 100),
        "Route chosen": row["route"],
    } for row in rows])
    st.dataframe(table, hide_index=True)

    st.pyplot(room1_plots.experiment_chart(rows, title=title,
                                           ylabel="mean return"))

    best = experiments.best_performing_row(rows, "mean_return")
    if best:
        st.info("Best-performing value among the tested configurations: "
                "**%s**, with a mean return of %.1f ± %.1f. This is not "
                "necessarily an optimal value — only the best of the ones tried."
                % (best["value"], best["mean_return"], best["std_return"]))


def _explanation(settings):
    """What the agent knows, what it is doing, and why that works here."""
    st.markdown("### What is actually happening in this room")

    first, second, third = st.columns(3)
    with first:
        st.markdown("""
**What does the agent know?**

Everything. Before it moves at all, R-5 has the full model of the chamber: every
cell, every laser, how slippery each floor is, exactly what each action can lead
to and what reward follows. That is the list `env.transitions(state, action)`
returns, and its probabilities always add up to 1.
        """)
    with second:
        st.markdown("""
**What is the agent learning?**

Nothing, in the usual sense. It never tries an action to find out what happens.
It computes the value of every state and the best action from the Bellman
equations, sweeping over all %d states until the numbers stop changing. This is
planning, not learning from experience.
        """ % len(_environment(settings).all_states()))
    with third:
        st.markdown("""
**Why can it avoid a laser it never touched?**

Because the laser is already in the model. A sweep can see that a sideways slip
on the frozen shaft leads to a beam worth %d, and that cost is folded into the
value of the cell before R-5 ever steps on it. An agent that had to learn from
experience would have to be hit first.
        """ % settings["laser_penalty"])

    st.markdown("**The Bellman optimality equation being applied:**")
    st.latex(BELLMAN)
    st.caption("Value Iteration applies this as an assignment, over and over, "
               "until the largest change in a sweep drops below theta. Policy "
               "Iteration instead alternates between solving for the value of "
               "the current policy and making the policy greedy with respect to "
               "those values. On this room both reach the same answer.")

    with st.expander("The state, the actions and the model in detail"):
        st.markdown("""
**State** — `(row, col, has_battery, previous_direction)`, giving
%d cells × 2 × 5 = %d states.

* `has_battery` is part of the state because the +10 bonus may only ever be paid
  once.
* `previous_direction` is part of the state because oil makes R-5 carry on the
  way it was already going. Without it the same cell would behave differently
  depending on hidden history, and the problem would no longer be Markovian.

**Actions** — `UP`, `DOWN`, `LEFT`, `RIGHT`. The action is what R-5 *tries*; the
floor it is standing on decides what actually happens.

**Reward** — every transition pays the step cost of −1, and event rewards are
added on top of it. So a laser hit is −1 + %d = %d, and reaching the control
panel is −1 + 100 = +99. Wall or one-way-door collisions cost −3, the battery
pays +10 once, and the teleporter pays +5.

**Slipping** — the surface *under* R-5 decides the outcome, because that is what
it pushes off from:

| Surface | Distribution |
|---|---|
| normal floor | 100%% the chosen action |
| wet floor `~` | %.0f%% chosen, the rest split evenly sideways |
| frozen floor `%%` | %.0f%% chosen, the rest split evenly sideways |
| oil `O` | %.0f%% carry on in the previous direction, 10%% sideways, the rest to the chosen action |

When there is no previous direction, oil gives the momentum share to the chosen
action instead, so the numbers still add to exactly 1.

**Lasers** are not obstacles to walk through: entering a beam sends R-5 back to
the start platform, keeping any battery it already had, and the episode carries
on. That is why the genuine danger in this room is slippery floor *next to* a
beam, and why the safe ring is safe even though beams exist nearby.
        """ % (len(map_data.walkable_cells()),
               len(_environment(settings).all_states()),
               settings["laser_penalty"], settings["laser_penalty"] - 1,
               settings["weak_ice_intended"] * 100,
               settings["strong_ice_intended"] * 100,
               settings["oil_momentum"] * 100))

    with st.expander("The chamber map and legend"):
        st.code("\n".join("%d  %s" % (index, row)
                          for index, row in enumerate(map_data.ROOM_MAP)),
                language="text")
        st.markdown("   ".join("`%s` %s" % (tile, name)
                               for tile, name in map_data.LEGEND))
        routes = map_data.route_lengths()
        st.markdown(
            "**Routes:** frozen shaft %s steps (short, most likely to fail) · "
            "teleporter %s steps (middle) · south ring %s steps (long, never "
            "beside a beam)." % (routes["shaft"], routes["teleport"],
                                 routes["ring"]))
