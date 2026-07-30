"""The Streamlit page for Room 3 — the Reactor Control Chamber.

Same layout as the earlier rooms, so all three read as one game.
Training happens only when Train is pressed; results live in `st.session_state`.
"""

import pandas as pd
import streamlit as st

from core import actions
from plots import room3_plots
from renderers import fallback_renderer, iso_canvas, room3_scene
from rooms.room3 import experiments, map_data, q_agent, simulate, storage
from rooms.room3.environment import Room3Env, STAGE_NAMES, STAGE_UNLOCKED
from ui import buttons, cards, game_state, inventory, nav, theme

ROOM_NUMBER = 3

BELLMAN = (r"Q_*(s,a) = \mathbb{E}\big[\,r + \gamma \max_{a'} Q_*(s',a')\,\big]")
Q_RULE = (r"Q(s,a) \leftarrow Q(s,a) + \alpha\big[\,r + \gamma \max_{a'}Q(s',a')"
          r" - Q(s,a)\,\big]")
SARSA_RULE = (r"Q(s,a) \leftarrow Q(s,a) + \alpha\big[\,r + \gamma\,Q(s',a')"
              r" - Q(s,a)\,\big]")

ALGORITHM_CHOICES = [q_agent.Q_LEARNING, q_agent.SARSA, "Compare both"]


# ----------------------------------------------------------------------
# Controls
# ----------------------------------------------------------------------

def _sidebar():
    sidebar = st.sidebar
    sidebar.markdown('<div class="r5-label">Room 3 controls</div>',
                     unsafe_allow_html=True)

    algorithm = sidebar.selectbox(
        "Algorithm", ALGORITHM_CHOICES, index=0, key="r3_algorithm",
        help="Q-Learning is this room's method. SARSA is here so the off-policy "
             "difference can be measured rather than only described.")

    sidebar.markdown("**Learning**")
    alpha = sidebar.slider("Learning rate (alpha)", 0.01, 1.00, 0.15, 0.01,
                           key="r3_alpha")
    gamma = sidebar.slider("Discount factor (gamma)", 0.50, 1.00, 0.98, 0.01,
                           key="r3_gamma",
                           help="The escape reward is 23 steps away from the move "
                                "that starts earning it, so this has to be high.")
    epsilon_start = sidebar.slider("Initial epsilon", 0.00, 1.00, 1.00, 0.05,
                                   key="r3_eps0")
    epsilon_min = sidebar.slider("Minimum epsilon", 0.00, 0.50, 0.05, 0.01,
                                 key="r3_epsmin")
    epsilon_decay = sidebar.select_slider(
        "Epsilon decay", options=[0.995, 0.998, 0.999, 0.9995, 1.000],
        value=0.999, format_func=lambda value: "%.4f" % value, key="r3_decay")
    episodes = sidebar.slider("Episodes", 500, 12000, 5000, 500, key="r3_episodes",
                              help="Below about 3000 the policy does not reliably "
                                   "finish the generator sequence.")
    seed = sidebar.number_input("Random seed", min_value=0, max_value=99999,
                                value=0, step=1, key="r3_seed")

    sidebar.markdown("**Rewards**")
    step_cost = sidebar.slider("Each step", -5, 0, -1, 1, key="r3_step")
    wall_penalty = sidebar.slider("Wall collision", -20, 0, -5, 1, key="r3_wall")
    hazard_penalty = sidebar.slider("Electrical hazard", -100, 0, -25, 5,
                                    key="r3_hazard")
    guard_penalty = sidebar.slider("Detected by the guard", -300, -10, -100, 10,
                                   key="r3_guard")
    generator_bonus = sidebar.slider("Generator activation", 0, 100, 20, 5,
                                     key="r3_gen")
    final_bonus = sidebar.slider("Final generator", 0, 150, 40, 5, key="r3_final")
    exit_reward = sidebar.slider("Reach the exit", 20, 400, 150, 10, key="r3_exit")

    sidebar.markdown("**Simulation and animation**")
    run_seed = sidebar.number_input("Run seed", min_value=0, max_value=99999,
                                    value=0, step=1, key="r3_runseed")
    run_epsilon = sidebar.slider("Exploration while running", 0.00, 0.50, 0.00,
                                 0.05, key="r3_runeps")
    speed = sidebar.select_slider("Animation speed", options=[0.5, 1.0, 2.0, 4.0],
                                 value=1.0, format_func=lambda v: "%.1gx" % v,
                                 key="r3_speed")
    evaluation_episodes = sidebar.slider("Episodes per measurement", 5, 60, 20, 5,
                                         key="r3_eval")

    sidebar.markdown("**Views**")
    view_mode = sidebar.radio("View", ["Gameplay", "Analysis"], horizontal=True,
                              key="r3_view")
    window = sidebar.slider("Moving-average window", 10, 400, 100, 10,
                            key="r3_window")
    slice_stage = sidebar.select_slider(
        "Policy view: stage", options=[0, 1, 2, 3], value=0,
        format_func=lambda value: "%d — %s" % (value, STAGE_NAMES[value]),
        key="r3_slice_stage")
    slice_guard = sidebar.slider("Policy view: guard patrol index", 0,
                                 map_data.PATROL_LENGTH - 1, 0, 1,
                                 key="r3_slice_guard")

    nav.sidebar_navigation(ROOM_NUMBER)

    return {
        "algorithm": algorithm, "alpha": alpha, "gamma": gamma,
        "epsilon_start": epsilon_start, "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay, "episodes": episodes, "seed": int(seed),
        "step_cost": step_cost, "wall_penalty": wall_penalty,
        "hazard_penalty": hazard_penalty, "guard_penalty": guard_penalty,
        "generator_bonus": generator_bonus, "final_generator_bonus": final_bonus,
        "exit_reward": exit_reward, "run_seed": int(run_seed),
        "run_epsilon": run_epsilon, "speed": speed,
        "evaluation_episodes": evaluation_episodes, "view_mode": view_mode,
        "window": window, "slice_stage": slice_stage, "slice_guard": slice_guard,
    }


def _environment(settings):
    return Room3Env(step_cost=settings["step_cost"],
                    wall_penalty=settings["wall_penalty"],
                    hazard_penalty=settings["hazard_penalty"],
                    guard_penalty=settings["guard_penalty"],
                    generator_bonus=settings["generator_bonus"],
                    final_generator_bonus=settings["final_generator_bonus"],
                    exit_reward=settings["exit_reward"])


def _hyperparameters(settings):
    return {"alpha": settings["alpha"], "gamma": settings["gamma"],
            "epsilon_start": settings["epsilon_start"],
            "epsilon_min": settings["epsilon_min"],
            "epsilon_decay": settings["epsilon_decay"], "seed": settings["seed"]}


def _fingerprint(settings):
    return (settings["algorithm"], settings["episodes"],
            tuple(sorted(_hyperparameters(settings).items())),
            settings["step_cost"], settings["wall_penalty"],
            settings["hazard_penalty"], settings["guard_penalty"],
            settings["generator_bonus"], settings["final_generator_bonus"],
            settings["exit_reward"])


# ----------------------------------------------------------------------
# Button actions
# ----------------------------------------------------------------------

def _train(store, settings):
    algorithms = ([q_agent.Q_LEARNING, q_agent.SARSA]
                  if settings["algorithm"] == "Compare both"
                  else [settings["algorithm"]])
    results = []
    bar = st.progress(0.0, text="Training…")
    for index, algorithm in enumerate(algorithms):
        env = _environment(settings)

        def report(fraction, index=index, algorithm=algorithm):
            bar.progress(min(1.0, (index + fraction) / len(algorithms)),
                         text="Training %s…" % algorithm)

        results.append(q_agent.train(env, algorithm=algorithm,
                                     episodes=settings["episodes"],
                                     progress=report,
                                     **_hyperparameters(settings)))
    bar.empty()
    store["results"] = results
    store["primary"] = results[0]
    store["fingerprint"] = _fingerprint(settings)
    store["run"] = None
    store["batch"] = None


def _run_agent(store, settings):
    result = store.get("primary")
    if not result:
        st.warning("Train the agent first, so it has a policy to follow.")
        return
    env = _environment(settings)
    store["run"] = simulate.run_episode(env, result["policy"],
                                        epsilon=settings["run_epsilon"],
                                        seed=settings["run_seed"])
    store["batch"] = simulate.run_many(env, result["policy"],
                                       episodes=settings["evaluation_episodes"],
                                       epsilon=settings["run_epsilon"])
    store["playing"] = True
    if store["run"]["success"]:
        game_state.mark_solved(ROOM_NUMBER)


# ----------------------------------------------------------------------
# The page
# ----------------------------------------------------------------------

def render():
    settings = _sidebar()
    store = game_state.room_store(ROOM_NUMBER)

    result = store.get("primary")
    run = store.get("run")
    stale = bool(result) and store.get("fingerprint") != _fingerprint(settings)

    status = "SOLVED" if game_state.is_solved(ROOM_NUMBER) else (
        "ACTIVE" if result else "STANDBY")
    if run:
        status = "ESCAPED" if run["success"] else "FAILED"
    note = ""
    if result:
        note = "%s · %d episodes" % (result["algorithm"], result["episodes"])

    nav.room_header(ROOM_NUMBER, status, note, algorithm=settings["algorithm"])
    nav.rail(ROOM_NUMBER)
    inventory.bar()

    if stale:
        st.info("The settings have changed since the agent was trained. Press "
                "**Train** to learn again — nothing is retrained until you do.",
                icon="ℹ️")

    _action_buttons(store, settings)

    with st.container(key="room3_main"):
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
    _explanation(store, settings)


def _action_buttons(store, settings):
    top = st.columns(4)
    with top[0]:
        if buttons.action("train", "r3"):
            _train(store, settings)
            st.rerun()
    with top[1]:
        if buttons.action("evaluate", "r3", disabled=not store.get("primary")):
            _run_agent(store, settings)
            st.rerun()
    with top[2]:
        if buttons.action("replay", "r3", disabled=not store.get("run")):
            store["playing"] = True
            st.rerun()
    with top[3]:
        playing = store.get("playing", True)
        if buttons.action("pause", "r3",
                          label="‖ Pause" if playing else "▶ Resume",
                          disabled=not store.get("run")):
            store["playing"] = not playing
            st.rerun()

    bottom = st.columns(4)
    with bottom[0]:
        if buttons.action("reset", "r3", disabled=not store.get("run")):
            store["run"] = None
            store["playing"] = True
            st.rerun()
    with bottom[1]:
        if buttons.action("experiment", "r3"):
            store["show_experiments"] = True
            st.rerun()
    with bottom[2]:
        if buttons.action("save", "r3", label="↓ Save Model",
                          disabled=not store.get("primary")):
            path = storage.save_model("room3_latest", _environment(settings),
                                      store["primary"])
            st.success("Saved to %s" % path.split("/")[-1])
    with bottom[3]:
        if buttons.action("load", "r3", label="↑ Load Model"):
            try:
                loaded = storage.load_model("room3_latest")
            except storage.ModelFileError as problem:
                st.error(str(problem))
            else:
                env = Room3Env()
                loaded["policy"] = q_agent.greedy_policy(loaded["q"], env)
                loaded.setdefault("runtime_seconds", 0.0)
                loaded.setdefault("history", {})
                store["primary"] = loaded
                store["results"] = [loaded]
                store["run"] = None
                store["batch"] = None
                store["fingerprint"] = None
                st.success("Loaded a %s model." % loaded["algorithm"])
                st.rerun()


def _chamber(store, settings):
    run = store.get("run")
    frames = run["frames"] if run else None

    if settings["view_mode"] == "Analysis":
        st.markdown("#### Analysis view")
        st.caption("Top-down chamber with the guard's patrol drawn in. The dashed "
                   "loop is the patrol; R-5 never sees it directly, it has to learn "
                   "the timing from experience.")
        path = [(frame["row"], frame["col"]) for frame in frames] if frames else None
        guard = ((frames[-1]["guard_row"], frames[-1]["guard_col"])
                 if frames else None)
        st.pyplot(room3_plots.chamber_map(highlight_path=path, guard_cell=guard))
        with st.expander("Isometric still frame (fallback renderer)"):
            still = room3_scene.build_scene(frames=frames)
            st.pyplot(fallback_renderer.render_frame(still, still["frames"][-1]))
        return

    scene = room3_scene.build_scene(
        frames=frames, status="OFFLINE",
        options={"speed": settings["speed"],
                 "autoplay": store.get("playing", True) and bool(frames)})
    iso_canvas.render(scene)

    if not store.get("primary"):
        st.caption("The reactor is offline. Press **Train** to learn the chamber, "
                   "then **Evaluate** to watch R-5 run it.")
    elif not run:
        st.caption("A policy is ready. Press **Evaluate** to send R-5 in.")
    else:
        st.caption("Playback controls are inside the chamber. The red unit is the "
                   "security robot — it patrols anticlockwise, one cell per step.")


def _cards(store, settings, status):
    run = store.get("run")
    result = store.get("primary")
    batch = store.get("batch")
    frames = run["frames"] if run else None
    frame = frames[-1] if frames else None

    stage = frame["stage"] if frame else 0
    theme.html(cards.mission_card(
        "Restore reactor power, then reach the exit",
        requirements=[("Generator A online", stage >= 1),
                      ("Generator B online", stage >= 2),
                      ("Generator C online — door unlocked", stage >= 3),
                      ("Reactor exit reached", bool(run and run["success"]))],
        complete=bool(run and run["success"])))

    if frame is None:
        state_rows = [("Position", "—"), ("Stage", "—"), ("Guard", "—")]
    else:
        state_rows = [
            ("Position", "row %d, col %d" % (frame["row"], frame["col"]),
             cards.TONE_INFO),
            ("Stage", "%d — %s" % (frame["stage"], STAGE_NAMES[frame["stage"]]),
             cards.TONE_GOOD if frame["stage"] >= STAGE_UNLOCKED
             else cards.TONE_WARN),
            ("Guard position", "row %d, col %d" % (frame["guard_row"],
                                                  frame["guard_col"]),
             cards.TONE_BAD),
            ("Guard patrol index", "%d / %d" % (frame["guard_index"],
                                                map_data.PATROL_LENGTH)),
            ("Sliding doors", "OPEN" if frame["doors_open"] else "SHUT",
             cards.TONE_GOOD if frame["doors_open"] else cards.TONE_WARN),
            ("Chosen action", "%s %s" % (
                actions.direction_arrow(frame["action"]) if frame["action"] else "·",
                frame["action_name"])),
        ]
    theme.html(cards.card("Agent State", state_rows))

    if frame is None:
        theme.html(cards.card("Reward", [("Last reward", "—"),
                                         ("Cumulative", "—")]))
    else:
        theme.html(cards.card("Reward", [
            ("Last reward", "%+.0f" % frame["reward"],
             cards.TONE_GOOD if frame["reward"] > 0 else
             (cards.TONE_BAD if frame["reward"] < -5 else cards.TONE_NEUTRAL)),
            ("Cumulative reward", "%+.0f" % frame["cumulative_reward"],
             cards.TONE_GOOD if frame["cumulative_reward"] > 0 else cards.TONE_BAD),
            ("Generators online", ", ".join(run["generators"]) or "none",
             cards.TONE_GOOD if run["generators"] else cards.TONE_NEUTRAL),
            ("Electrical hazards", str(run["hazards"]),
             cards.TONE_WARN if run["hazards"] else cards.TONE_NEUTRAL),
            ("Wall collisions", str(run["wall_collisions"])),
            ("Waits", str(run["waits"])),
        ]))

    theme.html(cards.algorithm_card(
        name=result["algorithm"] if result else settings["algorithm"],
        update_rule="Q(s,a) += a[r + g max Q(s',a') - Q(s,a)]",
        model_known=False,
        exploration="epsilon-greedy, decaying to a floor",
        hyperparameters=[
            ("alpha", "%.2f" % settings["alpha"]),
            ("gamma", "%.2f" % settings["gamma"]),
            ("epsilon now", "%.3f" % result["final_epsilon"] if result else "—"),
            ("episodes", str(result["episodes"]) if result else "—"),
            ("training time", "%.1f s" % result["runtime_seconds"]
             if result else "—"),
            ("mean |Q|", "%.2f" % result["mean_abs_q"] if result else "—"),
            ("states", "%d" % len(_environment(settings).all_states())),
        ]))

    status_rows = []
    if run:
        status_rows = [("Steps taken", str(run["steps"])),
                       ("Outcome", run["status"]),
                       ("Route used", simulate.describe_route(run["frames"])),
                       ("Seed", str(run["seed"]))]
    if batch:
        status_rows.append(("Success over %d runs" % batch["episodes"],
                            "%.0f%%" % (batch["success_rate"] * 100)))
        status_rows.append(("Mean return", "%.1f ± %.1f"
                            % (batch["mean_return"], batch["std_return"])))
        status_rows.append(("Detection rate", "%.0f%%"
                            % (batch["caught_rate"] * 100)))
    theme.html(cards.room_status_card(status, status_rows))

    st.markdown('<div class="r5-label" style="margin-top:10px">Event log</div>',
                unsafe_allow_html=True)
    theme.html(cards.event_log(frames or []))


def _graphs(store, settings):
    st.markdown("### Training results")
    results = store.get("results") or []
    if not results or not results[0].get("history", {}).get("reward"):
        st.caption("Train the agent to produce the graphs.")
        return

    window = settings["window"]
    pairs = [
        (room3_plots.episode_reward, room3_plots.moving_average_reward),
        (room3_plots.episode_length, room3_plots.success_rate),
        (room3_plots.generator_completion_rate, room3_plots.average_stage),
        (room3_plots.guard_detections, room3_plots.hazard_hits),
        (room3_plots.door_unlock_frequency, room3_plots.epsilon_decay),
    ]
    for left_builder, right_builder in pairs:
        left, right = st.columns(2)
        with left:
            st.pyplot(_call(left_builder, results, window))
        with right:
            st.pyplot(_call(right_builder, results, window))

    last_left, last_right = st.columns(2)
    with last_left:
        st.pyplot(room3_plots.q_value_magnitude(results))
    with last_right:
        st.pyplot(room3_plots.training_time(results))

    with st.expander("The learned policy, cell by cell"):
        st.pyplot(room3_plots.policy_arrows(results[0]["policy"],
                                            stage=settings["slice_stage"],
                                            guard_index=settings["slice_guard"]))
        st.caption("The state has four parts, so a flat picture has to fix the "
                   "stage and where the guard is standing. Both are set in the "
                   "sidebar. 'G' marks the guard's cell.")


def _call(builder, results, window):
    """Some graph builders take a window and some do not."""
    try:
        return builder(results, window)
    except TypeError:
        return builder(results)


def _experiments(store, settings):
    st.markdown("### Parameter experiments")
    st.caption("Every row trains from scratch several times with different seeds "
               "and reports mean ± standard deviation. A sweep can only identify "
               "the best-performing configuration among the settings tested.")

    sweep_tab, algorithm_tab = st.tabs(["Parameter sweep",
                                        "Q-Learning vs SARSA"])

    with sweep_tab:
        parameter = st.selectbox("Setting to sweep",
                                 list(experiments.SWEEPABLE_PARAMETERS.keys()),
                                 format_func=lambda name:
                                 experiments.READABLE_NAMES.get(name, name),
                                 key="r3_sweep_parameter")
        columns = st.columns(2)
        with columns[0]:
            episodes = st.slider("Episodes per run", 500, 6000, 3000, 500,
                                 key="r3_sweep_episodes")
        with columns[1]:
            repeats = st.slider("Seeds per value", 1, 5, 2, 1,
                                key="r3_sweep_repeats")
        if buttons.action("experiment", "r3_sweep", label="≣ Run sweep"):
            with st.spinner("Training %d configurations…"
                            % (len(experiments.SWEEPABLE_PARAMETERS[parameter])
                               * repeats)):
                store["sweep_rows"] = experiments.parameter_experiment(
                    parameter, episodes=episodes, repeats=repeats)
        _show_experiment(store.get("sweep_rows"),
                         experiments.READABLE_NAMES.get(parameter, parameter))

    with algorithm_tab:
        st.caption("Both methods, same chamber, same settings and seeds.")
        if buttons.action("compare", "r3_algo", label="⇄ Compare algorithms"):
            with st.spinner("Training Q-Learning and SARSA…"):
                store["algorithm_rows"] = experiments.compare_algorithms(
                    episodes=3000, repeats=2)
        _show_experiment(store.get("algorithm_rows"), "algorithm")


def _show_experiment(rows, title):
    if not rows:
        st.caption("Press the button above to run this experiment.")
        return

    table = pd.DataFrame([{
        "Value": str(row["value"]),
        "Mean reward": "%.1f ± %.1f" % (row["mean_reward"], row["std_reward"]),
        "Success": "%.0f%%" % (row["mean_success"] * 100),
        "Generators done": "%.0f%%" % (row["mean_generators"] * 100),
        "Avg stage": "%.2f" % row["mean_stage"],
        "Detected": "%.0f%%" % (row["mean_caught"] * 100),
        "Hazards": "%.2f" % row["mean_hazards"],
        "Steps": "%.1f" % row["mean_steps"],
        "Greedy run": "%+.0f" % row["mean_greedy_reward"],
        "Training time": "%.1f s" % row["mean_runtime"],
        "Route": row["route"],
    } for row in rows])
    st.dataframe(table, hide_index=True)

    columns = st.columns(2)
    with columns[0]:
        st.pyplot(room3_plots.experiment_chart(rows, title=title,
                                               ylabel="mean reward"))
    with columns[1]:
        st.pyplot(room3_plots.outcome_chart(rows))

    best = experiments.best_performing_row(rows, "mean_reward")
    if best:
        st.info("Best-performing configuration among the tested settings: **%s**, "
                "with a mean reward of %.1f ± %.1f. This is not necessarily an "
                "optimal value — only the best of the ones tried."
                % (best["value"], best["mean_reward"], best["std_reward"]))


def _explanation(store, settings):
    st.markdown("### What is actually happening in this room")

    first, second, third = st.columns(3)
    with first:
        st.markdown("""
**Why Q-Learning is off-policy**

The update bootstraps from `max Q(s',a')` — the value of the *best* action
available next, whether or not that is the action about to be taken. So while the
behaviour is still exploring, the values being learned are the values of behaving
optimally. Two different policies are in play at once: the one acting, and the one
being evaluated. That is what off-policy means.
        """)
    with second:
        st.markdown("""
**SARSA against Q-Learning**

SARSA uses `Q(s',a')` for the action it is *actually going to take*, so it learns
the value of its own exploratory behaviour. Q-Learning uses the maximum, so it
learns the value of the greedy policy regardless of what it does meanwhile.

One consequence: SARSA prices in its own mistakes and Q-Learning does not. Room 2
is built to show exactly that. Here both work, and the comparison tab measures
them on this chamber.
        """)
    with third:
        st.markdown("""
**Why it will learn a riskier optimum**

Because the target ignores the exploration that is still going on, Q-Learning is
happy to learn a route that only pays off if executed properly — squeezing past
the patrol, or crossing the service shaft on the step the doors happen to be open.
It converges on the best plan available rather than the safest one to stumble
through. In a room whose payoff is %d steps away, that is what is wanted.
        """ % (map_data.mission_route_length() or 23))

    st.markdown("**The Bellman optimality equation this is estimating:**")
    st.latex(BELLMAN)
    st.markdown("**The Q-Learning update:**")
    st.latex(Q_RULE)
    st.markdown("**SARSA, for comparison — one term differs:**")
    st.latex(SARSA_RULE)

    _worked_example(store)

    with st.expander("The state, the actions and the rewards in detail"):
        env = _environment(settings)
        st.markdown("""
**State** — `(row, col, stage, guard_index)`, giving %(states)d states over
%(actions)d actions.

* `stage` counts how far through the generator sequence R-5 is: 0 nothing, 1 A
  running, 2 A and B, 3 all three and the blast door unlocked.
* `guard_index` is where the security robot is on its %(patrol)d-cell patrol.
  Putting it in the state is what makes this a proper MDP — everything needed to
  predict the next step is visible.

**The sliding doors** are on a timer, which would normally break the Markov
property. Instead of adding a separate clock, the door cycle is derived from
`guard_index`, which is already in the state: they are open for %(half)d steps and
shut for %(half)d. The patrol is %(patrol)d cells and the cycle is %(period)d, so
the pattern repeats cleanly.

**Actions** — `UP`, `DOWN`, `LEFT`, `RIGHT` and `WAIT`. Waiting matters twice over:
it is how R-5 lets the patrol pass, and how it holds position until a door opens.

**The generator sequence** — A, then B, then C. Stepping on a generator out of turn
does nothing at all; the event log says so when it happens.

**Rewards** — every transition pays the step cost and event rewards are added on
top, the same convention as the earlier rooms:

| Event | Total |
|---|---|
| each step | %(step)d |
| wall, shut sliding door, or the locked blast door | %(step)d %(wall)+d = **%(wall_total)d** |
| electrical hazard | %(step)d %(hazard)+d = **%(hazard_total)d** |
| generator online (A or B) | %(step)d %(gen)+d = **%(gen_total)+d** |
| final generator (C) | %(step)d %(final)+d = **%(final_total)+d** |
| reach the exit | %(step)d %(exit)+d = **%(exit_total)+d** |
| detected by the guard | %(step)d %(guard)+d = **%(guard_total)d** |

**The order of a step** — the same in training, evaluation and replay: R-5 acts,
the cell it landed on takes effect, the guard advances one cell, and then a
collision is checked. A collision counts either when both end up on the same cell
or when the two swap places.
        """ % {
            "states": len(env.all_states()), "actions": len(env.actions()),
            "patrol": map_data.PATROL_LENGTH,
            "half": map_data.DOOR_HALF_PERIOD, "period": map_data.DOOR_PERIOD,
            "step": settings["step_cost"], "wall": settings["wall_penalty"],
            "wall_total": settings["step_cost"] + settings["wall_penalty"],
            "hazard": settings["hazard_penalty"],
            "hazard_total": settings["step_cost"] + settings["hazard_penalty"],
            "gen": settings["generator_bonus"],
            "gen_total": settings["step_cost"] + settings["generator_bonus"],
            "final": settings["final_generator_bonus"],
            "final_total": settings["step_cost"] + settings["final_generator_bonus"],
            "exit": settings["exit_reward"],
            "exit_total": settings["step_cost"] + settings["exit_reward"],
            "guard": settings["guard_penalty"],
            "guard_total": settings["step_cost"] + settings["guard_penalty"],
        })

    with st.expander("The chamber map and legend"):
        st.code("\n".join("%d  %s" % (index, row)
                          for index, row in enumerate(map_data.ROOM_MAP)),
                language="text")
        st.markdown("   ".join("`%s` %s" % (tile, name)
                               for tile, name in map_data.LEGEND))
        st.markdown(
            "**The mission** is a full clockwise lap of the service ring: A in the "
            "north-east, B in the south-east, C in the south-west, then the blast "
            "door on the south side. The shortest legal route is **%d steps**. The "
            "service shaft down the middle cuts the east-to-west trip from twelve "
            "steps and two shocks to six and none — when the doors happen to be "
            "open, and it is the only place the patrol cannot reach."
            % (map_data.mission_route_length() or 23))


def _worked_example(store):
    result = store.get("primary")
    example = result.get("example_update") if result else None
    if not example:
        st.caption("Train the agent to see one of its own updates worked through "
                   "with real numbers.")
        return

    is_q = example["algorithm"] == q_agent.Q_LEARNING
    best_next = example["best_next_action"]
    st.markdown("**One update from this training session, worked through:**")
    st.markdown("""
Taken from the final episode of the %(algorithm)s run.

| Term | Value |
|---|---|
| state `s` (row, col, stage, guard) | `%(state)s` |
| action `a` | `%(action)s` |
| reward `r` | `%(reward)+.0f` |
| next state `s'` | `%(next_state)s` |
| %(next_label)s | `%(next_action)s` |
| `Q(s,a)` before | `%(before).4f` |
| `%(bootstrap)s` | `%(next_q).4f` |
| target `r + γ·%(bootstrap)s` | `%(target).4f` |
| `Q(s,a)` after | `%(after).4f` |

so with α = %(alpha)s and γ = %(gamma)s:

`%(before).4f + %(alpha)s × (%(target).4f − %(before).4f) = %(after).4f`
    """ % {
        "algorithm": example["algorithm"],
        "state": example["state"],
        "action": actions.direction_name(example["action"]),
        "reward": example["reward"],
        "next_state": example["next_state"],
        "next_label": ("action with the highest value next" if is_q
                       else "next action, chosen by the same policy"),
        "next_action": (actions.direction_name(best_next)
                        if best_next is not None else "—"),
        "before": example["q_before"],
        "bootstrap": "max Q(s',·)" if is_q else "Q(s',a')",
        "next_q": example["max_next_q"],
        "target": example["target"],
        "after": example["q_after"],
        "alpha": example["alpha"],
        "gamma": example["gamma"],
    })
    if example["done"]:
        st.caption("This step ended the episode, so the target is just the reward: "
                   "there is no next state to bootstrap from.")
