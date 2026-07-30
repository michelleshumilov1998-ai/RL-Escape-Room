"""The Streamlit page for Room 4 — the Drone Wind Tunnel.

Same layout and design system as the earlier rooms. Training happens only when
Train is pressed; the weights, the flights and the experiments live in
`st.session_state`.
"""

import pandas as pd
import streamlit as st

from plots import room4_plots
from renderers import iso_canvas, room4_scene
from rooms.room4 import chamber, experiments, sarsa_agent, simulate, storage
from rooms.room4.environment import ACTION_NAMES, Room4Env
from ui import buttons, cards, game_state, inventory, nav, theme

ROOM_NUMBER = 4

Q_HAT = r"\hat{Q}(s,a,\mathbf{w}) = \mathbf{w}^{\top}\mathbf{x}(s,a)"
SARSA_RULE = (r"\delta = r + \gamma\,\hat{Q}(s',a',\mathbf{w}) - "
              r"\hat{Q}(s,a,\mathbf{w}) \qquad "
              r"\mathbf{w} \leftarrow \mathbf{w} + \alpha\,\delta\,"
              r"\mathbf{x}(s,a)")

ALGORITHM_CHOICES = [sarsa_agent.SARSA, sarsa_agent.Q_LEARNING, "Compare both"]


def _sidebar():
    sidebar = st.sidebar
    sidebar.markdown('<div class="r5-label">Room 4 controls</div>',
                     unsafe_allow_html=True)

    algorithm = sidebar.selectbox("Algorithm", ALGORITHM_CHOICES, index=0,
                                  key="r4_algorithm",
                                  help="Semi-Gradient SARSA is this room's method. "
                                       "Q-Learning uses the same features so the "
                                       "two can be compared directly.")

    sidebar.markdown("**Learning**")
    alpha = sidebar.slider("Learning rate (alpha)", 0.01, 1.00, 0.30, 0.01,
                           key="r4_alpha",
                           help="Divided by the number of tilings before use — "
                                "see the note in the explanation below.")
    gamma = sidebar.slider("Discount factor (gamma)", 0.80, 1.00, 0.99, 0.01,
                           key="r4_gamma")
    epsilon_start = sidebar.slider("Initial epsilon", 0.00, 1.00, 1.00, 0.05,
                                   key="r4_eps0")
    epsilon_min = sidebar.slider("Minimum epsilon", 0.00, 0.30, 0.02, 0.01,
                                 key="r4_epsmin")
    epsilon_decay = sidebar.select_slider(
        "Epsilon decay", options=[0.99, 0.995, 0.998, 1.000], value=0.995,
        format_func=lambda value: "%.3f" % value, key="r4_decay")
    episodes = sidebar.slider("Episodes", 100, 4000, 1200, 100, key="r4_episodes",
                              help="Below about 700 the drone has not learned to "
                                   "slow down for the pad.")
    max_steps = sidebar.slider("Maximum steps per episode", 60, 400, 220, 20,
                               key="r4_maxsteps")
    seed = sidebar.number_input("Random seed", min_value=0, max_value=99999,
                               value=0, step=1, key="r4_seed")

    sidebar.markdown("**Tile coding**")
    num_tilings = sidebar.select_slider("Number of tilings",
                                        options=[4, 6, 8, 12, 16], value=8,
                                        key="r4_tilings")
    tiles_per_dimension = sidebar.select_slider("Tiles per dimension",
                                                options=[4, 6, 8, 10, 12], value=8,
                                                key="r4_tiles")
    weights = num_tilings * tiles_per_dimension ** 4 * 5
    sidebar.caption("%s weights, %d active per update"
                    % ("{:,}".format(weights), num_tilings))

    sidebar.markdown("**Flight physics**")
    thrust = sidebar.slider("Thrust strength", 0.3, 2.5, 1.1, 0.1, key="r4_thrust")
    drag = sidebar.slider("Drag", 0.70, 1.00, 0.90, 0.01, key="r4_drag")
    wind_multiplier = sidebar.slider("Wind strength", 0.0, 2.0, 1.0, 0.1,
                                     key="r4_wind")
    turbulence = sidebar.slider("Turbulence", 0.00, 0.30, 0.00, 0.01,
                                key="r4_turb",
                                help="Seeded, so a replay of a turbulent flight is "
                                     "still exact.")
    safe_landing_speed = sidebar.slider("Safe landing speed (m/s)", 0.2, 1.5, 0.7,
                                        0.1, key="r4_safe")

    sidebar.markdown("**Simulation and animation**")
    run_seed = sidebar.number_input("Run seed", min_value=0, max_value=99999,
                                   value=0, step=1, key="r4_runseed")
    speed = sidebar.select_slider("Animation speed", options=[0.5, 1.0, 2.0, 4.0],
                                 value=1.0, format_func=lambda v: "%.1gx" % v,
                                 key="r4_speed")
    evaluation_episodes = sidebar.slider("Flights per measurement", 4, 40, 12, 2,
                                         key="r4_eval")

    sidebar.markdown("**Views**")
    view_mode = sidebar.radio("View", ["Gameplay", "Analysis", "Wind field"],
                              key="r4_view")
    window = sidebar.slider("Moving-average window", 5, 200, 40, 5, key="r4_window")
    slice_vx = sidebar.slider("Value slice: vx", -3.0, 3.0, 0.0, 0.5,
                              key="r4_slice_vx")
    slice_vy = sidebar.slider("Value slice: vy", -3.0, 3.0, 0.0, 0.5,
                              key="r4_slice_vy")

    nav.sidebar_navigation(ROOM_NUMBER)

    return {
        "algorithm": algorithm, "alpha": alpha, "gamma": gamma,
        "epsilon_start": epsilon_start, "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay, "episodes": episodes,
        "max_steps": max_steps, "seed": int(seed), "num_tilings": num_tilings,
        "tiles_per_dimension": tiles_per_dimension, "thrust": thrust, "drag": drag,
        "wind_multiplier": wind_multiplier, "turbulence": turbulence,
        "safe_landing_speed": safe_landing_speed, "run_seed": int(run_seed),
        "speed": speed, "evaluation_episodes": evaluation_episodes,
        "view_mode": view_mode, "window": window, "slice_vx": slice_vx,
        "slice_vy": slice_vy, "weights": weights,
    }


def _environment_parameters(settings):
    return {"thrust": settings["thrust"], "drag": settings["drag"],
            "wind_multiplier": settings["wind_multiplier"],
            "turbulence": settings["turbulence"],
            "safe_landing_speed": settings["safe_landing_speed"]}


def _hyperparameters(settings):
    return {"alpha": settings["alpha"], "gamma": settings["gamma"],
            "epsilon_start": settings["epsilon_start"],
            "epsilon_min": settings["epsilon_min"],
            "epsilon_decay": settings["epsilon_decay"],
            "num_tilings": settings["num_tilings"],
            "tiles_per_dimension": settings["tiles_per_dimension"],
            "max_steps": settings["max_steps"], "seed": settings["seed"]}


def _fingerprint(settings):
    return (settings["algorithm"], settings["episodes"],
            tuple(sorted(_hyperparameters(settings).items())),
            tuple(sorted(_environment_parameters(settings).items())))


# ----------------------------------------------------------------------

def _train(store, settings):
    algorithms = ([sarsa_agent.SARSA, sarsa_agent.Q_LEARNING]
                  if settings["algorithm"] == "Compare both"
                  else [settings["algorithm"]])
    results = []
    bar = st.progress(0.0, text="Training…")
    for index, algorithm in enumerate(algorithms):
        def report(fraction, index=index, algorithm=algorithm):
            bar.progress(min(1.0, (index + fraction) / len(algorithms)),
                         text="Training %s…" % algorithm)
        results.append(sarsa_agent.train(
            algorithm=algorithm, episodes=settings["episodes"], progress=report,
            environment_parameters=_environment_parameters(settings),
            **_hyperparameters(settings)))
    bar.empty()
    store["results"] = results
    store["primary"] = results[0]
    store["fingerprint"] = _fingerprint(settings)
    store["run"] = None
    store["batch"] = None


def _fly(store, settings):
    result = store.get("primary")
    if not result:
        st.warning("Train the drone first, so it has a policy to fly.")
        return
    store["run"] = simulate.run_episode(
        result["agent"], seed=settings["run_seed"],
        max_steps=settings["max_steps"],
        environment_parameters=_environment_parameters(settings))
    store["batch"] = simulate.run_many(
        result["agent"], episodes=settings["evaluation_episodes"],
        environment_parameters=_environment_parameters(settings))
    store["playing"] = True
    if store["run"]["success"]:
        game_state.mark_solved(ROOM_NUMBER)


# ----------------------------------------------------------------------

def render():
    settings = _sidebar()
    store = game_state.room_store(ROOM_NUMBER)

    result = store.get("primary")
    run = store.get("run")
    stale = bool(result) and store.get("fingerprint") != _fingerprint(settings)

    status = "SOLVED" if game_state.is_solved(ROOM_NUMBER) else (
        "ACTIVE" if result else "LOCKED")
    if run:
        status = "ESCAPED" if run["success"] else "FAILED"
    note = ""
    if result:
        note = "%s · %d episodes" % (result["algorithm"], result["episodes"])

    nav.room_header(ROOM_NUMBER, status, note, algorithm=settings["algorithm"])
    nav.rail(ROOM_NUMBER)
    inventory.bar()
    nav.progress_line()

    if stale:
        st.info("The settings have changed since the drone was trained. Press "
                "**Train** to learn again — nothing is retrained until you do.",
                icon="ℹ️")

    _action_buttons(store, settings)

    with st.container(key="room4_main"):
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
        if buttons.action("train", "r4"):
            _train(store, settings)
            st.rerun()
    with top[1]:
        if buttons.action("run", "r4", label="▶ Run Drone",
                          disabled=not store.get("primary")):
            _fly(store, settings)
            st.rerun()
    with top[2]:
        if buttons.action("replay", "r4", disabled=not store.get("run")):
            store["playing"] = True
            st.rerun()
    with top[3]:
        playing = store.get("playing", True)
        if buttons.action("pause", "r4",
                          label="‖ Pause" if playing else "▶ Resume",
                          disabled=not store.get("run")):
            store["playing"] = not playing
            st.rerun()

    bottom = st.columns(4)
    with bottom[0]:
        if buttons.action("reset", "r4", disabled=not store.get("run")):
            store["run"] = None
            store["playing"] = True
            st.rerun()
    with bottom[1]:
        if buttons.action("experiment", "r4"):
            store["show_experiments"] = True
            st.rerun()
    with bottom[2]:
        if buttons.action("save", "r4", label="↓ Save Model",
                          disabled=not store.get("primary")):
            path = storage.save_model("room4_latest", store["primary"])
            st.success("Saved to %s" % path.split("/")[-1])
    with bottom[3]:
        if buttons.action("load", "r4", label="↑ Load Model"):
            env = Room4Env(**_environment_parameters(settings))
            agent = sarsa_agent.SemiGradientAgent(
                env, settings["num_tilings"], settings["tiles_per_dimension"],
                settings["alpha"], settings["gamma"])
            try:
                payload = storage.load_into("room4_latest", agent)
            except storage.ModelFileError as problem:
                st.error(str(problem))
            else:
                store["primary"] = {
                    "algorithm": payload["algorithm"], "agent": agent,
                    "episodes": payload.get("episodes", 0),
                    "runtime_seconds": payload.get("runtime_seconds", 0.0),
                    "history": payload.get("history", {}),
                    "example_update": None,
                    "final_epsilon": 0.0,
                    "weight_norm": agent.weight_norm(),
                    "active_features": agent.active_feature_count(),
                    "tile_coder": payload["tile_coder"],
                }
                store["results"] = [store["primary"]]
                store["run"] = None
                store["batch"] = None
                store["fingerprint"] = None
                st.success("Loaded a %s model." % payload["algorithm"])
                st.rerun()


def _chamber(store, settings):
    run = store.get("run")
    frames = run["frames"] if run else None

    if settings["view_mode"] == "Wind field":
        st.markdown("#### Wind field")
        st.caption("The sum of all four fans. This is the force the drone is "
                   "fighting, and it depends only on position — never on time — "
                   "which is what keeps the environment Markovian.")
        st.pyplot(room4_plots.wind_field())
        return

    if settings["view_mode"] == "Analysis":
        st.markdown("#### Analysis view")
        st.caption("Top-down hall with the flight paths drawn over it. Green paths "
                   "landed safely, red ones did not.")
        batch = store.get("batch")
        runs = batch["runs"] if batch else ([run] if run else [])
        st.pyplot(room4_plots.trajectories(runs, show_wind=True))
        return

    scene = room4_scene.build_scene(
        frames=frames, status="TESTING",
        wind_multiplier=settings["wind_multiplier"],
        safe_landing_speed=settings["safe_landing_speed"],
        options={"speed": settings["speed"],
                 "autoplay": store.get("playing", True) and bool(frames)})
    iso_canvas.render(scene)

    if not store.get("primary"):
        st.caption("The drone link is up. Press **Train** to learn the hall, then "
                   "**Run Drone** to fly it.")
    elif not run:
        st.caption("Weights are ready. Press **Run Drone** to fly.")
    else:
        st.caption("Playback controls are inside the hall. Wind, Trail and Vectors "
                   "toggle the overlays — cyan is velocity, amber thrust, purple "
                   "wind.")


def _cards(store, settings, status):
    run = store.get("run")
    result = store.get("primary")
    batch = store.get("batch")
    frame = run["frames"][-1] if run else None

    theme.html(cards.mission_card(
        "Cross the tunnel and land at %.1f m/s or less"
        % settings["safe_landing_speed"],
        requirements=[
            ("Reached the landing zone",
             bool(run and (run["success"] or run["hard_landings"]
                           or run["crashed"]))),
            ("Landed safely", bool(run and run["success"])),
        ],
        complete=bool(run and run["success"])))

    if frame is None:
        state_rows = [("Position", "—"), ("Velocity", "—"), ("Speed", "—")]
    else:
        state_rows = [
            ("Position", "%.2f, %.2f m" % (frame["x"], frame["y"]),
             cards.TONE_INFO),
            ("Velocity", "%+.2f, %+.2f m/s" % (frame["vx"], frame["vy"])),
            ("Speed", "%.2f m/s" % frame["speed"],
             cards.TONE_GOOD if frame["speed"] <= settings["safe_landing_speed"]
             else cards.TONE_WARN),
            ("Distance to pad", "%.2f m" % frame["distance_to_goal"]),
            ("Thrust", "%+.2f, %+.2f" % (frame["thrust_x"], frame["thrust_y"])),
            ("Wind", "%+.2f, %+.2f" % (frame["wind_x"], frame["wind_y"]),
             cards.TONE_WARN),
            ("Action", frame["action_name"]),
        ]
    theme.html(cards.card("Drone State", state_rows))

    if frame is None:
        theme.html(cards.card("Reward", [("Last reward", "—"),
                                         ("Cumulative", "—")]))
    else:
        theme.html(cards.card("Reward", [
            ("Last reward", "%+.2f" % frame["reward"],
             cards.TONE_GOOD if frame["reward"] > 0 else cards.TONE_NEUTRAL),
            ("Cumulative reward", "%+.1f" % frame["cumulative_reward"],
             cards.TONE_GOOD if frame["cumulative_reward"] > 0 else cards.TONE_BAD),
            ("Collisions", str(run["collisions"]),
             cards.TONE_WARN if run["collisions"] else cards.TONE_NEUTRAL),
            ("Hard landings", str(run["hard_landings"]),
             cards.TONE_WARN if run["hard_landings"] else cards.TONE_NEUTRAL),
            ("Landing speed", "%.2f m/s" % run["landing_speed"]
             if run["landing_speed"] else "did not arrive"),
            ("Safe landing limit", "%.1f m/s" % settings["safe_landing_speed"]),
        ]))

    theme.html(cards.algorithm_card(
        name=result["algorithm"] if result else settings["algorithm"],
        update_rule="w += (a/tilings) * delta * x(s,a)",
        model_known=False,
        exploration="epsilon-greedy over tile-coded features",
        hyperparameters=[
            ("alpha", "%.2f" % settings["alpha"]),
            ("effective alpha", "%.4f" % (settings["alpha"]
                                          / settings["num_tilings"])),
            ("gamma", "%.2f" % settings["gamma"]),
            ("tilings x tiles", "%d x %d" % (settings["num_tilings"],
                                             settings["tiles_per_dimension"])),
            ("weights", "{:,}".format(settings["weights"])),
            ("weights used", "{:,}".format(result["active_features"])
             if result else "—"),
            ("||w||", "%.1f" % result["weight_norm"] if result else "—"),
            ("training time", "%.1f s" % result["runtime_seconds"]
             if result else "—"),
        ]))

    status_rows = []
    if run:
        status_rows = [("Steps flown", str(run["steps"])),
                       ("Outcome", run["status"]),
                       ("Final distance", "%.2f m" % run["final_distance"]),
                       ("Seed", str(run["seed"]))]
    if batch:
        status_rows.append(("Safe landings over %d flights" % batch["episodes"],
                            "%.0f%%" % (batch["success_rate"] * 100)))
        status_rows.append(("Mean return", "%.1f ± %.1f"
                            % (batch["mean_return"], batch["std_return"])))
        status_rows.append(("Crash rate", "%.0f%%" % (batch["crash_rate"] * 100)))
    theme.html(cards.room_status_card(status, status_rows))

    st.markdown('<div class="r5-label" style="margin-top:10px">Event log</div>',
                unsafe_allow_html=True)
    theme.html(cards.event_log(run["frames"] if run else []))


def _graphs(store, settings):
    st.markdown("### Training results")
    results = store.get("results") or []
    if not results or not results[0].get("history", {}).get("reward"):
        st.caption("Train the drone to produce the graphs.")
        return

    window = settings["window"]
    run = store.get("run")
    batch = store.get("batch")

    rows = [
        (lambda: room4_plots.episode_reward(results, window),
         lambda: room4_plots.moving_average_reward(results, window)),
        (lambda: room4_plots.episode_length(results, window),
         lambda: room4_plots.success_rate(results, window)),
        (lambda: room4_plots.final_distance(results, window),
         lambda: room4_plots.landing_speed(results,
                                           settings["safe_landing_speed"])),
        (lambda: room4_plots.collision_count(results, window),
         lambda: room4_plots.epsilon_decay(results)),
        (lambda: room4_plots.weight_norm(results),
         lambda: room4_plots.td_error(results, window)),
        (lambda: room4_plots.active_tiles(results),
         lambda: room4_plots.training_time(results)),
    ]
    for left_builder, right_builder in rows:
        left, right = st.columns(2)
        with left:
            st.pyplot(left_builder())
        with right:
            st.pyplot(right_builder())

    with st.expander("Flight analysis"):
        if run:
            first, second = st.columns(2)
            with first:
                st.pyplot(room4_plots.velocity_over_time(run))
            with second:
                st.pyplot(room4_plots.wind_and_thrust(run))
            st.pyplot(room4_plots.landing_analysis(
                run, settings["safe_landing_speed"]))
        if batch:
            st.pyplot(room4_plots.trajectories(batch["runs"],
                                               title="Every evaluation flight"))
        if not run and not batch:
            st.caption("Press **Run Drone** to produce the flight analysis.")

    with st.expander("A slice of the value function"):
        st.caption("The state is four-dimensional, so there is no complete picture "
                   "to show. This fixes the velocity and sweeps position, plotting "
                   "max Q(s,a) with the preferred action on top. The velocity "
                   "slice is set in the sidebar.")
        st.pyplot(room4_plots.value_slice(results[0]["agent"],
                                          vx=settings["slice_vx"],
                                          vy=settings["slice_vy"]))


def _experiments(store, settings):
    st.markdown("### Experiments")
    st.caption("Every row trains from scratch several times with different seeds "
               "and reports mean ± standard deviation. A sweep can only identify "
               "the best-performing configuration among the settings tested.")

    general_tab, tile_tab, wind_tab, sweep_tab = st.tabs(
        ["Generalisation", "Tile resolution", "Wind and landing", "Parameter sweep"])

    episodes = st.session_state.get("r4_exp_episodes", 800)

    with general_tab:
        st.caption("The headline experiment. The drone is trained from **one** "
                   "release point and then flown from points it has never left. "
                   "Nothing in training ever sees the test starts.")
        columns = st.columns(2)
        with columns[0]:
            episodes = st.slider("Episodes per run", 200, 2000, 800, 100,
                                 key="r4_exp_episodes")
        with columns[1]:
            repeats = st.slider("Seeds", 1, 4, 2, 1, key="r4_gen_repeats")
        if buttons.action("experiment", "r4_gen",
                          label="≣ Run generalisation test"):
            with st.spinner("Training and evaluating on three start groups…"):
                store["gen_rows"] = experiments.generalisation_test(
                    episodes=episodes, repeats=repeats,
                    environment_parameters=_environment_parameters(settings))
        _show_generalisation(store.get("gen_rows"))

    with tile_tab:
        st.caption("Coarse tilings generalise widely but cannot represent a "
                   "precise approach; fine ones are precise but need far more "
                   "experience and far more memory.")
        if buttons.action("experiment", "r4_tiles", label="≣ Compare resolutions"):
            with st.spinner("Training three tile configurations…"):
                store["tile_rows"] = experiments.tile_resolution_experiment(
                    episodes=episodes, repeats=1)
        _show_experiment(store.get("tile_rows"), "tile coding", show_weights=True)

    with wind_tab:
        columns = st.columns(2)
        with columns[0]:
            if buttons.action("experiment", "r4_wind", label="≣ Wind strength"):
                with st.spinner("Training three wind levels…"):
                    store["wind_rows"] = experiments.wind_strength_experiment(
                        episodes=episodes, repeats=1)
        with columns[1]:
            if buttons.action("experiment", "r4_land", label="≣ Landing threshold"):
                with st.spinner("Training three landing thresholds…"):
                    store["land_rows"] = experiments.landing_threshold_experiment(
                        episodes=episodes, repeats=1)
        _show_experiment(store.get("wind_rows"), "wind strength")
        _show_experiment(store.get("land_rows"), "safe landing speed")

    with sweep_tab:
        parameter = st.selectbox("Setting to sweep",
                                 list(experiments.SWEEPABLE_PARAMETERS.keys()),
                                 format_func=lambda name:
                                 experiments.READABLE_NAMES.get(name, name),
                                 key="r4_sweep_parameter")
        if buttons.action("experiment", "r4_sweep", label="≣ Run sweep"):
            with st.spinner("Sweeping %s…" % parameter):
                store["sweep_rows"] = experiments.parameter_experiment(
                    parameter, episodes=episodes, repeats=1,
                    environment_parameters=_environment_parameters(settings))
        _show_experiment(store.get("sweep_rows"),
                         experiments.READABLE_NAMES.get(parameter, parameter))


def _show_experiment(rows, title, show_weights=False):
    if not rows:
        st.caption("Press the button above to run this experiment.")
        return
    record = {}
    table = []
    for row in rows:
        record = {
            "Value": str(row["value"]),
            "Mean reward": "%.1f ± %.1f" % (row["mean_reward"],
                                            row["std_reward"]),
            "Safe landings": "%.0f%%" % (row["mean_success"] * 100),
            "Hard landings": "%.0f%%" % (row["mean_hard_landing"] * 100),
            "Crashes": "%.0f%%" % (row["mean_crash"] * 100),
            "Final distance": "%.2f m" % row["mean_final_distance"],
            "Landing speed": "%.2f m/s" % row["mean_landing_speed"],
            "Collisions": "%.2f" % row["mean_collisions"],
            "||w||": "%.1f" % row["mean_weight_norm"],
            "Training time": "%.1f s" % row["mean_runtime"],
        }
        if show_weights and "weights" in row:
            record["Weights"] = "{:,}".format(row["weights"])
        table.append(record)
    st.dataframe(pd.DataFrame(table), hide_index=True)

    columns = st.columns(2)
    with columns[0]:
        st.pyplot(room4_plots.experiment_chart(rows, title=title,
                                               ylabel="mean reward"))
    with columns[1]:
        st.pyplot(room4_plots.experiment_chart(
            rows, metric="mean_success", error="std_success",
            title="Safe-landing rate", ylabel="share of episodes"))

    best = experiments.best_performing_row(rows, "mean_reward")
    if best:
        st.info("Best-performing configuration among the tested settings: **%s**, "
                "with a mean reward of %.1f ± %.1f. This is not necessarily an "
                "optimal setting — only the best of the ones tried."
                % (best["value"], best["mean_reward"], best["std_reward"]))


def _show_generalisation(rows):
    if not rows:
        st.caption("Press the button above to run the generalisation test.")
        return
    st.dataframe(pd.DataFrame([{
        "Start group": row["value"],
        "Seen in training": "yes" if row["seen_during_training"] else "NO",
        "Release points": len(row["starts"]),
        "Safe landings": "%.0f%% ± %.0f%%" % (row["mean_eval_success"] * 100,
                                              row["std_eval_success"] * 100),
        "Mean return": "%.1f ± %.1f" % (row["mean_eval_return"],
                                        row["std_eval_return"]),
        "Final distance": "%.2f m" % row["mean_final_distance"],
        "Landing speed": "%.2f m/s" % row["mean_eval_landing_speed"],
        "Collisions": "%.2f" % row["mean_eval_collisions"],
    } for row in rows]), hide_index=True)
    st.pyplot(room4_plots.generalisation_chart(rows))

    unseen = [row for row in rows if not row["seen_during_training"]]
    seen = [row for row in rows if row["seen_during_training"]]
    if unseen and seen:
        drop = (seen[0]["mean_eval_success"]
                - max(row["mean_eval_success"] for row in unseen)) * 100
        st.info("Trained from %d release point(s); flown from %d it had never left. "
                "The safe-landing rate on the best unseen group is %.0f percentage "
                "points from the trained one. A tile-coded linear model shares "
                "features between nearby states, which is why it transfers at all."
                % (len(seen[0]["starts"]),
                   sum(len(row["starts"]) for row in unseen), drop))


def _explanation(store, settings):
    st.markdown("### What is actually happening in this room")

    first, second, third = st.columns(3)
    with first:
        st.markdown("""
**Why a Q-table will not do**

The state is `(x, y, vx, vy)` and every one of those is a real number. Between any
two positions there is another position, so there is no finite list of states to
keep a table over and no chance of visiting the same state twice. The first three
rooms could look a value up; this one has to **compute** it.
        """)
    with second:
        st.markdown("""
**What function approximation does**

The value is estimated from shared parameters rather than stored per state:

`Q(s,a) = w · x(s,a)`

`x(s,a)` is a long, mostly-zero feature vector and `w` is learned. Because nearby
states produce overlapping features, learning something at one position
automatically says something about its neighbours. That sharing is the whole point,
and it is what the generalisation test measures.
        """)
    with third:
        st.markdown("""
**What tile coding does**

Lay a grid over the state space and a state falls in one cell — but a single grid
has hard edges. So lay down %d grids, each offset by a fraction of a cell. Every
state now switches on %d features out of %s, and two nearby states share most but
not all of them, which makes the generalisation smooth instead of blocky.

Only those %d weights are touched per update, however large the vector is.
        """ % (settings["num_tilings"], settings["num_tilings"],
               "{:,}".format(settings["weights"]), settings["num_tilings"]))

    st.markdown("**The approximate action-value function:**")
    st.latex(Q_HAT)
    st.markdown("**The semi-gradient SARSA update:**")
    st.latex(SARSA_RULE)
    st.markdown("""
**Why "semi-gradient".** The target `r + γ Q(s',a')` contains the weights too, so a
true gradient would have to differentiate through it as well. Semi-gradient methods
deliberately do not: the target is treated as a fixed number and only the gradient
of the current estimate is used — which, for a linear model, is just `x(s,a)`.

**Why alpha is divided by the number of tilings.** All %d active features feed the
same prediction, so moving each of them by the full step size would overshoot by
that factor. The effective step size is `alpha / tilings` = **%.4f**, which keeps
`alpha` meaning roughly the same thing whatever the tiling count.
    """ % (settings["num_tilings"],
           settings["alpha"] / settings["num_tilings"]))

    _worked_example(store)

    with st.expander("The dynamics, the wind and the landing rule in detail"):
        st.markdown("""
**The chamber** is %.0f x %.0f metres of open hall with %d fixed obstacles, %d
ventilation fans, a release point at (%.1f, %.1f) and a landing pad at
(%.1f, %.1f) of radius %.1f m.

**Actions** — five discrete thrusts: coast, up, down, left, right. Thrust changes
*velocity*, not position.

**Each step**, in this order:

```
wind_x, wind_y = wind_at(x, y)              # position only, never time
vx = (vx + thrust_x + wind_x + turbulence_x) * drag
vy = (vy + thrust_y + wind_y + turbulence_y) * drag
vx, vy = clip(vx, vy, v_max)
x, y = x + vx*dt, y + vy*dt                 # walked in 4 substeps
```

The move is walked in substeps so a fast drone cannot tunnel through a thin
obstacle between one frame and the next. On contact the drone is pushed back
outside the surface and bounced along the normal, losing most of its speed.

**The wind** is the sum of %d fans, each with a position, a direction, a strength
and a radius, falling off as `strength * (1 - d/radius)^2`. It depends only on
position, which is what keeps the environment Markovian without a clock in the
state. Turbulence is off by default; when switched on it is drawn from a seeded
generator and the actual draw is stored in the replay, so a turbulent flight plays
back exactly.

**Landing** needs both conditions:

| Outcome | Condition | Reward |
|---|---|---|
| Safe landing | in the pad **and** speed ≤ %.1f m/s | **+%.0f**, episode ends |
| Hard landing | in the pad, speed between %.1f and %.1f | %.0f, bounces off and carries on |
| Crash | in the pad at ≥ %.1f m/s | %.0f, episode ends |

Reaching the pad quickly is therefore not enough, which is what forces the drone to
learn to slow down.

**The rest of the reward** — %.1f per step, plus %.1f x the reduction in distance
to the pad as shaping, %.2f for using thrust, %.0f for hitting an obstacle and
%.0f for hitting a wall.
        """ % (chamber.WIDTH, chamber.HEIGHT, len(chamber.OBSTACLES),
               len(chamber.FANS), chamber.START_POSITION[0],
               chamber.START_POSITION[1], chamber.GOAL_POSITION[0],
               chamber.GOAL_POSITION[1], chamber.GOAL_RADIUS, len(chamber.FANS),
               settings["safe_landing_speed"], 150.0,
               settings["safe_landing_speed"], 2.4, -40.0, 2.4, -100.0,
               -0.5, 2.0, -0.05, -30.0, -25.0))

    with st.expander("The release points, and which are held back"):
        st.markdown("""
The generalisation test only means anything if the evaluation points were never
trained on, so the split is fixed in `rooms/room4/chamber.py`:

| Group | Points | Used for training |
|---|---|---|
| Training | %s | **yes** |
| Validation | %s | no |
| Unseen test | %s | no |

Training draws only from the training group. The tests assert the three sets do
not overlap.
        """ % (", ".join("(%.1f, %.1f)" % start
                         for start in chamber.TRAINING_STARTS),
               ", ".join("(%.1f, %.1f)" % start
                         for start in chamber.VALIDATION_STARTS),
               ", ".join("(%.1f, %.1f)" % start
                         for start in chamber.TEST_STARTS)))


def _worked_example(store):
    result = store.get("primary")
    example = result.get("example_update") if result else None
    if not example:
        st.caption("Train the drone to see one of its own updates worked through "
                   "with real numbers.")
        return

    tiles = example["active_tiles"]
    st.markdown("**One real update from this training session:**")
    st.markdown("""
Taken from the final episode of the %(algorithm)s run.

| Term | Value |
|---|---|
| state `s` = (x, y, vx, vy) | `%(state)s` |
| action `a` | `%(action)s` |
| active tile indices `x(s,a)` | `%(tiles)s` |
| `Q(s,a)` before | `%(before).4f` |
| reward `r` | `%(reward)+.4f` |
| %(bootstrap)s | `%(next_q).4f` |
| TD target | `%(target).4f` |
| TD error `delta` | `%(delta)+.4f` |
| alpha / tilings | `%(effective).5f` |
| change applied to each active weight | `%(change)+.5f` |

Only those %(count)d weights moved. Everything else in the vector is untouched.
    """ % {
        "algorithm": example["algorithm"],
        "state": "(%.3f, %.3f, %.3f, %.3f)" % example["state"],
        "action": ACTION_NAMES[example["action"]],
        "tiles": ", ".join(str(index) for index in tiles),
        "before": example["q_before"],
        "reward": example["reward"],
        "bootstrap": ("`max Q(s',·)`" if example["algorithm"]
                      == sarsa_agent.Q_LEARNING else "`Q(s',a')`"),
        "next_q": example["next_q"],
        "target": example["target"],
        "delta": example["td_error"],
        "effective": example["effective_alpha"],
        "change": example["weight_change"],
        "count": len(tiles),
    })
    if example["done"]:
        st.caption("This step ended the episode, so the target is just the reward: "
                   "there is no bootstrap term at all.")
