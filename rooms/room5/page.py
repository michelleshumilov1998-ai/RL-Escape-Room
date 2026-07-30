"""The Streamlit page for Room 5 — the Adaptive Storage Facility.

Same layout and design system as the other four rooms. Training happens only when
Train is pressed; the weights, the layout pools and the experiments live in
`st.session_state`.
"""

import pandas as pd
import streamlit as st

from plots import room5_plots
from renderers import fallback_renderer, iso_canvas, room5_scene
from rooms.room5 import experiments, layout as L, q_agent, simulate, storage
from rooms.room5.environment import ACTION_NAMES, RADAR_NAMES, Room5Env
from rooms.room5.features import FEATURE_SETS, FeatureExtractor
from ui import buttons, cards, completion, game_state, inventory, nav, theme

ROOM_NUMBER = 5

Q_HAT = r"\hat{Q}(s,a,\mathbf{w}) = \mathbf{w}^{\top}\mathbf{x}(s,a)"
Q_RULE = (r"\delta = \big[r + \gamma \max_{a'}\hat{Q}(s',a',\mathbf{w})\big]"
          r" - \hat{Q}(s,a,\mathbf{w}) \qquad "
          r"\mathbf{w} \leftarrow \mathbf{w} + \alpha\,\delta\,\mathbf{x}(s,a)")

ALGORITHM_CHOICES = [q_agent.Q_LEARNING, q_agent.SARSA, "Compare both"]
VIEW_MODES = ["3D Gameplay", "Tactical", "Radar", "Full analysis"]


def _sidebar():
    sidebar = st.sidebar
    sidebar.markdown('<div class="r5-label">Room 5 controls</div>',
                     unsafe_allow_html=True)

    algorithm = sidebar.selectbox("Algorithm", ALGORITHM_CHOICES, index=0,
                                 key="r5_algorithm")
    difficulty = sidebar.selectbox("Difficulty", list(L.DIFFICULTIES.keys()),
                                   index=list(L.DIFFICULTIES).index(
                                       L.DEFAULT_DIFFICULTY),
                                   key="r5_difficulty",
                                   help="Easy is the default because it is where "
                                        "the agent demonstrably generalises. "
                                        "Medium and Hard are harder and it does "
                                        "measurably worse.")
    feature_set = sidebar.selectbox("Feature representation",
                                    list(FEATURE_SETS.keys()),
                                    index=list(FEATURE_SETS).index(
                                        q_agent.DEFAULT_FEATURE_SET),
                                    key="r5_features")

    sidebar.markdown("**Learning**")
    alpha = sidebar.slider("Learning rate (alpha)", 0.01, 0.60, 0.10, 0.01,
                           key="r5_alpha")
    gamma = sidebar.slider("Discount factor (gamma)", 0.80, 1.00, 0.97, 0.01,
                           key="r5_gamma")
    epsilon_start = sidebar.slider("Initial epsilon", 0.00, 1.00, 1.00, 0.05,
                                   key="r5_eps0")
    epsilon_min = sidebar.slider("Minimum epsilon", 0.00, 0.30, 0.05, 0.01,
                                 key="r5_epsmin")
    epsilon_decay = sidebar.select_slider(
        "Epsilon decay", options=[0.995, 0.998, 0.999, 0.9995], value=0.999,
        format_func=lambda value: "%.4f" % value, key="r5_decay")
    episodes = sidebar.slider("Episodes", 200, 8000, 2500, 100, key="r5_episodes",
                              help="Below about 1500 the policy has not learned "
                                   "to route around the racking.")
    seed = sidebar.number_input("Random seed", min_value=0, max_value=99999,
                               value=0, step=1, key="r5_seed")

    sidebar.markdown("**Layout pools**")
    training_layouts = sidebar.slider("Training layouts", 1, 60, 20, 1,
                                      key="r5_train_layouts",
                                      help="More variety generalises better — the "
                                           "layout-count experiment measures it.")
    validation_layouts = sidebar.slider("Validation layouts", 2, 20, 8, 1,
                                       key="r5_val_layouts")
    test_layouts = sidebar.slider("Unseen test layouts", 2, 30, 12, 1,
                                  key="r5_test_layouts")
    validation_every = sidebar.slider("Validate every N episodes", 0, 1000, 250,
                                      50, key="r5_val_every")

    sidebar.markdown("**Warehouse**")
    preset = L.DIFFICULTIES[difficulty]
    radar_range = sidebar.slider("Radar range", 2, 8, preset["radar_range"], 1,
                                 key="r5_radar")
    robots = sidebar.slider("Maintenance robots", 0, 4, preset["robots"], 1,
                            key="r5_robots")
    crates = sidebar.slider("Crates (obstacle density)", 0, 30, preset["crates"], 1,
                            key="r5_crates")
    conveyors = sidebar.slider("Conveyor cells", 0, 10, preset["conveyors"], 1,
                               key="r5_conveyors")

    sidebar.markdown("**Simulation and animation**")
    run_split = sidebar.radio("Run on a layout from", ["test", "validation",
                                                       "training"],
                              key="r5_run_split",
                              help="'test' seeds were never trained on.")
    run_index = sidebar.number_input("Layout index", min_value=0, max_value=29,
                                    value=0, step=1, key="r5_run_index")
    speed = sidebar.select_slider("Animation speed", options=[0.5, 1.0, 2.0, 4.0],
                                 value=1.0, format_func=lambda v: "%.1gx" % v,
                                 key="r5_speed")
    fog = sidebar.checkbox("Fog of war in gameplay view", value=True, key="r5_fog",
                           help="Off reveals the whole warehouse. The agent never "
                                "receives it either way.")

    sidebar.markdown("**Views**")
    view_mode = sidebar.radio("View", VIEW_MODES, key="r5_view")
    window = sidebar.slider("Moving-average window", 10, 400, 100, 10,
                            key="r5_window")

    nav.sidebar_navigation(ROOM_NUMBER)

    return {
        "algorithm": algorithm, "difficulty": difficulty,
        "feature_set": feature_set, "alpha": alpha, "gamma": gamma,
        "epsilon_start": epsilon_start, "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay, "episodes": episodes, "seed": int(seed),
        "training_layouts": training_layouts,
        "validation_layouts": validation_layouts, "test_layouts": test_layouts,
        "validation_every": validation_every, "radar_range": radar_range,
        "robots": robots, "crates": crates, "conveyors": conveyors,
        "run_split": run_split, "run_index": int(run_index), "speed": speed,
        "fog": fog, "view_mode": view_mode, "window": window,
    }


def _layout_overrides(settings):
    return {"radar_range": settings["radar_range"], "robots": settings["robots"],
            "crates": settings["crates"], "conveyors": settings["conveyors"]}


def _environment_parameters(settings):
    return {"radar_range": settings["radar_range"]}


def _hyperparameters(settings):
    return {"alpha": settings["alpha"], "gamma": settings["gamma"],
            "epsilon_start": settings["epsilon_start"],
            "epsilon_min": settings["epsilon_min"],
            "epsilon_decay": settings["epsilon_decay"],
            "training_layouts": settings["training_layouts"],
            "validation_layouts": settings["validation_layouts"],
            "seed": settings["seed"]}


def _fingerprint(settings):
    return (settings["algorithm"], settings["episodes"], settings["difficulty"],
            settings["feature_set"],
            tuple(sorted(_hyperparameters(settings).items())),
            tuple(sorted(_layout_overrides(settings).items())))


def _pool(settings, split, count):
    return L.generate_pool(split, count, settings["difficulty"],
                           _layout_overrides(settings))


# ----------------------------------------------------------------------

def _train(store, settings):
    algorithms = ([q_agent.Q_LEARNING, q_agent.SARSA]
                  if settings["algorithm"] == "Compare both"
                  else [settings["algorithm"]])
    results = []
    bar = st.progress(0.0, text="Training…")
    for index, algorithm in enumerate(algorithms):
        def report(fraction, index=index, algorithm=algorithm):
            bar.progress(min(1.0, (index + fraction) / len(algorithms)),
                         text="Training %s…" % algorithm)
        results.append(q_agent.train(
            algorithm=algorithm, episodes=settings["episodes"], progress=report,
            difficulty=settings["difficulty"], feature_set=settings["feature_set"],
            environment_parameters=_environment_parameters(settings),
            validation_every=settings["validation_every"],
            layout_overrides=_layout_overrides(settings),
            **_hyperparameters(settings)))
    bar.empty()
    store["results"] = results
    store["primary"] = results[0]
    store["fingerprint"] = _fingerprint(settings)
    store["run"] = None
    store["splits"] = None


def _run_agent(store, settings):
    result = store.get("primary")
    if not result:
        st.warning("Train the agent first.")
        return
    pool = _pool(settings, settings["run_split"],
                 max(settings["run_index"] + 1, 4))
    chosen = pool[min(settings["run_index"], len(pool) - 1)]
    store["layout"] = chosen
    store["run"] = simulate.run_episode(result["agent"], chosen,
                                        _environment_parameters(settings))
    store["playing"] = True
    if store["run"]["success"]:
        game_state.mark_solved(ROOM_NUMBER)


def _evaluate(store, settings):
    result = store.get("primary")
    if not result:
        st.warning("Train the agent first.")
        return
    parameters = _environment_parameters(settings)
    store["splits"] = []
    for split, count in (("training", settings["training_layouts"]),
                         ("validation", settings["validation_layouts"]),
                         ("test", settings["test_layouts"])):
        outcome = simulate.run_many(result["agent"], _pool(settings, split, count),
                                    parameters)
        outcome["split"] = split
        outcome["seen_during_training"] = split == "training"
        store["splits"].append(outcome)


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
        note = "%s · %d episodes · %d layouts" % (
            result["algorithm"], result["episodes"],
            len(result["training_seeds"]))

    nav.room_header(ROOM_NUMBER, status, note, algorithm=settings["algorithm"])
    nav.rail(ROOM_NUMBER)
    inventory.bar()

    if stale:
        st.info("The settings have changed since the agent was trained. Press "
                "**Train** to learn again — nothing is retrained until you do.",
                icon="ℹ️")

    _split_banner(settings)
    _action_buttons(store, settings)

    with st.container(key="room5_main"):
        left, right = st.columns([2.05, 1], gap="medium")
        with left:
            _warehouse(store, settings)
        with right:
            _cards(store, settings, status)

    st.divider()
    _graphs(store, settings)
    st.divider()
    _experiments(store, settings)
    st.divider()
    _explanation(store, settings)

    if game_state.all_solved():
        st.divider()
        left, middle, right = st.columns([1, 1, 1])
        with middle:
            if buttons.link("▸ Final report", "r5_complete", role="success"):
                game_state.go_to_complete()
                st.rerun()


def _split_banner(settings):
    """Which seeds belong to which split, stated up front."""
    theme.html(
        '<div class="r5-card" style="margin-bottom:6px">'
        '  <div class="r5-card-title"><span class="r5-card-accent"></span>'
        '     Layout split</div>'
        '  <div class="r5-row"><span class="r5-row-key">Training seeds</span>'
        '    <span class="r5-row-value info">%d–%d (%d layouts)</span></div>'
        '  <div class="r5-row"><span class="r5-row-key">Validation seeds</span>'
        '    <span class="r5-row-value warn">%d–%d (%d layouts)</span></div>'
        '  <div class="r5-row"><span class="r5-row-key">Unseen test seeds</span>'
        '    <span class="r5-row-value pos">%d–%d (%d layouts)</span></div>'
        '</div>' % (
            L.TRAINING_SEED_BASE,
            L.TRAINING_SEED_BASE + settings["training_layouts"] - 1,
            settings["training_layouts"],
            L.VALIDATION_SEED_BASE,
            L.VALIDATION_SEED_BASE + settings["validation_layouts"] - 1,
            settings["validation_layouts"],
            L.TEST_SEED_BASE,
            L.TEST_SEED_BASE + settings["test_layouts"] - 1,
            settings["test_layouts"]))


def _action_buttons(store, settings):
    top = st.columns(4)
    with top[0]:
        if buttons.action("train", "r5"):
            _train(store, settings)
            st.rerun()
    with top[1]:
        if buttons.action("evaluate", "r5", disabled=not store.get("primary")):
            _evaluate(store, settings)
            st.rerun()
    with top[2]:
        if buttons.action("run", "r5", label="▶ Run Agent",
                          disabled=not store.get("primary")):
            _run_agent(store, settings)
            st.rerun()
    with top[3]:
        playing = store.get("playing", True)
        if buttons.action("pause", "r5",
                          label="‖ Pause" if playing else "▶ Resume",
                          disabled=not store.get("run")):
            store["playing"] = not playing
            st.rerun()

    bottom = st.columns(4)
    with bottom[0]:
        if buttons.link("⟳ Generate Layout", "r5_layout", role="secondary"):
            pool = _pool(settings, settings["run_split"],
                         settings["run_index"] + 1)
            store["layout"] = pool[-1]
            store["run"] = None
            st.rerun()
    with bottom[1]:
        if buttons.action("reset", "r5", disabled=not store.get("run")):
            store["run"] = None
            store["playing"] = True
            st.rerun()
    with bottom[2]:
        if buttons.action("save", "r5", label="↓ Save Model",
                          disabled=not store.get("primary")):
            path = storage.save_model("room5_latest", store["primary"])
            st.success("Saved to %s" % path.split("/")[-1])
    with bottom[3]:
        if buttons.action("load", "r5", label="↑ Load Model"):
            extractor = FeatureExtractor(groups=FEATURE_SETS[settings["feature_set"]])
            agent = q_agent.LinearAgent(extractor)
            try:
                payload = storage.load_into("room5_latest", agent,
                                            settings["radar_range"])
            except storage.ModelFileError as problem:
                st.error(str(problem))
            else:
                store["primary"] = {
                    "algorithm": payload["algorithm"], "agent": agent,
                    "episodes": payload.get("episodes", 0),
                    "runtime_seconds": payload.get("runtime_seconds", 0.0),
                    "history": payload.get("history", {}),
                    "example_update": None, "final_epsilon": 0.0,
                    "weight_norm": agent.weight_norm(),
                    "active_features": agent.active_feature_count(),
                    "features": payload["features"],
                    "feature_set": payload["feature_set"],
                    "difficulty": payload.get("difficulty",
                                              settings["difficulty"]),
                    "training_seeds": payload.get("training_seeds", []),
                    "validation_seeds": payload.get("validation_seeds", []),
                }
                store["results"] = [store["primary"]]
                store["run"] = None
                store["splits"] = None
                store["fingerprint"] = None
                st.success("Loaded a %s model." % payload["algorithm"])
                st.rerun()


def _current_layout(store, settings):
    if store.get("layout") is None:
        pool = _pool(settings, settings["run_split"], settings["run_index"] + 1)
        store["layout"] = pool[-1]
    return store["layout"]


def _warehouse(store, settings):
    layout = _current_layout(store, settings)
    run = store.get("run")
    frames = run["frames"] if run else None

    if settings["view_mode"] == "Full analysis":
        st.markdown("#### Full analysis view")
        st.warning("This shows the **entire** warehouse, the patrol routes and the "
                   "trajectory. The agent never receives any of it — it sees only "
                   "the radar readings shown in the panel on the right.", icon="👁")
        st.pyplot(room5_plots.layout_map(layout, run))
        with st.expander("Isometric still frame (fallback renderer)"):
            still = room5_scene.build_scene(layout, frames=frames, fog_of_war=False)
            st.pyplot(fallback_renderer.render_frame(still, still["frames"][-1]))
        return

    if settings["view_mode"] == "Radar":
        st.markdown("#### Radar view")
        st.caption("The eight static rays and the nearest maintenance robot, as the "
                   "agent received them.")
        if run:
            st.pyplot(room5_plots.radar_over_time(run))
            st.pyplot(room5_plots.nearest_robot_distance(run))
        else:
            st.caption("Press **Run Agent** to record a mission first.")
        return

    camera = "tactical" if settings["view_mode"] == "Tactical" else "iso"
    scene = room5_scene.build_scene(
        layout, frames=frames, status="RECONFIGURING",
        fog_of_war=settings["fog"],
        options={"speed": settings["speed"], "camera": camera,
                 "autoplay": store.get("playing", True) and bool(frames)})
    iso_canvas.render(scene)

    if settings["fog"]:
        st.caption("Cells the radar has never reached are unlit; cells it has seen "
                   "but cannot see now are dimmed. Toggle **Wind** in the sector to "
                   "show the radar rays. Layout seed %d (%s split)."
                   % (layout.seed, layout.split))
    else:
        st.caption("Fog of war is off, so the whole warehouse is drawn. The agent "
                   "still only receives its radar. Layout seed %d (%s split)."
                   % (layout.seed, layout.split))


def _cards(store, settings, status):
    layout = _current_layout(store, settings)
    run = store.get("run")
    result = store.get("primary")
    splits = store.get("splits")
    frame = run["frames"][-1] if run else None

    stage = frame["stage"] if frame else 0
    theme.html(cards.mission_card(
        "Activate the terminal, then reach the final gate",
        requirements=[("Access terminal activated", stage >= 1),
                      ("Final gate reached", bool(run and run["success"]))],
        complete=bool(run and run["success"])))

    if frame is None:
        state_rows = [("Layout seed", str(layout.seed)),
                      ("Split", layout.split), ("Position", "—")]
    else:
        state_rows = [
            ("Layout seed", "%d (%s)" % (layout.seed, layout.split),
             cards.TONE_INFO),
            ("Position", "row %d, col %d" % (frame["row"], frame["col"])),
            ("Stage", "%d — %s" % (frame["stage"],
                                   "exit unlocked" if frame["stage"] >= 1
                                   else "terminal offline"),
             cards.TONE_GOOD if frame["stage"] >= 1 else cards.TONE_WARN),
            ("Current target", "row %d, col %d" % tuple(frame["current_target"])),
            ("Nearest robot", "%.2f" % frame["nearest_robot"],
             cards.TONE_BAD if frame["nearest_robot"] < 0.35 else cards.TONE_NEUTRAL),
            ("Action", frame["action_name"]),
        ]
    theme.html(cards.card("Agent State", state_rows))

    if frame is not None:
        theme.html(cards.card("Radar (what the agent sees)", [
            (RADAR_NAMES[index], "%.2f" % value)
            for index, value in enumerate(frame["radar_observation"])
        ]))

    if frame is None:
        theme.html(cards.card("Reward", [("Last reward", "—"),
                                         ("Cumulative", "—")]))
    else:
        theme.html(cards.card("Reward", [
            ("Last reward", "%+.1f" % frame["reward"],
             cards.TONE_GOOD if frame["reward"] > 0 else cards.TONE_NEUTRAL),
            ("Cumulative reward", "%+.0f" % frame["cumulative_reward"],
             cards.TONE_GOOD if frame["cumulative_reward"] > 0 else cards.TONE_BAD),
            ("Static collisions", str(run["collisions"]),
             cards.TONE_WARN if run["collisions"] else cards.TONE_NEUTRAL),
            ("Conveyor events", str(run["conveyor_events"])),
            ("Waits", str(run["waits"])),
            ("Path efficiency", "%.2f x shortest" % run["path_efficiency"]
             if run["path_efficiency"] else "—"),
        ]))

    theme.html(cards.algorithm_card(
        name=result["algorithm"] if result else settings["algorithm"],
        update_rule="w += a * delta * x(s,a),  delta uses max Q(s',a')",
        model_known=False,
        exploration="epsilon-greedy over local radar features",
        hyperparameters=[
            ("alpha", "%.2f" % settings["alpha"]),
            ("gamma", "%.2f" % settings["gamma"]),
            ("features", settings["feature_set"].split(".")[0]),
            ("weights", "{:,}".format(result["features"]["total_features"])
             if result else "—"),
            ("weights used", "{:,}".format(result["active_features"])
             if result else "—"),
            ("||w||", "%.1f" % result["weight_norm"] if result else "—"),
            ("radar range", str(settings["radar_range"])),
            ("training layouts", str(len(result["training_seeds"]))
             if result else "—"),
        ]))

    status_rows = []
    if run:
        status_rows = [("Steps", str(run["steps"])),
                       ("Outcome", run["status"]),
                       ("Shortest possible", str(run["reference_path"]))]
    if splits:
        for outcome in splits:
            status_rows.append(
                ("%s layouts%s" % (outcome["split"].title(),
                                   "" if outcome["seen_during_training"]
                                   else " (unseen)"),
                 "%.0f%% of %d" % (outcome["success_rate"] * 100,
                                   outcome["layouts"]),
                 cards.TONE_GOOD if outcome["seen_during_training"]
                 else cards.TONE_INFO))
    theme.html(cards.room_status_card(status, status_rows))

    st.markdown('<div class="r5-label" style="margin-top:10px">Event log</div>',
                unsafe_allow_html=True)
    theme.html(cards.event_log(run["frames"] if run else []))


def _graphs(store, settings):
    st.markdown("### Training results")
    results = store.get("results") or []
    if not results or not results[0].get("history", {}).get("reward"):
        st.caption("Train the agent to produce the graphs.")
        return

    window = settings["window"]
    pairs = [
        (lambda: room5_plots.episode_reward(results, window),
         lambda: room5_plots.moving_average_reward(results, window)),
        (lambda: room5_plots.episode_length(results, window),
         lambda: room5_plots.training_success(results, window)),
        (lambda: room5_plots.validation_success(results),
         lambda: room5_plots.terminal_rate(results, window)),
        (lambda: room5_plots.robot_collisions(results, window),
         lambda: room5_plots.static_collisions(results, window)),
        (lambda: room5_plots.timeout_rate(results, window),
         lambda: room5_plots.epsilon_decay(results)),
        (lambda: room5_plots.td_error(results, window),
         lambda: room5_plots.weight_norm(results)),
        (lambda: room5_plots.feature_activity(results),
         lambda: room5_plots.training_time(results)),
    ]
    for left_builder, right_builder in pairs:
        left, right = st.columns(2)
        with left:
            st.pyplot(left_builder())
        with right:
            st.pyplot(right_builder())

    splits = store.get("splits")
    if splits:
        st.markdown("#### Generalisation — the measurement that matters")
        st.dataframe(pd.DataFrame([{
            "Split": outcome["split"].title(),
            "Seen in training": "yes" if outcome["seen_during_training"] else "NO",
            "Layouts": outcome["layouts"],
            "Success": "%.0f%%" % (outcome["success_rate"] * 100),
            "Terminal reached": "%.0f%%" % (outcome["terminal_rate"] * 100),
            "Mean return": "%.1f ± %.1f" % (outcome["mean_return"],
                                            outcome["std_return"]),
            "Robot collisions": "%.0f%%" % (outcome["caught_rate"] * 100),
            "Timeouts": "%.0f%%" % (outcome["timeout_rate"] * 100),
            "Path efficiency": ("%.2f" % outcome["path_efficiency"]
                                if outcome["path_efficiency"] == outcome["path_efficiency"]
                                else "—"),
        } for outcome in splits]), hide_index=True)
        rows = [{"value": outcome["split"].title(),
                 "mean_training_success": (outcome["success_rate"]
                                           if outcome["seen_during_training"]
                                           else 0.0),
                 "mean_validation_success": (outcome["success_rate"]
                                             if outcome["split"] == "validation"
                                             else 0.0),
                 "mean_test_success": (outcome["success_rate"]
                                       if outcome["split"] == "test" else 0.0),
                 "parameter": "split"} for outcome in splits]
        st.pyplot(room5_plots.generalisation_chart(rows))
    else:
        st.caption("Press **Evaluate** to measure training, validation and unseen "
                   "test layouts side by side.")

    run = store.get("run")
    if run:
        with st.expander("Radar and trajectory analysis"):
            first, second = st.columns(2)
            with first:
                st.pyplot(room5_plots.radar_over_time(run))
            with second:
                st.pyplot(room5_plots.nearest_robot_distance(run))
            st.pyplot(room5_plots.action_distribution([run]))


def _experiments(store, settings):
    st.markdown("### Experiments")
    st.caption("Every row trains from scratch and is evaluated on the same unseen "
               "test layouts. A sweep can only identify the best-performing "
               "configuration among the settings tested.")

    tabs = st.tabs(["Layout count", "Feature set", "Radar range", "Stress test",
                    "Baselines", "Parameter sweep"])
    episodes = st.session_state.get("r5_exp_episodes", 1500)

    with tabs[0]:
        st.caption("The headline experiment: training on few layouts memorises "
                   "them, training on many generalises. Every agent is tested on "
                   "the same unseen layouts.")
        episodes = st.slider("Episodes per run", 300, 4000, 1500, 100,
                             key="r5_exp_episodes")
        if buttons.action("experiment", "r5_layouts", label="≣ Run layout count"):
            with st.spinner("Training %d agents…" % len(experiments.LAYOUT_COUNTS)):
                store["layout_rows"] = experiments.layout_count_experiment(
                    episodes=episodes, difficulty=settings["difficulty"],
                    layout_overrides=_layout_overrides(settings),
                    environment_parameters=_environment_parameters(settings))
        _show_experiment(store.get("layout_rows"), "training layouts")

    with tabs[1]:
        st.caption("From target-direction only up to the full representation.")
        if buttons.action("experiment", "r5_features", label="≣ Compare feature sets"):
            with st.spinner("Training %d feature sets…" % len(FEATURE_SETS)):
                store["feature_rows"] = experiments.feature_set_experiment(
                    episodes=episodes, difficulty=settings["difficulty"],
                    layout_overrides=_layout_overrides(settings),
                    environment_parameters=_environment_parameters(settings))
        _show_experiment(store.get("feature_rows"), "feature set")

    with tabs[2]:
        st.caption("More information against more features to learn about.")
        if buttons.action("experiment", "r5_radar", label="≣ Compare radar ranges"):
            with st.spinner("Training three radar ranges…"):
                store["radar_rows"] = experiments.radar_range_experiment(
                    episodes=episodes, difficulty=settings["difficulty"])
        _show_experiment(store.get("radar_rows"), "radar range")

    with tabs[3]:
        st.warning("This is a **distribution shift**, not just unseen layouts: the "
                   "agent trains on Easy and is then tested on Medium and Hard "
                   "warehouses. A collapse here is a real result, not a bug.",
                   icon="⚠")
        if buttons.action("experiment", "r5_stress", label="≣ Run stress test"):
            with st.spinner("Training and testing across difficulties…"):
                store["stress_rows"] = experiments.generalisation_stress_test(
                    episodes=episodes)
        _show_experiment(store.get("stress_rows"), "evaluation difficulty")

    with tabs[4]:
        st.caption("What the learned policy is being compared against. The greedy "
                   "target policy walks straight at the objective and ignores the "
                   "racking, which is exactly the behaviour the features have to "
                   "improve on.")
        if buttons.action("experiment", "r5_base", label="≣ Run baselines"):
            with st.spinner("Running the baseline policies…"):
                store["baseline_rows"] = experiments.baseline_policies(
                    _pool(settings, "test", settings["test_layouts"]))
        _show_experiment(store.get("baseline_rows"), "baseline")

    with tabs[5]:
        parameter = st.selectbox("Setting to sweep",
                                 list(experiments.SWEEPABLE_PARAMETERS.keys()),
                                 format_func=lambda name:
                                 experiments.READABLE_NAMES.get(name, name),
                                 key="r5_sweep_parameter")
        if buttons.action("experiment", "r5_sweep", label="≣ Run sweep"):
            with st.spinner("Sweeping %s…" % parameter):
                store["sweep_rows"] = experiments.parameter_experiment(
                    parameter, episodes=episodes,
                    difficulty=settings["difficulty"],
                    layout_overrides=_layout_overrides(settings),
                    environment_parameters=_environment_parameters(settings))
        _show_experiment(store.get("sweep_rows"),
                         experiments.READABLE_NAMES.get(parameter, parameter))


def _show_experiment(rows, title):
    if not rows:
        st.caption("Press the button above to run this experiment.")
        return

    def percent(value):
        return "—" if value != value else "%.0f%%" % (value * 100)

    st.dataframe(pd.DataFrame([{
        "Value": str(row["value"]),
        "Training": percent(row.get("mean_training_success", float("nan"))),
        "Validation (unseen)": percent(row.get("mean_validation_success",
                                               float("nan"))),
        "Test (unseen)": "%s ± %s" % (percent(row.get("mean_test_success", 0.0)),
                                      percent(row.get("std_test_success", 0.0))),
        "Test return": "%.1f" % row.get("mean_test_return", 0.0),
        "Path efficiency": ("%.2f" % row["mean_test_efficiency"]
                            if row.get("mean_test_efficiency", float("nan"))
                            == row.get("mean_test_efficiency", float("nan"))
                            else "—"),
        "Robot collisions": percent(row.get("mean_test_caught", 0.0)),
        "Timeouts": percent(row.get("mean_test_timeout", 0.0)),
        "Training time": ("%.1f s" % row["mean_runtime"]
                          if row.get("mean_runtime") else "—"),
    } for row in rows]), hide_index=True)

    columns = st.columns(2)
    with columns[0]:
        st.pyplot(room5_plots.experiment_chart(rows, title=title))
    with columns[1]:
        st.pyplot(room5_plots.generalisation_chart(rows))

    best = experiments.best_performing_row(rows, "mean_test_success")
    if best:
        st.info("Best-performing configuration among the tested settings: **%s**, "
                "with %.0f%% success on the unseen test layouts. This is not "
                "necessarily an optimal setting — only the best of the ones tried."
                % (best["value"], best.get("mean_test_success", 0.0) * 100))


def _explanation(store, settings):
    st.markdown("### What is actually happening in this room")

    first, second, third = st.columns(3)
    with first:
        st.markdown("""
**Why one Q-table cannot do this**

A table keyed on absolute position learns *this* warehouse. The moment the crates
move it is worthless — the same cell now means something completely different, and
a cell in a layout it has never seen has no entry at all. Room 5 changes the
warehouse every episode, so memorising one is not an option.
        """)
    with second:
        st.markdown("""
**What partial observability means**

R-5 does not receive the warehouse. It receives eight radar distances, four
robot-radar distances, where its objective is, and the mission stage. That is
genuinely less than the full state: two different spots in two different
warehouses can produce **identical** readings, and no linear model can tell them
apart. This is a partially observable problem, not a tidy MDP, and the limitations
below follow from that.
        """)
    with third:
        st.markdown("""
**Why these features transfer**

Nothing in the feature vector mentions a position, a layout or a seed. "A shelf two
cells north" and "the objective is south-east and far" mean the same thing in every
warehouse, so weights learned in one apply in the next. That is the whole claim,
and the layout-count experiment is what tests it.
        """)

    st.markdown("**The approximate action-value function:**")
    st.latex(Q_HAT)
    st.markdown("**The semi-gradient Q-Learning update:**")
    st.latex(Q_RULE)
    st.markdown("""
**Why this is off-policy.** The target uses `max Q(s',a')` — the best action
available next — while the behaviour is still taking random exploratory steps. The
policy being evaluated and the policy being followed are not the same one.
    """)

    _worked_example(store)

    st.markdown("#### Known limitations, stated plainly")
    st.markdown("""
* **The observation is not Markov.** Local radar cannot distinguish two identical
  looking corners of two different warehouses, so some situations are genuinely
  ambiguous and no amount of training fixes that.
* **A linear model has limits.** It can learn "go towards the objective, and turn
  when something is close". It cannot plan around a long dead-end aisle that
  requires backtracking, because that needs memory of where it has been.
* **Patrol timing cannot be inferred.** The dynamic radar says how close a robot
  is, not which way it is going or when it will arrive.
* **Performance drops on harder layouts.** The default is Easy because that is
  where the agent demonstrably generalises. Medium and Hard are measurably worse,
  and the stress test shows it rather than hiding it.
    """)

    with st.expander("The observation, the actions and the rewards in detail"):
        extractor = FeatureExtractor(groups=FEATURE_SETS[settings["feature_set"]])
        st.markdown("""
**Observation** — %d numbers, all local:

| Group | Contents |
|---|---|
| static radar | 8 normalised distances to the nearest shelf, crate or wall |
| dynamic radar | 4 normalised distances to the nearest maintenance robot |
| objective | relative row, relative column, normalised distance |
| stage | 0 terminal offline, 1 exit unlocked |
| previous action | one of five |

**Features** — each group gets its own small tile coder and the active indices are
concatenated, then offset per action. The chosen set (**%s**) gives
**%s** weights, of which **%d** are active for any one state-action pair.

**Actions** — up, down, left, right and **wait**. Waiting matters because the
robots and conveyors move whether R-5 does or not.

**The order of a step** — identical in training, evaluation and replay: R-5 moves;
racking blocks it; a conveyor may shove it one more square; the robots each advance
one step; collisions are checked (same cell *or* a straight swap); then the
terminal, charger and exit are processed.

**Rewards**

| Event | Reward |
|---|---|
| each step | %.1f |
| blocked by racking | %.1f |
| carried by a conveyor | %.1f extra |
| activate the terminal | +%.0f |
| reach the gate before the terminal | %.1f |
| struck by a maintenance robot | %.0f, episode ends |
| out of time | %.0f |
| reach the unlocked gate | **+%.0f**, episode ends |
| charging station, once | +%.0f |

Progress shaping adds %.1f x the reduction in distance to the current objective. It
is deliberately small enough that a round trip always loses money, so it cannot be
farmed — there is a test for that.
        """ % (8 + 4 + 3 + 1 + 5, settings["feature_set"],
               "{:,}".format(extractor.total_features),
               len(extractor.active_features(
                   Room5Env(layout=_current_layout(
                       store, settings)).reset(), 0)),
               -1.0, -5.0, -1.0, 40.0, -2.0, -100.0, -30.0, 200.0, 5.0, 1.5))

    with st.expander("Layout generation and the seed split"):
        layout = _current_layout(store, settings)
        st.markdown("""
Layouts are generated from a seed, so a seed **is** a layout. Shelves go down in
aisles, then crates, conveyors, a charger and the patrol routes. A layout is only
accepted once breadth-first search confirms **both** legs are walkable —
start to terminal, and terminal to exit — so an impossible warehouse can never
reach the agent.

| Split | Seeds | Trained on |
|---|---|---|
| Training | %d onwards | **yes** |
| Validation | %d onwards | no |
| Test | %d onwards | no |

The current layout is seed **%d** (%s split), %dx%d, obstacle density %.2f,
%d maintenance robot(s), radar range %d. The shortest possible mission is **%s
steps** — used only to compute path efficiency, never given to the agent.
        """ % (L.TRAINING_SEED_BASE, L.VALIDATION_SEED_BASE, L.TEST_SEED_BASE,
               layout.seed, layout.split, layout.size, layout.size,
               layout.obstacle_density(), len(layout.robot_templates),
               layout.radar_range, layout.shortest_mission_length()))
        st.code("\n".join("".join(row) for row in layout.grid), language="text")
        st.markdown("   ".join("`%s` %s" % (tile, name) for tile, name in L.LEGEND))


def _worked_example(store):
    result = store.get("primary")
    example = result.get("example_update") if result else None
    if not example:
        st.caption("Train the agent to see one of its own updates worked through "
                   "with real numbers.")
        return

    features = example["active_features"]
    observation = example["observation"]
    st.markdown("**One real update from this training session:**")
    st.markdown("""
Taken from the final episode of the %(algorithm)s run.

| Term | Value |
|---|---|
| static radar | `%(radar)s` |
| objective (drow, dcol, distance) | `%(target)s` |
| stage | `%(stage)d` |
| action `a` | `%(action)s` |
| active feature indices | `%(features)s`%(more)s |
| `Q(s,a)` before | `%(before).4f` |
| reward `r` | `%(reward)+.2f` |
| `max Q(s',·)` | `%(next_q).4f` |
| TD target | `%(target_value).4f` |
| TD error `delta` | `%(delta)+.4f` |
| change applied to each active weight | `%(change)+.5f` |

Only those %(count)d weights moved.
    """ % {
        "algorithm": example["algorithm"],
        "radar": ", ".join("%.2f" % value
                           for value in observation["static_radar"]),
        "target": ", ".join("%.2f" % value for value in observation["target"]),
        "stage": int(observation["stage"]),
        "action": ACTION_NAMES[example["action"]],
        "features": ", ".join(str(index) for index in features[:12]),
        "more": " …" if len(features) > 12 else "",
        "before": example["q_before"],
        "reward": example["reward"],
        "next_q": example["max_next_q"],
        "target_value": example["target"],
        "delta": example["td_error"],
        "change": example["weight_change"],
        "count": len(features),
    })
    if example["done"]:
        st.caption("This step ended the episode, so the target is just the reward.")
