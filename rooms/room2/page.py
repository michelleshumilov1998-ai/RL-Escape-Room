"""The Streamlit page for Room 2 — the Broken Bridge Sector.

Same layout as Room 1, so the two rooms read as the same game:

  * the laboratory nameplate, the navigation rail and the inventory bar
  * the action buttons
  * the sector itself, with the information cards beside it
  * the training graphs
  * the experiments
  * an explanation of what SARSA is doing, with a worked update from this session

Nothing is trained by accident. Training, running and experiments only happen
when the matching button is pressed, and the results live in `st.session_state`,
so moving a slider never throws a trained model away.
"""

import pandas as pd
import streamlit as st

from core import actions
from plots import room2_plots
from renderers import fallback_renderer, iso_canvas, room2_scene
from rooms.room2 import experiments, map_data, sarsa_agent, simulate, storage
from rooms.room2.environment import Room2Env
from ui import buttons, cards, game_state, inventory, nav, theme

ROOM_NUMBER = 2

SARSA_RULE = (r"Q(s,a) \leftarrow Q(s,a) + \alpha\big[\,r + \gamma\,Q(s',a')"
              r" - Q(s,a)\,\big]")
Q_RULE = (r"Q(s,a) \leftarrow Q(s,a) + \alpha\big[\,r + \gamma\,\max_{a'}Q(s',a')"
          r" - Q(s,a)\,\big]")

ALGORITHM_CHOICES = [sarsa_agent.SARSA, sarsa_agent.Q_LEARNING,
                     "Compare both"]


# ----------------------------------------------------------------------
# The controls
# ----------------------------------------------------------------------

def _sidebar():
    """Draw the sidebar and return the current settings as a dictionary."""
    sidebar = st.sidebar
    sidebar.markdown('<div class="r5-label">Room 2 controls</div>',
                     unsafe_allow_html=True)

    algorithm = sidebar.selectbox(
        "Algorithm", ALGORITHM_CHOICES, index=0, key="r2_algorithm",
        help="SARSA is this room's method. Q-Learning is here so the difference "
             "between them can be measured rather than just described.")

    sidebar.markdown("**Learning**")
    alpha = sidebar.slider("Learning rate (alpha)", 0.01, 1.00, 0.10, 0.01,
                           key="r2_alpha",
                           help="How much of each new estimate to keep.")
    gamma = sidebar.slider("Discount factor (gamma)", 0.50, 1.00, 0.99, 0.01,
                           key="r2_gamma",
                           help="How much a future reward is worth now. This room "
                                "needs a high value: the thing being avoided is "
                                "rare and expensive.")
    epsilon_start = sidebar.slider("Initial epsilon", 0.00, 1.00, 1.00, 0.05,
                                   key="r2_eps0")
    epsilon_min = sidebar.slider("Minimum epsilon", 0.00, 0.50, 0.10, 0.01,
                                 key="r2_epsmin",
                                 help="SARSA and Q-Learning agree once exploration "
                                      "stops, so leaving a floor here is what "
                                      "keeps the difference visible.")
    epsilon_decay = sidebar.select_slider(
        "Epsilon decay", options=[0.980, 0.990, 0.995, 0.999, 1.000],
        value=0.995, format_func=lambda value: "%.3f" % value, key="r2_decay")
    episodes = sidebar.slider("Episodes", 200, 5000, 1500, 100, key="r2_episodes")
    q_init = sidebar.slider("Initial Q value", 0.0, 150.0, 120.0, 10.0,
                            key="r2_qinit",
                            help="Starting the table high pushes the agent to try "
                                 "every action before settling — without it the "
                                 "safe route is found first and never questioned.")
    seed = sidebar.number_input("Random seed", min_value=0, max_value=99999,
                                value=0, step=1, key="r2_seed")

    sidebar.markdown("**State representation**")
    include_bridge_state = sidebar.checkbox(
        "Include collapsed bridges in the state", value=False,
        key="r2_bridge_state",
        help="Off is the assignment's state (row, col, has_keycard). On adds which "
             "bridges have already fallen, which makes the room Markovian — and is "
             "what makes SARSA and Q-Learning disagree.")

    sidebar.markdown("**Rewards**")
    step_cost = sidebar.slider("Each step", -5, 0, -1, 1, key="r2_step")
    wall_penalty = sidebar.slider("Wall collision", -20, 0, -5, 1, key="r2_wall")
    pit_penalty = sidebar.slider("Fall into a pit", -300, -10, -100, 10,
                                 key="r2_pit")
    keycard_bonus = sidebar.slider("Collect the keycard", 0, 100, 25, 5,
                                   key="r2_key")
    exit_reward = sidebar.slider("Reach the exit", 10, 300, 100, 10, key="r2_exit")
    collapsing_penalty = sidebar.slider("Cross a collapsing bridge", -20, 0, -2, 1,
                                        key="r2_collapse")

    sidebar.markdown("**Simulation and animation**")
    run_seed = sidebar.number_input("Run seed", min_value=0, max_value=99999,
                                    value=1, step=1, key="r2_runseed")
    run_epsilon = sidebar.slider("Exploration while running", 0.00, 0.50, 0.00,
                                 0.05, key="r2_runeps",
                                 help="Leave at 0 to watch the learned policy. "
                                      "Raise it to watch the behaviour policy — "
                                      "the one SARSA was actually valuing.")
    speed = sidebar.select_slider("Animation speed", options=[0.5, 1.0, 2.0, 4.0],
                                  value=1.0, format_func=lambda v: "%.1gx" % v,
                                  key="r2_speed")
    evaluation_episodes = sidebar.slider("Episodes per measurement", 10, 100, 30,
                                         10, key="r2_eval")

    sidebar.markdown("**Views**")
    view_mode = sidebar.radio("View", ["Gameplay", "Analysis"], horizontal=True,
                              key="r2_view")
    window = sidebar.slider("Moving-average window", 5, 200, 50, 5, key="r2_window")
    slice_keycard = sidebar.checkbox("Policy view: keycard collected", value=True,
                                     key="r2_slice_key")

    nav.sidebar_navigation(ROOM_NUMBER)

    return {
        "algorithm": algorithm, "alpha": alpha, "gamma": gamma,
        "epsilon_start": epsilon_start, "epsilon_min": epsilon_min,
        "epsilon_decay": epsilon_decay, "episodes": episodes, "q_init": q_init,
        "seed": int(seed), "include_bridge_state": include_bridge_state,
        "step_cost": step_cost, "wall_penalty": wall_penalty,
        "pit_penalty": pit_penalty, "keycard_bonus": keycard_bonus,
        "exit_reward": exit_reward, "collapsing_penalty": collapsing_penalty,
        "run_seed": int(run_seed), "run_epsilon": run_epsilon, "speed": speed,
        "evaluation_episodes": evaluation_episodes, "view_mode": view_mode,
        "window": window, "slice_keycard": slice_keycard,
    }


def _environment(settings):
    """Build an environment from the current settings."""
    return Room2Env(include_bridge_state=settings["include_bridge_state"],
                    step_cost=settings["step_cost"],
                    wall_penalty=settings["wall_penalty"],
                    pit_penalty=settings["pit_penalty"],
                    keycard_bonus=settings["keycard_bonus"],
                    exit_reward=settings["exit_reward"],
                    collapsing_penalty=settings["collapsing_penalty"])


def _hyperparameters(settings):
    return {
        "alpha": settings["alpha"], "gamma": settings["gamma"],
        "epsilon_start": settings["epsilon_start"],
        "epsilon_min": settings["epsilon_min"],
        "epsilon_decay": settings["epsilon_decay"],
        "q_init": settings["q_init"], "seed": settings["seed"],
    }


def _fingerprint(settings):
    """The settings a trained model actually depends on."""
    return (settings["algorithm"], settings["episodes"],
            tuple(sorted(_hyperparameters(settings).items())),
            settings["include_bridge_state"], settings["step_cost"],
            settings["wall_penalty"], settings["pit_penalty"],
            settings["keycard_bonus"], settings["exit_reward"],
            settings["collapsing_penalty"])


# ----------------------------------------------------------------------
# The actions behind the buttons
# ----------------------------------------------------------------------

def _train(store, settings):
    """Train, showing a progress bar because this actually takes a moment."""
    algorithms = ([sarsa_agent.SARSA, sarsa_agent.Q_LEARNING]
                  if settings["algorithm"] == "Compare both"
                  else [settings["algorithm"]])

    results = []
    bar = st.progress(0.0, text="Training…")
    for index, algorithm in enumerate(algorithms):
        env = _environment(settings)

        def report(fraction, index=index, algorithm=algorithm):
            share = (index + fraction) / len(algorithms)
            bar.progress(min(1.0, share), text="Training %s…" % algorithm)

        results.append(sarsa_agent.train(
            env, algorithm=algorithm, episodes=settings["episodes"],
            progress=report, **_hyperparameters(settings)))
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
    """Draw the whole Room 2 page."""
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

    nav.room_header(ROOM_NUMBER, status, note,
                    algorithm=settings["algorithm"])
    nav.rail(ROOM_NUMBER)
    inventory.bar()
    nav.progress_line()

    if stale:
        st.info("The settings have changed since the agent was trained. Press "
                "**Train** to learn again with the new numbers — nothing is "
                "retrained until you do.", icon="ℹ️")

    _action_buttons(store, settings)

    with st.container(key="room2_main"):
        left, right = st.columns([2.05, 1], gap="medium")
        with left:
            _sector(store, settings)
        with right:
            _cards(store, settings, status)

    st.divider()
    _graphs(store, settings)
    st.divider()
    _experiments(store, settings)
    st.divider()
    _explanation(store, settings)


def _action_buttons(store, settings):
    """The standard actions, two rows of four so no label has to wrap."""
    top = st.columns(4)
    with top[0]:
        if buttons.action("train", "r2"):
            _train(store, settings)
            st.rerun()
    with top[1]:
        if buttons.action("run", "r2", disabled=not store.get("primary")):
            _run_agent(store, settings)
            st.rerun()
    with top[2]:
        if buttons.action("replay", "r2", disabled=not store.get("run")):
            store["playing"] = True
            st.rerun()
    with top[3]:
        playing = store.get("playing", True)
        label = "‖ Pause" if playing else "▶ Resume"
        if buttons.action("pause", "r2", label=label,
                          disabled=not store.get("run")):
            store["playing"] = not playing
            st.rerun()

    bottom = st.columns(4)
    with bottom[0]:
        if buttons.action("reset", "r2", disabled=not store.get("run")):
            store["run"] = None
            store["playing"] = True
            st.rerun()
    with bottom[1]:
        if buttons.action("experiment", "r2"):
            store["show_experiments"] = True
            st.rerun()
    with bottom[2]:
        if buttons.action("save", "r2", label="↓ Save Model",
                          disabled=not store.get("primary")):
            env = _environment(settings)
            path = storage.save_model("room2_latest", env, store["primary"])
            st.success("Saved to %s" % path.split("/")[-1])
    with bottom[3]:
        if buttons.action("load", "r2", label="↑ Load Model"):
            try:
                loaded = storage.load_model("room2_latest")
            except storage.ModelFileError as problem:
                st.error(str(problem))
            else:
                _adopt_loaded_model(store, loaded)
                st.success("Loaded a %s model trained for %d episodes."
                           % (loaded["algorithm"], loaded.get("episodes", 0)))
                st.rerun()


def _adopt_loaded_model(store, loaded):
    """Turn a loaded file back into something the page can use.

    The policy is rebuilt against a live environment whose state representation
    matches the file, so a model saved with the bridge state cannot be read back
    against the shorter state by accident.
    """
    env = Room2Env(include_bridge_state=loaded.get("include_bridge_state", False))
    loaded["policy"] = sarsa_agent.greedy_policy(loaded["q"], env)
    loaded.setdefault("runtime_seconds", 0.0)
    loaded.setdefault("history", {})
    store["primary"] = loaded
    store["results"] = [loaded]
    store["run"] = None
    store["batch"] = None
    store["fingerprint"] = None


def _sector(store, settings):
    """The sector itself: the 3D view, or the top-down analysis views."""
    run = store.get("run")
    frames = run["frames"] if run else None

    if settings["view_mode"] == "Analysis":
        st.markdown("#### Analysis view")
        st.caption("Top-down maps used to read the plan. The sector, the policy "
                   "and the graphs all use the same grid orientation.")
        path = [(frame["row"], frame["col"]) for frame in frames] if frames else None
        st.pyplot(room2_plots.sector_map(highlight_path=path))
        with st.expander("Isometric still frame (fallback renderer)"):
            st.caption("The same sector drawn in Python. This is what is shown if "
                       "the interactive canvas cannot run in the browser.")
            still = room2_scene.build_scene(frames=frames)
            st.pyplot(fallback_renderer.render_frame(still, still["frames"][-1]))
        return

    scene = room2_scene.build_scene(
        frames=frames, status="DAMAGED",
        options={"speed": settings["speed"],
                 "autoplay": store.get("playing", True) and bool(frames)})
    iso_canvas.render(scene)

    if not store.get("primary"):
        st.caption("R-5 is powering up on the start platform. Press **Train** to "
                   "learn the sector, then **Run Agent** to watch it.")
    elif not run:
        st.caption("A policy is ready. Press **Run Agent** to send R-5 in.")
    else:
        st.caption("Playback controls are inside the sector, so using them never "
                   "reloads the page. Drag to pan, scroll to zoom.")


def _cards(store, settings, status):
    """The information cards and the event log."""
    run = store.get("run")
    result = store.get("primary")
    batch = store.get("batch")
    frames = run["frames"] if run else None
    frame = frames[-1] if frames else None

    routes = map_data.ROUTE_STEPS_WITH_KEYCARD
    theme.html(cards.mission_card(
        "Exit door at row %d, col %d" % map_data.exit_cell(),
        requirements=[
            ("Security keycard collected", bool(run and run["keycard_collected"])),
            ("Exit door reached", bool(run and run["success"])),
        ],
        complete=bool(run and run["success"])))

    # Room 2's agent-state card is written by hand, because the keycard and the
    # bridges are different facts from Room 1's battery and slipping.
    if frame is None:
        state_rows = [("Position", "—"), ("Keycard", "—")]
    else:
        state_rows = [
            ("Position", "row %d, col %d" % (frame["row"], frame["col"]),
             cards.TONE_INFO),
            ("Keycard", "YES" if frame["has_keycard"] else "no",
             cards.TONE_GOOD if frame["has_keycard"] else cards.TONE_NEUTRAL),
            ("Chosen action", "%s %s" % (
                actions.direction_arrow(frame["action"]) if frame["action"] else "·",
                frame["action_name"])),
            ("Bridges collapsed", str(len(frame["collapsed"])),
             cards.TONE_WARN if frame["collapsed"] else cards.TONE_NEUTRAL),
            ("Bridge route", "%d steps" % routes["span"]),
            ("Walkway route", "%d steps" % routes["walkway"]),
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
            ("Pit falls", str(run["pit_falls"]),
             cards.TONE_BAD if run["pit_falls"] else cards.TONE_NEUTRAL),
            ("Wall collisions", str(run["wall_collisions"]),
             cards.TONE_WARN if run["wall_collisions"] else cards.TONE_NEUTRAL),
            ("Bridges crossed", "%d  (%+d each)" % (run["bridges_collapsed"],
                                                    settings["collapsing_penalty"])
             if run["bridges_collapsed"] else "0"),
            ("Keycard bonus", "%+d collected" % settings["keycard_bonus"]
             if run["keycard_collected"] else "not collected",
             cards.TONE_GOOD if run["keycard_collected"] else cards.TONE_NEUTRAL),
        ]))

    theme.html(cards.algorithm_card(
        name=result["algorithm"] if result else settings["algorithm"],
        update_rule="Q(s,a) += a[r + g Q(s',a') - Q(s,a)]",
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
            ("state", "with bridges" if settings["include_bridge_state"]
             else "row, col, keycard"),
        ]))

    status_rows = []
    if run:
        status_rows = [
            ("Steps taken", str(run["steps"])),
            ("Outcome", run["status"]),
            ("Route used", simulate.describe_route(run["frames"])),
            ("Seed", str(run["seed"])),
        ]
    if batch:
        status_rows.append(("Success over %d runs" % batch["episodes"],
                            "%.0f%%" % (batch["success_rate"] * 100)))
        status_rows.append(("Mean return", "%.1f ± %.1f"
                            % (batch["mean_return"], batch["std_return"])))
        status_rows.append(("Pit-fall rate", "%.0f%%"
                            % (batch["pit_fall_rate"] * 100)))
    theme.html(cards.room_status_card(status, status_rows))

    st.markdown('<div class="r5-label" style="margin-top:10px">Event log</div>',
                unsafe_allow_html=True)
    theme.html(cards.event_log(frames or []))


def _graphs(store, settings):
    """The ten training graphs the assignment asks for."""
    st.markdown("### Training results")
    results = store.get("results") or []
    if not results or not results[0].get("history", {}).get("reward"):
        st.caption("Train the agent to produce the graphs.")
        return

    window = settings["window"]

    first, second = st.columns(2)
    with first:
        st.pyplot(room2_plots.episode_reward(results, window))
    with second:
        st.pyplot(room2_plots.moving_average_reward(results, window))

    third, fourth = st.columns(2)
    with third:
        st.pyplot(room2_plots.episode_length(results, window))
    with fourth:
        st.pyplot(room2_plots.success_rate(results, window))

    fifth, sixth = st.columns(2)
    with fifth:
        st.pyplot(room2_plots.epsilon_decay(results))
    with sixth:
        st.pyplot(room2_plots.q_value_convergence(results))

    seventh, eighth = st.columns(2)
    with seventh:
        st.pyplot(room2_plots.pit_falls(results, window))
    with eighth:
        st.pyplot(room2_plots.bridge_usage(results, window))

    ninth, tenth = st.columns(2)
    with ninth:
        st.pyplot(room2_plots.keycard_rate(results, window))
    with tenth:
        st.pyplot(room2_plots.training_time(results))

    with st.expander("The learned policy, cell by cell"):
        mask = 0 if settings["include_bridge_state"] else None
        st.pyplot(room2_plots.policy_arrows(
            results[0]["policy"], has_keycard=settings["slice_keycard"],
            collapsed_mask=mask))
        st.caption("The arrow is the action with the highest learned value. With "
                   "the bridge state switched on, this is the slice where no "
                   "bridge has collapsed yet.")


def _experiments(store, settings):
    """The parameter sweeps and the two headline comparisons."""
    st.markdown("### Parameter experiments")
    st.caption("Every row trains from scratch several times with different seeds "
               "and reports mean ± standard deviation. A sweep can only identify "
               "the best-performing value among the values that were tested.")

    sweep_tab, algorithm_tab, state_tab = st.tabs(
        ["Parameter sweep", "SARSA vs Q-Learning", "State representation"])

    with sweep_tab:
        parameter = st.selectbox("Setting to sweep",
                                 list(experiments.SWEEPABLE_PARAMETERS.keys()),
                                 format_func=lambda name:
                                 experiments.READABLE_NAMES.get(name, name),
                                 key="r2_sweep_parameter")
        columns = st.columns(2)
        with columns[0]:
            episodes = st.slider("Episodes per run", 200, 2000, 800, 100,
                                 key="r2_sweep_episodes")
        with columns[1]:
            repeats = st.slider("Seeds per value", 1, 5, 3, 1,
                                key="r2_sweep_repeats")
        if buttons.action("experiment", "r2_sweep", label="≣ Run sweep"):
            with st.spinner("Training %d configurations…"
                            % (len(experiments.SWEEPABLE_PARAMETERS[parameter])
                               * repeats)):
                store["sweep_rows"] = experiments.parameter_experiment(
                    parameter,
                    environment_parameters={
                        "include_bridge_state": settings["include_bridge_state"]},
                    episodes=episodes, repeats=repeats)
        _show_experiment(store.get("sweep_rows"),
                         experiments.READABLE_NAMES.get(parameter, parameter), "")

    with algorithm_tab:
        st.caption("Both methods, same room, same settings. With the assignment's "
                   "state they behave the same; with the bridge state switched on "
                   "they do not.")
        if buttons.action("compare", "r2_algo", label="⇄ Compare algorithms"):
            with st.spinner("Training SARSA and Q-Learning…"):
                comparison = experiments.compare_algorithms(
                    environment_parameters={
                        "include_bridge_state": settings["include_bridge_state"]},
                    episodes=1200, repeats=3)
                store["algorithm_rows"] = [
                    dict(row, value=name) for name, row in comparison.items()]
        _show_experiment(store.get("algorithm_rows"), "algorithm", "")

    with state_tab:
        st.caption("The headline experiment of this room. It trains both methods "
                   "twice — once with the assignment's state and once with the "
                   "collapsed bridges added — and shows that what the agent is "
                   "allowed to observe decides whether the two methods differ "
                   "at all.")
        if buttons.action("experiment", "r2_state", label="≣ Run state comparison"):
            with st.spinner("Training four configurations…"):
                store["state_rows"] = experiments.compare_state_representations(
                    episodes=1200, repeats=3)
        _show_experiment(store.get("state_rows"), "state representation", "")


def _show_experiment(rows, title, explanation):
    """One experiment's table, chart and best-performing row."""
    if not rows:
        st.caption("Press the button above to run this experiment.")
        return
    if explanation:
        st.caption(explanation)

    table = pd.DataFrame([{
        "Value": str(row["value"]),
        "Mean reward": "%.1f ± %.1f" % (row["mean_reward"], row["std_reward"]),
        "Success": "%.0f%%" % (row["mean_success"] * 100),
        "Steps": "%.1f ± %.1f" % (row["mean_steps"], row["std_steps"]),
        "Pit falls": "%.0f%%" % (row["mean_pit_falls"] * 100),
        "Bridge use": "%.0f%%" % (row["mean_span_use"] * 100),
        "Greedy return": "%.1f ± %.1f" % (row["mean_greedy_return"],
                                          row["std_greedy_return"]),
        "Training time": "%.1f s" % row["mean_runtime"],
        "Route": row["route"],
    } for row in rows])
    st.dataframe(table, hide_index=True)

    columns = st.columns(2)
    with columns[0]:
        st.pyplot(room2_plots.experiment_chart(rows, title=title,
                                               ylabel="mean reward"))
    with columns[1]:
        st.pyplot(room2_plots.algorithm_comparison_chart(rows))

    best = experiments.best_performing_row(rows, "mean_reward")
    if best:
        st.info("Best-performing value among the tested configurations: **%s**, "
                "with a mean reward of %.1f ± %.1f. This is not necessarily an "
                "optimal value — only the best of the ones tried."
                % (best["value"], best["mean_reward"], best["std_reward"]))


def _explanation(store, settings):
    """What SARSA is doing, and why it behaves the way it does here."""
    st.markdown("### What is actually happening in this room")

    first, second, third = st.columns(3)
    with first:
        st.markdown("""
**Why SARSA suits this room**

R-5 has no model of the sector. It cannot look up what is on the other side of a
bridge; it has to step there and find out. That rules out the planning used in
Room 1 and calls for a method that learns from experience — and because the
sector contains a drop that ends the run, it calls for one that takes its own
mistakes into account.
        """)
    with second:
        st.markdown("""
**On-policy learning**

SARSA updates towards `Q(s',a')`, where `a'` is the action it is *actually going
to take next* — chosen by the same epsilon-greedy policy, exploration and all. So
the value it learns is the value of the policy it is really following, holes
included. That is what "on-policy" means.

Q-Learning updates towards `max Q(s',a')` instead: the value of behaving
perfectly from the next state on. It is **off-policy**, and it never charges
itself for the random steps it is still taking.
        """)
    with third:
        st.markdown("""
**Why that avoids dangerous shortcuts**

The bridge is worth more than the walkway when crossed perfectly (%+d against
%+d). Q-Learning prices it at that, so it takes it — and falls in, because it is
still exploring. SARSA prices in those falls, which drags the bridge below the
walkway, and walks over the top instead.

The catch is that this only shows up when the agent can *see* which bridges have
already gone. With the assignment's state it cannot, and both methods avoid the
bridge for a different reason entirely — see the state-representation experiment
above.
        """ % (map_data.perfect_return("span"),
               map_data.perfect_return("walkway")))

    st.markdown("**The SARSA update:**")
    st.latex(SARSA_RULE)
    st.markdown("**Q-Learning, for comparison — one term differs:**")
    st.latex(Q_RULE)

    _worked_example(store)

    with st.expander("The state, the actions and the rewards in detail"):
        env = _environment(settings)
        st.markdown("""
**State** — `(row, col, has_keycard)`%(extra)s, giving %(count)d states.
`has_keycard` is part of it because the exit door stays locked until the card is
held, so the same cell means two different things.

**Actions** — `UP`, `DOWN`, `LEFT`, `RIGHT`. The sector is **deterministic**:
an action always moves R-5 the way it was aimed. Nothing slips. The danger here
comes from the agent's own exploration, not from the environment.

**Reward** — every transition pays the step cost, and event rewards are added on
top, the same convention Room 1 uses:

| Event | Total |
|---|---|
| each step | %(step)d |
| wall collision, or the locked door | %(step)d %(wall)+d = **%(wall_total)d** |
| cross a collapsing bridge | %(step)d %(collapse)+d = **%(collapse_total)d** |
| collect the keycard | %(step)d %(key)+d = **%(key_total)+d** |
| reach the exit | %(step)d %(exit)+d = **%(exit_total)+d** |
| fall into the shaft | %(step)d %(pit)+d = **%(pit_total)d** |

**Collapsing bridges** — crossing one breaks it. For the rest of that episode the
cell behaves as a pit, so the sector's layout changes while the episode runs and
`reset` rebuilds it. There are %(collapsing)d of them, and they are the only way
across the shaft on the lower route.

**Pits** — entering one ends the episode immediately. %(pits)d of the shaft cells
can actually be fallen into; the rest are sealed off and are scenery.

**The exit door** — locked until the keycard is held. Walking into it without the
card costs the wall penalty and R-5 stays put.
        """ % {
            "extra": (", plus a bit per collapsing bridge saying whether it has "
                      "already fallen" if settings["include_bridge_state"] else ""),
            "count": len(env.all_states()),
            "step": settings["step_cost"],
            "wall": settings["wall_penalty"],
            "wall_total": settings["step_cost"] + settings["wall_penalty"],
            "collapse": settings["collapsing_penalty"],
            "collapse_total": settings["step_cost"] + settings["collapsing_penalty"],
            "key": settings["keycard_bonus"],
            "key_total": settings["step_cost"] + settings["keycard_bonus"],
            "exit": settings["exit_reward"],
            "exit_total": settings["step_cost"] + settings["exit_reward"],
            "pit": settings["pit_penalty"],
            "pit_total": settings["step_cost"] + settings["pit_penalty"],
            "collapsing": len(map_data.collapsing_cells()),
            "pits": len(map_data.enterable_pits()),
        })

    with st.expander("The sector map and legend"):
        st.code("\n".join("%d  %s" % (index, row)
                          for index, row in enumerate(map_data.ROOM_MAP)),
                language="text")
        st.markdown("   ".join("`%s` %s" % (tile, name)
                               for tile, name in map_data.LEGEND))
        st.markdown(
            "**Routes:** lower bridge %d steps, crossing %d collapsing bridges "
            "with the shaft on both sides · upper walkway %d steps with nothing "
            "to fall into anywhere on it."
            % (map_data.ROUTE_STEPS_WITH_KEYCARD["span"],
               map_data.ROUTE_COLLAPSING_CROSSINGS["span"],
               map_data.ROUTE_STEPS_WITH_KEYCARD["walkway"]))


def _worked_example(store):
    """One real update from the session, with the numbers filled in."""
    result = store.get("primary")
    example = result.get("example_update") if result else None
    if not example:
        st.caption("Train the agent to see one of its own updates worked through "
                   "with real numbers.")
        return

    is_sarsa = example["algorithm"] == sarsa_agent.SARSA
    next_action_name = (actions.direction_name(example["next_action"])
                        if example["next_action"] is not None else "—")

    st.markdown("**One update from this training session, worked through:**")
    st.markdown("""
Taken from the final episode of the %(algorithm)s run.

| Term | Value |
|---|---|
| state `s` | `%(state)s` |
| action `a` | `%(action)s` |
| reward `r` | `%(reward)+.0f` |
| next state `s'` | `%(next_state)s` |
| %(next_label)s | `%(next_action)s` |
| `Q(s,a)` before | `%(before).4f` |
| %(target_label)s | `%(next_q).4f` |
| target `r + γ·%(target_short)s` | `%(target).4f` |
| `Q(s,a)` after | `%(after).4f` |

so with α = %(alpha)s and γ = %(gamma)s:

`%(before).4f + %(alpha)s × (%(target).4f − %(before).4f) = %(after).4f`
    """ % {
        "algorithm": example["algorithm"],
        "state": example["state"],
        "action": actions.direction_name(example["action"]),
        "reward": example["reward"],
        "next_state": example["next_state"],
        "next_label": ("next action `a'` (chosen by the same policy)" if is_sarsa
                       else "next action taken (not used in the target)"),
        "next_action": next_action_name,
        "before": example["q_before"],
        "target_label": ("`Q(s',a')`" if is_sarsa else "`max Q(s',·)`"),
        "target_short": "Q(s',a')" if is_sarsa else "max Q(s',·)",
        "next_q": example["next_q"],
        "target": example["target"],
        "after": example["q_after"],
        "alpha": example["alpha"],
        "gamma": example["gamma"],
    })
    if example["done"]:
        st.caption("This step ended the episode, so the target is just the reward: "
                   "there is no next state to bootstrap from.")
