'use strict';

/* =====================================================================
   The room screen's shell.

   It owns the DOM and nothing else.  It does not know how the agent
   moves, how fast playback is going, or what any number means: it reads
   the RoomDefinition to build the sidebar, hands the batch to the
   playback controller, and paints whatever comes back.

   Two views of the same room are drawn: the world, filling the viewport,
   and the small still grid in the step inspector.  They are separate
   Renderer instances over separate canvases and share nothing but the
   palette.

   Every control's enabled state is derived from the playback state in one
   function — `paintControls` — so no flag anywhere can drift out of step
   with what is actually happening.
   ===================================================================== */

(function () {

  const C = window.ROOM_CONFIG;
  const P = window.Playback;

  const dom = {
    canvas: document.getElementById('world'),
    legend: document.getElementById('legend'),
    strip: {
      episodes: document.getElementById('strip-episodes'),
      objective: document.getElementById('strip-objective'),
      environment: document.getElementById('strip-environment'),
      state: document.getElementById('strip-state'),
      metric: document.getElementById('strip-metric'),
      cleared: document.getElementById('strip-cleared'),
      stale: document.getElementById('strip-stale'),
    },
    exit: document.getElementById('exit'),
    toggle: document.getElementById('toggle'),
    toggleGlyph: document.getElementById('toggle-glyph'),
    sidebar: document.getElementById('sidebar'),
    sector: document.getElementById('room-sector'),
    name: document.getElementById('room-name'),
    play: document.getElementById('play'),
    step: document.getElementById('step'),
    reset: document.getElementById('reset'),
    finalRoute: document.getElementById('final-route'),
    finalRouteHint: document.getElementById('final-route-hint'),
    speeds: document.getElementById('speeds'),
    speedHint: document.getElementById('speed-hint'),
    parameters: document.getElementById('parameters'),
    roomInfo: document.getElementById('room-info'),
    algorithmInfo: document.getElementById('algorithm-info'),
    charts: document.getElementById('charts'),
    episodeList: document.getElementById('episode-list'),
    episodeDetail: document.getElementById('episode-detail'),
    scrub: document.getElementById('scrub'),
    replayControls: document.getElementById('replay-controls'),
    replayPlay: document.getElementById('replay-play'),
    replayStep: document.getElementById('replay-step'),
    replayExit: document.getElementById('replay-exit'),
    inspectEpisode: document.getElementById('inspect-episode'),
    inspectStep: document.getElementById('inspect-step'),
    inspectCount: document.getElementById('inspect-count'),
    inspectBack: document.getElementById('inspect-back'),
    inspectForward: document.getElementById('inspect-forward'),
    inspectGrid: document.getElementById('inspect-grid'),
    inspectReadout: document.getElementById('inspect-readout'),
    inspectEmpty: document.getElementById('inspect-empty'),
    compare: document.getElementById('compare'),
    generalisation: document.getElementById('generalisation-section'),
    generalisationHint: document.getElementById('generalisation-hint'),
    evaluate: document.getElementById('evaluate'),
    testRoom: document.getElementById('test-room'),
    anotherTestRoom: document.getElementById('another-test-room'),
    testRoomHint: document.getElementById('test-room-hint'),
    evaluationTable: document.getElementById('evaluation-table'),
    ending: document.getElementById('ending'),
    cleared: document.getElementById('cleared'),
  };

  /* Only reached if a room offers no algorithm at all, which no built room
     does. Kept as a shape for the section rather than as copy anyone should
     expect to read. */
  const ALGORITHM_PLACEHOLDER = {
    label: 'No method',
    summary: 'This chamber has no method selected.',
    updateRule: '—',
    watchFor: 'Nothing to watch until a method is chosen.',
  };

  const OUTCOME_LABELS = {
    success: 'escaped', failure: 'lost', timeout: 'ran out',
  };

  /* What the chamber holds before anything has been learned, and again after
     Reset. Shared rather than written out at each of the three places that
     need it, so they cannot drift apart. */
  const EMPTY_BATCH = { episodes: [], metrics: [] };

  const STATE_LABELS = {
    IDLE: 'Idle', RUNNING: 'Running', FINISHED: 'Finished',
    REPLAYING: 'Replaying',
  };

  const ui = {
    roomId: 'room2',
    room: null,
    world: null,          // Renderer instance for the viewport
    inspector: null,      // Renderer instance for the sidebar still
    playback: null,
    /* Set for real by `setSidebar` during boot, which also applies the body
       class the layout reads. */
    open: false,
    stale: false,
    staleReason: '',
    parameters: {},
    repaintCharts: null,
    inspect: { episode: null, step: 0 },
    dirty: false,
    /* Whether this chamber has been cleared at any point during this
       visit. Latched rather than read live, because Reset throws the run
       away and pressing it should not un-solve a chamber that was
       genuinely solved a minute ago. */
    everCleared: false,
    leaving: false,
    /* True while a request to the far end is outstanding — opening a room
       or training one. Nothing may be pressed during it. */
    working: false,
    /* Whether this run has reached its episode target. Until it has, Play
       means "train, and let me watch"; afterwards it is a transport control
       over the recorded episodes. */
    trained: false,
    /* Whether training is running right now. */
    running: false,
    /* The live loop's animation frame, separate from `ui.playback`'s: one
       draws training as it happens, the other walks a finished recording. */
    liveFrame: null,
    lastLive: 0,
    /* Throttling for the periodic pull of the recording during training. */
    lastBatchAt: 0,
    fetchingBatch: false,
    recorded: null,
    /* The learned route: one run of the policy with the exploration taken
       out, fetched once when training finishes. Null until then, because
       until then there is no settled policy to run. */
    greedy: null,
    /* The last unseen-room run, if one has been asked for. Until there is
       one, "another unseen room" has nothing to be another of. */
    testedRoom: null,
    /* The mission stage the live view last saw, and how many frames ago it
       changed. See `liveDetail`. Null until a room with a mission reports one. */
    liveStage: null,
    liveSince: null,
    /* The detail of whichever frame was drawn last — live or replayed.
       The objective line reads this so it always describes the frame on
       screen rather than whichever source happens to be newer. */
    lastDetail: null,
    /* Whether the collapse sound has already played for the frame being
       shown. See `soundCollapse`. */
    lastCollapse: false,
    /* The ending sequence, for the last chamber only. `endingSeen`
       latches so the sequence arrives once rather than every time the
       run is repainted. */
    ending: null,
    endingSeen: false,
    /* The sector-cleared flash, and whether it has been shown for this
       run. Latched, because `isCleared` is asked on every repaint. */
    clearedFlash: null,
    clearedFlashSeen: false,
    /* Steps owed but not yet whole, carried between frames so the speeds
       mean what they say per second. */
    owed: 0,
  };

  /* ---------------------------------------------------------------------
     Small helpers
     ------------------------------------------------------------------- */

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function roomFromQuery() {
    const raw = new URLSearchParams(window.location.search).get('room');
    if (!raw) return 'room2';
    // A number is the chamber number; anything else is an id, which is how
    // the continuous renderer demo is reached.
    if (/^\d+$/.test(raw)) return 'room' + raw;
    return raw;
  }

  function sidebarInset() {
    if (!ui.open) return 0;
    return dom.sidebar.getBoundingClientRect().width;
  }

  function refit() {
    ui.world.fit(ui.room, sidebarInset());
    ui.inspector.fit(ui.room, 0, C.stepReplay.padding);
    paintInspector();
    redrawWorld();
  }

  /**
   * Put the world back on the canvas after anything that cleared it.
   *
   * WHY THIS HAS TO EXIST
   * `fit` assigns `canvas.width`, and assigning that wipes the canvas — so
   * every re-fit blanks the world. That is fine while something is animating,
   * because the next frame paints it again. It is not fine at rest: opening
   * this screen fits, draws once, and then `setSidebar` schedules a second
   * fit 220ms later for the end of the sidebar transition, which wiped the
   * room and left nothing to redraw it. The chamber was simply blank until
   * Train was pressed and the live loop took over.
   *
   * Draws whichever description of the world is current — a replayed frame if
   * one is selected, the live scene if training has produced one, and the bare
   * room otherwise. Never invents anything: with nothing to show it draws the
   * chamber with no agent in it, which is exactly right before a run starts.
   */
  function redrawWorld() {
    if (!ui.room || !ui.world) return;
    // A running loop owns the canvas; leave it alone rather than fighting it
    // for the same frame.
    if (ui.liveFrame !== null) return;
    if (ui.playback && (ui.playback.state === P.RUNNING
                        || ui.playback.state === P.REPLAYING)) {
      return;
    }
    const replayed = ui.playback && ui.playback.selected !== null
      ? ui.playback.snapshot() : null;
    if (replayed && replayed.position) {
      ui.world.draw(replayed);
      return;
    }
    const live = liveSnapshot();
    ui.world.draw(live && live.position ? live
                                       : { room: ui.room, position: null });
  }

  /* ---------------------------------------------------------------------
     The legend, drawn with the same recipes as the world
     ------------------------------------------------------------------- */

  function buildLegend() {
    dom.legend.innerHTML = '';
    window.Renderer.legend(ui.room).forEach(entry => {
      const item = element('li');

      // A canvas, not an SVG swatch: the key is drawn by the very code
      // that draws the world, so it cannot show something the grid does
      // not.
      const swatch = document.createElement('canvas');
      swatch.className = 'swatch';
      swatch.setAttribute('aria-hidden', 'true');
      item.appendChild(swatch);
      item.appendChild(element('span', null, entry.label));
      dom.legend.appendChild(item);

      // Drawn after it is in the document, so it has a measurable size.
      window.Renderer.swatch(swatch, ui.room, entry.type);
    });
  }

  /* ---------------------------------------------------------------------
     Speeds
     ------------------------------------------------------------------- */

  function buildSpeeds() {
    dom.speeds.innerHTML = '';
    Object.keys(C.speeds).forEach(key => {
      const chip = element('button', 'chip', C.speeds[key].label);
      chip.type = 'button';
      chip.setAttribute('role', 'radio');
      chip.dataset.speed = key;
      chip.addEventListener('click', () => {
        ui.playback.setSpeed(key);
        paintSpeeds();
      });
      dom.speeds.appendChild(chip);
    });
    ui.playback.setSpeed(C.speedDefault);
    paintSpeeds();
  }

  function paintSpeeds() {
    Array.prototype.forEach.call(dom.speeds.children, chip => {
      chip.setAttribute('aria-checked',
                        String(chip.dataset.speed === ui.playback.speed));
    });
    dom.speedHint.textContent = C.speeds[ui.playback.speed].animated
      ? 'Every step is drawn, and movement is interpolated between them.'
      : 'Consumes episodes inside a ' + C.turboBudgetMs
        + 'ms budget per frame and draws only where it ended up.';
  }

  /* ---------------------------------------------------------------------
     Parameters, built entirely from the schema
     ------------------------------------------------------------------- */

  function buildParameters() {
    dom.parameters.innerHTML = '';

    const groups = [
      { live: true, label: 'Live — applies immediately' },
      { live: false, label: 'Requires reset' },
    ];

    groups.forEach(group => {
      const members = ui.room.parameterSchema.filter(
        parameter => parameter.appliesLive === group.live);
      if (!members.length) return;

      const heading = element('p', 'group-label', group.label);
      if (!group.live) heading.classList.add('is-reset');
      dom.parameters.appendChild(heading);

      members.forEach(specification => {
        dom.parameters.appendChild(parameterControl(specification));
      });

      // One restore per group, because the two groups mean different
      // things and restoring them together would hide that.
      const restore = element('button', 'parameter-default',
                              'restore these to defaults');
      restore.type = 'button';
      restore.addEventListener('click', () => {
        members.forEach(specification => {
          setParameter(specification, specification.default);
        });
      });
      dom.parameters.appendChild(restore);
    });
  }

  function parameterControl(specification) {
    const wrapper = element('div', 'parameter');
    wrapper.dataset.parameter = specification.key;

    const head = element('div', 'parameter-head');
    const title = specification.symbol
      ? specification.label + '  ' + specification.symbol
      : specification.label;
    head.appendChild(element('span', null, title));
    const value = element('span', 'parameter-value');
    head.appendChild(value);
    wrapper.appendChild(head);

    const slider = document.createElement('input');
    slider.className = 'field';
    slider.type = 'range';
    slider.min = String(specification.min);
    slider.max = String(specification.max);
    slider.step = String(specification.step || 0.01);
    slider.setAttribute('aria-label', specification.label);
    wrapper.appendChild(slider);

    wrapper.appendChild(element('p', 'hint', specification.description));

    slider.addEventListener('input', () => {
      setParameter(specification, Number(slider.value));
    });

    wrapper.show = function (current) {
      slider.value = String(current);
      value.textContent = format(current);
    };
    wrapper.show(ui.parameters[specification.key]);
    return wrapper;
  }

  function format(value) {
    if (Number.isInteger(value)) return String(value);
    const size = Math.abs(value);
    if (size !== 0 && size < 0.001) return value.toExponential(1);
    return String(Number(value.toFixed(4)));
  }

  function setParameter(specification, next) {
    ui.parameters[specification.key] = next;

    const control = dom.parameters.querySelector(
      '[data-parameter="' + specification.key + '"]');
    if (control && control.show) control.show(next);

    // Every change invalidates the run, `appliesLive` or not, and that is
    // a property of this screen rather than of the parameter: training has
    // already finished by the time a control can be touched, so there is
    // nothing left for a live parameter to apply to. The planner screen is
    // the other way round — it drives training a sweep at a time, so a live
    // parameter there really does take effect mid-run.
    //
    // The run is not thrown away here; that is the point of making Reset
    // explicit.
    ui.stale = true;
    /* An environment parameter is not the same kind of change as a learning
       one: it alters the room itself, so the policy is not merely out of date
       but was learned against a different world. Said plainly, because the
       generic "X changed" did not convey that the results on screen no longer
       describe the room the player is now looking at. */
    ui.staleReason = specification.appliesLive
      ? specification.label + ' changed'
      : 'Environment changed — retrain to update the policy';
    if (ui.compare) ui.compare.setParameters(currentParameters());
    // Some parameters shape the room itself rather than the learning, so
    // what the room says about itself has to keep up even before Reset.
    buildRoomInfo();
    paintStrip();
    paintControls();
  }

  function currentParameters() {
    const values = {};
    ui.room.parameterSchema.forEach(specification => {
      values[specification.key] = ui.parameters[specification.key];
    });
    return values;
  }

  /* ---------------------------------------------------------------------
     Room info and the algorithm shell
     ------------------------------------------------------------------- */

  function block(heading, text) {
    const wrapper = element('div', 'info-block');
    wrapper.appendChild(element('p', 'group-label', heading));
    wrapper.appendChild(element('p', null, text));
    return wrapper;
  }

  /** The description of the method this run is actually using. */
  function algorithmDescription() {
    const running = window.Producer.snapshot;
    const key = running ? running.algorithm : window.Producer.algorithmDefault;
    const found = window.Producer.algorithms.filter(
      entry => entry.key === key);
    return found.length ? found[0] : null;
  }

  function buildRoomInfo() {
    const info = ui.room.info;

    dom.sector.textContent = ui.room.sector || '';
    dom.name.textContent = ui.room.name;

    dom.roomInfo.innerHTML = '';
    dom.roomInfo.appendChild(block('Objective', info.objective));

    // The choice the room poses, when it poses one. Stated as a table,
    // because the whole point is that two things are being compared.
    if (info.routes) {
      const routes = element('div', 'info-block');
      routes.appendChild(element('p', 'group-label', 'The choice'));
      const table = element('table', 'routes');

      const head = element('tr');
      ['Route', 'Steps', 'If it works', 'Risk'].forEach(name => {
        head.appendChild(element('th', null, name));
      });
      table.appendChild(head);

      info.routes.forEach(route => {
        const row = element('tr');
        row.appendChild(element('td', 'route-name', route.name));
        row.appendChild(element('td', null, String(route.steps)));
        row.appendChild(element('td', 'route-best', route.best));
        row.appendChild(element('td', 'route-risk', route.risk));
        table.appendChild(row);
      });
      routes.appendChild(table);
      dom.roomInfo.appendChild(routes);
    }

    const obstacles = element('div', 'info-block');
    obstacles.appendChild(element('p', 'group-label', 'Obstacles'));
    const list = element('ul');
    info.obstacles.forEach(line => list.appendChild(element('li', null, line)));
    obstacles.appendChild(list);
    dom.roomInfo.appendChild(obstacles);

    dom.roomInfo.appendChild(block('Action set', info.actionSet));

    const rewards = element('div', 'info-block');
    rewards.appendChild(element('p', 'group-label', 'Rewards'));
    const table = element('table', 'rewards');
    info.rewards.forEach(pair => {
      const row = element('tr');
      row.appendChild(element('td', null, pair[0]));
      row.appendChild(element('td', null, pair[1]));
      table.appendChild(row);
    });
    rewards.appendChild(table);
    dom.roomInfo.appendChild(rewards);

    dom.roomInfo.appendChild(block('Terminal condition', info.terminal));
    if (info.note) dom.roomInfo.appendChild(block('Note', info.note));
  }

  function buildAlgorithmInfo() {
    // The room names its own method — they are not all SARSA — and falls
    // back to the placeholder only if it does not.
    const text = algorithmDescription() || ALGORITHM_PLACEHOLDER;
    dom.algorithmInfo.innerHTML = '';
    dom.algorithmInfo.appendChild(block(text.label, text.summary));

    const rule = element('div', 'info-block');
    rule.appendChild(element('p', 'group-label', 'Update rule'));
    rule.appendChild(element('pre', 'rule', text.updateRule));
    dom.algorithmInfo.appendChild(rule);

    dom.algorithmInfo.appendChild(block('What to watch', text.watchFor));
  }

  /* ---------------------------------------------------------------------
     The final route
     ------------------------------------------------------------------- */

  /**
   * The best of the recorded *training* episodes.
   *
   * The last one that actually reached the exit, rather than simply the
   * last one played: with exploration never decaying to zero, the final
   * episode of a run is quite often a fall, and that is not the thing
   * anybody wants to watch. Falls back to the last episode played when
   * nothing has succeeded, because the most recent failure is more use
   * than refusing to show anything.
   *
   * This is about what training managed while it was still exploring, which
   * is the right question for `isCleared` and the wrong one for the route to
   * display. See `finalRoute`.
   */
  /**
   * Whether an episode *began* part-way through the mission.
   *
   * A share of training episodes start at the control terminal already in
   * stage 1, so the second half of the mission gets practised instead of
   * starving — see `WarehouseWorld.reset`. Browsing those without knowing it
   * is genuinely misleading: the episode opens with the terminal spent, the
   * blast door open and the beams down, which looks exactly like a room that
   * has been won. It is a practice start, and it is labelled as one.
   *
   * Read off the recorded first frame, so it is a fact about the episode
   * rather than a guess about the run.
   */
  function startedMidMission(episode) {
    if (!episode || !episode.steps || !episode.steps.length) return false;
    const first = episode.steps[0].detail;
    return Boolean(first && first.stage);
  }

  function bestRecordedRoute() {
    const played = recordedMetrics();
    if (!played.length) return null;

    for (let at = played.length - 1; at >= 0; at -= 1) {
      const episode = ui.playback.episodeAt(played[at].episode);
      if (episode && episode.outcome === 'success') {
        return { index: episode.index, succeeded: true, greedy: false };
      }
    }
    return { index: played[played.length - 1].episode, succeeded: false,
             greedy: false };
  }

  /**
   * The route to show as the one the agent settled on.
   *
   * The greedy run — the learned policy followed with the exploration taken
   * out — and *not* a recorded episode, which is the distinction this was
   * getting wrong. ε stops at `epsilon_min`, 0.05 by default, rather than
   * decaying to zero, so every episode that was trained on still takes a
   * random step about one time in twenty and the route visibly doubles back
   * on itself. Room 2 at the defaults is the plain case: the last recorded
   * episode that reached the exit wanders through 22 steps with a step into
   * the west wall in the middle of it, and the policy behind that episode
   * walks the same route cleanly in 21. Showing the first and calling it the
   * optimal route is not a presentation quibble — it is showing exploration
   * noise and labelling it as the answer.
   *
   * Falls back to the best recorded attempt while training is still running,
   * since until it finishes there is no settled policy to run.
   */
  function finalRoute() {
    if (ui.greedy) {
      return {
        index: ui.greedy.index,
        succeeded: ui.greedy.outcome === 'success',
        greedy: true,
      };
    }
    return bestRecordedRoute();
  }

  /**
   * Whether the chamber counts as cleared.
   *
   * One episode that actually reached the goal is the whole test. Not
   * "the batch finished" — a run that trains to the end and never once
   * gets out has not solved anything — and not the mean reward either,
   * which can be respectable for an agent that has merely learned to
   * survive.
   */
  function isCleared() {
    /* NO EARLY RETURN ON `everCleared`.
       There used to be one at the top of this function, and it was safe only
       while becoming cleared and showing the overlay happened in the same
       call. They no longer do: the chamber is cleared as soon as the run
       earns it, and the overlay waits until R-5 has been watched arriving.
       An early return here means the second of those is never reached, and
       the overlay never appears at all. The work below is a couple of array
       lookups per repaint, and it is guarded so it only runs while there is
       still something left to find out. */
    if (!ui.everCleared) {
      // Deliberately the recorded route and not the greedy one: the test is
      // whether the agent ever actually got out, which is a fact about the
      // run that happened. A policy that would get out is a different claim,
      // and it is not this one.
      const route = bestRecordedRoute();
      if (route && route.succeeded) ui.everCleared = true;
    }

    /* THE PLANNER'S ROOM HAS NO METRICS, AND WAS THEREFORE NEVER CLEARABLE
       `bestRecordedRoute` walks `recordedMetrics()`, and room 1 keeps none:
       Value Iteration is not episodic, so it records the single greedy walk
       as an episode and no per-episode metrics beside it. The route lookup
       therefore returned null there however well the room went, which meant
       room 1 could never be marked cleared and never unlocked room 2.

       So a recorded escape is also looked for directly. `solved` is the far
       end's own verdict and it is the same question — for every room it
       requires an episode that actually reached the exit — so this agrees
       with the route test wherever both can answer. */
    if (!ui.everCleared) {
      const snapshot = window.Producer && window.Producer.snapshot;
      if (snapshot && snapshot.solved) ui.everCleared = true;
    }

    maybeFlashCleared();
    return ui.everCleared;
  }

  /**
   * Raise Mission Complete, once, when R-5 has been seen to arrive.
   *
   * WHEN THE OVERLAY IS ALLOWED ON SCREEN
   * Not when the chamber becomes cleared — that is a fact about the recorded
   * run, and it becomes true during *training*, while the robot on screen
   * may be part-way across the room or not moving at all. The overlay used
   * to arrive then, over a walking robot.
   *
   * It waits for all three of the things that make an arrival:
   *   1. a recorded episode reached the exit            `everCleared`
   *   2. playback walked that episode to its last step, so R-5 was drawn
   *      arriving and its final stride landed           `witnessedEscape`
   *   3. it has not already been shown for this run     `clearedFlashSeen`
   *
   * `everCleared` is deliberately left alone, so unlocking the next chamber
   * and the cleared marker in the strip still happen the moment the run
   * earns them. Only the overlay waits.
   *
   * CALLED FROM THE FRAME LOOP, NOT ONLY FROM THE STRIP PAINT
   * This lived inside `isCleared`, which runs from `paintStrip` — and the
   * strip is only repainted when `ui.dirty` is set, which a replay looping
   * one episode never sets. So the one moment this is waiting for was the
   * one moment nothing asked. It is its own function now and the frame
   * handler calls it; once latched it is three boolean reads per frame.
   */
  function maybeFlashCleared() {
    if (!ui.everCleared) return;
    if (ui.clearedFlashSeen || !ui.clearedFlash) return;
    if (!(ui.playback && ui.playback.witnessedEscape)) return;

    ui.clearedFlashSeen = true;
    ui.clearedFlash.show({
      sector: ui.room && ui.room.sector,
      last: Boolean(ui.room && ui.room.isFinal),
    });
  }

  function showFinalRoute() {
    const route = finalRoute();
    if (!route) return;
    // Legal from IDLE, RUNNING and FINISHED alike.
    ui.playback.selectEpisode(route.index);
    paintAll();
  }

  function paintFinalRouteHint() {
    const route = finalRoute();
    if (!route) {
      dom.finalRouteHint.textContent =
        'Nothing has been recorded yet, so there is no route to show.';
      return;
    }

    // The greedy run, once training has finished: say what it is, because
    // "the learned policy with no exploration" is exactly what makes it
    // different from the episodes listed below it.
    if (route.greedy) {
      const steps = ui.greedy.steps.length - 1;
      if (!route.succeeded) {
        dom.finalRouteHint.textContent =
          'The learned policy, followed with no exploration — and it does not '
          + 'reach the exit (' + OUTCOME_LABELS[ui.greedy.outcome] + ' after '
          + steps + ' steps). Training has not converged on a route out.';
        return;
      }
      dom.finalRouteHint.textContent =
        'The route it settled on: the learned policy followed with the '
        + 'exploration taken out, ' + steps + ' steps for '
        + ui.greedy.totalReward.toFixed(0) + '. The training episodes below '
        + 'still wander, because ε never reaches zero.';
      return;
    }

    if (!route.succeeded) {
      dom.finalRouteHint.textContent =
        'No episode has reached the exit yet — this shows the most recent '
        + 'attempt, episode ' + (route.index + 1) + '.';
      return;
    }
    dom.finalRouteHint.textContent =
      'Episode ' + (route.index + 1) + ', the last one that reached the exit. '
      + 'Training is still going, so this is an exploratory episode rather '
      + 'than the settled route.';
  }

  /* ---------------------------------------------------------------------
     The episode browser
     ------------------------------------------------------------------- */

  /**
   * The episodes there are to look at: everything recorded, in order.
   *
   * Deliberately not `ui.playback.metrics`, which counts episodes *played
   * back*. An episode can be taken apart because it was recorded, not
   * because it has been watched — and now that training is watched live
   * rather than replayed, nothing is played back at all unless you ask for
   * it. Reading the played count was what left the step inspector empty
   * after a run finished.
   */
  function recordedMetrics() {
    if (ui.recorded) return ui.recorded.metrics;
    return ui.playback.metrics;
  }

  /**
   * What the charts are drawn from.
   *
   * The whole batch rather than one array, because the two things graphed here
   * have different lengths and different x-axes. `history` is one row per
   * episode ever run — that is what a learning curve is about. `metrics`
   * covers only the couple of dozen episodes kept frame by frame, and a curve
   * over those is a picture of the sampling rather than of the run.
   * `checkpoints` is different again: held-out measurements taken every few
   * hundred episodes with the weights frozen.
   */
  function chartData() {
    if (ui.recorded) return ui.recorded;
    return ui.playback.metrics;
  }

  function buildEpisodeList(metrics) {
    const played = metrics || recordedMetrics();
    dom.episodeList.innerHTML = '';

    if (!played.length) {
      dom.episodeList.appendChild(
        element('p', 'hint', 'Episodes appear here as they are recorded.'));
      paintEpisodeDetail();
      return;
    }

    // Newest first: the interesting episodes are the recent ones.
    const newestFirst = played.slice().reverse();
    const shown = newestFirst.slice(0, C.episodes.maxRows);

    shown.forEach(entry => {
      const episode = ui.playback.episodeAt(entry.episode);
      const item = element('button', 'episode');
      item.type = 'button';
      item.dataset.episode = String(entry.episode);

      item.appendChild(element('span', 'episode-index',
                               String(entry.episode + 1).padStart(3, '0')));
      const outcome = episode ? episode.outcome : 'timeout';
      item.appendChild(element('span', 'episode-outcome is-' + outcome,
                               OUTCOME_LABELS[outcome] || outcome));
      if (startedMidMission(episode)) {
        const mark = element('span', 'episode-tag', 'practice');
        mark.title = 'Began at the control terminal, in stage 2 of the '
                   + 'mission, so the second half gets practised.';
        item.appendChild(mark);
      }
      item.appendChild(element('span', 'episode-reward',
                               entry.reward.toFixed(0)));

      item.addEventListener('click', () => {
        ui.playback.selectEpisode(entry.episode);
        paintAll();
      });
      dom.episodeList.appendChild(item);
    });

    // Never truncate quietly: a list that stops at 120 rows without
    // saying so reads as a list of everything there was.
    if (newestFirst.length > shown.length) {
      dom.episodeList.appendChild(element(
        'p', 'hint',
        'and ' + (newestFirst.length - shown.length)
        + ' earlier episodes, not listed.'));
    }

    paintEpisodeDetail();
  }

  function paintEpisodeDetail() {
    const selected = ui.playback.selected;
    const replaying = selected !== null;

    dom.replayControls.hidden = !replaying;
    dom.scrub.hidden = !replaying;
    dom.episodeDetail.innerHTML = '';

    Array.prototype.forEach.call(dom.episodeList.children, item => {
      if (!item.dataset) return;
      item.setAttribute('aria-current',
                        String(Number(item.dataset.episode) === selected));
    });

    if (!replaying) return;

    const episode = ui.playback.episodeAt(selected);
    if (!episode) return;

    // The learned route sits in the batch beside the training episodes but it
    // is not one of them, and numbering it as though it were would claim an
    // episode that never ran.
    const rows = episode.greedy
      ? [
        ['Route', 'Learned policy'],
        ['Outcome', OUTCOME_LABELS[episode.outcome] || episode.outcome],
        ['Reward', episode.totalReward.toFixed(0)],
        ['Exploration', 'none'],
        ['Steps', String(episode.steps.length - 1)],
      ]
      : [
        ['Episode', String(selected + 1)],
        ['Began', startedMidMission(episode)
          ? 'at the terminal (practice start)' : 'at the start, door locked'],
        ['Outcome', OUTCOME_LABELS[episode.outcome] || episode.outcome],
        ['Reward', episode.totalReward.toFixed(0)],
        ['Exploration', episode.epsilon.toFixed(3)],
        ['Steps', String(episode.steps.length - 1)],
      ];
    /* The environment this episode was recorded under.
       Room 2's bridge risk is a slider, so two episodes with the same reward
       can describe completely different rooms. The value is stored on the
       episode itself by the recorder, so it is what that episode really ran
       at rather than whatever the slider says now. */
    if (typeof episode.collapseChance === 'number') {
      rows.push(['Bridge collapse probability',
                 episode.collapseChance.toFixed(2)]);
    }

    const list = element('dl', 'readout');
    rows.forEach(pair => {
      list.appendChild(element('dt', null, pair[0]));
      list.appendChild(element('dd', null, pair[1]));
    });
    dom.episodeDetail.appendChild(list);
  }

  /* ---------------------------------------------------------------------
     Generalisation: held-out layouts and unseen rooms

     Only a chamber with layout pools has any of this, which is room 5
     alone. Every number shown here is measured with the weights frozen —
     the far end never calls an update from these paths — so an unseen
     layout stays unseen however many times it is measured.
     ------------------------------------------------------------------- */

  function hasPools() {
    return Boolean(ui.room && ui.room.layoutPools
                   && ui.room.layoutPools.test);
  }

  function buildGeneralisation() {
    dom.generalisation.hidden = !hasPools();
    if (!hasPools()) return;

    const pools = ui.room.layoutPools;
    dom.generalisationHint.textContent =
      'Training draws from ' + pools.train.count + ' warehouses (seeds '
      + pools.train.first + '–' + pools.train.last + '). '
      + pools.validation.count + ' more are kept back to check against, and '
      + pools.test.count + ' are never trained on at all. The three pools do '
      + 'not overlap, and nothing measured here updates a weight.';
    paintGeneralisation();
  }

  function paintGeneralisation() {
    if (!hasPools()) return;
    // Nothing to evaluate until a policy exists. Before training these would
    // measure untouched weights and report zeroes as though they meant
    // something.
    const ready = ui.trained && !ui.working;
    disable(dom.evaluate, !ready);
    disable(dom.testRoom, !ready);
    disable(dom.anotherTestRoom, !ready || !ui.testedRoom);

    if (!ui.trained) {
      dom.testRoomHint.textContent =
        'Train the agent first — there is no learned policy to test yet.';
    }
  }

  /** The evaluation table, once it has been asked for. */
  function paintEvaluation(report) {
    dom.evaluationTable.innerHTML = '';
    if (!report) return;

    const rows = [
      ['Training layouts', report.train],
      ['Validation layouts', report.validation],
      ['Unseen test layouts', report.test],
      ['Unseen — random policy', report.randomTest],
    ];

    const table = element('table', 'rewards');
    const head = element('tr');
    ['', 'Escaped', 'Terminal', 'Collided', 'Reward'].forEach(name => {
      head.appendChild(element('th', null, name));
    });
    table.appendChild(head);

    rows.forEach(pair => {
      const report_ = pair[1];
      if (!report_) return;
      const row = element('tr');
      row.appendChild(element('td', null,
                              pair[0] + ' (' + report_.layouts + ')'));
      row.appendChild(element('td', null, percent(report_.escapeRate)));
      row.appendChild(element('td', null, percent(report_.terminalRate)));
      row.appendChild(element('td', null, percent(report_.collisionRate)));
      row.appendChild(element('td', null,
                              report_.meanReward.toFixed(1) + ' ± '
                              + report_.rewardDeviation.toFixed(0)));
      table.appendChild(row);
    });
    dom.evaluationTable.appendChild(table);

    // The comparison that decides whether any of this means anything. A
    // learned policy that does not beat a random one on held-out layouts has
    // not generalised, however respectable the training numbers look.
    const learned = report.test.escapeRate;
    const chance = report.randomTest.escapeRate;
    dom.evaluationTable.appendChild(element(
      'p', 'hint',
      learned > chance
        ? 'On layouts it never trained on the learned policy escapes '
          + percent(learned) + ' against ' + percent(chance)
          + ' for a random one.'
        : 'On layouts it never trained on the learned policy does not beat a '
          + 'random one (' + percent(learned) + ' against ' + percent(chance)
          + '). Whatever it learned has not transferred.'));
  }

  function percent(share) {
    return Math.round(share * 100) + '%';
  }

  async function runEvaluation() {
    if (ui.working) return;
    ui.working = true;
    paintGeneralisation();
    dom.testRoomHint.textContent = 'Measuring every layout…';
    try {
      const report = await window.Producer.evaluate();
      paintEvaluation(report);
      dom.testRoomHint.textContent =
        'Measured over ' + report.episodes + ' episodes of training, with the '
        + 'weights frozen.';
    } catch (problem) {
      dom.testRoomHint.textContent = String(
        (problem && problem.message) || problem);
    }
    ui.working = false;
    paintGeneralisation();
    paintControls();
  }

  /**
   * Build an unseen warehouse, fly the learned policy in it, and replay it.
   *
   * The episode comes back shaped like any other, so it is handed to the
   * playback controller and watched through the same transport controls. It
   * is labelled unseen on the way in, because a run on a held-out layout is a
   * different claim from a run on a trained one and the screen must not let
   * the two be confused.
   */
  async function runTestRoom() {
    if (ui.working) return;
    ui.working = true;
    paintGeneralisation();
    dom.testRoomHint.textContent = 'Generating an unseen warehouse…';
    try {
      const episode = await window.Producer.testRoom();
      ui.testedRoom = episode;

      // Into the batch, so every reader that resolves an episode by number
      // reaches it. Kept out of `metrics` on purpose: it is not a training
      // episode and must not appear in the graphs of training.
      const batch = ui.recorded || EMPTY_BATCH;
      batch.episodes = batch.episodes
        .filter(entry => entry.index !== episode.index)
        .concat([episode]);
      ui.recorded = batch;
      ui.playback.load(ui.room, batch);
      ui.playback.selectEpisode(episode.index);
      ui.playback.start();

      dom.testRoomHint.textContent =
        'Unseen warehouse, seed ' + episode.layoutSeed + ' — never trained on. '
        + 'The learned policy '
        + (episode.outcome === 'success'
            ? 'escaped in ' + (episode.steps.length - 1) + ' steps for '
              + episode.totalReward.toFixed(0) + '.'
            : (OUTCOME_LABELS[episode.outcome] || episode.outcome)
              + ' after ' + (episode.steps.length - 1) + ' steps for '
              + episode.totalReward.toFixed(0) + '.');
      buildEpisodeList();
      buildInspectorEpisodes();
    } catch (problem) {
      dom.testRoomHint.textContent = String(
        (problem && problem.message) || problem);
    }
    ui.working = false;
    paintGeneralisation();
    paintAll();
  }

  /* ---------------------------------------------------------------------
     The ending

     Only the last chamber has one, and only a real escape earns it: the
     test is `isCleared`, which is true when a recorded episode reached the
     *blast door*. Activating the control terminal cannot make it true —
     `info["goal"]` is set on `escaped` alone — so the sequence cannot
     appear half-way through the mission.
     ------------------------------------------------------------------- */

  function hasEnding() {
    return Boolean(ui.room && ui.room.isFinal && ui.ending);
  }

  /** The episode to offer as the final escape: the greedy route if it got
   *  out, otherwise the last recorded episode that did. */
  function escapeRoute() {
    if (ui.greedy && ui.greedy.outcome === 'success') return ui.greedy.index;
    const route = bestRecordedRoute();
    return route && route.succeeded ? route.index : null;
  }

  /**
   * Show the ending, if this chamber has one and has genuinely been escaped.
   *
   * Returns whether it was shown, so `leave` can hold the navigation back:
   * the sequence appears *instead of* returning to the chamber select, which
   * is the whole point of it.
   */
  function maybeShowEnding(force) {
    if (!hasEnding()) return false;
    if (!isCleared()) return false;
    if (ui.endingSeen && !force) return false;

    ui.endingSeen = true;
    // Both loops stopped: the world behind the overlay is not being looked
    // at, and a replay left running would go on asking for frames.
    ui.playback.pause();
    stopLive();
    if (ui.running) {
      ui.running = false;
      window.Producer.hold().catch(() => {});
    }

    const batch = ui.recorded || EMPTY_BATCH;
    const description = algorithmDescription();
    const figures = window.Ending.summary(
      batch, window.Producer.snapshot,
      description ? description.label : null);
    ui.ending.show(figures, { canReplay: escapeRoute() !== null });
    paintControls();
    return true;
  }

  /** Dismiss the ending and hand the room back. */
  function closeEnding() {
    if (!ui.ending) return;
    ui.ending.hide();
    paintAll();
  }

  /* ---------------------------------------------------------------------
     The step inspector

     Any completed episode, any single step, held still. It reads the
     batch through `frameAt` and changes nothing, so it can be scrubbed
     while a run is going without disturbing it.
     ------------------------------------------------------------------- */

  function buildInspectorEpisodes() {
    const played = recordedMetrics();
    const previous = ui.inspect.episode;

    dom.inspectEpisode.innerHTML = '';
    played.slice().reverse().forEach(entry => {
      const episode = ui.playback.episodeAt(entry.episode);
      if (!episode) return;
      const option = element('option');
      option.value = String(entry.episode);
      option.textContent = String(entry.episode + 1).padStart(3, '0')
        + '  ·  ' + (OUTCOME_LABELS[episode.outcome] || episode.outcome)
        + '  ·  ' + entry.reward.toFixed(0);
      dom.inspectEpisode.appendChild(option);
    });

    if (!played.length) {
      ui.inspect.episode = null;
      return;
    }

    // Keep the episode being looked at if it is still there; otherwise
    // start from the most recent one.
    const stillThere = played.some(entry => entry.episode === previous);
    ui.inspect.episode = stillThere ? previous
                                    : played[played.length - 1].episode;
    dom.inspectEpisode.value = String(ui.inspect.episode);
  }

  function paintInspector() {
    const empty = ui.inspect.episode === null;
    dom.inspectEpisode.disabled = empty;
    dom.inspectStep.disabled = empty;
    disable(dom.inspectBack, empty);
    disable(dom.inspectForward, empty);
    dom.inspectReadout.innerHTML = '';

    if (empty) {
      dom.inspectEmpty.textContent =
        'Play some episodes and any one of them can be taken apart here, '
        + 'a step at a time.';
      dom.inspectCount.textContent = '';
      // Nothing to draw, so the still is cleared rather than left stale.
      ui.inspector.draw({ room: ui.room, position: null });
      return;
    }
    dom.inspectEmpty.textContent = '';

    const frame = ui.playback.frameAt(ui.inspect.episode, ui.inspect.step);
    if (!frame) return;

    // Clamp, in case a shorter episode was selected while the slider was
    // near the end of a longer one.
    ui.inspect.step = frame.index;
    dom.inspectStep.max = String(frame.lastIndex);
    dom.inspectStep.value = String(frame.index);
    dom.inspectCount.textContent = frame.index + ' / ' + frame.lastIndex;

    ui.inspector.draw(frame.snapshot);

    const step = frame.step;
    const rows = [
      ['State', 'row ' + Math.floor(step.position.y)
                + ', col ' + Math.floor(step.position.x)],
      ['Action taken to get here', frame.index === 0 ? '— (start)' : step.action],
      ['Reward for this step', formatReward(step.reward)],
      ['Total reward so far', formatReward(frame.cumulativeReward)],
      ['Steps so far', String(frame.index)],
      ['Episode outcome',
       OUTCOME_LABELS[frame.episode.outcome] || frame.episode.outcome],
      ['Episode total', formatReward(frame.episode.totalReward)],
      ['Exploration then', frame.episode.epsilon.toFixed(3)],
    ];
    if (step.velocity) {
      rows.splice(1, 0, ['Velocity', step.velocity.x.toFixed(2) + ', '
                                   + step.velocity.y.toFixed(2)]);
    }
    rows.forEach(pair => {
      dom.inspectReadout.appendChild(element('dt', null, pair[0]));
      dom.inspectReadout.appendChild(element('dd', null, pair[1]));
    });
  }

  function formatReward(value) {
    return (value > 0 ? '+' : '') + value.toFixed(0);
  }

  function stepInspector(by) {
    if (ui.inspect.episode === null) return;
    ui.inspect.step += by;
    if (ui.inspect.step < 0) ui.inspect.step = 0;
    paintInspector();
  }

  /* ---------------------------------------------------------------------
     Painting
     ------------------------------------------------------------------- */

  /**
   * The current mission objective, for a room that has more than one.
   *
   * The stage comes off the frame being drawn, which came off the
   * environment — so the line changes at the same instant the objective
   * marker moves and the door unlocks, because all three read the same
   * number. The wording is the room's (`definition.objectives`); this only
   * indexes it.
   */
  function paintObjective() {
    const names = ui.room && ui.room.objectives;
    const detail = ui.lastDetail;
    const stage = detail && detail.stage;
    if (!names || typeof stage !== 'number' || !names[stage]) {
      dom.strip.objective.hidden = true;
      return;
    }
    dom.strip.objective.hidden = false;
    dom.strip.objective.textContent = names[stage];
    dom.strip.objective.dataset.stage = String(stage);
  }

  function paintStrip() {
    /* The environment settings are painted here as well as from the live
       strip, because `paintLiveStrip` only runs while training is going. At
       rest — which is what the player sees the moment the room opens, and
       again after Reset — nothing was calling it, so the collapse probability
       was blank exactly when someone would first look for it. */
    paintEnvironment(window.Producer.snapshot);

    // While training is what is on screen, the strip counts episodes trained
    // rather than episodes replayed, and there is no batch to count anyway.
    if (!ui.trained) {
      const snapshot = window.Producer.snapshot;
      if (snapshot && snapshot.progress.count) {
        paintLiveStrip();
      } else {
        dom.strip.episodes.textContent = 'press Train to start';
        dom.strip.state.textContent = 'Ready';
        dom.strip.metric.textContent = '';
      }
      dom.strip.cleared.hidden = true;
      dom.strip.stale.hidden = !ui.stale;
      if (ui.stale) {
        dom.strip.stale.textContent = staleMessage();
      }
      return;
    }

    // Counted against the episodes that were recorded, not against the batch,
    // because the batch also carries the learned route — which is not an
    // episode and must not inflate the total.
    const played = ui.playback.episodesPlayed;
    dom.strip.episodes.textContent = played + ' / ' + recordedMetrics().length
                                   + ' episodes';
    dom.strip.state.textContent = STATE_LABELS[ui.playback.state];

    // One metric, as text. The window is the recent past, because the mean
    // over a whole run hides the improvement the screen is about.
    const metrics = ui.playback.metrics;
    if (!metrics.length) {
      dom.strip.metric.textContent = '';
    } else {
      const window_ = metrics.slice(-50);
      const key = ui.room.metric.key;
      const mean = window_.reduce((total, entry) => total + entry[key], 0)
                 / window_.length;
      dom.strip.metric.textContent = ui.room.metric.label + ' '
                                   + mean.toFixed(1);
    }

    const cleared = isCleared();
    dom.strip.cleared.hidden = !cleared;
    if (cleared) {
      dom.strip.cleared.textContent = 'Chamber cleared — the next one opens '
                                    + 'when you leave';
    }

    dom.strip.stale.hidden = !ui.stale;
    if (ui.stale) {
      dom.strip.stale.textContent = staleMessage();
    }
  }

  /**
   * Every control's enabled state, derived from the playback state alone.
   *
   * This is the only function that decides what may be pressed.
   */
  function paintControls() {
    const state = ui.playback.state;
    const replaying = state === P.REPLAYING;
    const finished = state === P.FINISHED;

    // Nothing may be pressed while the far end is training: there is no
    // batch to act on yet, and a second request would race the first.
    const busy = ui.working;

    // Until the episode target is reached, the transport controls drive the
    // training itself rather than a recording of it.
    if (!ui.trained) {
      dom.play.textContent = ui.running ? 'Pause' : 'Train';
      disable(dom.play, busy || ui.stale);
      disable(dom.step, busy || ui.stale || ui.running);
      disable(dom.reset, busy);
      // There is no finished route to show until there is a recording.
      disable(dom.finalRoute, true);
      disable(dom.replayPlay, false);
      disable(dom.replayStep, false);
      return;
    }

    dom.play.textContent = (state === P.RUNNING || replaying) ? 'Pause' : 'Play';
    // A stale run may not be continued; it may only be reset.
    disable(dom.play, finished || ui.stale || busy);
    disable(dom.step, finished || ui.stale || busy);
    disable(dom.reset, busy
            || (state === P.IDLE && !ui.playback.inProgress && !ui.stale));

    // Deliberately still available once FINISHED — that is exactly when
    // the finished route is what you want to look at.
    //
    // The test is whether there is a route to show. It used to be
    // `episodesPlayed === 0`, which counted episodes *played back* — and now
    // that training finishes straight into the learned route, nothing is ever
    // played back, so that count stays zero and the button would be disabled
    // exactly when it is most wanted.
    disable(dom.finalRoute, busy || finalRoute() === null);

    disable(dom.replayPlay, false);
    disable(dom.replayStep, false);
  }

  /**
   * What the stale banner says.
   *
   * A reason that already tells the player what to do is left alone; anything
   * else gets the instruction appended. Without this the environment message
   * read "Environment changed — retrain to update the policy — reset to
   * apply", which is two instructions and one too many.
   */
  function staleMessage() {
    const reason = ui.staleReason || 'Something changed';
    if (/retrain|reset/i.test(reason)) return reason;
    return reason + ' — reset to apply';
  }

  /** Disabled is the real attribute; aria-disabled only describes it. */
  function disable(node, off) {
    node.disabled = Boolean(off);
    node.setAttribute('aria-disabled', String(Boolean(off)));
  }

  function paintAll() {
    paintStrip();
    paintObjective();
    paintControls();
    paintGeneralisation();
    paintEpisodeDetail();
    paintFinalRouteHint();
    paintInspector();
    if (ui.repaintCharts) ui.repaintCharts(chartData());
  }

  /* ---------------------------------------------------------------------
     Controls
     ------------------------------------------------------------------- */

  /* ---------------------------------------------------------------------
     Training, watched

     Play runs the agent in the chamber and draws it as it goes: one slice
     of work per frame, then the world that came back. Nothing is trained
     out of sight, and nothing has to finish before there is something to
     look at. Replaying a particular episode afterwards is a separate thing
     — see `ui.playback`, which owns that and nothing else now.
     ------------------------------------------------------------------- */

  /**
   * The live frame's detail, with the activation flourish counted in.
   *
   * A replay can look backwards through the frames it recorded to find when
   * the mission stage changed; the live view has no such history, so the
   * transition is noticed as it goes past and counted from there. Either way
   * the count starts on the frame where the *environment* first reported the
   * new stage — nothing here decides that an activation happened.
   *
   * The counter resets whenever the stage goes back to 0, which is every new
   * episode that starts before the terminal.
   */
  function liveDetail(scene) {
    const detail = scene.detail;
    if (!detail || typeof detail.stage !== 'number') return detail || null;

    if (ui.liveStage !== detail.stage) {
      // Only a change *into* a later stage is an activation. Dropping back to
      // stage 0 is a new episode beginning, not the terminal un-activating.
      ui.liveSince = (ui.liveStage !== null && detail.stage > ui.liveStage)
        ? 0 : null;
      ui.liveStage = detail.stage;
    } else if (ui.liveSince !== null) {
      ui.liveSince += 1;
    }
    return Object.assign({}, detail, { sinceActivation: ui.liveSince });
  }

  /**
   * Play the collapse sound on the frame a bridge section fails.
   *
   * `collapseAgo` is 0 on exactly the frame it happened, so the sound is tied
   * to the recording rather than to a wall clock — it lands with the shake and
   * on the same step every time the episode is replayed. `lastCollapse` stops
   * a held frame or a repaint from playing it twice, and `Impact` itself is
   * idempotent within a tenth of a second as a second line of defence.
   */
  function soundCollapse(snapshot) {
    const ago = snapshot && snapshot.collapseAgo;
    if (ago !== 0) {
      // Away from the collapse frame, so the next one may sound again.
      if (ago === null || ago === undefined || ago > 1) ui.lastCollapse = false;
      return;
    }
    if (ui.lastCollapse) return;
    ui.lastCollapse = true;
    if (window.Impact) window.Impact.collapse();
  }

  /** The live world, in the shape the renderer draws frames in. */
  function liveSnapshot() {
    const snapshot = window.Producer.snapshot;
    if (!ui.room || !snapshot || !snapshot.scene) return null;

    const scene = snapshot.scene;
    // The renderer wants these keyed by entity id; the contract carries them
    // as lists, because a list is the honest shape for "what changed".
    const states = {};
    (scene.entityStates || []).forEach(entry => {
      states[entry.id] = entry.state;
    });
    const moved = {};
    (scene.entityPositions || []).forEach(entry => {
      moved[entry.id] = entry.position;
    });

    return {
      room: ui.room,
      position: scene.position,
      entityStates: states,
      entityPositions: moved,
      /* A continuous room reports a velocity with the moment; the three grid
         rooms report null and the renderer takes its usual branch. This is the
         same field a recorded frame carries, which is the property that lets
         one renderer draw both. */
      /* The warehouse as it stands now. Null in every room whose furniture
         does not move, and the renderer falls back to the room's own list. */
      entities: scene.entities || null,
      /* The same field a recorded frame carries, for the live view: the
         mission stage, the sensors and the visible obstacles. Sending it from
         the far end rather than deriving it here is what makes the sensor fan
         drawn during training identical to the one drawn in a replay. */
      detail: liveDetail(scene),
      velocity: scene.velocity || null,
      facing: scene.facing === undefined ? null : scene.facing,
      landingSpeed: ui.room.landingSpeed,
      speedLimit: ui.room.speedLimit,
      /* Live, the clock may as well be the step count the server reports: it
         advances with the simulation rather than with the frame rate, so the
         fans turn at a speed that means something. Pinned under reduced
         motion, exactly as in playback. */
      phase: window.matchMedia('(prefers-reduced-motion: reduce)').matches
        ? 0 : (scene.steps || 0) * 0.16,
    };
  }

  function liveTick(now) {
    const dt = Math.min(0.05, (now - ui.lastLive) / 1000) || 0;
    ui.lastLive = now;

    // One speed setting for both loops, kept on the playback controller so
    // the chips have a single thing to read and write.
    const speed = C.speeds[ui.playback.speed];
    let options = null;

    if (speed.animated) {
      /* Steps are owed at a rate per *second*. A frame is not a unit the
         simulation cares about, and asking for a whole step on every frame
         would make the slowest speed twenty times too fast at 60fps — so the
         fraction is carried over rather than rounded away.

         The rate is scaled by whatever the room says one of its steps is
         worth, because "steps per second" does not mean the same thing in
         every room. A step in a grid room moves the agent a whole cell; a step
         in the wind tunnel is 0.02 s of flight and moves it a five-hundredth
         of the chamber, so the tiers on their own leave it apparently
         motionless. The number comes from the room definition, so the pace of
         a room stays the room's business — see `rooms.py`. Absent, it is 1,
         which is every grid room and is what they had before. */
      const room = ui.room;
      const scale = (room && room.playback && room.playback.liveStepScale) || 1;
      ui.owed += speed.stepsPerSecond * scale * dt;
      const steps = Math.floor(ui.owed);
      if (steps >= 1) {
        ui.owed -= steps;
        options = { steps: steps };
      }
    } else {
      // Turbo is not paced at all: as much as fits in the budget, drawn once.
      options = { budgetMs: C.turboBudgetMs };
    }

    if (options) {
      // Not awaited: a frame must not wait on the network. `slice` resolves
      // null while a request is outstanding, so slow frames simply skip.
      window.Producer.slice(options).then(snapshot => {
        if (!snapshot) return;
        if (snapshot.state !== 'TRAINING') finishedTraining();
      }).catch(failed);
    }

    const live = liveSnapshot();
    ui.world.draw(live);
    ui.lastDetail = live && live.detail;
    paintLiveStrip();
    paintObjective();
    refreshRecordings(now);

    ui.liveFrame = window.requestAnimationFrame(liveTick);
  }

  /**
   * Pull the recorded episodes in every so often while training runs.
   *
   * The graphs and the episode list are meant to fill in *as* the agent
   * learns, not once it has finished — but the recording is a whole batch in
   * one response, so fetching it every frame would be absurd. Every couple of
   * seconds is often enough to watch a curve move and cheap enough to ignore.
   */
  function refreshRecordings(now) {
    if (ui.fetchingBatch) return;
    if (now - ui.lastBatchAt < C.liveBatchRefreshMs) return;
    ui.lastBatchAt = now;
    ui.fetchingBatch = true;

    window.Producer.episodes().then(batch => {
      ui.fetchingBatch = false;
      if (ui.trained) return;
      ui.recorded = batch;
      // Handed to the playback controller so the step inspector can reach
      // individual frames, but its loop stays stopped: the world on screen is
      // the live one, and two loops drawing into one canvas is one too many.
      ui.playback.load(ui.room, batch);
      buildEpisodeList();
      buildInspectorEpisodes();
      paintInspector();
      if (ui.repaintCharts) ui.repaintCharts(batch);
    }).catch(() => { ui.fetchingBatch = false; });
  }

  function startLive() {
    if (ui.liveFrame !== null) return;
    ui.lastLive = window.performance ? window.performance.now() : 0;
    ui.owed = 0;
    ui.liveFrame = window.requestAnimationFrame(liveTick);
  }

  function stopLive() {
    if (ui.liveFrame === null) return;
    window.cancelAnimationFrame(ui.liveFrame);
    ui.liveFrame = null;
  }

  /**
   * Training has reached its episode target.
   *
   * Two things are fetched, and they are not the same thing. The recording is
   * what the agent *did* on the way here, which the graphs and the episode
   * browser are about. The greedy run is what it would do now if it stopped
   * exploring — the route it settled on — and that is what Play shows from
   * here on, because it is the only one of the two that can honestly be
   * called the route it learned.
   */
  function finishedTraining() {
    if (ui.trained) return;
    ui.trained = true;
    ui.running = false;
    stopLive();
    Promise.all([window.Producer.episodes(), window.Producer.replay()])
      .then(results => {
        const batch = results[0];
        const learned = results[1];
        ui.recorded = batch;
        ui.greedy = learned;
        // The learned route rides along inside the batch, so every reader
        // that resolves an episode by number — the renderer, the scrubber,
        // the step inspector — reaches it without knowing it is any
        // different. It stays out of `metrics` on purpose: the episode list
        // and the charts describe training, and this did not happen during
        // training.
        if (learned) batch.episodes = batch.episodes.concat([learned]);
        ui.playback.load(ui.room, batch);
        buildEpisodeList();
        buildInspectorEpisodes();
        // The recording's own metrics, not the played count — nothing has been
        // played back yet, and the graphs are about what happened in training.
        if (ui.repaintCharts) ui.repaintCharts(batch);
        // Straight to the learned route, rather than to the top of a batch of
        // exploratory episodes. Selecting it is what puts playback into
        // REPLAYING, so Play and Pause drive that one route on a loop.
        if (learned) ui.playback.selectEpisode(learned.index);
        // Now the world belongs to the playback controller, so its loop starts.
        ui.playback.start();
        paintAll();
        /* The payoff, at the moment it is earned: the last chamber, escaped.
           `maybeShowEnding` stops the playback loop it just started, which is
           correct — the overlay is what is being looked at now. */
        maybeShowEnding();
      }).catch(failed);
  }

  async function togglePlay() {
    if (ui.trained) {
      // Training is over; Play is now a transport control over the replay.
      const state = ui.playback.state;
      if (state === P.RUNNING || state === P.REPLAYING) ui.playback.pause();
      else ui.playback.play();
      paintAll();
      return;
    }

    if (ui.running) {
      ui.running = false;
      stopLive();
      await window.Producer.hold();
      paintAll();
      return;
    }

    ui.running = true;
    paintControls();
    try {
      await window.Producer.begin();
    } catch (problem) {
      failed(problem);
      return;
    }
    startLive();
  }

  /** One step of training, then stopped. */
  async function stepOnce() {
    if (ui.trained) { ui.playback.step(); paintAll(); return; }
    try {
      await window.Producer.begin();
      const snapshot = await window.Producer.slice({ steps: 1 });
      if (snapshot && snapshot.state !== 'TRAINING') finishedTraining();
      await window.Producer.hold();
    } catch (problem) {
      failed(problem);
      return;
    }
    const stepped = liveSnapshot();
    ui.world.draw(stepped);
    ui.lastDetail = stepped && stepped.detail;
    paintLiveStrip();
    paintObjective();
    paintControls();
  }

  /** The strip while training is being watched. */
  function paintLiveStrip() {
    const snapshot = window.Producer.snapshot;
    if (!snapshot) return;
    const target = snapshot.parameters.episodes;
    dom.strip.episodes.textContent = target
      ? snapshot.progress.count + ' / ' + Math.round(target) + ' episodes'
      : snapshot.progress.count + ' sweeps';
    dom.strip.state.textContent = ui.running ? 'Training' : 'Paused';

    const metric = snapshot.metric;
    if (metric && metric.value !== null && metric.value !== undefined) {
      dom.strip.metric.textContent = metric.label + ' '
                                   + Number(metric.value).toFixed(1);
    }
    paintEnvironment(snapshot);
  }

  /**
   * The environment settings in force, in the status strip.
   *
   * Room 2's subject is a risk/return trade and the risk is a slider, so a
   * mean reward with no indication of the collapse probability that produced
   * it cannot be compared with anything. The value comes from the readout the
   * far end builds, which reads the *environment* rather than the stored
   * parameter — so a slider that has been moved but not yet applied shows the
   * value actually being trained against, not the pending one.
   *
   * Silent in every room that has no such setting.
   */
  function paintEnvironment(snapshot) {
    if (!dom.strip.environment) return;
    const rows = (snapshot && snapshot.readout) || [];
    let shown = '';
    rows.forEach(pair => {
      if (pair[0] === 'Bridge collapse probability') {
        shown = 'Collapse probability ' + pair[1];
      }
    });
    dom.strip.environment.textContent = shown;
    dom.strip.environment.hidden = shown === '';
  }

  /* ---------------------------------------------------------------------
     Talking to the far end

     Training happens on the Python side and takes real time, so the two
     things that trigger it — opening the chamber and resetting it — are
     the only asynchronous functions on this screen. Everything else
     operates on a batch that has already arrived.
     ------------------------------------------------------------------- */

  /**
   * A request failed, and the screen has to say so rather than sit on
   * "Training" forever. The most likely cause by far is that `serve.py`
   * is not running, which the message should make obvious.
   */
  function failed(problem) {
    dom.strip.state.textContent = 'Failed';
    dom.strip.episodes.textContent = String(
      (problem && problem.message) || problem);
    ui.working = false;
    paintControls();
  }

  async function reset() {
    if (ui.working) return;
    ui.working = true;
    // Both loops stopped first: one is animating a batch about to be
    // replaced, the other is asking for training about to be thrown away.
    ui.playback.stop();
    stopLive();
    ui.running = false;
    paintControls();

    let room;
    try {
      // The room itself is rebuilt, because some parameters shape it rather
      // than the learning, and Reset is where those take effect.
      room = await window.Producer.restart(currentParameters());
    } catch (problem) {
      failed(problem);
      return;
    }
    ui.working = false;
    // Back to how the chamber opened: nothing learned, waiting on Play.
    ui.trained = false;

    ui.room = room;
    ui.playback.reset();
    ui.playback.load(ui.room, EMPTY_BATCH);

    ui.stale = false;
    ui.staleReason = '';
    ui.inspect = { episode: null, step: 0 };
    ui.recorded = null;
    // The route belonged to the policy that was just thrown away. Keeping it
    // would leave the previous run's route on screen beside a fresh chamber.
    ui.greedy = null;
    ui.testedRoom = null;
    // A fresh run has not been escaped yet, so the ending is owed again.
    ui.endingSeen = false;
    ui.clearedFlashSeen = false;
    if (ui.clearedFlash) ui.clearedFlash.hide();
    ui.lastBatchAt = 0;

    buildLegend();
    buildRoomInfo();
    buildAlgorithmInfo();
    buildEpisodeList();
    buildInspectorEpisodes();
    buildGeneralisation();
    paintEvaluation(null);
    if (ui.repaintCharts) ui.repaintCharts([]);
    refit();
    paintAll();
    ui.world.draw({ room: ui.room, position: null });
  }

  function setSidebar(open) {
    ui.open = open;
    document.body.classList.toggle('is-open', open);
    dom.sidebar.setAttribute('aria-hidden', String(!open));
    dom.toggle.setAttribute('aria-expanded', String(open));
    dom.toggle.setAttribute('aria-label', open ? 'Close the sidebar'
                                               : 'Open the sidebar');
    dom.toggleGlyph.innerHTML = open ? '&rsaquo;' : '&lsaquo;';
    // Re-fitted during the slide and again after it, so the world never
    // jumps at the end of the transition.
    refit();
    window.setTimeout(refit, C.timing.sidebarMs + 20);
  }

  function leave() {
    if (ui.leaving) return;

    /* The last chamber ends the story rather than returning to the select.
       Shown here rather than navigated past, so Exit on a cleared final room
       reaches the ending instead of the menu. */
    if (maybeShowEnding()) return;

    const unfinished = (ui.playback && ui.playback.inProgress) || ui.running;
    if (unfinished
        && !window.confirm('This run has not finished. Leaving discards it.')) {
      return;
    }

    goToChamberSelect();
  }

  /**
   * Hand the run back and navigate to the chamber select.
   *
   * Split out of `leave` because the ending's Main Menu button has to do the
   * identical thing — including reporting the chamber as cleared, which is
   * what unlocks progress. Two copies of that would eventually have differed,
   * and the copy that forgot would silently lose the player's progress.
   */
  function goToChamberSelect() {
    ui.leaving = true;
    // Both loops stopped before navigating, so nothing is still drawing into
    // a canvas that is about to go away.
    ui.playback.stop();
    stopLive();
    // A room that never opened — the far end was not running — has nothing
    // to unload, and the exit must still work in that case.
    if (ui.room) ui.playback.load(ui.room, { episodes: [], metrics: [] });

    // Read from the query rather than the room, which may never have loaded.
    const number = ui.roomId.replace('room', '');
    try {
      // Come back to the chamber just left, rather than to whichever one
      // is furthest along.
      window.sessionStorage.setItem('projectR5:returnTo', number);

      // Report completion. The chamber select owns the progress integer,
      // so all that is handed over is which chamber was cleared; it does
      // the writing and plays the unlock sequence. Without this the next
      // chamber stays locked however well this one went.
      if (isCleared()) {
        window.sessionStorage.setItem('projectR5:justCompleted', number);
      }
    } catch (problem) {
      // Storage unavailable: the chamber simply is not remembered.
    }
    window.location.href = '../levels/';
  }

  /* ---------------------------------------------------------------------
     Wiring
     ------------------------------------------------------------------- */

  dom.play.addEventListener('click', togglePlay);
  dom.step.addEventListener('click', stepOnce);
  dom.reset.addEventListener('click', reset);
  dom.evaluate.addEventListener('click', runEvaluation);
  dom.testRoom.addEventListener('click', runTestRoom);
  dom.anotherTestRoom.addEventListener('click', runTestRoom);
  dom.finalRoute.addEventListener('click', showFinalRoute);
  dom.toggle.addEventListener('click', () => setSidebar(!ui.open));
  dom.exit.addEventListener('click', leave);

  dom.replayPlay.addEventListener('click', togglePlay);
  dom.replayStep.addEventListener('click', () => {
    ui.playback.step();
    paintAll();
  });
  dom.replayExit.addEventListener('click', () => {
    ui.playback.clearEpisode();
    paintAll();
  });
  dom.scrub.addEventListener('input', () => {
    ui.playback.scrubTo(Number(dom.scrub.value) / 1000);
  });

  dom.inspectEpisode.addEventListener('change', () => {
    ui.inspect.episode = Number(dom.inspectEpisode.value);
    ui.inspect.step = 0;
    paintInspector();
  });
  dom.inspectStep.addEventListener('input', () => {
    ui.inspect.step = Number(dom.inspectStep.value);
    paintInspector();
  });
  dom.inspectBack.addEventListener('click', () => stepInspector(-1));
  dom.inspectForward.addEventListener('click', () => stepInspector(1));

  document.addEventListener('keydown', event => {
    /* While the ending is up it owns the keyboard: its own buttons are
       focusable and tabbable, and none of the room's shortcuts should reach
       past it. Escape in particular would otherwise navigate to the chamber
       select from behind the overlay. */
    if (ui.ending && ui.ending.isOpen) return;

    const tag = (event.target.tagName || '').toLowerCase();
    // Never hijack a key while the player is inside a control.
    if (tag === 'input' || tag === 'select' || tag === 'textarea') {
      if (event.key !== 'Escape') return;
    }

    if (event.key === ' ' || event.code === 'Space') {
      event.preventDefault();
      togglePlay();
      return;
    }
    if (event.key === 's' || event.key === 'S') {
      if (!dom.step.disabled) dom.step.click();
      return;
    }
    if (event.key === 'r' || event.key === 'R') {
      if (!dom.reset.disabled) dom.reset.click();
      return;
    }
    if (event.key === 'Escape') {
      leave();
      return;
    }
    // Tab opens the sidebar and then behaves normally, so keyboard
    // navigation is never trapped.
    if (event.key === 'Tab' && !ui.open) setSidebar(true);
  });

  window.addEventListener('resize', refit);

  document.addEventListener('visibilitychange', () => {
    if (document.hidden) {
      // The loop is stopped and playback is paused, so nothing advances
      // unseen and the position is exactly where it was on return.
      ui.playback.stop();
      // Training is paused rather than merely left undrawn: it runs on the
      // far end, and a hidden page must not keep asking for work.
      stopLive();
      if (ui.running) {
        ui.running = false;
        window.Producer.hold().catch(() => {});
      }
      if (ui.playback.state === P.RUNNING || ui.playback.state === P.REPLAYING) {
        ui.playback.pause();
        paintAll();
      }
      return;
    }
    ui.playback.start();
  });

  /* ---------------------------------------------------------------------
     Boot
     ------------------------------------------------------------------- */

  async function boot() {
    ui.roomId = roomFromQuery();
    ui.working = true;

    // The chamber number is what the API takes; the id is this screen's way
    // of naming the same thing.
    const number = Number(ui.roomId.replace('room', ''));

    window.Renderer.readPalette();
    ui.world = window.Renderer.create(dom.canvas);
    ui.inspector = window.Renderer.create(dom.inspectGrid);

    ui.playback = P.create({
      frame: function (snapshot) {
        ui.world.draw(snapshot);
        ui.lastDetail = snapshot && snapshot.detail;
        paintObjective();
        soundCollapse(snapshot);
        // The frame R-5 finishes arriving is a frame, not an episode boundary
        // and not a strip repaint — so this is asked here, where frames are.
        maybeFlashCleared();

        // A completed episode changes the strip, the charts and the lists,
        // but Turbo completes many of them between two frames — so the
        // work is marked and done once here rather than once per episode.
        if (ui.dirty) {
          ui.dirty = false;
          paintStrip();
          paintControls();
          paintFinalRouteHint();
          buildEpisodeList();
          buildInspectorEpisodes();
          paintInspector();
          if (ui.repaintCharts) ui.repaintCharts(chartData());
        }

        if (dom.scrub && !dom.scrub.hidden) {
          dom.scrub.value = String(Math.round(ui.playback.progress * 1000));
        }
      },
      episode: function () { ui.dirty = true; },
      state: function () { paintControls(); paintStrip(); },
    });

    buildSpeeds();
    // Its own run on the far end, so comparing never disturbs this one.
    ui.compare = window.Compare.mount(dom.compare, number);
    ui.clearedFlash = window.Cleared.mount(dom.cleared);

    /* The ending sequence. Mounted for every room and shown by none but the
       last: `maybeShowEnding` checks `room.isFinal`, which only the final
       chamber sends. The four buttons are wired here because what each of
       them means belongs to the screen that owns the run — the module draws
       and animates, and navigates nothing itself. */
    ui.ending = window.Ending.mount(dom.ending, {
      /* The recorded escape, replayed through the ordinary path: the same
         frames, renderer and transport controls as any other episode. */
      replay: function () {
        const index = escapeRoute();
        closeEnding();
        if (index === null) return;
        ui.playback.selectEpisode(index);
        ui.playback.start();
        paintAll();
      },
      /* The graphs of the run that produced it. The section is opened rather
         than anything being rebuilt — it has been filled in all along. */
      results: function () {
        closeEnding();
        const charts = dom.charts.closest('details');
        if (charts) charts.open = true;
        if (dom.generalisation && !dom.generalisation.hidden) {
          dom.generalisation.open = true;
        }
        if (!ui.open) setSidebar(true);
        if (charts) charts.scrollIntoView({ block: 'nearest' });
      },
      /* A fresh run of the same chamber, which is exactly what Reset is. */
      again: function () {
        closeEnding();
        reset();
      },
      menu: function () {
        closeEnding();
        goToChamberSelect();
      },
    });

    dom.strip.state.textContent = 'Opening';
    paintControls();

    let room;
    try {
      room = await window.Producer.open(number);
    } catch (problem) {
      failed(problem);
      return;
    }
    ui.working = false;

    ui.room = room;
    ui.room.parameterSchema.forEach(specification => {
      ui.parameters[specification.key] = specification.default;
    });

    /* Built here rather than before the room arrives, because a room may name
       its own graphs. Rooms 1 to 4 name none and get the shared four; room 5
       declares ten, including the one that compares the frozen policy on the
       layouts it trained on against the layouts it never saw. */
    ui.repaintCharts = window.Charts.build(dom.charts, ui.room.charts);

    // An empty batch, on purpose: nothing has been learned yet. The renderer
    // draws a room with no episode in it perfectly well — the chamber, its
    // legend and what it says about itself are all up immediately, and there
    // is simply no agent walking about until Play is pressed.
    ui.playback.load(ui.room, EMPTY_BATCH);

    buildLegend();
    buildParameters();
    buildRoomInfo();
    buildAlgorithmInfo();
    buildEpisodeList();
    buildInspectorEpisodes();
    buildGeneralisation();

    // Open, not closed. Everything that makes the chamber legible — what it
    // is, its parameters, its graphs — lives in there, and a screen that
    // opens with all of it hidden behind one handle reads as an empty grid.
    // `setSidebar` re-fits the world into what is left, so this replaces the
    // bare `refit()` rather than sitting beside it.
    setSidebar(true);
    paintAll();
    // Drawn once, rather than by starting the playback loop. That loop draws
    // whatever the batch says, and there is no batch yet — and once training
    // begins the live loop owns this canvas. Two loops painting into one
    // canvas is one too many, which is what made the agent flicker.
    ui.world.draw({ room: ui.room, position: null });
  }

  /* The session is the far end's to keep only as long as this page holds
     it, so it is handed back on the way out however the page goes away. */
  window.addEventListener('pagehide', () => {
    window.Producer.release();
    if (ui.compare) ui.compare.release();
  });

  boot();
})();
