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
  };

  /* Placeholder copy for the algorithm section. The real text arrives with
     the algorithms; the section exists now so the shape of the sidebar is
     settled and nothing has to be rearranged later. */
  const ALGORITHM_PLACEHOLDER = {
    label: 'SARSA',
    summary: 'Learns the value of the policy it is actually following, '
           + 'including the exploratory steps that policy takes.',
    updateRule: 'Q(s,a) ← Q(s,a) + α[ r + γ Q(s′,a′) − Q(s,a) ]',
    watchFor: 'Awaiting the algorithm layer. Until it is wired up, the '
            + 'trajectories on this screen come from the mock producer and '
            + 'nothing here is learned.',
  };

  const OUTCOME_LABELS = {
    success: 'escaped', failure: 'lost', timeout: 'ran out',
  };

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

    // A parameter that cannot apply live invalidates what the run has
    // done so far. The run is not thrown away here — that is the point of
    // making Reset explicit.
    if (!specification.appliesLive && ui.playback.inProgress) {
      ui.stale = true;
      ui.staleReason = specification.label + ' changed';
    }
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

  function buildRoomInfo() {
    // Rebuilt rather than patched, because a parameter can change what
    // this section says about the room.
    const info = window.Mock.room(ui.roomId, ui.parameters).info;

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
    const text = ui.room.algorithm || ALGORITHM_PLACEHOLDER;
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
   * The episode that counts as "the route it settled on".
   *
   * The last one that actually reached the exit, rather than simply the
   * last one played: with exploration never decaying to zero, the final
   * episode of a run is quite often a fall, and that is not the thing
   * anybody wants to watch. Falls back to the last episode played when
   * nothing has succeeded, because the most recent failure is more use
   * than refusing to show anything.
   */
  function finalRoute() {
    const played = ui.playback.metrics;
    if (!played.length) return null;

    for (let at = played.length - 1; at >= 0; at -= 1) {
      const episode = ui.playback.episodeAt(played[at].episode);
      if (episode && episode.outcome === 'success') {
        return { index: episode.index, succeeded: true };
      }
    }
    return { index: played[played.length - 1].episode, succeeded: false };
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
    if (ui.everCleared) return true;
    const route = finalRoute();
    if (route && route.succeeded) ui.everCleared = true;
    return ui.everCleared;
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
        'Nothing has been played yet, so there is no route to show.';
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
      + 'Played at a readable speed whatever the training speed was.';
  }

  /* ---------------------------------------------------------------------
     The episode browser
     ------------------------------------------------------------------- */

  function buildEpisodeList() {
    const played = ui.playback.metrics;
    dom.episodeList.innerHTML = '';

    if (!played.length) {
      dom.episodeList.appendChild(
        element('p', 'hint', 'Episodes appear here as they are played.'));
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

    const rows = [
      ['Episode', String(selected + 1)],
      ['Outcome', OUTCOME_LABELS[episode.outcome] || episode.outcome],
      ['Reward', episode.totalReward.toFixed(0)],
      ['Exploration', episode.epsilon.toFixed(3)],
      ['Steps', String(episode.steps.length - 1)],
    ];
    const list = element('dl', 'readout');
    rows.forEach(pair => {
      list.appendChild(element('dt', null, pair[0]));
      list.appendChild(element('dd', null, pair[1]));
    });
    dom.episodeDetail.appendChild(list);
  }

  /* ---------------------------------------------------------------------
     The step inspector

     Any completed episode, any single step, held still. It reads the
     batch through `frameAt` and changes nothing, so it can be scrubbed
     while a run is going without disturbing it.
     ------------------------------------------------------------------- */

  function buildInspectorEpisodes() {
    const played = ui.playback.metrics;
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

  function paintStrip() {
    const played = ui.playback.episodesPlayed;
    dom.strip.episodes.textContent = played + ' / ' + ui.playback.total
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
      dom.strip.stale.textContent = ui.staleReason + ' — reset to apply';
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

    dom.play.textContent = (state === P.RUNNING || replaying) ? 'Pause' : 'Play';
    // A stale run may not be continued; it may only be reset.
    disable(dom.play, finished || ui.stale);
    disable(dom.step, finished || ui.stale);
    disable(dom.reset, state === P.IDLE && !ui.playback.inProgress && !ui.stale);

    // Deliberately still available once FINISHED — that is exactly when
    // the finished route is what you want to look at.
    disable(dom.finalRoute, ui.playback.episodesPlayed === 0);

    disable(dom.replayPlay, false);
    disable(dom.replayStep, false);
  }

  /** Disabled is the real attribute; aria-disabled only describes it. */
  function disable(node, off) {
    node.disabled = Boolean(off);
    node.setAttribute('aria-disabled', String(Boolean(off)));
  }

  function paintAll() {
    paintStrip();
    paintControls();
    paintEpisodeDetail();
    paintFinalRouteHint();
    paintInspector();
    if (ui.repaintCharts) ui.repaintCharts(ui.playback.metrics);
  }

  /* ---------------------------------------------------------------------
     Controls
     ------------------------------------------------------------------- */

  function togglePlay() {
    const state = ui.playback.state;
    if (state === P.RUNNING || state === P.REPLAYING) ui.playback.pause();
    else ui.playback.play();
    paintAll();
  }

  function reset() {
    // The room itself is rebuilt: some parameters shape it rather than the
    // learning, and Reset is where they take effect. When a real producer
    // is behind this, it is the same two calls.
    ui.room = window.Mock.room(ui.roomId, ui.parameters);
    ui.playback.reset();
    ui.playback.load(ui.room, window.Mock.batch(ui.room, currentParameters()));

    ui.stale = false;
    ui.staleReason = '';
    ui.inspect = { episode: null, step: 0 };

    buildLegend();
    buildRoomInfo();
    buildEpisodeList();
    buildInspectorEpisodes();
    refit();
    paintAll();
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

    if (ui.playback && ui.playback.inProgress
        && !window.confirm('This run has not finished. Leaving discards it.')) {
      return;
    }

    ui.leaving = true;
    // Stopped before navigating, so nothing is still drawing into a canvas
    // that is about to go away.
    ui.playback.stop();
    ui.playback.load(ui.room, { episodes: [], metrics: [] });

    const number = ui.room.id.replace('room', '');
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
  dom.step.addEventListener('click', () => { ui.playback.step(); paintAll(); });
  dom.reset.addEventListener('click', reset);
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

  function boot() {
    ui.roomId = roomFromQuery();
    ui.room = window.Mock.room(ui.roomId);

    ui.room.parameterSchema.forEach(specification => {
      ui.parameters[specification.key] = specification.default;
    });
    // Built once more now the parameters are known, since some of them
    // shape what the room says about itself.
    ui.room = window.Mock.room(ui.roomId, ui.parameters);

    window.Renderer.readPalette();
    ui.world = window.Renderer.create(dom.canvas);
    ui.inspector = window.Renderer.create(dom.inspectGrid);

    ui.playback = P.create({
      frame: function (snapshot) {
        ui.world.draw(snapshot);

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
          if (ui.repaintCharts) ui.repaintCharts(ui.playback.metrics);
        }

        if (dom.scrub && !dom.scrub.hidden) {
          dom.scrub.value = String(Math.round(ui.playback.progress * 1000));
        }
      },
      episode: function () { ui.dirty = true; },
      state: function () { paintControls(); paintStrip(); },
    });

    ui.playback.load(ui.room, window.Mock.batch(ui.room, currentParameters()));

    buildLegend();
    buildSpeeds();
    buildParameters();
    buildRoomInfo();
    buildAlgorithmInfo();
    ui.repaintCharts = window.Charts.build(dom.charts);
    buildEpisodeList();
    buildInspectorEpisodes();

    refit();
    paintAll();
    ui.playback.start();
  }

  boot();
})();
