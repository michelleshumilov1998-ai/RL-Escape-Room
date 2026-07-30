'use strict';

/* =====================================================================
   The level screen's shell.

   It reads snapshots and writes intents, and that is all.  It has no
   opinion about how the simulation works, and every control's enabled
   state is derived from the run state that came back in the last snapshot
   rather than from a flag kept here — so an illegal transition is not
   something this file has to remember to prevent.

       IDLE       built, nothing done
       TRAINING   learning; may advance
       TRAINED    finished; a greedy replay is available
       REPLAYING  showing the learned policy, learning nothing
   ===================================================================== */

const STATE_LABELS = {
  IDLE: 'Idle',
  TRAINING: 'Training',
  TRAINED: 'Trained',
  REPLAYING: 'Replaying',
};

const dom = {
  canvas: document.getElementById('grid'),
  strip: {
    state: document.getElementById('strip-state'),
    progress: document.getElementById('strip-progress'),
    metric: document.getElementById('strip-metric'),
    stale: document.getElementById('strip-stale'),
  },
  legend: document.getElementById('legend'),
  exit: document.getElementById('exit'),
  toggle: document.getElementById('toggle'),
  toggleGlyph: document.getElementById('toggle-glyph'),
  sidebar: document.getElementById('sidebar'),
  sector: document.getElementById('room-sector'),
  name: document.getElementById('room-name'),
  play: document.getElementById('play'),
  step: document.getElementById('step'),
  reset: document.getElementById('reset'),
  speeds: document.getElementById('speeds'),
  speedHint: document.getElementById('speed-hint'),
  algorithm: document.getElementById('algorithm'),
  parameters: document.getElementById('parameters'),
  levelInfo: document.getElementById('level-info'),
  algorithmInfo: document.getElementById('algorithm-info'),
  charts: document.getElementById('charts'),
  readout: document.getElementById('readout'),
};

/* What this screen plots.
 *
 * A planner measures itself per sweep rather than per episode, so these
 * are not the learner's four series — but they are drawn by the same
 * module, from the same shaped array. Every one of them is derived from
 * the snapshots that already arrive; none of it asks the simulation for
 * anything new.
 */
const PLANNER_SERIES = [
  { key: 'delta', label: 'Largest value change per sweep', scale: 'log',
    threshold: context => (context && context.threshold) || 1e-4 },
  { key: 'startValue', label: 'V(start) per sweep' },
  { key: 'meanAbsValue', label: 'Mean |V| across the chamber' },
  { key: 'policyChanges', label: 'Cells whose best action changed' },
];

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

const ui = {
  room: 1,
  speed: 'normal',
  open: false,
  // Fractional sweeps owed to a planner, so the animated tiers can pace it
  // in sweeps per second rather than per frame.
  owed: 0,
  frame: null,
  lastTime: 0,
  // Replay playback, driven here so it runs at one readable rate whatever
  // the training speed was set to.
  replay: { index: 0, elapsed: 0, waiting: 0 },
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
  const number = Number.parseInt(raw, 10);
  return Number.isFinite(number) && number >= 1 && number <= 5 ? number : 1;
}

function sidebarInset() {
  if (!ui.open) return 0;
  return dom.sidebar.getBoundingClientRect().width;
}

function refit() {
  const describe = Sim.describe;
  Renderer.resize(describe ? describe.layout : null, sidebarInset());
}

/* ---------------------------------------------------------------------
   Building the sidebar out of the room's data
   ------------------------------------------------------------------- */

function buildLegend() {
  dom.legend.innerHTML = '';
  Renderer.legend().forEach(entry => {
    const item = element('li');

    // A canvas, not an SVG swatch: the key is drawn by the very code that
    // draws the chamber, so it cannot show something the grid does not.
    const swatch = document.createElement('canvas');
    swatch.className = 'swatch';
    swatch.setAttribute('aria-hidden', 'true');
    item.appendChild(swatch);
    item.appendChild(element('span', null, entry.label));
    dom.legend.appendChild(item);

    // Drawn once it is in the document, so it has a measurable size.
    Renderer.swatch(swatch, entry.kind);
  });
}

function buildSpeeds() {
  const config = Sim.config;
  dom.speeds.innerHTML = '';
  Object.keys(config.speeds).forEach(key => {
    const chip = element('button', 'chip', config.speeds[key].label);
    chip.type = 'button';
    chip.setAttribute('role', 'radio');
    chip.dataset.speed = key;
    chip.addEventListener('click', () => {
      ui.speed = key;
      ui.owed = 0;
      paintSpeeds();
    });
    dom.speeds.appendChild(chip);
  });
  ui.speed = config.speedDefault;
  paintSpeeds();
}

function paintSpeeds() {
  const config = Sim.config;
  Array.prototype.forEach.call(dom.speeds.children, chip => {
    const on = chip.dataset.speed === ui.speed;
    chip.setAttribute('aria-checked', String(on));
  });
  const animated = config.speeds[ui.speed].animated;
  dom.speedHint.textContent = animated
    ? 'Every sweep is drawn.'
    : 'Runs a batch of sweeps per frame inside a ' +
      config.turboBudgetMs + 'ms budget and draws only the result.';
}

function buildAlgorithms() {
  const describe = Sim.describe;
  dom.algorithm.innerHTML = '';
  describe.algorithms.forEach(entry => {
    const option = element('option');
    option.value = entry.key;
    const isDefault = entry.key === describe.algorithmDefault;
    option.textContent = entry.label + (isDefault ? '  (assignment default)' : '');
    dom.algorithm.appendChild(option);
  });
  dom.algorithm.value = Sim.snapshot.algorithm;
  dom.algorithm.addEventListener('change', () => {
    Sim.setAlgorithm(dom.algorithm.value).then(paint);
  });
}

function buildParameters() {
  const describe = Sim.describe;
  dom.parameters.innerHTML = '';

  const groups = [
    { scope: 'live', label: 'Live — applies immediately' },
    { scope: 'reset', label: 'Requires reset' },
  ];

  groups.forEach(group => {
    const members = describe.parameters.filter(p => p.scope === group.scope);
    if (!members.length) return;

    const heading = element('p', 'group-label', group.label);
    if (group.scope === 'reset') heading.classList.add('is-reset');
    dom.parameters.appendChild(heading);

    members.forEach(specification => {
      dom.parameters.appendChild(parameterControl(specification));
    });
  });
}

function parameterControl(specification) {
  const wrapper = element('div', 'parameter');
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
  const choices = specification.choices || null;
  if (choices) {
    slider.min = '0';
    slider.max = String(choices.length - 1);
    slider.step = '1';
  } else {
    slider.min = String(specification.minimum);
    slider.max = String(specification.maximum);
    slider.step = String(specification.step || 0.01);
  }
  wrapper.appendChild(slider);

  wrapper.appendChild(element('p', 'hint', specification.explanation));

  const footer = element('div', 'parameter-head');
  const range = choices
    ? choices[0] + ' – ' + choices[choices.length - 1]
    : specification.minimum + ' – ' + specification.maximum;
  footer.appendChild(element('span', 'hint', 'valid ' + range));
  const restore = element('button', 'parameter-default', 'default');
  restore.type = 'button';
  restore.addEventListener('click', () => {
    Sim.resetParameter(specification.name).then(paint);
  });
  footer.appendChild(restore);
  wrapper.appendChild(footer);

  function show(current) {
    const digits = current < 0.01 ? 6 : 3;
    value.textContent = Number(current).toFixed(digits).replace(/0+$/, '')
      .replace(/\.$/, '');
    slider.value = choices
      ? String(nearestChoice(choices, current))
      : String(current);
  }

  slider.addEventListener('input', () => {
    const next = choices
      ? choices[Number(slider.value)]
      : Number(slider.value);
    value.textContent = String(next);
    Sim.setParameters({ [specification.name]: next }).then(paint);
  });

  wrapper.dataset.parameter = specification.name;
  wrapper.show = show;
  show(Sim.snapshot.parameters[specification.name]);
  return wrapper;
}

function nearestChoice(choices, current) {
  let best = 0;
  for (let index = 1; index < choices.length; index += 1) {
    if (Math.abs(choices[index] - current) < Math.abs(choices[best] - current)) {
      best = index;
    }
  }
  return best;
}

function buildInfo() {
  const describe = Sim.describe;
  const info = describe.room.info;

  dom.sector.textContent = describe.room.sector;
  dom.name.textContent = describe.room.name;

  dom.levelInfo.innerHTML = '';
  dom.levelInfo.appendChild(block('Objective', info.objective));

  const obstacles = element('div', 'info-block');
  obstacles.appendChild(element('p', 'group-label', 'Obstacles'));
  const list = element('ul');
  info.obstacles.forEach(line => list.appendChild(element('li', null, line)));
  obstacles.appendChild(list);
  dom.levelInfo.appendChild(obstacles);

  dom.levelInfo.appendChild(block('Actions', info.actions));

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
  dom.levelInfo.appendChild(rewards);

  dom.levelInfo.appendChild(block('Termination', info.termination));
  if (info.note) dom.levelInfo.appendChild(block('Note', info.note));
}

function block(heading, text) {
  const wrapper = element('div', 'info-block');
  wrapper.appendChild(element('p', 'group-label', heading));
  wrapper.appendChild(element('p', null, text));
  return wrapper;
}

function paintAlgorithmInfo() {
  const describe = Sim.describe;
  const current = describe.algorithms.filter(
    entry => entry.key === Sim.snapshot.algorithm)[0];
  if (!current) return;

  dom.algorithmInfo.innerHTML = '';
  dom.algorithmInfo.appendChild(block(current.label, current.summary));

  const rule = element('div', 'info-block');
  rule.appendChild(element('p', 'group-label', 'Update rule'));
  rule.appendChild(element('pre', 'rule', current.updateRule));
  dom.algorithmInfo.appendChild(rule);

  dom.algorithmInfo.appendChild(block('What to watch', current.watchFor));
  dom.algorithmInfo.appendChild(block(
    'Model',
    current.needsModel
      ? 'This method is given the whole model of the chamber and plans from '
        + 'it. It never takes a step while it is learning.'
      : 'This method is given no model. The only way it finds out what an '
        + 'action does is to take it.'));
}

/* ---------------------------------------------------------------------
   Painting the current snapshot
   ------------------------------------------------------------------- */

function paint() {
  const snapshot = Sim.snapshot;
  if (!snapshot) return;

  const state = snapshot.state;
  const replaying = state === 'REPLAYING';

  dom.strip.state.textContent = STATE_LABELS[state] || state;
  dom.strip.progress.textContent =
    snapshot.progress.count + ' ' + snapshot.progress.unit;

  if (replaying && snapshot.replay) {
    dom.strip.metric.textContent =
      snapshot.replay.steps + ' steps · ' +
      snapshot.replay.totalReward.toFixed(0) + ' reward';
  } else if (snapshot.metric.value === null
             || snapshot.metric.value === undefined) {
    dom.strip.metric.textContent = '';
  } else {
    dom.strip.metric.textContent =
      snapshot.metric.label + ' ' +
      Number(snapshot.metric.value).toFixed(2);
  }

  dom.strip.stale.hidden = !snapshot.stale;
  if (snapshot.stale) {
    dom.strip.stale.textContent = snapshot.staleReason + ' — reset to apply';
  }

  // Every control's state comes from the run state, not from a local flag.
  dom.play.textContent = playLabel(snapshot);
  dom.play.disabled = snapshot.stale && state !== 'IDLE';
  dom.step.disabled = replaying || snapshot.progress.converged;
  dom.reset.disabled = state === 'IDLE' && snapshot.progress.count === 0
    && !snapshot.stale;
  dom.algorithm.value = snapshot.algorithm;
  dom.algorithm.disabled = false;

  Array.prototype.forEach.call(
    dom.parameters.querySelectorAll('[data-parameter]'), node => {
      if (node.show) node.show(snapshot.parameters[node.dataset.parameter]);
    });

  paintAlgorithmInfo();
  paintAnalysis();
}

function playLabel(snapshot) {
  if (snapshot.state === 'REPLAYING') return 'Continue training';
  if (snapshot.state === 'TRAINED') return 'Train further';
  return snapshot.playing ? 'Pause' : 'Play';
}

function paintAnalysis() {
  const snapshot = Sim.snapshot;
  const learned = snapshot.learned || {};
  const threshold = snapshot.parameters.theta || 1e-4;

  // The threshold is a parameter, so it is passed on every repaint rather
  // than baked into the series when the charts were built.
  if (ui.repaintCharts) {
    ui.repaintCharts(ui.metrics || [], { threshold: threshold });
  }

  dom.readout.innerHTML = '';
  const rows = [
    ['Sweeps', learned.sweeps === undefined ? '—' : String(learned.sweeps)],
    ['Largest change', learned.delta === undefined ? '—'
      : Number(learned.delta).toExponential(2)],
    ['Threshold', Number(threshold).toExponential(0)],
    ['V(start)', learned.startValue === undefined ? '—'
      : Number(learned.startValue).toFixed(3)],
    ['Converged', learned.converged ? 'yes' : 'no'],
  ];
  if (learned.rounds !== undefined) {
    rows.splice(1, 0, ['Improvement rounds', String(learned.rounds)]);
  }
  rows.forEach(pair => {
    dom.readout.appendChild(element('dt', null, pair[0]));
    dom.readout.appendChild(element('dd', null, pair[1]));
  });
}

/* ---------------------------------------------------------------------
   The loop
   ------------------------------------------------------------------- */

function agentPosition() {
  const snapshot = Sim.snapshot;
  const replay = snapshot.replay;

  if (snapshot.state !== 'REPLAYING' || !replay) {
    const cell = snapshot.grid.agent;
    return { row: cell[0], col: cell[1] };
  }

  const frames = replay.frames;
  const index = Math.min(ui.replay.index, frames.length - 1);
  const current = frames[index].cell;
  const next = frames[Math.min(index + 1, frames.length - 1)].cell;

  if (reduceMotion.matches) return { row: current[0], col: current[1] };

  const share = Math.min(1, ui.replay.elapsed);
  return {
    row: current[0] + (next[0] - current[0]) * share,
    col: current[1] + (next[1] - current[1]) * share,
  };
}

function advanceReplay(dt) {
  const replay = Sim.snapshot.replay;
  if (!replay) return;
  const config = Sim.config.render;

  if (ui.replay.waiting > 0) {
    ui.replay.waiting -= dt;
    if (ui.replay.waiting > 0) return;
    ui.replay.index = 0;
    ui.replay.elapsed = 0;
  }

  ui.replay.elapsed += dt * config.replay_steps_per_second;
  while (ui.replay.elapsed >= 1) {
    ui.replay.elapsed -= 1;
    ui.replay.index += 1;
    if (ui.replay.index >= replay.frames.length - 1) {
      // Hold on the final frame before going round again.
      ui.replay.index = replay.frames.length - 1;
      ui.replay.elapsed = 0;
      ui.replay.waiting = config.replay_pause_seconds;
      break;
    }
  }
}

function replayTrail() {
  const snapshot = Sim.snapshot;
  if (snapshot.state !== 'REPLAYING' || !snapshot.replay) return null;
  return snapshot.replay.frames
    .slice(0, ui.replay.index + 1)
    .map(frame => frame.cell);
}

async function trainingWork(dt) {
  const snapshot = Sim.snapshot;
  if (snapshot.state !== 'TRAINING' || !snapshot.playing) return;
  if (Sim.busy) return;

  const config = Sim.config;
  const tier = config.speeds[ui.speed];

  if (!tier.animated) {
    // Turbo: one time-budgeted batch per frame, result drawn once.
    await Sim.advance({ budgetMs: config.turboBudgetMs });
    recordSweep();
    paint();
    return;
  }

  if (snapshot.progress.unit === 'sweeps') {
    // A planner is paced in sweeps per second, because room 1 converges in
    // under thirty of them and one per frame would be over instantly.
    const rate = config.plannerSweepsPerSecond[ui.speed] || 8;
    ui.owed += rate * dt;
    if (ui.owed < 1) return;
    const steps = Math.floor(ui.owed);
    ui.owed -= steps;
    await Sim.advance({ steps: steps });
  } else {
    await Sim.advance({ steps: tier.steps_per_frame });
  }

  recordSweep();
  paint();
}

/**
 * Bank one sweep's worth of measurements.
 *
 * Four numbers, all read or derived from the snapshot that just arrived:
 * the largest change, the value at the start cell, the mean size of the
 * whole table, and how many cells changed their best action since the
 * last sweep. The last of those is the interesting one — the plan
 * normally stops changing well before the numbers do.
 */
function recordSweep() {
  const learned = Sim.snapshot.learned || {};
  if (learned.delta === undefined || learned.delta === null) return;

  ui.metrics = ui.metrics || [];
  ui.metrics.push({
    sweep: learned.sweeps,
    delta: learned.delta,
    startValue: learned.startValue,
    meanAbsValue: meanAbsolute(learned.values),
    policyChanges: countPolicyChanges(learned.policy),
  });

  // A long run is bucketed by the charts anyway; this only bounds memory.
  if (ui.metrics.length > 4000) ui.metrics.shift();
}

function meanAbsolute(values) {
  if (!values) return 0;
  const keys = Object.keys(values);
  if (!keys.length) return 0;
  let total = 0;
  keys.forEach(key => { total += Math.abs(values[key]); });
  return total / keys.length;
}

/** How many cells changed their best action, against the previous sweep. */
function countPolicyChanges(policy) {
  if (!policy) return 0;
  const previous = ui.previousPolicy;
  ui.previousPolicy = policy;
  if (!previous) return 0;

  let changed = 0;
  Object.keys(policy).forEach(cell => {
    if (previous[cell] !== policy[cell]) changed += 1;
  });
  return changed;
}

async function maybeEnterReplay() {
  if (Sim.snapshot.state !== 'TRAINED') return;
  if (Sim.busy) return;
  ui.replay = { index: 0, elapsed: 0, waiting: 0 };
  await Sim.replay();
  paint();
}

function tick(now) {
  const dt = Math.min(0.05, (now - ui.lastTime) / 1000) || 0;
  ui.lastTime = now;

  const state = Sim.state;
  if (state === 'TRAINING') trainingWork(dt);
  else if (state === 'TRAINED') maybeEnterReplay();
  else if (state === 'REPLAYING') advanceReplay(dt);

  Renderer.draw({
    layout: Sim.describe.layout,
    learned: Sim.snapshot.learned,
    showPolicy: true,
    agent: agentPosition(),
    trail: replayTrail(),
  });

  ui.frame = window.requestAnimationFrame(tick);
}

function startLoop() {
  if (ui.frame !== null) return;
  ui.lastTime = performance.now();
  ui.frame = window.requestAnimationFrame(tick);
}

function stopLoop() {
  if (ui.frame === null) return;
  window.cancelAnimationFrame(ui.frame);
  ui.frame = null;
}

/* ---------------------------------------------------------------------
   Controls
   ------------------------------------------------------------------- */

function togglePlay() {
  const snapshot = Sim.snapshot;
  if (snapshot.playing) {
    Sim.pause().then(paint);
    return;
  }
  Sim.play().then(paint);
}

function setSidebar(open) {
  ui.open = open;
  document.body.classList.toggle('is-open', open);
  dom.sidebar.setAttribute('aria-hidden', String(!open));
  dom.toggle.setAttribute('aria-expanded', String(open));
  dom.toggle.setAttribute('aria-label', open ? 'Close the sidebar'
                                             : 'Open the sidebar');
  dom.toggleGlyph.innerHTML = open ? '&rsaquo;' : '&lsaquo;';
  // The sidebar overlays the canvas, so the grid is re-fitted into what is
  // left. It is re-fitted after the slide, and during it, so the grid never
  // jumps at the end.
  refit();
  window.setTimeout(refit, Sim.config.render.sidebar_ms + 20);
}

function leave() {
  if (ui.leaving) return;

  // Deliberately tolerant of a page that never finished loading. Getting
  // out must always work, so nothing above the navigation is allowed to
  // depend on the simulation having started.
  const snapshot = Sim.snapshot;

  if (snapshot) {
    const inProgress = snapshot.state === 'TRAINING'
      && snapshot.progress.count > 0
      && !snapshot.progress.converged;
    if (inProgress && !window.confirm(
        'Training is still in progress. Leaving discards this run.')) {
      return;
    }
  }

  ui.leaving = true;
  stopLoop();

  try {
    // Come back to the chamber just left, rather than to whichever one is
    // furthest along.
    window.sessionStorage.setItem('projectR5:returnTo', String(ui.room));
    // Report completion. The chamber select owns the progress integer, so
    // all that is handed over is which room was solved; it does the writing
    // and plays the unlock sequence.
    if (snapshot && snapshot.solved) {
      window.sessionStorage.setItem('projectR5:justCompleted', String(ui.room));
    }
  } catch (problem) {
    // Storage unavailable: the run simply is not remembered.
  }

  Sim.release();
  window.location.href = '../levels/';
}

/* ---------------------------------------------------------------------
   Wiring
   ------------------------------------------------------------------- */

dom.play.addEventListener('click', togglePlay);
dom.step.addEventListener('click', () => Sim.step().then(() => {
  recordSweep();
  paint();
}));
dom.reset.addEventListener('click', () => Sim.reset().then(() => {
  ui.metrics = [];
  ui.previousPolicy = null;
  ui.owed = 0;
  ui.replay = { index: 0, elapsed: 0, waiting: 0 };
  paint();
}));
dom.toggle.addEventListener('click', () => setSidebar(!ui.open));
dom.exit.addEventListener('click', leave);

document.addEventListener('keydown', event => {
  // Never hijack a key while the player is typing into a control.
  const tag = (event.target.tagName || '').toLowerCase();
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
  // Tab opens the sidebar, then lets focus move into it as normal — the
  // default is not prevented, so keyboard navigation is never trapped.
  if (event.key === 'Tab' && !ui.open) setSidebar(true);
});

window.addEventListener('resize', refit);

document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    stopLoop();
    // Paused rather than merely not drawn, so nothing advances unseen.
    if (Sim.playing) Sim.pause().then(paint);
    return;
  }
  startLoop();
});

window.addEventListener('pagehide', () => Sim.release());

/* ---------------------------------------------------------------------
   Boot
   ------------------------------------------------------------------- */

async function boot() {
  ui.room = roomFromQuery();
  ui.metrics = [];
  ui.previousPolicy = null;

  // The whole of the start-up is guarded, not only the request. A throw
  // while building the sidebar used to leave the strip reading "Loading"
  // with no indication of what had gone wrong.
  try {
    await Sim.open(ui.room);

    Renderer.attach(dom.canvas, Sim.config);
    buildLegend();
    buildSpeeds();
    buildAlgorithms();
    buildParameters();
    buildInfo();
    ui.repaintCharts = Charts.build(dom.charts, PLANNER_SERIES);
    refit();
    paint();
    startLoop();
  } catch (problem) {
    dom.strip.state.textContent = 'Unavailable';
    dom.strip.metric.textContent = String(problem && problem.message || problem);
    // The exit button is wired before this runs, and `leave` copes with
    // there being no session, so the way out still works.
    throw problem;
  }
}

boot();
