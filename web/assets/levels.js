'use strict';

/* =====================================================================
   PROJECT R-5 — chamber select.

   Five chambers, one on screen at a time, navigable in both directions.
   Locked chambers are still browsable and still occupy exactly the same
   space; only their appearance and their entry button change.

   Progress is one integer.  `highestUnlocked` is the only thing stored,
   and every locked / cleared state on this screen is derived from it:

       locked   = level.index > highestUnlocked
       cleared  = level.index < highestUnlocked

   The LEVELS array below is content and nothing else.  It has no idea
   whether anything has been played.
   ===================================================================== */

/* ---------------------------------------------------------------------
   Config. Every tunable number, and the one dev switch.
   ------------------------------------------------------------------- */
const CONFIG = {
  // Unlocks every chamber without playing through. Never writes to
  // storage, so turning it off leaves real progress exactly as it was.
  debugUnlockAll: false,

  storage: {
    key: 'projectR5:highestUnlocked',
    // Set by a chamber as it is completed, read and cleared by this screen.
    completedKey: 'projectR5:justCompleted',
    // Set by any chamber on the way out, so leaving one comes back to it.
    returnKey: 'projectR5:returnTo',
    // Accepted for development: ?completed=3 plays the unlock sequence.
    completedParam: 'completed',
  },

  unlock: {
    holdSeconds: 0.45,     // beat on the chamber that was just cleared
    resolveSeconds: 0.70,  // locked appearance lifting on the new one
    buttonSwapAt: 0.62,    // share of the resolve when LOCKED becomes ENTER
  },

  swapSeconds: 0.12,       // cross-fade when stepping between chambers

  scene: {
    scrollSpeed: 22,       // px/s the ambience drifts, forever
    lockedDim: 0.42,       // how much further a locked chamber is dimmed
    dimEase: 3.2,          // how quickly the dimming follows the selection
    motifFade: 2.6,        // how quickly one chamber's motif becomes another's
    far: { factor: 0.45, spacing: 190, width: 110, minHeight: 80, maxHeight: 260 },
    mid: { factor: 1.0, spacing: 140, width: 78, minHeight: 34, maxHeight: 120 },
    backdropMix: 0.09,
    farMix: 0.06,
    midMix: 0.03,
  },
};

/* ---------------------------------------------------------------------
   The five chambers. Content only.
   ------------------------------------------------------------------- */

/* The artwork is inline SVG built from the same three colours as
   everything else, so `filter: grayscale()` is all a locked chamber
   needs — the accent is the only thing there is to drain. */

const ART = {
  1: `
    <svg viewBox="0 0 320 200" role="presentation">
      <rect x="0" y="30" width="320" height="3" fill="var(--muted)" opacity=".7"/>
      <rect x="0" y="166" width="320" height="3" fill="var(--muted)" opacity=".7"/>
      <rect x="52" y="94" width="26" height="72" fill="var(--muted)" opacity=".38"/>
      <rect x="120" y="70" width="24" height="96" fill="var(--muted)" opacity=".38"/>
      <rect x="186" y="108" width="22" height="58" fill="var(--muted)" opacity=".38"/>
      <rect x="266" y="106" width="34" height="60" fill="var(--muted)" opacity=".62"/>
      <rect x="86" y="33" width="12" height="7" fill="var(--muted)" opacity=".8"/>
      <rect x="158" y="33" width="12" height="7" fill="var(--muted)" opacity=".8"/>
      <rect x="228" y="33" width="12" height="7" fill="var(--muted)" opacity=".8"/>
      <line x1="92" y1="40" x2="118" y2="166" stroke="var(--accent)"
            stroke-width="1.4" opacity=".8"/>
      <line x1="164" y1="40" x2="146" y2="166" stroke="var(--accent)"
            stroke-width="1.4" opacity=".8"/>
      <line x1="234" y1="40" x2="252" y2="166" stroke="var(--accent)"
            stroke-width="1.4" opacity=".8"/>
      <rect x="22" y="142" width="15" height="24" fill="var(--muted)"/>
      <circle cx="34" cy="148" r="2.6" fill="var(--accent)"/>
    </svg>`,

  2: `
    <svg viewBox="0 0 320 200" role="presentation">
      <rect x="0" y="120" width="96" height="4" fill="var(--muted)" opacity=".7"/>
      <rect x="224" y="120" width="96" height="4" fill="var(--muted)" opacity=".7"/>
      <rect x="0" y="124" width="96" height="46" fill="var(--muted)" opacity=".22"/>
      <rect x="224" y="124" width="96" height="46" fill="var(--muted)" opacity=".22"/>
      <rect x="92" y="106" width="6" height="18" fill="var(--muted)" opacity=".6"/>
      <rect x="222" y="106" width="6" height="18" fill="var(--muted)" opacity=".6"/>
      <rect x="102" y="120" width="26" height="4" fill="var(--muted)" opacity=".75"/>
      <rect x="134" y="120" width="26" height="4" fill="var(--muted)" opacity=".75"/>
      <g transform="rotate(24 178 138)">
        <rect x="166" y="136" width="26" height="4" fill="var(--muted)" opacity=".45"/>
      </g>
      <g transform="rotate(52 206 158)">
        <rect x="194" y="156" width="26" height="4" fill="var(--muted)" opacity=".3"/>
      </g>
      <line x1="98" y1="176" x2="222" y2="176" stroke="var(--accent)"
            stroke-width="1.2" opacity=".45" stroke-dasharray="3 7"/>
      <rect x="58" y="96" width="15" height="24" fill="var(--muted)"/>
      <circle cx="70" cy="102" r="2.6" fill="var(--accent)"/>
    </svg>`,

  3: `
    <svg viewBox="0 0 320 200" role="presentation">
      <rect x="0" y="166" width="320" height="3" fill="var(--muted)" opacity=".7"/>
      <rect x="60" y="96" width="30" height="70" fill="var(--muted)" opacity=".5"/>
      <rect x="145" y="80" width="30" height="86" fill="var(--muted)" opacity=".5"/>
      <rect x="230" y="110" width="30" height="56" fill="var(--muted)" opacity=".5"/>
      <rect x="68" y="106" width="14" height="8" fill="var(--accent)" opacity=".9"/>
      <rect x="153" y="90" width="14" height="8" fill="var(--accent)" opacity=".9"/>
      <rect x="238" y="120" width="14" height="8" fill="var(--accent)" opacity=".45"/>
      <path d="M74 96 L74 70 L246 70 L246 110" fill="none" stroke="var(--muted)"
            stroke-width="1.6" opacity=".3"/>
      <rect x="128" y="140" width="64" height="26" fill="var(--muted)" opacity=".3"/>
      <path d="M112 150 l8 -8 l-4 12 l10 -6" fill="none" stroke="var(--accent)"
            stroke-width="1.2" opacity=".55"/>
      <rect x="26" y="142" width="15" height="24" fill="var(--muted)"/>
      <circle cx="38" cy="148" r="2.6" fill="var(--accent)"/>
    </svg>`,

  4: `
    <svg viewBox="0 0 320 200" role="presentation">
      <rect x="0" y="176" width="320" height="3" fill="var(--muted)" opacity=".55"/>
      <circle cx="52" cy="100" r="34" fill="none" stroke="var(--muted)"
              stroke-width="3" opacity=".55"/>
      <path d="M52 100 L52 70 M52 100 L78 114 M52 100 L26 114"
            stroke="var(--muted)" stroke-width="4" opacity=".55"/>
      <path d="M96 74 q60 -14 118 6" fill="none" stroke="var(--accent)"
            stroke-width="1.3" opacity=".6"/>
      <path d="M96 104 q66 -16 132 8" fill="none" stroke="var(--accent)"
            stroke-width="1.3" opacity=".45"/>
      <path d="M96 134 q58 -12 112 4" fill="none" stroke="var(--accent)"
            stroke-width="1.3" opacity=".3"/>
      <circle cx="150" cy="88" r="8" fill="var(--muted)"/>
      <rect x="140" y="78" width="20" height="3" fill="var(--muted)"/>
      <circle cx="156" cy="86" r="2.4" fill="var(--accent)"/>
      <ellipse cx="262" cy="150" rx="30" ry="9" fill="none" stroke="var(--muted)"
               stroke-width="2" opacity=".6"/>
      <ellipse cx="262" cy="150" rx="12" ry="4" fill="var(--accent)" opacity=".35"/>
    </svg>`,

  5: `
    <svg viewBox="0 0 320 200" role="presentation">
      <rect x="0" y="170" width="320" height="3" fill="var(--muted)" opacity=".55"/>
      <g fill="var(--muted)" opacity=".4">
        <rect x="70" y="46" width="14" height="104"/>
        <rect x="130" y="70" width="14" height="80"/>
        <rect x="190" y="46" width="14" height="60"/>
        <rect x="250" y="92" width="14" height="58"/>
      </g>
      <g fill="var(--muted)" opacity=".26">
        <rect x="104" y="132" width="18" height="18"/>
        <rect x="222" y="120" width="18" height="18"/>
        <rect x="162" y="46" width="18" height="18"/>
      </g>
      <rect x="278" y="60" width="22" height="30" fill="var(--muted)" opacity=".6"/>
      <rect x="284" y="68" width="10" height="6" fill="var(--accent)" opacity=".8"/>
      <g fill="none" stroke="var(--accent)" stroke-width="1.2">
        <path d="M56 118 a34 34 0 0 1 0 -44" opacity=".6"/>
        <path d="M64 128 a52 52 0 0 1 0 -64" opacity=".35"/>
        <path d="M72 138 a70 70 0 0 1 0 -84" opacity=".18"/>
      </g>
      <rect x="30" y="122" width="15" height="24" fill="var(--muted)"/>
      <circle cx="42" cy="128" r="2.6" fill="var(--accent)"/>
    </svg>`,
};

const LEVELS = [
  { index: 1, name: 'Laser Security Chamber', method: 'Value Iteration',
    blurb: 'The model is known. Plan a route around beams R-5 has never touched.',
    art: ART[1], built: true },
  { index: 2, name: 'Broken Bridge Sector', method: 'SARSA',
    blurb: 'A short span worth more than the long way round — until the walking is imperfect.',
    art: ART[2], built: true },
  { index: 3, name: 'Reactor Control Chamber', method: 'Q-Learning',
    blurb: 'Three generators in order, a patrol walking the other way.',
    art: ART[3], built: true },
  { index: 4, name: 'Drone Wind Tunnel', method: 'Semi-Gradient SARSA',
    blurb: 'Position and velocity are real numbers. No table fits, so approximate.',
    art: ART[4], built: true },
  { index: 5, name: 'Adaptive Storage Facility', method: 'Semi-Gradient Q-Learning',
    blurb: 'The warehouse is rearranged every time. Radar contact only.',
    art: ART[5], built: true },
];

const LAST = LEVELS.length;

/* ---------------------------------------------------------------------
   Progress: read, derive, write
   ------------------------------------------------------------------- */

/** The stored integer, or 1 if storage is missing, blocked or corrupt. */
function readProgress() {
  try {
    const raw = window.localStorage.getItem(CONFIG.storage.key);
    if (raw === null) return 1;
    const value = Number.parseInt(raw, 10);
    // Anything outside the range a real run can produce is treated as
    // corrupt, not clamped, so a bad value cannot quietly open chambers.
    if (!Number.isFinite(value) || value < 1 || value > LAST + 1) return 1;
    return value;
  } catch (problem) {
    return 1;
  }
}

function writeProgress(value) {
  try {
    window.localStorage.setItem(CONFIG.storage.key, String(value));
  } catch (problem) {
    // Storage being unavailable must not break the screen; the run simply
    // will not be remembered.
  }
}

/** Completing chamber N opens N+1. It can never take progress away. */
function completeLevel(number) {
  if (!Number.isFinite(number) || number < 1 || number > LAST) return;
  const next = Math.min(LAST + 1, number + 1);
  const current = readProgress();
  if (next > current) writeProgress(next);
  refresh();
}

function resetProgress() {
  try {
    window.localStorage.removeItem(CONFIG.storage.key);
  } catch (problem) { /* nothing to undo */ }
  refresh();
}

/**
 * How far through the game the player is, in words.
 *
 * Progress survives closing the browser, which is right — and used to be
 * completely invisible, so a restored run looked like a bug. Cleared count and
 * a way to start over, both stated.
 */
function paintProgress() {
  if (!dom.progressNote) return;
  const cleared = Math.max(0, Math.min(LAST, state.highestUnlocked - 1));
  dom.progressNote.textContent = cleared === 0
    ? 'New run · no sectors cleared yet'
    : cleared + ' of ' + LAST + ' sectors cleared';
  // Nothing to reset on a clean run, so the control is not offered.
  if (dom.newGame) dom.newGame.hidden = cleared === 0;
}

/**
 * Start over from chamber 1.
 *
 * Confirmed, because it throws away real progress — and offered at all because
 * without it a player whose stored progress is wrong (or who simply wants
 * another run) has no way back to the beginning. `resetProgress` clears the one
 * stored integer and nothing else.
 */
function startNewGame() {
  const cleared = Math.max(0, state.highestUnlocked - 1);
  if (cleared > 0 && !window.confirm(
      'Start a new run? This clears ' + cleared
      + ' cleared sector' + (cleared === 1 ? '' : 's') + '.')) {
    return;
  }
  resetProgress();
  select(1, { instant: true, force: true });
}

/** Everything the screen knows about one chamber, derived — never stored. */
function derive(level, highestUnlocked) {
  if (CONFIG.debugUnlockAll) {
    return { locked: false, cleared: level.index < highestUnlocked };
  }
  return {
    locked: level.index > highestUnlocked,
    cleared: level.index < highestUnlocked,
  };
}

/* ---------------------------------------------------------------------
   Elements and state
   ------------------------------------------------------------------- */

const dom = {
  stage: document.getElementById('stage'),
  count: document.getElementById('count'),
  dots: document.getElementById('dots'),
  level: document.querySelector('.level'),
  art: document.getElementById('art'),
  name: document.getElementById('name'),
  method: document.getElementById('method'),
  blurb: document.getElementById('blurb'),
  prev: document.getElementById('prev'),
  next: document.getElementById('next'),
  enter: document.getElementById('enter'),
  back: document.getElementById('back'),
  note: document.getElementById('note'),
  progressNote: document.getElementById('progress-note'),
  newGame: document.getElementById('new-game'),
  canvas: document.getElementById('scene'),
};

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

const state = {
  highestUnlocked: 1,
  selected: 1,
  renderedIndex: null,
  unlockTarget: null,
  // While an unlock is playing, the chamber is already open in storage but
  // is deliberately still drawn locked until the transition resolves.
  forceLocked: false,
  buttonLocked: false,
  unlocking: false,
  timers: [],
  swapTimer: null,
};

function clearTimers() {
  state.timers.forEach(window.clearTimeout);
  state.timers.length = 0;
}

function later(callback, seconds) {
  const id = window.setTimeout(callback, seconds * 1000);
  state.timers.push(id);
  return id;
}

function levelAt(index) {
  return LEVELS[index - 1];
}

function pad(number) {
  return String(number).padStart(2, '0');
}

/* ---------------------------------------------------------------------
   Rendering
   ------------------------------------------------------------------- */

function buildDots() {
  dom.dots.innerHTML = LEVELS.map(() => '<span class="dot"></span>').join('');
}

function render() {
  const level = levelAt(state.selected);
  const status = derive(level, state.highestUnlocked);
  const locked = status.locked || state.forceLocked;

  // Content. The artwork is only re-parsed when the chamber changes.
  if (state.renderedIndex !== level.index) {
    dom.art.innerHTML = level.art;
    dom.name.textContent = level.name;
    dom.method.textContent = level.method;
    dom.blurb.textContent = level.blurb;
    state.renderedIndex = level.index;
  }

  dom.count.textContent = pad(level.index) + ' / ' + pad(LAST);
  Array.prototype.forEach.call(dom.dots.children, (dot, position) => {
    const other = derive(levelAt(position + 1), state.highestUnlocked);
    dot.classList.toggle('is-cleared', other.cleared);
    dot.classList.toggle('is-current', position + 1 === level.index);
  });

  dom.stage.classList.toggle('is-locked', locked);

  // A chamber can be shut for two different reasons, and they are not the
  // same thing: locked means it has not been earned yet, unbuilt means it
  // does not exist. Saying "locked" for the second one would be a lie the
  // player can never resolve.
  const unbuilt = !level.built;
  const buttonLocked = locked || unbuilt || state.buttonLocked;
  // The entry button is the same element either way, at the same size.
  dom.enter.textContent = locked ? 'Locked' : (unbuilt ? 'Not built' : 'Enter');
  dom.enter.disabled = buttonLocked;
  dom.enter.setAttribute('aria-disabled', String(buttonLocked));

  // The line underneath: why it is shut, or that it has been cleared.
  let note = '';
  if (locked) {
    const prerequisite = levelAt(level.index - 1);
    note = 'Complete ' + prerequisite.method + ' to unlock';
  } else if (unbuilt) {
    note = 'Not implemented yet';
  } else if (status.cleared) {
    note = 'Cleared';
  }
  dom.note.textContent = note;
  dom.note.classList.toggle('is-shown', note !== '');

  paintProgress();

  dom.prev.disabled = level.index <= 1;
  dom.next.disabled = level.index >= LAST;

  sceneSelect(level.index, locked);
}

/* ---------------------------------------------------------------------
   Navigation
   ------------------------------------------------------------------- */

function select(index, options) {
  const settings = options || {};
  const target = Math.min(LAST, Math.max(1, index));
  if (target === state.selected && !settings.force) return;

  state.selected = target;

  if (settings.instant || reduceMotion.matches
      || CONFIG.swapSeconds <= 0) {
    render();
    return;
  }

  // A short cross-fade, and nothing more: stepping between chambers is
  // not the moment that earns an animation.
  window.clearTimeout(state.swapTimer);
  dom.level.classList.add('is-swapping');
  state.swapTimer = window.setTimeout(() => {
    render();
    dom.level.classList.remove('is-swapping');
  }, CONFIG.swapSeconds * 1000);
}

function step(direction) {
  select(state.selected + direction);
}

/* ---------------------------------------------------------------------
   The unlock moment
   ------------------------------------------------------------------- */

/** Jump straight to the resolved state. Returns true if it had work to do. */
function skipUnlock() {
  if (!state.unlocking) return false;
  clearTimers();
  window.clearTimeout(state.swapTimer);
  const target = state.unlockTarget;
  state.unlocking = false;
  state.forceLocked = false;
  state.buttonLocked = false;
  state.unlockTarget = null;
  dom.stage.classList.add('is-instant');
  select(target, { instant: true, force: true });
  // Let the resolved frame land before transitions are allowed again.
  window.requestAnimationFrame(() => dom.stage.classList.remove('is-instant'));
  return true;
}

/**
 * Hold on the chamber just cleared, then open the next one.
 *
 * The chamber being unlocked is already open as far as storage is
 * concerned by the time this runs, so the sequence starts by drawing it
 * locked on purpose and then letting that appearance lift.
 */
function playUnlock(completed, opened) {
  const unlock = CONFIG.unlock;
  state.unlocking = true;
  state.unlockTarget = opened;

  // 1. the chamber that was just finished, marked cleared
  select(completed, { instant: true, force: true });

  if (reduceMotion.matches) {
    skipUnlock();
    return;
  }

  later(() => {
    // 2. arrive on the new chamber, still wearing its locked appearance
    dom.stage.classList.add('is-instant');
    state.forceLocked = true;
    state.buttonLocked = true;
    select(opened, { instant: true, force: true });

    window.requestAnimationFrame(() => {
      // 3. release it, and let the stylesheet interpolate the rest
      dom.stage.classList.remove('is-instant');
      state.forceLocked = false;
      render();

      later(() => {
        state.buttonLocked = false;
        render();
      }, unlock.resolveSeconds * unlock.buttonSwapAt);

      later(() => {
        state.unlocking = false;
        state.unlockTarget = null;
      }, unlock.resolveSeconds);
    });
  }, unlock.holdSeconds);
}

/** Which chamber the player has just stepped out of, if any. */
function takeReturnSignal() {
  try {
    const raw = window.sessionStorage.getItem(CONFIG.storage.returnKey);
    if (raw === null) return null;
    // Read once: arriving here again later is not a return.
    window.sessionStorage.removeItem(CONFIG.storage.returnKey);
    const number = Number.parseInt(raw, 10);
    if (Number.isFinite(number) && number >= 1 && number <= LAST) return number;
  } catch (problem) {
    return null;
  }
  return null;
}


/** Which chamber, if any, the player has just come back from having cleared. */
function takeCompletedSignal() {
  const storage = CONFIG.storage;

  const parameter = new URLSearchParams(window.location.search)
    .get(storage.completedParam);
  if (parameter !== null) {
    const number = Number.parseInt(parameter, 10);
    if (Number.isFinite(number) && number >= 1 && number <= LAST) return number;
  }

  try {
    const raw = window.sessionStorage.getItem(storage.completedKey);
    if (raw === null) return null;
    // Read once: coming back to this screen later is not a completion.
    window.sessionStorage.removeItem(storage.completedKey);
    const number = Number.parseInt(raw, 10);
    if (Number.isFinite(number) && number >= 1 && number <= LAST) return number;
  } catch (problem) {
    return null;
  }
  return null;
}

/* ---------------------------------------------------------------------
   Freshness: progress is re-read every time the screen is shown
   ------------------------------------------------------------------- */

function refresh() {
  state.highestUnlocked = readProgress();
  render();
}

/* ---------------------------------------------------------------------
   The ambience behind it all
   ------------------------------------------------------------------- */

const ctx = dom.canvas.getContext('2d');

let palette = null;
const view = { width: 0, height: 0, groundY: 0, scale: 1 };
const scene = {
  time: 0, camX: 0,
  motif: 1, previousMotif: 1, mix: 1,
  dim: 1, dimTarget: 1, locked: false,
};

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

function hash(n) {
  let x = Math.imul(n ^ 0x9E3779B9, 0x85EBCA6B);
  x ^= x >>> 13;
  x = Math.imul(x, 0xC2B2AE35);
  x ^= x >>> 16;
  return (x >>> 0) / 4294967296;
}

function between(n, low, high) {
  return low + hash(n) * (high - low);
}

function parseHex(value) {
  let hex = value.trim().replace('#', '');
  if (hex.length === 3) {
    hex = hex[0] + hex[0] + hex[1] + hex[1] + hex[2] + hex[2];
  }
  const number = parseInt(hex, 16);
  return [(number >> 16) & 255, (number >> 8) & 255, number & 255];
}

function mixHex(fromHex, toHex, amount) {
  const a = parseHex(fromHex);
  const b = parseHex(toHex);
  const channel = index => Math.round(a[index] + (b[index] - a[index]) * amount);
  return 'rgb(' + channel(0) + ',' + channel(1) + ',' + channel(2) + ')';
}

function rgba(hex, alpha) {
  const [red, green, blue] = parseHex(hex);
  return 'rgba(' + red + ',' + green + ',' + blue + ',' + alpha + ')';
}

function readPalette() {
  const css = getComputedStyle(document.documentElement);
  const base = css.getPropertyValue('--base').trim() || '#07090C';
  const muted = css.getPropertyValue('--muted').trim() || '#7C8797';
  const accent = css.getPropertyValue('--accent').trim() || '#38D9FF';
  const settings = CONFIG.scene;
  return {
    base: base, muted: muted, accent: accent,
    backdrop: mixHex(base, muted, settings.backdropMix),
    far: mixHex(base, muted, settings.farMix),
    mid: mixHex(base, muted, settings.midMix),
    near: base,
  };
}

function resize() {
  const ratio = window.devicePixelRatio || 1;
  view.width = dom.canvas.clientWidth;
  view.height = dom.canvas.clientHeight;
  dom.canvas.width = Math.round(view.width * ratio);
  dom.canvas.height = Math.round(view.height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  view.groundY = Math.round(view.height * 0.82);
  view.scale = clamp(view.height / 900, 0.8, 1.15);
}

/** Point the ambience at a chamber. Its motif cross-fades in. */
function sceneSelect(index, locked) {
  if (index !== scene.motif) {
    scene.previousMotif = scene.motif;
    scene.motif = index;
    scene.mix = 0;
  }
  scene.locked = locked;
  scene.dimTarget = locked ? CONFIG.scene.lockedDim : 1;
}

function drawLayer(settings, colour, salt) {
  const spacing = settings.spacing * view.scale;
  const offset = scene.camX * settings.factor;
  const first = Math.floor(offset / spacing) - 1;
  const last = Math.ceil((offset + view.width) / spacing) + 1;

  ctx.fillStyle = colour;
  for (let index = first; index <= last; index += 1) {
    const seed = index * 13 + salt;
    if (hash(seed) < 0.2) continue;
    const width = settings.width * view.scale * between(seed + 1, 0.5, 1.3);
    const height = between(seed + 2, settings.minHeight, settings.maxHeight)
      * view.scale;
    const x = index * spacing - offset + between(seed + 3, -0.2, 0.2) * spacing;
    ctx.fillRect(Math.round(x), Math.round(view.groundY - height),
                 Math.round(width), Math.round(height));
  }
}

/* One motif per chamber, each a couple of strokes. Drawn in the accent
   normally and in the mid-tone when the chamber is locked, which is what
   "desaturated" means when there is only one colour to lose. */
const MOTIFS = {
  1: function lasers(colour, alpha) {
    ctx.strokeStyle = rgba(colour, alpha * 0.5);
    ctx.lineWidth = 1.2;
    for (let beam = 0; beam < 3; beam += 1) {
      const base = view.width * (0.24 + beam * 0.26);
      const sweep = Math.sin(scene.time * 0.7 + beam * 2.1) * view.width * 0.06;
      ctx.beginPath();
      ctx.moveTo(base, view.height * 0.1);
      ctx.lineTo(base + sweep, view.groundY);
      ctx.stroke();
    }
  },

  2: function span(colour, alpha) {
    const y = view.height * 0.5;
    ctx.strokeStyle = rgba(colour, alpha * 0.4);
    ctx.lineWidth = 1.4;
    ctx.setLineDash([16, 22]);
    ctx.beginPath();
    ctx.moveTo(view.width * 0.12, y);
    ctx.lineTo(view.width * 0.88, y);
    ctx.stroke();
    ctx.setLineDash([]);
    // A plank or two, always falling, never landing.
    for (let plank = 0; plank < 3; plank += 1) {
      const drop = (scene.time * 0.22 + plank * 0.33) % 1;
      ctx.save();
      ctx.globalAlpha = alpha * 0.3 * (1 - drop);
      ctx.fillStyle = rgba(colour, 1);
      ctx.translate(view.width * (0.3 + plank * 0.22),
                    y + drop * view.height * 0.4);
      ctx.rotate(drop * 1.6);
      ctx.fillRect(-14, -2, 28, 3);
      ctx.restore();
    }
  },

  3: function reactor(colour, alpha) {
    const pulse = 0.5 + 0.5 * Math.sin(scene.time * 1.5);
    ctx.fillStyle = rgba(colour, alpha * (0.1 + pulse * 0.12));
    ctx.fillRect(0, view.groundY - 3 * view.scale, view.width, 3 * view.scale);
    for (let core = 0; core < 3; core += 1) {
      const phase = 0.5 + 0.5 * Math.sin(scene.time * 1.5 - core * 0.8);
      ctx.fillStyle = rgba(colour, alpha * (0.16 + phase * 0.3));
      const x = view.width * (0.24 + core * 0.26);
      const height = view.height * 0.05 * (0.6 + phase * 0.6);
      ctx.fillRect(x, view.groundY - height, 10 * view.scale, height);
    }
  },

  4: function airflow(colour, alpha) {
    ctx.strokeStyle = rgba(colour, alpha * 0.32);
    ctx.lineWidth = 1.2;
    for (let streak = 0; streak < 6; streak += 1) {
      const y = view.height * (0.16 + streak * 0.11);
      const drift = ((scene.time * 0.16 + streak * 0.17) % 1) * view.width * 1.3
        - view.width * 0.15;
      ctx.beginPath();
      ctx.moveTo(drift, y);
      ctx.quadraticCurveTo(drift + view.width * 0.09, y - 16 * view.scale,
                           drift + view.width * 0.18, y);
      ctx.stroke();
    }
  },

  5: function radar(colour, alpha) {
    const originX = view.width * 0.22;
    const originY = view.groundY - view.height * 0.1;
    for (let ring = 0; ring < 3; ring += 1) {
      const share = ((scene.time * 0.3 + ring * 0.34) % 1);
      ctx.strokeStyle = rgba(colour, alpha * 0.3 * (1 - share));
      ctx.lineWidth = 1.1;
      ctx.beginPath();
      ctx.arc(originX, originY, share * view.width * 0.42,
              -Math.PI * 0.75, -Math.PI * 0.05);
      ctx.stroke();
    }
  },
};

function drawScene() {
  ctx.clearRect(0, 0, view.width, view.height);

  const gradient = ctx.createLinearGradient(0, 0, 0, view.height);
  gradient.addColorStop(0, palette.base);
  gradient.addColorStop(0.5, palette.backdrop);
  gradient.addColorStop(1, palette.far);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, view.width, view.height);

  ctx.save();
  ctx.globalAlpha = scene.dim;

  drawLayer(CONFIG.scene.far, palette.far, 101);
  drawLayer(CONFIG.scene.mid, palette.mid, 211);

  ctx.fillStyle = palette.near;
  ctx.fillRect(0, view.groundY, view.width, view.height - view.groundY);
  ctx.fillStyle = palette.mid;
  ctx.fillRect(0, view.groundY, view.width, 1);

  const colour = scene.locked ? palette.muted : palette.accent;
  const outgoing = MOTIFS[scene.previousMotif];
  const incoming = MOTIFS[scene.motif];
  if (outgoing && scene.mix < 1) outgoing(colour, 1 - scene.mix);
  if (incoming) incoming(colour, scene.mix);

  ctx.restore();
}

function advanceScene(dt) {
  scene.time += dt;
  scene.camX += CONFIG.scene.scrollSpeed * dt;
  scene.mix = Math.min(1, scene.mix + dt * CONFIG.scene.motifFade);
  scene.dim += (scene.dimTarget - scene.dim)
    * Math.min(1, dt * CONFIG.scene.dimEase);
}

let frame = null;
let lastTime = 0;

function tick(now) {
  const dt = Math.min(0.05, (now - lastTime) / 1000) || 0;
  lastTime = now;
  advanceScene(dt);
  drawScene();
  frame = window.requestAnimationFrame(tick);
}

function startScene() {
  if (frame !== null || reduceMotion.matches) return;
  lastTime = performance.now();
  frame = window.requestAnimationFrame(tick);
}

function stopScene() {
  if (frame === null) return;
  window.cancelAnimationFrame(frame);
  frame = null;
}

function drawStillScene() {
  scene.mix = 1;
  scene.dim = scene.dimTarget;
  advanceScene(6);
  drawScene();
}

/* ---------------------------------------------------------------------
   Wiring
   ------------------------------------------------------------------- */

dom.prev.addEventListener('click', () => step(-1));
dom.next.addEventListener('click', () => step(1));

if (dom.newGame) dom.newGame.addEventListener('click', startNewGame);

dom.back.addEventListener('click', () => {
  // Progress is already in storage, written the moment a chamber was
  // cleared, so there is nothing to save on the way out.
  window.location.href = '../';
});

dom.enter.addEventListener('click', () => {
  // Disabled buttons do not fire this, so reaching here means the chamber
  // is genuinely open.
  //
  // Two screens exist while the rebuild is part-way through. Room 1 is a
  // planner and still uses the original one; rooms 2 and up are learners
  // and use the new one, which is built around episodes rather than
  // sweeps. When room 1 moves across, this becomes one path again.
  const screen = state.selected === 1 ? '../level/' : '../room/';
  window.location.href = screen + '?room=' + state.selected;
});

document.addEventListener('keydown', event => {
  if (event.key === 'ArrowLeft') { step(-1); return; }
  if (event.key === 'ArrowRight') { step(1); return; }
});

/* Any input at all skips the unlock sequence.
   Registered in the capture phase so it runs before the arrow keys and the
   buttons below, and the input that did the skipping is consumed rather
   than also being acted on — otherwise the press that resolved the screen
   would immediately navigate away from it. */
document.addEventListener('keydown', event => {
  if (skipUnlock()) event.stopPropagation();
}, true);

document.addEventListener('click', event => {
  if (!skipUnlock()) return;
  event.stopPropagation();
  event.preventDefault();
}, true);

window.addEventListener('resize', () => {
  resize();
  if (reduceMotion.matches) drawStillScene();
});

// Progress is re-read whenever this screen comes back into view, not only
// on first load: a chamber cleared in between must not still read locked.
document.addEventListener('visibilitychange', () => {
  if (document.hidden) {
    stopScene();
    return;
  }
  refresh();
  startScene();
});

window.addEventListener('pageshow', refresh);
window.addEventListener('focus', refresh);

// Another tab finishing a chamber counts too.
window.addEventListener('storage', event => {
  if (event.key === null || event.key === CONFIG.storage.key) refresh();
});

if (typeof reduceMotion.addEventListener === 'function') {
  reduceMotion.addEventListener('change', boot);
}

function boot() {
  palette = readPalette();
  resize();
  buildDots();

  // The one place the resolve timing is written down is the config, so the
  // stylesheet is told about it rather than repeating it.
  document.documentElement.style.setProperty(
    '--resolve-ms', Math.round(CONFIG.unlock.resolveSeconds * 1000) + 'ms');

  const completed = takeCompletedSignal();
  // This screen owns the write, so arriving back from a finished chamber
  // records the progress even if the chamber itself did not.
  const before = readProgress();
  if (completed !== null) completeLevel(completed);
  state.highestUnlocked = readProgress();

  const opened = completed === null ? null : completed + 1;
  // Something to unlock only if there is a next chamber at all, and it was
  // genuinely still shut before this run.
  const unlocked = opened !== null && opened <= LAST && opened > before;

  if (unlocked) {
    playUnlock(completed, opened);
  } else if (completed !== null) {
    // Either a replay of a chamber already cleared, or the last chamber:
    // nothing opens, so there is nothing to animate.
    select(completed, { instant: true, force: true });
  } else {
    /* Coming back out of a chamber returns to that chamber. Arriving from the
       main menu opens on the FIRST chamber.

       WHY NOT THE FURTHEST ONE, WHICH IS WHAT IT USED TO DO
       `Math.min(LAST, highestUnlocked)` meant a player who had finished the
       game — or whose stored progress said so — pressed PLAY and landed on
       chamber 5 with no explanation. That is indistinguishable from the game
       starting in the wrong room, and it is what was reported.

       Progress is not thrown away: chambers stay unlocked, the readout above
       says how many are cleared, and the player can walk right to any of them.
       What changes is that the game always *starts* at the beginning, so
       arriving at chamber 5 is a decision rather than a surprise. */
    const returned = takeReturnSignal();
    const opening = returned === null ? 1 : returned;
    select(opening, { instant: true, force: true });
  }

  if (reduceMotion.matches) {
    stopScene();
    drawStillScene();
  } else {
    startScene();
  }
}

/* The seam for the rest of the game: the About screen's reset action, and
   whatever launches a chamber, both need a way in. */
window.R5Chambers = {
  refresh: refresh,
  complete: completeLevel,
  reset: resetProgress,
  progress: readProgress,
};

boot();
