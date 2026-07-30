'use strict';

/* =====================================================================
   PROJECT R-5 — the start screen's background animation.

   A side-scrolling silhouette of a robot escaping a laboratory, drawn as
   a training loop: it fails at a hazard, dissolves, respawns at the left
   edge and gets one hazard further on the next attempt, until it runs
   clean and escapes.  Then it starts over.

   There is no restart cut anywhere, because the camera never stops or
   jumps: it advances at a constant speed forever, and the world ahead of
   it is generated on demand from the hazard index.  The only thing that
   resets is the robot, which is the point of the scene.

   Nothing is random.  Every "random" value comes from `hash`, so the same
   viewport always produces the same run — which is also what lets the
   reduced-motion path fast-forward to a good-looking still frame.
   ===================================================================== */

/* ---------------------------------------------------------------------
   Every tunable number, in one place.
   ------------------------------------------------------------------- */
const CONFIG = {
  world: {
    scrollSpeed: 140,        // px/s the camera advances — always, forever
    robotSpeed: 190,         // px/s the robot runs while it is alive
    sprintFactor: 2.6,       // how much faster the escaping robot runs
    hazardSpacing: 300,      // world px between hazards
    hazardJitter: 34,        // +/- variation; must stay under spacing / 2
    firstHazardGap: 1.1,     // spacings before the very first hazard
    spawnMargin: 120,        // screen px from the left a robot appears at
    groundRatio: 0.74,       // floor line, as a share of viewport height
    ceilingRatio: 0.15,      // ceiling line, likewise
    scaleFrom: 900,          // the viewport height these sizes suit
    scaleMin: 0.8,           // clamped, so the pacing cannot drift far
    scaleMax: 1.1,
  },

  loop: {
    attemptsBeforeEscape: 4, // four failures, then one clean run
    dissolveSeconds: 0.85,
    respawnDelay: 0.45,
    fadeInSeconds: 0.35,
    exitMargin: 90,          // screen px past the right edge counts as away
  },

  ghosts: {
    max: 6,
    fadeSeconds: 9,
    sampleInterval: 0.07,
    maxPoints: 420,
    alpha: 0.30,
    width: 1.2,
  },

  robot: {
    width: 26,
    height: 40,
    duckFactor: 0.52,        // ducked height, as a share of the full one
    sensor: 3.1,
    strideRate: 11,          // leg cycles per second
    gravity: 2600,
    jumpImpulse: 1320,       // clears pit.width with ~30px to spare
    shortJumpFactor: 0.62,   // the jump that does not make it
    jumpLead: 34,            // how far before the edge it leaves the ground
    fallKill: 150,           // px below the floor before it is gone
  },

  hazards: {
    // The cycle of hazard types, in order, repeating forever.
    order: ['laser', 'pit', 'drone', 'bridge', 'gap'],
    // The sweep is quick relative to the robot, so a robot that does not
    // duck is certain to meet the beam while it is inside the corridor.
    laser:  { corridor: 250, spread: 78, period: 0.8, tolerance: 0.55 },
    pit:    { width: 130 },
    drone:  { range: 95, period: 2.4, hover: 74, size: 22,
              holdBuffer: 150, safeGap: 60, holdMax: 1.3, contact: 0.8 },
    bridge: { width: 300, planks: 5, fallDelay: 0.28, fallGravity: 900,
              rebuildSeconds: 0.5 },
    gap:    { panelWidth: 46, clearance: 25 },  // clearance above the floor
  },

  parallax: {
    far: { factor: 0.24, spacing: 210, width: 120, minHeight: 90, maxHeight: 300 },
    mid: { factor: 0.55, spacing: 150, width: 84, minHeight: 40, maxHeight: 130 },
  },

  render: {
    backdropMix: 0.10,       // how far each tone sits from base towards muted
    farMix: 0.07,
    midMix: 0.035,
    nearMix: 0.0,            // the near silhouettes are base itself
    particles: 16,
  },

  still: { fastForwardSeconds: 12.4 },  // where the reduced-motion frame sits
};

/* ---------------------------------------------------------------------
   Small helpers
   ------------------------------------------------------------------- */

const canvas = document.getElementById('scene');
const ctx = canvas.getContext('2d');

const reduceMotion = window.matchMedia('(prefers-reduced-motion: reduce)');

function clamp(value, low, high) {
  return Math.max(low, Math.min(high, value));
}

/** A stable pseudo-random number in 0..1 for an integer key. */
function hash(n) {
  let x = Math.imul(n ^ 0x9E3779B9, 0x85EBCA6B);
  x ^= x >>> 13;
  x = Math.imul(x, 0xC2B2AE35);
  x ^= x >>> 16;
  return (x >>> 0) / 4294967296;
}

/** hash, mapped onto a range. */
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

/** `amount` of the way from `fromHex` to `toHex`, as an rgb() string. */
function mix(fromHex, toHex, amount) {
  const a = parseHex(fromHex);
  const b = parseHex(toHex);
  const channel = index => Math.round(a[index] + (b[index] - a[index]) * amount);
  return 'rgb(' + channel(0) + ',' + channel(1) + ',' + channel(2) + ')';
}

function rgba(hex, alpha) {
  const [red, green, blue] = parseHex(hex);
  return 'rgba(' + red + ',' + green + ',' + blue + ',' + alpha + ')';
}

/* ---------------------------------------------------------------------
   The palette, read from the stylesheet
   ------------------------------------------------------------------- */

let palette = null;

function readPalette() {
  const css = getComputedStyle(document.documentElement);
  const base = css.getPropertyValue('--base').trim() || '#07090C';
  const muted = css.getPropertyValue('--muted').trim() || '#7C8797';
  const accent = css.getPropertyValue('--accent').trim() || '#38D9FF';
  const render = CONFIG.render;

  return {
    base: base,
    muted: muted,
    accent: accent,
    // Depth is one colour lifted by four different amounts. The near
    // silhouettes are base itself, so they read as holes in a backdrop
    // that has been lifted very slightly off black.
    backdrop: mix(base, muted, render.backdropMix),
    far: mix(base, muted, render.farMix),
    mid: mix(base, muted, render.midMix),
    near: mix(base, muted, render.nearMix),
  };
}

/* ---------------------------------------------------------------------
   Viewport metrics
   ------------------------------------------------------------------- */

const metrics = {
  width: 0, height: 0, scale: 1, groundY: 0, ceilingY: 0,
  robotWidth: 0, robotHeight: 0, speed: 0, scroll: 0,
};

function measure() {
  const world = CONFIG.world;
  metrics.width = canvas.clientWidth;
  metrics.height = canvas.clientHeight;
  // One scale factor, driven by height, so a jump always fits under the
  // ceiling and the pacing still feels the same on a laptop and a wall.
  metrics.scale = clamp(metrics.height / world.scaleFrom,
                        world.scaleMin, world.scaleMax);
  metrics.groundY = Math.round(metrics.height * world.groundRatio);
  metrics.ceilingY = Math.round(metrics.height * world.ceilingRatio);
  metrics.robotWidth = CONFIG.robot.width * metrics.scale;
  metrics.robotHeight = CONFIG.robot.height * metrics.scale;
  metrics.speed = world.robotSpeed * metrics.scale;
  metrics.scroll = world.scrollSpeed * metrics.scale;
}

function resize() {
  const ratio = window.devicePixelRatio || 1;
  const width = canvas.clientWidth;
  const height = canvas.clientHeight;
  canvas.width = Math.round(width * ratio);
  canvas.height = Math.round(height * ratio);
  ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
  measure();
}

/* ---------------------------------------------------------------------
   The world: hazards, generated from their index
   ------------------------------------------------------------------- */

/** Where hazard `index` sits in world coordinates. */
function hazardX(index) {
  const world = CONFIG.world;
  const jitter = (hash(index * 7 + 3) * 2 - 1) * world.hazardJitter;
  return (index + world.firstHazardGap) * world.hazardSpacing + jitter;
}

function hazardType(index) {
  const order = CONFIG.hazards.order;
  // Modulo that behaves for negative indices too.
  return order[((index % order.length) + order.length) % order.length];
}

/** The half-width of the stretch of floor a hazard occupies. */
function hazardHalfWidth(type) {
  const hazards = CONFIG.hazards;
  const scale = metrics.scale;
  if (type === 'pit') return hazards.pit.width * scale / 2;
  if (type === 'bridge') return hazards.bridge.width * scale / 2;
  if (type === 'laser') return hazards.laser.corridor * scale / 2;
  if (type === 'gap') return hazards.gap.panelWidth * scale / 2;
  return hazards.drone.range * scale;
}

function hazardAt(index) {
  const type = hazardType(index);
  return { index: index, type: type, x: hazardX(index),
           half: hazardHalfWidth(type) };
}

function hazardsBetween(fromX, toX) {
  const spacing = CONFIG.world.hazardSpacing;
  const first = Math.floor(fromX / spacing) - 2;
  const last = Math.ceil(toX / spacing) + 2;
  const found = [];
  for (let index = first; index <= last; index += 1) {
    const hazard = hazardAt(index);
    if (hazard.x + hazard.half > fromX && hazard.x - hazard.half < toX) {
      found.push(hazard);
    }
  }
  return found;
}

/** The first hazard whose centre lies ahead of a world position. */
function nextHazardIndex(fromX) {
  let index = Math.floor(fromX / CONFIG.world.hazardSpacing) - 2;
  while (hazardX(index) <= fromX) index += 1;
  return index;
}

/* Mutable per-hazard state — fallen planks, how long the robot has been
   waiting for a drone. Kept beside the hazards rather than inside them,
   because the hazards themselves are regenerated every frame. Entries are
   pruned once the camera has left them behind. */
const hazardState = new Map();

function stateFor(hazard) {
  let entry = hazardState.get(hazard.index);
  if (!entry) {
    entry = { born: state.time, jumped: false, holdT: 0, planks: null,
              falling: [] };
    hazardState.set(hazard.index, entry);
  }
  if (hazard.type === 'bridge' && !entry.planks) {
    entry.planks = [];
    for (let plank = 0; plank < CONFIG.hazards.bridge.planks; plank += 1) {
      entry.planks.push({ passedAt: null, gone: false });
    }
  }
  return entry;
}

function pruneHazardState(camX) {
  hazardState.forEach((entry, index) => {
    if (hazardX(index) < camX - 800) hazardState.delete(index);
  });
}

/** Advance whatever a hazard is doing on its own, robot or no robot.
 *
 *  Falling planks have to keep falling after the robot that knocked them
 *  loose has gone, or a collapsed bridge would freeze in mid-air for the
 *  rest of the time it is on screen.
 */
function updateHazards(dt) {
  const bridge = CONFIG.hazards.bridge;
  const scale = metrics.scale;
  const visible = hazardsBetween(state.camX - 400 * scale,
                                 state.camX + metrics.width + 400 * scale);

  visible.forEach(hazard => {
    const entry = stateFor(hazard);
    for (let piece = entry.falling.length - 1; piece >= 0; piece -= 1) {
      const plank = entry.falling[piece];
      plank.vy += bridge.fallGravity * scale * dt;
      plank.y += plank.vy * dt;
      if (plank.y > CONFIG.robot.fallKill * 3 * scale) {
        entry.falling.splice(piece, 1);
      }
    }
  });
}

/** Put the chamber back the way it was, ready for the next attempt.
 *
 *  Only the stretch ahead of the new robot is rebuilt: a bridge it is
 *  about to cross is whole again, while what it has already passed is left
 *  alone.  Rebuilt planks fade back in, which reads as the laboratory
 *  resetting itself between trials rather than as a glitch.
 */
function rebuildHazardsAhead(fromX) {
  hazardState.forEach((entry, index) => {
    if (hazardX(index) > fromX - 120 * metrics.scale) hazardState.delete(index);
  });
}

/* ---------------------------------------------------------------------
   The scene's live state
   ------------------------------------------------------------------- */

const state = {
  time: 0,          // seconds since the scene began; drives every cycle
  camX: 0,          // world x at the left edge of the screen
  attempt: 1,       // 1..attemptsBeforeEscape, then the clean run
  phase: 'run',     // run | dissolve | waiting
  phaseT: 0,
  robot: null,
  ghosts: [],
  particles: [],
  particleSeed: 0,
};

function spawnRobot() {
  const loop = CONFIG.loop;
  const escaping = state.attempt > loop.attemptsBeforeEscape;
  const x = state.camX + CONFIG.world.spawnMargin * metrics.scale;
  const first = nextHazardIndex(x + 20);

  // Every one-shot the last attempt tripped — a jump already taken, a
  // plank already dropped — has to be cleared, or this attempt would meet
  // a chamber that has already been used and fail at the wrong hazard.
  rebuildHazardsAhead(x);

  state.robot = {
    x: x,
    y: 0,                 // height above the floor
    vy: 0,
    airborne: false,
    falling: false,
    duck: 0,
    stride: 0,
    holding: false,
    sprinting: false,
    fadeIn: 0,
    escaping: escaping,
    // Fate: this attempt dies at the n-th hazard ahead of the spawn. The
    // clean run has no fated hazard at all.
    fatedIndex: escaping ? null : first + (state.attempt - 1),
    sprintFromIndex: first + loop.attemptsBeforeEscape - 1,
    trail: [],
    trailT: 0,
  };
  state.phase = 'run';
  state.phaseT = 0;
}

function reset() {
  state.time = 0;
  state.camX = 0;
  state.attempt = 1;
  state.ghosts.length = 0;
  state.particles.length = 0;
  state.particleSeed = 0;
  hazardState.clear();
  spawnRobot();
}

/* ---------------------------------------------------------------------
   Dying
   ------------------------------------------------------------------- */

function kill(robot) {
  if (state.phase !== 'run') return;
  state.phase = 'dissolve';
  state.phaseT = 0;

  if (!reduceMotion.matches) {
    state.ghosts.push({ points: robot.trail.slice(), born: state.time });
    if (state.ghosts.length > CONFIG.ghosts.max) state.ghosts.shift();
  }

  const count = CONFIG.render.particles;
  const centreY = metrics.groundY - robot.y - metrics.robotHeight * 0.5;
  for (let index = 0; index < count; index += 1) {
    const seed = state.particleSeed += 1;
    const angle = between(seed * 3, -Math.PI, Math.PI);
    const speed = between(seed * 5 + 1, 30, 150) * metrics.scale;
    state.particles.push({
      x: robot.x + between(seed * 7 + 2, -0.5, 0.5) * metrics.robotWidth,
      y: centreY + between(seed * 11 + 4, -0.5, 0.5) * metrics.robotHeight,
      vx: Math.cos(angle) * speed,
      vy: Math.sin(angle) * speed - 40 * metrics.scale,
      life: between(seed * 13 + 6, 0.45, 0.9),
      age: 0,
      // One piece is the optical sensor, and it is allowed to keep its
      // colour on the way out.
      accent: index === 0,
    });
  }
}

/* ---------------------------------------------------------------------
   The floor: which stretches are missing, and what stands in them
   ------------------------------------------------------------------- */

function floorGap(x) {
  const spacing = CONFIG.world.hazardSpacing;
  const index = Math.round(x / spacing - CONFIG.world.firstHazardGap);
  for (let offset = -1; offset <= 1; offset += 1) {
    const hazard = hazardAt(index + offset);
    if (hazard.type !== 'pit' && hazard.type !== 'bridge') continue;
    if (x > hazard.x - hazard.half && x < hazard.x + hazard.half) return hazard;
  }
  return null;
}

/** True when there is something under this point to stand on. */
function supported(x) {
  const hazard = floorGap(x);
  if (!hazard) return true;
  if (hazard.type === 'pit') return false;

  const bridge = CONFIG.hazards.bridge;
  const entry = stateFor(hazard);
  const plankWidth = hazard.half * 2 / bridge.planks;
  for (let plank = 0; plank < entry.planks.length; plank += 1) {
    if (entry.planks[plank].gone) continue;
    const centre = hazard.x - hazard.half + (plank + 0.5) * plankWidth;
    if (Math.abs(x - centre) < plankWidth * 0.6) return true;
  }
  return false;
}

/* ---------------------------------------------------------------------
   Hazard behaviour
   ------------------------------------------------------------------- */

function laserBeamX(hazard) {
  const laser = CONFIG.hazards.laser;
  const phase = hash(hazard.index * 17 + 5);
  return hazard.x + Math.sin((state.time / laser.period + phase) * Math.PI * 2)
    * laser.spread * metrics.scale;
}

function dronePosition(hazard) {
  const drone = CONFIG.hazards.drone;
  const phase = hash(hazard.index * 23 + 9);
  return hazard.x + Math.sin((state.time / drone.period + phase) * Math.PI * 2)
    * drone.range * metrics.scale;
}

/** Let each nearby hazard act on the robot: duck, jump, wait, or die. */
function negotiate(robot, dt) {
  const hazards = CONFIG.hazards;
  const scale = metrics.scale;
  let duckWanted = 0;
  let holding = false;

  const nearby = hazardsBetween(robot.x - 260 * scale, robot.x + 420 * scale);

  for (let index = 0; index < nearby.length; index += 1) {
    const hazard = nearby[index];
    const fated = robot.fatedIndex === hazard.index;
    const entry = stateFor(hazard);
    const inside = Math.abs(robot.x - hazard.x) < hazard.half;

    if (hazard.type === 'pit') {
      const left = hazard.x - hazard.half;
      const lead = CONFIG.robot.jumpLead * scale;
      if (!robot.airborne && !entry.jumped
          && robot.x > left - lead && robot.x < left) {
        entry.jumped = true;
        robot.airborne = true;
        robot.vy = CONFIG.robot.jumpImpulse * scale
          * (fated ? CONFIG.robot.shortJumpFactor : 1);
      }

    } else if (hazard.type === 'bridge') {
      const bridge = hazards.bridge;
      const plankWidth = hazard.half * 2 / bridge.planks;
      for (let plank = 0; plank < entry.planks.length; plank += 1) {
        const record = entry.planks[plank];
        const centre = hazard.x - hazard.half + (plank + 0.5) * plankWidth;
        // The plank the fated robot is counting on has already gone.
        if (fated && plank === Math.floor(bridge.planks / 2)
            && record.passedAt === null && robot.x > hazard.x - hazard.half) {
          record.passedAt = -Infinity;
        }
        if (record.passedAt === null && robot.x > centre) {
          record.passedAt = state.time;
        }
        if (!record.gone && record.passedAt !== null
            && state.time - record.passedAt > bridge.fallDelay) {
          record.gone = true;
          entry.falling.push({ x: centre, y: 0, vy: 0,
                               spin: between(hazard.index * 31 + plank, -2, 2) });
        }
      }

    } else if (hazard.type === 'laser') {
      if (inside && !fated) duckWanted = 1;
      if (fated && inside) {
        const beamX = laserBeamX(hazard);
        const tolerance = metrics.robotWidth * hazards.laser.tolerance;
        // The sweep is quick enough to cross the corridor while the robot
        // is inside it; the second test is only there so a fate always
        // resolves and the attempt count cannot drift.
        if (Math.abs(beamX - robot.x) < tolerance
            || robot.x > hazard.x + hazard.half * 0.5) {
          kill(robot);
          return { duck: duckWanted, holding: holding };
        }
      }

    } else if (hazard.type === 'gap') {
      const reach = hazard.half + metrics.robotWidth * 0.6;
      if (Math.abs(robot.x - hazard.x) < reach && !fated) duckWanted = 1;
      if (fated && Math.abs(robot.x - hazard.x) < hazard.half) {
        kill(robot);
        return { duck: duckWanted, holding: holding };
      }

    } else if (hazard.type === 'drone') {
      const drone = hazards.drone;
      const droneX = dronePosition(hazard);
      const ahead = droneX - robot.x;
      if (fated) {
        if (Math.abs(ahead) < metrics.robotWidth * drone.contact
            || robot.x > hazard.x + hazard.half * 0.5) {
          kill(robot);
          return { duck: duckWanted, holding: holding };
        }
      } else if (ahead > 0 && ahead < drone.holdBuffer * scale
                 && entry.holdT < drone.holdMax) {
        // Stand still and let it sweep past.
        entry.holdT += dt;
        holding = true;
      } else if (ahead <= 0 || ahead > drone.safeGap * scale) {
        entry.holdT = Math.max(0, entry.holdT - dt);
      }
    }
  }

  return { duck: duckWanted, holding: holding };
}

/* ---------------------------------------------------------------------
   One step of the simulation
   ------------------------------------------------------------------- */

function step(dt) {
  const loop = CONFIG.loop;
  const ghosts = CONFIG.ghosts;

  state.time += dt;
  state.camX += metrics.scroll * dt;
  state.phaseT += dt;
  pruneHazardState(state.camX);
  updateHazards(dt);

  const robot = state.robot;

  if (state.phase === 'run' && robot) {
    const outcome = negotiate(robot, dt);
    robot.duck += (outcome.duck - robot.duck) * Math.min(1, dt * 12);
    robot.holding = outcome.holding;
    robot.fadeIn = Math.min(1, robot.fadeIn + dt / loop.fadeInSeconds);

    if (state.phase === 'run') {
      // The clean run breaks into a sprint once it is past the last hazard
      // that has ever stopped it.
      if (robot.escaping && !robot.sprinting
          && robot.x > hazardX(robot.sprintFromIndex) + 60 * metrics.scale) {
        robot.sprinting = true;
      }

      let speed = metrics.speed;
      if (robot.holding && !robot.airborne) speed = 0;
      if (robot.sprinting) speed = metrics.speed * CONFIG.world.sprintFactor;
      robot.x += speed * dt;
      robot.stride += speed > 0 ? dt * CONFIG.robot.strideRate : 0;

      // Vertical: gravity, then landing or falling out of the world.
      if (!robot.airborne && !supported(robot.x)) {
        robot.airborne = true;
        robot.falling = true;
        robot.vy = 0;
      }
      if (robot.airborne) {
        robot.vy -= CONFIG.robot.gravity * metrics.scale * dt;
        robot.y += robot.vy * dt;
        if (robot.vy <= 0 && robot.y <= 0 && supported(robot.x)) {
          robot.y = 0;
          robot.vy = 0;
          robot.airborne = false;
          robot.falling = false;
        } else if (robot.y < -CONFIG.robot.fallKill * metrics.scale) {
          kill(robot);
        }
      }

      // The path, sampled for the ghost trail it will leave behind.
      robot.trailT += dt;
      if (robot.trailT >= ghosts.sampleInterval) {
        robot.trailT = 0;
        robot.trail.push({ x: robot.x, y: robot.y });
        if (robot.trail.length > ghosts.maxPoints) robot.trail.shift();
      }

      // Away clean: the next attempt starts from the first hazard again.
      if (robot.x - state.camX > metrics.width + loop.exitMargin) {
        state.phase = 'waiting';
        state.phaseT = 0;
        state.attempt = 1;
        state.robot = null;
      }
    }
  }

  if (state.phase === 'dissolve' && state.phaseT >= loop.dissolveSeconds) {
    state.phase = 'waiting';
    state.phaseT = 0;
    state.robot = null;
    state.attempt += 1;
  }

  if (state.phase === 'waiting' && state.phaseT >= loop.respawnDelay) {
    spawnRobot();
  }

  // Particles and ghosts age on their own.
  for (let index = state.particles.length - 1; index >= 0; index -= 1) {
    const particle = state.particles[index];
    particle.age += dt;
    if (particle.age >= particle.life) {
      state.particles.splice(index, 1);
      continue;
    }
    particle.vy -= 260 * metrics.scale * dt;
    particle.x += particle.vx * dt;
    particle.y -= particle.vy * dt;
  }
  for (let index = state.ghosts.length - 1; index >= 0; index -= 1) {
    if (state.time - state.ghosts[index].born > ghosts.fadeSeconds) {
      state.ghosts.splice(index, 1);
    }
  }
}

/* ---------------------------------------------------------------------
   Drawing
   ------------------------------------------------------------------- */

function screenX(worldX) {
  return worldX - state.camX;
}

function drawBackdrop() {
  const gradient = ctx.createLinearGradient(0, 0, 0, metrics.height);
  gradient.addColorStop(0, palette.base);
  gradient.addColorStop(0.42, palette.backdrop);
  gradient.addColorStop(1, palette.far);
  ctx.fillStyle = gradient;
  ctx.fillRect(0, 0, metrics.width, metrics.height);
}

/** One procedural layer of lab architecture, standing on the floor line. */
function drawParallaxLayer(settings, colour, salt) {
  const scale = metrics.scale;
  const spacing = settings.spacing * scale;
  const offset = state.camX * settings.factor;
  const first = Math.floor(offset / spacing) - 1;
  const last = Math.ceil((offset + metrics.width) / spacing) + 1;

  ctx.fillStyle = colour;
  for (let index = first; index <= last; index += 1) {
    const seed = index * 13 + salt;
    if (hash(seed) < 0.18) continue;                 // gaps in the skyline
    const width = settings.width * scale * between(seed + 1, 0.55, 1.25);
    const height = between(seed + 2, settings.minHeight, settings.maxHeight) * scale;
    const x = index * spacing - offset
      + between(seed + 3, -0.2, 0.2) * spacing;
    const top = metrics.groundY - height;
    ctx.fillRect(Math.round(x), Math.round(top), Math.round(width),
                 Math.round(height));

    // A tank or a vent stack on top of roughly one block in three.
    if (hash(seed + 4) > 0.66) {
      const stackWidth = width * between(seed + 5, 0.18, 0.34);
      const stackHeight = height * between(seed + 6, 0.12, 0.3);
      ctx.fillRect(Math.round(x + width * 0.2), Math.round(top - stackHeight),
                   Math.round(stackWidth), Math.round(stackHeight));
    }
  }
}

function drawCeilingAndFloor() {
  const scale = metrics.scale;
  ctx.fillStyle = palette.near;
  ctx.fillRect(0, 0, metrics.width, metrics.ceilingY);

  // The floor, drawn as the stretches that are still there.
  const fromX = state.camX;
  const toX = state.camX + metrics.width;
  const holes = hazardsBetween(fromX - 400 * scale, toX + 400 * scale)
    .filter(hazard => hazard.type === 'pit' || hazard.type === 'bridge')
    .sort((first, second) => first.x - second.x);

  let cursor = fromX - 40;
  const segments = [];
  holes.forEach(hazard => {
    const left = hazard.x - hazard.half;
    const right = hazard.x + hazard.half;
    if (left > cursor) segments.push([cursor, left]);
    cursor = Math.max(cursor, right);
  });
  segments.push([cursor, toX + 40]);

  segments.forEach(segment => {
    const left = screenX(segment[0]);
    const width = segment[1] - segment[0];
    if (width <= 0) return;
    ctx.fillStyle = palette.near;
    ctx.fillRect(Math.round(left), metrics.groundY,
                 Math.ceil(width), metrics.height - metrics.groundY);
    // A hairline along the surface, so the floor edge stays readable
    // against the backdrop without the silhouette gaining any detail.
    ctx.fillStyle = palette.mid;
    ctx.fillRect(Math.round(left), metrics.groundY, Math.ceil(width), 1);
  });
}

function drawHazards() {
  const scale = metrics.scale;
  const hazards = CONFIG.hazards;
  const visible = hazardsBetween(state.camX - 400 * scale,
                                 state.camX + metrics.width + 400 * scale);

  visible.forEach(hazard => {
    const x = screenX(hazard.x);
    const entry = stateFor(hazard);

    if (hazard.type === 'laser') {
      // Emitter housing on the ceiling.
      const housing = 14 * scale;
      ctx.fillStyle = palette.near;
      ctx.fillRect(Math.round(x - housing / 2), metrics.ceilingY,
                   Math.round(housing), Math.round(9 * scale));

      const beamX = screenX(laserBeamX(hazard));
      const top = metrics.ceilingY + 9 * scale;
      ctx.save();
      ctx.strokeStyle = rgba(palette.accent, 0.5);
      ctx.lineWidth = Math.max(1, 1.1 * scale);
      ctx.shadowColor = rgba(palette.accent, 0.55);
      ctx.shadowBlur = 10 * scale;
      ctx.beginPath();
      ctx.moveTo(x, top);
      ctx.lineTo(beamX, metrics.groundY);
      ctx.stroke();
      ctx.restore();

      ctx.fillStyle = rgba(palette.accent, 0.8);
      ctx.beginPath();
      ctx.arc(x, top, Math.max(1.2, 1.8 * scale), 0, Math.PI * 2);
      ctx.fill();

    } else if (hazard.type === 'gap') {
      // A blast panel hanging low enough that the robot has to duck.
      const clearance = hazards.gap.clearance * scale;
      const width = hazards.gap.panelWidth * scale;
      const top = metrics.ceilingY;
      const bottom = metrics.groundY - clearance;
      ctx.fillStyle = palette.near;
      ctx.fillRect(Math.round(x - width / 2), Math.round(top),
                   Math.round(width), Math.round(bottom - top));

    } else if (hazard.type === 'drone') {
      const droneX = screenX(dronePosition(hazard));
      const size = hazards.drone.size * scale;
      const y = metrics.groundY - hazards.drone.hover * scale;
      ctx.fillStyle = palette.near;
      ctx.beginPath();
      ctx.ellipse(droneX, y, size * 0.62, size * 0.34, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillRect(Math.round(droneX - size * 0.5), Math.round(y - size * 0.42),
                   Math.round(size), Math.round(size * 0.16));
      // A dim scan cone, in the silhouette tone rather than the accent.
      ctx.fillStyle = rgba(palette.muted, 0.07);
      ctx.beginPath();
      ctx.moveTo(droneX, y + size * 0.3);
      ctx.lineTo(droneX - size * 0.9, metrics.groundY);
      ctx.lineTo(droneX + size * 0.9, metrics.groundY);
      ctx.closePath();
      ctx.fill();

    } else if (hazard.type === 'bridge') {
      const bridge = hazards.bridge;
      const plankWidth = hazard.half * 2 / bridge.planks;
      const thickness = 5 * scale;
      // A span that has just been rebuilt for a new attempt fades in.
      const rebuilt = clamp((state.time - entry.born) / bridge.rebuildSeconds,
                            0, 1);
      ctx.save();
      ctx.globalAlpha = rebuilt;
      ctx.fillStyle = palette.near;
      entry.planks.forEach((record, plank) => {
        if (record.gone) return;
        const left = hazard.x - hazard.half + plank * plankWidth;
        ctx.fillRect(Math.round(screenX(left) + plankWidth * 0.08),
                     metrics.groundY,
                     Math.ceil(plankWidth * 0.84), Math.round(thickness));
      });
      ctx.restore();
      // Planks on their way down.
      entry.falling.forEach(plank => {
        ctx.save();
        ctx.translate(screenX(plank.x), metrics.groundY + plank.y);
        ctx.rotate(plank.spin * plank.y * 0.002);
        ctx.fillStyle = palette.near;
        ctx.fillRect(Math.round(-plankWidth * 0.42), 0,
                     Math.ceil(plankWidth * 0.84), Math.round(thickness));
        ctx.restore();
      });
      // Anchor posts either side of the span.
      [-hazard.half, hazard.half].forEach(side => {
        ctx.fillStyle = palette.near;
        ctx.fillRect(Math.round(screenX(hazard.x + side) - 3 * scale),
                     metrics.groundY - 16 * scale,
                     Math.round(6 * scale), Math.round(16 * scale));
      });
    }
  });
}

function drawGhosts() {
  const ghosts = CONFIG.ghosts;
  state.ghosts.forEach(ghost => {
    const age = state.time - ghost.born;
    const alpha = ghosts.alpha * (1 - clamp(age / ghosts.fadeSeconds, 0, 1));
    if (alpha <= 0.005 || ghost.points.length < 2) return;

    ctx.strokeStyle = rgba(palette.muted, alpha);
    ctx.lineWidth = ghosts.width;
    ctx.beginPath();
    ghost.points.forEach((point, index) => {
      const x = screenX(point.x);
      const y = metrics.groundY - point.y - metrics.robotHeight * 0.5;
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    // A small mark where that attempt ended.
    const last = ghost.points[ghost.points.length - 1];
    ctx.fillStyle = rgba(palette.muted, alpha * 1.4);
    ctx.beginPath();
    ctx.arc(screenX(last.x),
            metrics.groundY - last.y - metrics.robotHeight * 0.5,
            2.2 * metrics.scale, 0, Math.PI * 2);
    ctx.fill();
  });
}

function drawRobot(robot) {
  const scale = metrics.scale;
  const width = metrics.robotWidth;
  const full = metrics.robotHeight;
  const height = full * (1 - robot.duck * (1 - CONFIG.robot.duckFactor));
  const x = screenX(robot.x);
  const feet = metrics.groundY - robot.y;
  const top = feet - height;
  const alpha = robot.fadeIn;

  ctx.save();
  ctx.globalAlpha = alpha;

  // Legs. Tucked in the air, striding on the ground.
  ctx.strokeStyle = palette.near;
  ctx.lineWidth = Math.max(1.4, 3 * scale);
  ctx.lineCap = 'round';
  const swing = robot.airborne ? 0.25
    : Math.sin(robot.stride) * (robot.holding ? 0.1 : 1);
  [-1, 1].forEach(side => {
    const spread = swing * side * width * 0.42;
    ctx.beginPath();
    ctx.moveTo(x + side * width * 0.16, feet - height * 0.32);
    ctx.lineTo(x + side * width * 0.16 + spread, feet);
    ctx.stroke();
  });

  // Body and head.
  ctx.fillStyle = palette.near;
  const bodyHeight = height * 0.7;
  ctx.fillRect(Math.round(x - width / 2), Math.round(top + height * 0.02),
               Math.round(width), Math.round(bodyHeight));
  ctx.fillRect(Math.round(x - width * 0.3), Math.round(top - height * 0.1),
               Math.round(width * 0.6), Math.round(height * 0.14));

  // The optical sensor — the only accent on the robot.
  ctx.shadowColor = rgba(palette.accent, 0.9);
  ctx.shadowBlur = 12 * scale;
  ctx.fillStyle = palette.accent;
  ctx.beginPath();
  ctx.arc(x + width * 0.26, top + height * 0.16,
          CONFIG.robot.sensor * scale, 0, Math.PI * 2);
  ctx.fill();

  ctx.restore();
}

function drawParticles() {
  state.particles.forEach(particle => {
    const share = 1 - particle.age / particle.life;
    const size = (particle.accent ? 2.4 : 1.9) * metrics.scale * share;
    ctx.fillStyle = particle.accent
      ? rgba(palette.accent, share * 0.9)
      : rgba(palette.muted, share * 0.5);
    ctx.fillRect(screenX(particle.x) - size / 2, particle.y - size / 2,
                 size, size);
  });
}

function draw() {
  ctx.clearRect(0, 0, metrics.width, metrics.height);
  drawBackdrop();
  drawParallaxLayer(CONFIG.parallax.far, palette.far, 101);
  drawParallaxLayer(CONFIG.parallax.mid, palette.mid, 211);
  drawCeilingAndFloor();
  drawGhosts();
  drawHazards();
  if (state.robot && state.phase === 'run') drawRobot(state.robot);
  drawParticles();
}

/* ---------------------------------------------------------------------
   The loop
   ------------------------------------------------------------------- */

let frame = null;
let lastTime = 0;

function tick(now) {
  const dt = Math.min(0.05, (now - lastTime) / 1000) || 0;
  lastTime = now;
  step(dt);
  draw();
  frame = window.requestAnimationFrame(tick);
}

function startAnimating() {
  if (frame !== null) return;
  lastTime = performance.now();
  frame = window.requestAnimationFrame(tick);
}

function stopAnimating() {
  if (frame === null) return;
  window.cancelAnimationFrame(frame);
  frame = null;
}

/** Reduced motion: one frame, fast-forwarded to somewhere interesting. */
function drawStillFrame() {
  reset();
  const dt = 1 / 60;
  const steps = Math.round(CONFIG.still.fastForwardSeconds / dt);
  for (let index = 0; index < steps; index += 1) step(dt);
  draw();
}

function boot() {
  palette = readPalette();
  resize();
  if (reduceMotion.matches) {
    stopAnimating();
    drawStillFrame();
    return;
  }
  reset();
  startAnimating();
}

window.addEventListener('resize', () => {
  resize();
  if (reduceMotion.matches) drawStillFrame();
});

// Nothing runs while the tab is in the background.
document.addEventListener('visibilitychange', () => {
  if (reduceMotion.matches) return;
  if (document.hidden) stopAnimating();
  else startAnimating();
});

// Following the setting if it is changed while the page is open.
if (typeof reduceMotion.addEventListener === 'function') {
  reduceMotion.addEventListener('change', boot);
}

/* ---- the two buttons ------------------------------------------------ */

document.getElementById('start').addEventListener('click', () => {
  window.location.href = 'levels/';
});

// ABOUT is where reset-progress will live. Nothing behind it yet.
document.getElementById('about').addEventListener('click', () => {});

boot();
