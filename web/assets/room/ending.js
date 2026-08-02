'use strict';

/* =====================================================================
   The ending sequence.

   Shown once R-5 has actually reached the unlocked blast door in the last
   chamber, in place of returning to the chamber select.

   WHAT IT DRAWS, AND WHY IT DRAWS IT THAT WAY
   The laboratory going dark, with the blast door standing open, R-5 flying
   out through it, and the emergency lamps shutting down one by one.  The
   door and the drone are drawn by `window.Shapes` — the very recipes the
   room itself uses — so the thing that flies away is recognisably the same
   drone the player has been watching, and the door is the same door.  A
   second set of drawings of the same two objects would have been free to
   drift out of step with them.

   Nothing here reads the simulation.  It is handed a summary of the run
   that finished and four callbacks, and it owns a canvas and a panel.

   The timeline is a function of one clock, so the whole sequence is a pure
   function of elapsed time.  Under reduced motion the clock is pinned to a
   settled frame near the end and never advances: the lamps are out, the
   door is open, and R-5 is already away.
   ===================================================================== */

window.Ending = (function () {

  /* Every number the sequence is paced by, in seconds from its start. The
     phases overlap on purpose — the lamps are already dying while R-5 is
     still on its way out, which is what makes it read as one event rather
     than as four cues in a row. */
  const TIMELINE = {
    hold: 0.9,          // the laboratory, lit, before anything moves
    flightFrom: 0.9,    // R-5 sets off
    flightTo: 5.1,      // and reaches the doorway
    vanishFor: 1.1,     // how long it takes to disappear into the light
    lampsFrom: 2.6,     // the emergency lamps begin shutting down
    lampsTo: 6.4,       // and the last one goes out
    darkFrom: 4.2,      // the laboratory begins fading
    darkTo: 8.6,        // and is gone
    settle: 7.4,        // the frame reduced motion shows
    loopAt: 13.0,       // then round again, after a long look at the dark
  };

  const SCENE = {
    lamps: 5,
    shelves: 7,
    floorRatio: 0.78,     // the floor line, as a share of the canvas height
    ceilingRatio: 0.17,
    doorRatio: 0.845,     // where the doorway is, across the canvas
    doorHeight: 0.34,     // and how tall, as a share of the canvas
    droneFrom: 0.12,      // R-5 starts here and flies to the doorway
    droneSize: 34,        // px at the reference height below
    scaleFrom: 900,
  };

  /** A deterministic 0..1 from an integer. No Math.random anywhere. */
  function hash(n) {
    const x = Math.sin(n * 12.9898) * 43758.5453;
    return x - Math.floor(x);
  }

  function between(n, low, high) {
    return low + hash(n) * (high - low);
  }

  function clamp(value, low, high) {
    return value < low ? low : value > high ? high : value;
  }

  /** 0 before `from`, 1 after `to`, smoothly eased in between. */
  function ramp(now, from, to) {
    if (to <= from) return now >= to ? 1 : 0;
    const share = clamp((now - from) / (to - from), 0, 1);
    return share * share * (3 - 2 * share);
  }

  function reducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function palette() {
    const computed = window.getComputedStyle(document.documentElement);
    const read = name => computed.getPropertyValue(name).trim();
    return {
      base: read('--base'),
      muted: read('--muted'),
      accent: read('--accent'),
      goal: read('--goal'),
      hazard: read('--hazard'),
      hairlineFaint: read('--hairline-faint'),
      // The blast door is bricked with the wall's stone; see `layBricks`.
      cellWall: read('--cell-wall'),
      warn: read('--warn'),
    };
  }

  /* ---------------------------------------------------------------------
     The run's statistics
     ------------------------------------------------------------------- */

  /**
   * The four figures the panel shows, from the run that just finished.
   *
   * Every one of them is read off what the project already collected. A
   * figure the run does not have is returned absent rather than invented,
   * and the panel prints an em dash for it — a made-up number on a screen
   * that says "mission complete" would be the worst place in the project
   * to have one.
   *
   * The escape rate says which layouts it was measured over, because the
   * two possible answers mean very different things: the share of training
   * episodes that escaped, or — if the player has run an evaluation — the
   * share of held-out warehouses the finished policy escapes. The second is
   * the better number and is preferred when it exists.
   */
  function summary(batch, snapshot, algorithmLabel) {
    const history = (batch && batch.history) || [];
    const evaluation = batch && batch.evaluation;

    const episodes = snapshot && snapshot.progress
      ? snapshot.progress.count : null;

    let meanReward = null;
    if (history.length) {
      const rewards = history.map(entry => entry.reward)
        .filter(value => typeof value === 'number' && isFinite(value));
      if (rewards.length) {
        meanReward = rewards.reduce((total, value) => total + value, 0)
                   / rewards.length;
      }
    }

    let escapeRate = null;
    let escapeOver = null;
    if (evaluation && evaluation.test
        && typeof evaluation.test.escapeRate === 'number') {
      escapeRate = evaluation.test.escapeRate;
      escapeOver = evaluation.test.layouts + ' unseen layouts';
    } else if (history.length) {
      const escapes = history.filter(entry => entry.success).length;
      escapeRate = escapes / history.length;
      escapeOver = history.length + ' training episodes';
    }

    return {
      episodes: episodes,
      meanReward: meanReward,
      escapeRate: escapeRate,
      escapeOver: escapeOver,
      algorithm: algorithmLabel || null,
    };
  }

  function integer(value) {
    return value === null || value === undefined
      ? null : String(Math.round(value));
  }

  /* ---------------------------------------------------------------------
     Mounting
     ------------------------------------------------------------------- */

  /**
   * Take over the ending section.
   *
   * `handlers` are the four buttons: replay, results, again, menu. Each is
   * called with nothing and is expected to do the navigating itself, because
   * what "play again" means belongs to the screen that owns the run.
   */
  function mount(host, handlers) {
    const on = handlers || {};
    const canvas = host.querySelector('#ending-scene');
    const statsHost = host.querySelector('#ending-stats');
    const ctx = canvas.getContext('2d');

    const view = { width: 0, height: 0, scale: 1 };
    let colours = palette();
    let frame = null;
    let began = 0;
    let open = false;

    function resize() {
      const ratio = window.devicePixelRatio || 1;
      const width = host.clientWidth || window.innerWidth;
      const height = host.clientHeight || window.innerHeight;
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      view.width = width;
      view.height = height;
      view.scale = clamp(height / SCENE.scaleFrom, 0.7, 1.35);
    }

    /* ---- the scene ---------------------------------------------------- */

    function floorY() { return view.height * SCENE.floorRatio; }
    function ceilingY() { return view.height * SCENE.ceilingRatio; }

    function doorBox() {
      const height = view.height * SCENE.doorHeight;
      const width = height * 0.62;
      return {
        left: view.width * SCENE.doorRatio - width / 2,
        top: floorY() - height,
        width: width,
        height: height,
      };
    }

    /** The laboratory shell: floor, ceiling and a run of shelves. */
    function drawLaboratory(lit) {
      const floor = floorY();
      const ceiling = ceilingY();

      ctx.save();
      ctx.globalAlpha = lit;

      // The two structural lines, which is all the architecture this needs.
      ctx.strokeStyle = colours.hairlineFaint;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, floor);
      ctx.lineTo(view.width, floor);
      ctx.moveTo(0, ceiling);
      ctx.lineTo(view.width, ceiling);
      ctx.stroke();

      // Floor plates, receding towards the door, so the room has a depth
      // for R-5 to fly out of.
      ctx.globalAlpha = lit * 0.34;
      ctx.beginPath();
      for (let index = 0; index <= 14; index += 1) {
        const x = (index / 14) * view.width;
        ctx.moveTo(x, floor);
        ctx.lineTo(view.width * 0.5 + (x - view.width * 0.5) * 0.62,
                   floor + view.height * 0.12);
      }
      ctx.stroke();

      // Shelving, in silhouette. Deterministic from `hash`, so the same
      // window always draws the same warehouse.
      ctx.globalAlpha = lit * 0.5;
      ctx.fillStyle = colours.base;
      ctx.strokeStyle = colours.hairlineFaint;
      for (let index = 0; index < SCENE.shelves; index += 1) {
        const width = between(index * 3 + 1, 0.03, 0.055) * view.width;
        const height = between(index * 3 + 2, 0.1, 0.28) * view.height;
        const x = between(index * 3 + 3, 0.04, 0.72) * view.width;
        ctx.fillRect(x, floor - height, width, height);
        ctx.strokeRect(x, floor - height, width, height);
        // One shelf line, so it reads as storage rather than as a block.
        ctx.beginPath();
        ctx.moveTo(x, floor - height * 0.5);
        ctx.lineTo(x + width, floor - height * 0.5);
        ctx.stroke();
      }
      ctx.restore();
    }

    /**
     * The emergency lamps, shutting down one after another.
     *
     * They go out left to right over `lampsFrom`..`lampsTo`, each with its
     * own moment, so the shutdown sweeps through the building instead of
     * everything darkening at once.
     */
    function drawLamps(now, lit) {
      const ceiling = ceilingY();
      const total = SCENE.lamps;

      for (let index = 0; index < total; index += 1) {
        const share = total === 1 ? 0 : index / (total - 1);
        const diesAt = TIMELINE.lampsFrom
          + (TIMELINE.lampsTo - TIMELINE.lampsFrom) * share;
        // 1 while alive, 0 once it has gone out, over half a second.
        const alive = 1 - ramp(now, diesAt, diesAt + 0.5);
        const x = view.width * (0.1 + 0.72 * share);
        // A dying lamp flickers; a healthy one pulses slowly.
        const flicker = alive < 1 && alive > 0
          ? 0.35 + 0.65 * Math.abs(Math.sin(now * 22 + index))
          : 0.55 + 0.45 * Math.sin(now * 1.7 + index * 1.3);

        const strength = lit * alive * flicker;
        if (strength <= 0.01) {
          // Out: the housing stays visible, unlit, so the room still has
          // its fittings in the dark.
          ctx.save();
          ctx.globalAlpha = lit * 0.22;
          ctx.fillStyle = colours.muted;
          ctx.fillRect(x - 4 * view.scale, ceiling,
                       8 * view.scale, 3 * view.scale);
          ctx.restore();
          continue;
        }

        ctx.save();
        // The wash it throws down into the room.
        const reach = view.height * 0.3 * (0.6 + 0.4 * alive);
        const wash = ctx.createRadialGradient(x, ceiling, 0, x, ceiling, reach);
        wash.addColorStop(0, colours.hazard);
        wash.addColorStop(1, 'transparent');
        ctx.globalAlpha = strength * 0.2;
        ctx.fillStyle = wash;
        ctx.beginPath();
        ctx.arc(x, ceiling, reach, 0, Math.PI * 2);
        ctx.fill();

        // The lamp itself.
        ctx.globalAlpha = Math.min(1, strength);
        ctx.fillStyle = colours.hazard;
        ctx.shadowColor = colours.hazard;
        ctx.shadowBlur = 14 * view.scale;
        ctx.fillRect(x - 4 * view.scale, ceiling - 1 * view.scale,
                     8 * view.scale, 4 * view.scale);
        ctx.restore();
      }
    }

    /**
     * The blast door, open, with the outside coming through it.
     *
     * Drawn by the room's own `blastdoor` recipe, in its open state, so this
     * is the same door the player watched unlock. The daylight behind it is
     * this file's, because in the room there is no outside to draw.
     */
    function drawDoorway(now, lit) {
      const box = doorBox();
      // The doorway keeps its light long after the room has gone dark: it is
      // the way out, and the last thing left on screen.
      const glow = Math.max(lit, 0.55);

      ctx.save();
      // The light spilling in across the floor.
      const spill = ctx.createRadialGradient(
        box.left + box.width / 2, box.top + box.height * 0.55, 0,
        box.left + box.width / 2, box.top + box.height * 0.55,
        box.height * 1.5);
      spill.addColorStop(0, colours.goal);
      spill.addColorStop(1, 'transparent');
      ctx.globalAlpha = 0.26 * glow;
      ctx.fillStyle = spill;
      ctx.beginPath();
      ctx.arc(box.left + box.width / 2, box.top + box.height * 0.55,
              box.height * 1.5, 0, Math.PI * 2);
      ctx.fill();

      ctx.globalAlpha = glow;
      window.Shapes.draw(ctx, 'blastdoor', box,
                         { color: colours.goal, phase: now },
                         box.height, colours);
      ctx.restore();
    }

    /**
     * R-5, flying out.
     *
     * The same `drone` recipe the room draws, handed a velocity so it tilts
     * into its travel the way it does in flight. It fades and shrinks into
     * the doorway rather than sliding off the edge, because what the story
     * says is that it got out — not that it left the frame.
     */
    function drawDrone(now) {
      const flight = ramp(now, TIMELINE.flightFrom, TIMELINE.flightTo);
      const box = doorBox();
      const fromX = view.width * SCENE.droneFrom;
      const toX = box.left + box.width / 2;
      const floor = floorY();
      const x = fromX + (toX - fromX) * flight;
      // Rising as it goes, and bobbing, so it is flying rather than sliding.
      const y = floor - view.height * (0.1 + 0.14 * flight)
              + Math.sin(now * 2.1) * 5 * view.scale;

      // Gone into the light: fully faded one `vanishFor` after arriving.
      const away = ramp(now, TIMELINE.flightTo,
                        TIMELINE.flightTo + TIMELINE.vanishFor);
      if (away >= 1) return;

      const size = SCENE.droneSize * view.scale * (1 - away * 0.65);

      ctx.save();
      ctx.globalAlpha = 1 - away;

      // The trail it has left, fading behind it.
      if (flight > 0.02) {
        const trail = ctx.createLinearGradient(fromX, y, x, y);
        trail.addColorStop(0, 'transparent');
        trail.addColorStop(1, colours.accent);
        ctx.globalAlpha = (1 - away) * 0.28;
        ctx.strokeStyle = trail;
        ctx.lineWidth = 1.6 * view.scale;
        ctx.beginPath();
        ctx.moveTo(fromX, floor - view.height * 0.1);
        ctx.lineTo(x, y);
        ctx.stroke();
        ctx.globalAlpha = 1 - away;
      }

      // Travelling right, and rising: the velocity the recipe tilts by.
      window.Shapes.drone(ctx, { x: x, y: y }, size,
                          { x: 1, y: -0.18 },
                          { phase: now, speedLimit: 1 }, colours);
      ctx.restore();
    }

    function draw(now) {
      // How lit the laboratory is: full until `darkFrom`, gone by `darkTo`.
      const lit = 1 - ramp(now, TIMELINE.darkFrom, TIMELINE.darkTo);

      ctx.clearRect(0, 0, view.width, view.height);
      drawLaboratory(lit);
      drawLamps(now, lit);
      drawDoorway(now, lit);
      drawDrone(now);
    }

    function tick(stamp) {
      if (!open) return;
      const now = (stamp - began) / 1000;
      draw(now % TIMELINE.loopAt);
      frame = window.requestAnimationFrame(tick);
    }

    function start() {
      if (reducedMotion()) {
        // One settled frame: lamps out, door open, R-5 already away. No
        // loop at all, so nothing on this screen moves.
        resize();
        colours = palette();
        draw(TIMELINE.settle);
        return;
      }
      if (frame !== null) return;
      began = window.performance ? window.performance.now() : 0;
      frame = window.requestAnimationFrame(tick);
    }

    function stop() {
      if (frame === null) return;
      window.cancelAnimationFrame(frame);
      frame = null;
    }

    /* ---- the statistics panel ---------------------------------------- */

    function element(tag, className, text) {
      const node = document.createElement(tag);
      if (className) node.className = className;
      if (text !== undefined) node.textContent = text;
      return node;
    }

    function paintStats(figures) {
      statsHost.innerHTML = '';
      const rows = [
        ['Training Episodes', integer(figures.episodes), null],
        ['Mean Reward', figures.meanReward === null
          ? null : figures.meanReward.toFixed(1), null],
        ['Escape Rate', figures.escapeRate === null
          ? null : Math.round(figures.escapeRate * 100) + '%',
         figures.escapeOver],
        ['Algorithm Used', figures.algorithm, null],
      ];

      rows.forEach(row => {
        const cell = element('div');
        const term = element('dt', null, row[0]);
        // What the escape rate was measured over, so the number cannot be
        // read as a claim about layouts it was not measured on.
        if (row[2]) term.title = 'Measured over ' + row[2];
        cell.appendChild(term);
        const value = element('dd', row[1] === null ? 'is-absent' : null,
                              row[1] === null ? '—' : row[1]);
        if (row[1] === null) {
          value.title = 'This run did not record that figure.';
        }
        cell.appendChild(value);
        statsHost.appendChild(cell);
      });
    }

    /* ---- wiring ------------------------------------------------------ */

    const buttons = [
      ['#ending-replay', 'replay'],
      ['#ending-results', 'results'],
      ['#ending-again', 'again'],
      ['#ending-menu', 'menu'],
    ];
    buttons.forEach(pair => {
      const node = host.querySelector(pair[0]);
      if (!node) return;
      node.addEventListener('click', () => {
        if (typeof on[pair[1]] === 'function') on[pair[1]]();
      });
    });

    window.addEventListener('resize', () => {
      if (!open) return;
      resize();
      if (reducedMotion()) draw(TIMELINE.settle);
    });

    return {
      /** Show it, with the figures of the run that just finished. */
      show: function (figures, options) {
        const settings = options || {};
        paintStats(figures);
        // Which of the four make sense for this run. A replay needs an
        // escape to replay; the others always do.
        const replay = host.querySelector('#ending-replay');
        if (replay) {
          const can = settings.canReplay !== false;
          replay.disabled = !can;
          replay.setAttribute('aria-disabled', String(!can));
        }

        host.hidden = false;
        host.setAttribute('aria-hidden', 'false');
        open = true;
        resize();
        colours = palette();
        // Drawn once before the fade begins, so the overlay never appears
        // over an empty canvas.
        draw(reducedMotion() ? TIMELINE.settle : 0);
        start();
        // Next frame, so the transition has a value to animate from.
        window.requestAnimationFrame(() => host.classList.add('is-open'));

        const first = host.querySelector('.ending-btn:not(:disabled)');
        if (first) first.focus();
      },

      hide: function () {
        open = false;
        stop();
        host.classList.remove('is-open');
        host.hidden = true;
        host.setAttribute('aria-hidden', 'true');
      },

      get isOpen() { return open; },
    };
  }

  return {
    mount: mount,
    // Exported because it is the part worth checking directly: the figures
    // are read from the run, and an absent one has to stay absent.
    summary: summary,
    TIMELINE: TIMELINE,
  };
})();
