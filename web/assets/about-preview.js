'use strict';

/* =====================================================================
   ABOUT — the looping sector previews.

   One small animation per sector, played on the About slide beside the
   words, so "ice sends it sideways" and "the gantry gives way" are things
   the reader watches happen rather than things they are told.

   THESE ARE DEMONSTRATIONS. THEY ARE NOT THE ENVIRONMENT.
   Nothing here imports, instantiates, steps or samples anything from
   `game/`.  There is no agent, no policy, no reward and no transition
   model — each preview is a hand-scripted loop of positions against a
   normalised clock, the same way a diagram in a manual is drawn rather
   than measured.  Two consequences worth being explicit about:

     - a preview can never disagree with the room in a way that matters,
       because it makes no claim about what the agent *would* do; it
       illustrates the hazard the room's own text describes.
     - nothing here can perturb a run.  The About panel is reached from
       the start screen, where no session exists at all.

   Everything is drawn from the shared palette — no colour is introduced
   here — so a preview looks like the room it stands for.

   ONE ANIMATION FRAME LOOP, AND ONLY WHILE VISIBLE
   `About` starts the preview belonging to the slide the reader is on and
   stops every other one, so there is exactly one loop running while the
   panel is open and none at all once it closes.
   ===================================================================== */

window.AboutPreview = (function () {

  /* Beats are labelled so the reader can follow what they are looking at:
     the caption under the canvas changes as the loop reaches each stage.
     `at` is the fraction of the loop where that stage begins. */
  const SCRIPTS = {
    1: { seconds: 7.5, beats: [
           { at: 0.00, text: 'Slides on ice' },
           { at: 0.42, text: 'Uses the teleport pad' },
           { at: 0.68, text: 'Avoids the laser' }] },
    2: { seconds: 9.0, beats: [
           { at: 0.00, text: 'Crosses the gantry' },
           { at: 0.26, text: 'A plank gives way' },
           { at: 0.46, text: 'Learns from the fall' },
           { at: 0.66, text: 'Crosses it safely' }] },
    3: { seconds: 8.0, beats: [
           { at: 0.00, text: 'Generators, in order' },
           { at: 0.72, text: 'The exit unlocks' }] },
    4: { seconds: 8.0, beats: [
           { at: 0.00, text: 'Enters the wind tunnel' },
           { at: 0.34, text: 'Corrects against the downdraught' },
           { at: 0.70, text: 'Sets down on the pad' }] },
    5: { seconds: 9.0, beats: [
           { at: 0.00, text: 'Drones on patrol' },
           { at: 0.34, text: 'Diagonal beams, still armed' },
           { at: 0.70, text: 'Terminal reached — blast door open' }] },
  };

  /* ---- small drawing helpers, shared by every preview -------------- */

  function palette() {
    const style = window.getComputedStyle(document.documentElement);
    const read = name => style.getPropertyValue(name).trim();
    return {
      base: read('--base'),
      muted: read('--muted'),
      accent: read('--accent'),
      goal: read('--goal'),
      hazard: read('--hazard'),
      warn: read('--warn'),
      hairline: read('--hairline'),
      floor: read('--cell-floor'),
      wall: read('--cell-wall'),
      ice: read('--cell-slippery'),
    };
  }

  function glow(ctx, colour, blur, draw) {
    ctx.save();
    ctx.shadowColor = colour;
    ctx.shadowBlur = blur;
    draw();
    ctx.restore();
  }

  /** The laboratory floor every preview stands on: a faint square grid. */
  function floor(ctx, w, h, p) {
    ctx.fillStyle = p.base;
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = p.hairline;
    ctx.globalAlpha = 0.16;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = 0; x <= w; x += 20) { ctx.moveTo(x, 0); ctx.lineTo(x, h); }
    for (let y = 0; y <= h; y += 20) { ctx.moveTo(0, y); ctx.lineTo(w, y); }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  /** R-5 itself: a small lit chip, the same silhouette in every preview. */
  function robot(ctx, x, y, p, colour) {
    const tone = colour || p.accent;
    glow(ctx, tone, 14, () => {
      ctx.fillStyle = tone;
      ctx.fillRect(x - 6, y - 6, 12, 12);
    });
    ctx.fillStyle = p.base;
    ctx.fillRect(x - 2.5, y - 2.5, 5, 5);
  }

  function ease(t) { return t * t * (3 - 2 * t); }
  function lerp(a, b, t) { return a + (b - a) * t; }
  /** Map t from [a,b] onto [0,1], clamped — the workhorse of every script. */
  function span(t, a, b) {
    if (t <= a) return 0;
    if (t >= b) return 1;
    return (t - a) / (b - a);
  }

  /* ---- the five scripts -------------------------------------------- *
     Each takes the normalised loop position `t` and draws one frame.    */

  /* Room 1 — ice, a teleport pad, and a beam wall.
     Ice carries it sideways; the pad skips the beams entirely. */
  function room1(ctx, w, h, t, p) {
    const midY = h * 0.62;
    const iceFrom = w * 0.14, iceTo = w * 0.34;

    // The ice sheet.
    ctx.fillStyle = p.ice;
    ctx.globalAlpha = 0.5;
    ctx.fillRect(iceFrom, midY - 16, iceTo - iceFrom, 32);
    ctx.globalAlpha = 1;
    ctx.strokeStyle = p.ice;
    ctx.globalAlpha = 0.65;
    ctx.strokeRect(iceFrom, midY - 16, iceTo - iceFrom, 32);
    ctx.globalAlpha = 1;

    // The beam wall, always lit — the point is that it is never touched.
    const beamX = w * 0.66;
    for (let i = 0; i < 3; i += 1) {
      const x = beamX + i * 14;
      glow(ctx, p.hazard, 10, () => {
        ctx.strokeStyle = p.hazard;
        ctx.lineWidth = 2;
        ctx.globalAlpha = 0.55 + 0.45 * Math.abs(Math.sin(t * 12 + i));
        ctx.beginPath();
        ctx.moveTo(x, h * 0.16);
        ctx.lineTo(x, h * 0.92);
        ctx.stroke();
        ctx.globalAlpha = 1;
      });
    }

    // The two pads.
    const padIn = w * 0.46, padOut = w * 0.86;
    [padIn, padOut].forEach(x => {
      glow(ctx, p.warn, 12, () => {
        ctx.strokeStyle = p.warn;
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.arc(x, midY, 11, 0, Math.PI * 2);
        ctx.stroke();
      });
    });

    // The walk: slide across the ice, drift sideways, reach the pad,
    // vanish, and reappear beyond the beams.
    let x, y = midY, tone = null;
    if (t < 0.42) {
      const k = ease(span(t, 0.02, 0.42));
      x = lerp(w * 0.06, padIn, k);
      // The drift: only while it is actually on the ice.
      if (x > iceFrom && x < iceTo) y = midY + Math.sin(k * 14) * 9;
    } else if (t < 0.6) {
      // In transit through the pad: a dissolve, then a re-form.
      const k = span(t, 0.42, 0.6);
      x = k < 0.5 ? padIn : padOut;
      ctx.globalAlpha = k < 0.5 ? 1 - k * 2 : (k - 0.5) * 2;
      tone = p.warn;
    } else {
      const k = ease(span(t, 0.6, 0.96));
      x = lerp(padOut, w * 0.95, k);
      tone = p.goal;
    }
    robot(ctx, x, y, p, tone);
    ctx.globalAlpha = 1;
  }

  /* Room 2 — the gantry, a plank giving way, and a successful crossing.
     The loop is the lesson: fall, remember, cross. */
  function room2(ctx, w, h, t, p) {
    const deckY = h * 0.6;
    const from = w * 0.12, to = w * 0.88;
    const plankFrom = w * 0.36, plankTo = w * 0.64;

    // The void under the gantry.
    const shaft = ctx.createLinearGradient(0, deckY, 0, h);
    shaft.addColorStop(0, p.wall);
    shaft.addColorStop(1, p.base);
    ctx.fillStyle = shaft;
    ctx.globalAlpha = 0.55;
    ctx.fillRect(plankFrom - 10, deckY + 6, plankTo - plankFrom + 20, h - deckY);
    ctx.globalAlpha = 1;

    const broken = t >= 0.26 && t < 0.62;

    // The sound sections either end.
    ctx.fillStyle = p.wall;
    ctx.fillRect(from, deckY, plankFrom - from, 7);
    ctx.fillRect(plankTo, deckY, to - plankTo, 7);

    // The four planks between them.
    for (let i = 0; i < 4; i += 1) {
      const pw = (plankTo - plankFrom) / 4;
      const px = plankFrom + i * pw;
      const gone = broken && i === 2;
      if (gone) {
        // Falling, tumbling, with a spark where it tore away.
        const k = span(t, 0.26, 0.44);
        ctx.save();
        ctx.translate(px + pw / 2, deckY + k * (h - deckY) * 1.1);
        ctx.rotate(k * 2.4);
        ctx.globalAlpha = 1 - k;
        ctx.fillStyle = p.wall;
        ctx.fillRect(-pw / 2, -3, pw - 3, 6);
        ctx.restore();
        ctx.globalAlpha = 1;
      } else {
        ctx.fillStyle = p.wall;
        ctx.fillRect(px, deckY, pw - 3, 7);
        // Hazard stripes, so a plank reads as a plank.
        ctx.strokeStyle = p.warn;
        ctx.globalAlpha = 0.5;
        ctx.lineWidth = 1;
        ctx.beginPath();
        for (let s = 0; s < pw - 3; s += 5) {
          ctx.moveTo(px + s, deckY + 7);
          ctx.lineTo(px + s + 3, deckY);
        }
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
    }

    // The crossing.
    let x, y = deckY - 10, tone = null;
    if (t < 0.26) {
      x = lerp(from + 6, plankFrom + (plankTo - plankFrom) * 0.55,
               ease(span(t, 0.0, 0.26)));
    } else if (t < 0.46) {
      // The fall.
      const k = span(t, 0.26, 0.46);
      x = plankFrom + (plankTo - plankFrom) * 0.55;
      y = deckY - 10 + k * k * (h - deckY) * 1.4;
      tone = p.hazard;
      ctx.globalAlpha = 1 - k * 0.9;
    } else if (t < 0.62) {
      // Remembering: back at the start, dimmed.
      x = from + 6;
      tone = p.muted;
      ctx.globalAlpha = 0.35 + 0.5 * span(t, 0.46, 0.62);
    } else {
      const k = ease(span(t, 0.62, 0.97));
      x = lerp(from + 6, to - 6, k);
      tone = k > 0.9 ? p.goal : p.accent;
    }
    robot(ctx, x, y, p, tone);
    ctx.globalAlpha = 1;
  }

  /* Room 3 — three generators brought up in order, then the door. */
  function room3(ctx, w, h, t, p) {
    const y = h * 0.52;
    const xs = [w * 0.2, w * 0.42, w * 0.64];
    const doorX = w * 0.88;
    // Which generators are lit: one, then two, then three.
    const lit = xs.map((unused, i) => t > 0.14 + i * 0.19);
    const open = t > 0.72;

    // The ring the errands run round.
    ctx.strokeStyle = p.hairline;
    ctx.globalAlpha = 0.5;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(w * 0.1, y + 26);
    ctx.lineTo(w * 0.94, y + 26);
    ctx.stroke();
    ctx.globalAlpha = 1;

    xs.forEach((x, i) => {
      const on = lit[i];
      const tone = on ? p.goal : p.muted;
      glow(ctx, tone, on ? 16 : 0, () => {
        ctx.strokeStyle = tone;
        ctx.lineWidth = 2;
        ctx.strokeRect(x - 10, y - 14, 20, 28);
        if (on) {
          ctx.fillStyle = tone;
          ctx.globalAlpha = 0.22;
          ctx.fillRect(x - 10, y - 14, 20, 28);
          ctx.globalAlpha = 1;
        }
      });
      ctx.fillStyle = on ? p.goal : p.hairline;
      ctx.font = '600 9px ' + 'monospace';
      ctx.textAlign = 'center';
      ctx.fillText(String(i + 1), x, y + 25);
    });

    // The blast door: shut and red until all three are up.
    const doorTone = open ? p.goal : p.hazard;
    glow(ctx, doorTone, open ? 20 : 8, () => {
      ctx.strokeStyle = doorTone;
      ctx.lineWidth = 2;
      ctx.strokeRect(doorX - 12, y - 20, 24, 40);
      if (!open) {
        ctx.beginPath();
        ctx.moveTo(doorX - 12, y);
        ctx.lineTo(doorX + 12, y);
        ctx.stroke();
      }
    });

    // R-5 running the lap, pausing at each generator.
    const stops = [w * 0.1].concat(xs).concat([doorX - 22]);
    const legs = stops.length - 1;
    const k = Math.min(0.999, t / 0.94) * legs;
    const leg = Math.floor(k);
    robot(ctx, lerp(stops[leg], stops[leg + 1], ease(k - leg)), y + 26, p,
          open ? p.goal : null);
  }

  /* Room 4 — a drone pushed down by fans, correcting, then landing. */
  function room4(ctx, w, h, t, p) {
    const padX = w * 0.84, padY = h * 0.8;

    // Two banks of fans, and the air they drive down.
    [w * 0.34, w * 0.58].forEach((x, bank) => {
      for (let i = 0; i < 4; i += 1) {
        const drift = ((t * 300 + i * 22 + bank * 11) % 88);
        ctx.strokeStyle = p.accent;
        ctx.globalAlpha = 0.16;
        ctx.lineWidth = 1;
        ctx.beginPath();
        ctx.moveTo(x - 12 + i * 8, h * 0.06 + drift);
        ctx.lineTo(x - 12 + i * 8, h * 0.06 + drift + 12);
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      // The housing.
      ctx.strokeStyle = p.hairline;
      ctx.lineWidth = 1;
      ctx.strokeRect(x - 18, h * 0.04, 36, 9);
    });

    // The landing pad.
    glow(ctx, p.goal, 14, () => {
      ctx.strokeStyle = p.goal;
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(padX - 20, padY);
      ctx.lineTo(padX + 20, padY);
      ctx.stroke();
    });

    // The flight: in, shoved down by the fans, corrected, then settled.
    const k = ease(span(t, 0.02, 0.94));
    const x = lerp(w * 0.08, padX, k);
    // The downdraught bites in the middle of the tunnel and is fought off.
    const push = Math.sin(Math.min(1, span(t, 0.2, 0.72)) * Math.PI) * 26;
    const glide = lerp(h * 0.34, padY - 8, k);
    const y = glide + push;

    // A thin trail, so the correction is visible as a shape rather than a
    // position.
    ctx.strokeStyle = p.accent;
    ctx.globalAlpha = 0.25;
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let s = 0; s <= 40; s += 1) {
      const st = (s / 40) * Math.min(t, 0.94);
      const sk = ease(span(st, 0.02, 0.94));
      const sp = Math.sin(Math.min(1, span(st, 0.2, 0.72)) * Math.PI) * 26;
      const sx = lerp(w * 0.08, padX, sk);
      const sy = lerp(h * 0.34, padY - 8, sk) + sp;
      if (s === 0) ctx.moveTo(sx, sy); else ctx.lineTo(sx, sy);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;

    robot(ctx, x, y, p, t > 0.9 ? p.goal : null);
  }

  /* Room 5 — patrolling drones, diagonal beams, the terminal, the door. */
  function room5(ctx, w, h, t, p) {
    const terminalX = w * 0.5, terminalY = h * 0.22;
    const doorX = w * 0.92;
    const reached = t > 0.7;

    // Shelving, so it reads as a warehouse rather than an empty box.
    ctx.fillStyle = p.wall;
    ctx.globalAlpha = 0.35;
    [[0.2, 0.55], [0.66, 0.6]].forEach(([sx, sy]) => {
      ctx.fillRect(w * sx, h * sy, w * 0.12, 8);
    });
    ctx.globalAlpha = 1;

    // Diagonal beams: lit until the terminal is reached, then dark.
    if (!reached) {
      [[0.3, 0.72], [0.62, 0.34]].forEach(([bx, by], i) => {
        glow(ctx, p.hazard, 10, () => {
          ctx.strokeStyle = p.hazard;
          ctx.lineWidth = 1.6;
          ctx.globalAlpha = 0.5 + 0.4 * Math.abs(Math.sin(t * 10 + i));
          ctx.beginPath();
          ctx.moveTo(w * bx, h * by);
          ctx.lineTo(w * bx + 46, h * by - 34);
          ctx.stroke();
          ctx.globalAlpha = 1;
        });
      });
    }

    // Two drones on closed patrol circuits.
    [[0.26, 0.34, 1], [0.7, 0.78, -1]].forEach(([cx, cy, dir]) => {
      const a = t * Math.PI * 2 * dir;
      const dx = w * cx + Math.cos(a) * 26;
      const dy = h * cy + Math.sin(a) * 14;
      glow(ctx, p.hazard, 10, () => {
        ctx.strokeStyle = p.hazard;
        ctx.lineWidth = 1.4;
        ctx.beginPath();
        ctx.arc(dx, dy, 5, 0, Math.PI * 2);
        ctx.stroke();
      });
    });

    // The terminal, then the blast door.
    glow(ctx, reached ? p.goal : p.warn, 14, () => {
      ctx.strokeStyle = reached ? p.goal : p.warn;
      ctx.lineWidth = 2;
      ctx.strokeRect(terminalX - 9, terminalY - 9, 18, 18);
    });
    glow(ctx, reached ? p.goal : p.hairline, reached ? 18 : 0, () => {
      ctx.strokeStyle = reached ? p.goal : p.hairline;
      ctx.lineWidth = 2;
      ctx.strokeRect(doorX - 8, h * 0.6 - 18, 16, 36);
    });

    // R-5: out to the terminal, then across to the door.
    let x, y;
    if (t < 0.7) {
      const k = ease(span(t, 0.02, 0.7));
      x = lerp(w * 0.08, terminalX, k);
      y = lerp(h * 0.78, terminalY + 20, k);
    } else {
      const k = ease(span(t, 0.7, 0.97));
      x = lerp(terminalX, doorX - 18, k);
      y = lerp(terminalY + 20, h * 0.6, k);
    }
    robot(ctx, x, y, p, reached ? p.goal : null);
  }

  const DRAW = { 1: room1, 2: room2, 3: room3, 4: room4, 5: room5 };

  /**
   * Attach a looping preview to a canvas.
   *
   * Returns `start`/`stop`; nothing runs until `start` is called, and `stop`
   * cancels the frame so a closed panel costs nothing.
   */
  function create(canvas, number, caption) {
    const script = SCRIPTS[number];
    const draw = DRAW[number];
    if (!canvas || !script || !draw) {
      return { start: function () {}, stop: function () {} };
    }

    const ctx = canvas.getContext('2d');
    let frame = null;
    let began = 0;
    let beatShown = -1;

    function size() {
      const ratio = window.devicePixelRatio || 1;
      const box = canvas.getBoundingClientRect();
      const w = Math.max(1, Math.round(box.width));
      const h = Math.max(1, Math.round(box.height));
      if (canvas.width !== w * ratio || canvas.height !== h * ratio) {
        canvas.width = w * ratio;
        canvas.height = h * ratio;
      }
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      return { w: w, h: h };
    }

    function paint(stamp) {
      const box = size();
      const p = palette();
      const elapsed = ((stamp - began) / 1000) % script.seconds;
      const t = elapsed / script.seconds;

      floor(ctx, box.w, box.h, p);
      draw(ctx, box.w, box.h, t, p);

      // The caption follows the loop, so the words and the picture agree.
      if (caption) {
        let index = 0;
        script.beats.forEach((beat, i) => { if (t >= beat.at) index = i; });
        if (index !== beatShown) {
          beatShown = index;
          caption.textContent = script.beats[index].text;
        }
      }
      frame = window.requestAnimationFrame(paint);
    }

    return {
      start: function () {
        if (frame !== null) return;
        began = window.performance ? window.performance.now() : 0;
        beatShown = -1;
        frame = window.requestAnimationFrame(paint);
      },
      stop: function () {
        if (frame === null) return;
        window.cancelAnimationFrame(frame);
        frame = null;
      },
      /** One still frame, for reduced-motion readers. */
      still: function () {
        const box = size();
        const p = palette();
        floor(ctx, box.w, box.h, p);
        draw(ctx, box.w, box.h, 0.95, p);
        if (caption) {
          caption.textContent =
            script.beats[script.beats.length - 1].text;
        }
      },
    };
  }

  return { create: create, SCRIPTS: SCRIPTS };
})();
