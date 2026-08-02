'use strict';

/* =====================================================================
   The renderer. Stateless: it is handed a snapshot and it draws it.

   It never mutates anything it is given, never advances the simulation and
   never decides anything about the run.  The only state it keeps is the
   canvas, the device pixel ratio and the fitted cell size — all of which
   describe the screen, not the game.

   For a planner there is nothing walking about the grid while it trains,
   so what there is to watch is the value table: `values` tints each cell
   and `policy` puts an arrow in it.  Value spreading outwards from the
   goal, one sweep at a time, is Dynamic Programming made visible.

   What a wall or a laser actually looks like is not here: that is
   `shapes.js`, which the room screen draws from as well, so a wall is the
   same wall on both screens.  This file decides only where each shape
   goes, and lays the value wash down underneath it.
   ===================================================================== */

window.Renderer = (function () {

  let canvas = null;
  let ctx = null;
  let entities = {};
  let colours = {};

  // The fitted geometry, recomputed on resize and when the sidebar moves.
  const view = { width: 0, height: 0, cell: 0, originX: 0, originY: 0,
                 rows: 0, cols: 0, inset: 0 };

  let settings = { cellSize: 44, cellMin: 14, padding: 28 };

  function resolveColours() {
    const style = getComputedStyle(document.documentElement);
    colours = {};
    Object.keys(entities).forEach(kind => {
      const property = entities[kind].colour;
      colours[kind] = style.getPropertyValue(property).trim() || '#333';
    });
    colours.base = style.getPropertyValue('--base').trim();
    colours.accent = style.getPropertyValue('--accent').trim();
    colours.muted = style.getPropertyValue('--muted').trim();
    colours.goal = style.getPropertyValue('--goal').trim();
    colours.hazard = style.getPropertyValue('--hazard').trim();
    colours.hairline = style.getPropertyValue('--hairline').trim();
    colours.hairlineFaint = style.getPropertyValue('--hairline-faint').trim();
  }

  function fit(rows, cols) {
    view.rows = rows;
    view.cols = cols;
    // The sidebar overlays the canvas, so the grid is fitted into what is
    // left of it rather than into the whole viewport. Nothing important
    // ends up underneath.
    const usableWidth = view.width - view.inset - settings.padding * 2;
    const usableHeight = view.height - settings.padding * 2;
    const cell = Math.min(usableWidth / cols, usableHeight / rows,
                          settings.cellSize * 1.8);
    view.cell = Math.max(settings.cellMin, cell);
    view.originX = (view.width - view.inset - view.cols * view.cell) / 2;
    view.originY = (view.height - view.rows * view.cell) / 2;
  }

  function cellRect(row, col) {
    return {
      x: view.originX + col * view.cell,
      y: view.originY + row * view.cell,
      size: view.cell,
    };
  }

  /* Said once per load, not once per cell per frame. */
  let warnedAboutShapes = false;

  /**
   * One cell's furniture, drawn by the shared recipe for its kind.
   *
   * The box is a pixel short on each side, which is what leaves the thin
   * dark gridlines between cells that this screen has always had.
   */
  function drawShape(kind, row, col, layout) {
    const definition = entities[kind];
    if (!definition) return;
    const box = cellRect(row, col);

    /* An entity table from before shapes existed — in practice, a server
       that has not been restarted since `game/config.py` changed. Drawing
       a plain fill is the right thing to fall back to, but doing it
       silently is not: the agent is drawn by a direct call and so changes
       anyway, and "everything except the robot looks the same" is a very
       confusing thing to be left staring at. */
    if (!definition.shape) {
      if (!warnedAboutShapes) {
        warnedAboutShapes = true;
        // eslint-disable-next-line no-console
        console.warn('Renderer: /api/config sent entity definitions with no '
                     + '"shape", so the chamber is being drawn flat. Restart '
                     + 'serve.py to pick up game/config.py.');
      }
      // Floor is left alone: it has already been laid down, and painting
      // it again would rub out the value wash on top of it.
      if (kind === 'floor') return;
      ctx.fillStyle = colours[kind];
      ctx.fillRect(box.x, box.y, box.size - 1, box.size - 1);
      return;
    }

    // The whole definition, plus the two things only this file knows: the
    // resolved colour and which way a run of these lies. Passing a fresh
    // object of just those would throw away anything else the definition
    // carries, which is a mistake worth not making twice.
    window.Shapes.draw(ctx, definition.shape, {
      left: box.x, top: box.y, width: box.size - 1, height: box.size - 1,
    }, Object.assign({}, definition, {
      color: colours[kind],
      orientation: runDirection(kind, row, col, layout),
    }), box.size, colours);
  }

  /**
   * Which way a run of identical cells lies, or null if it is alone.
   *
   * A laser is the reason this exists: a row of beam cells should read as
   * one beam across the chamber rather than as a row of separate strokes,
   * and only the layout knows which way the run goes. Horizontal wins a
   * tie, because a barrier that spans the width is the more common of the
   * two and the one worth reading at a glance.
   */
  function runDirection(kind, row, col, layout) {
    if (!layout) return null;
    const same = (r, c) => (
      r >= 0 && c >= 0 && r < layout.rows && c < layout.cols
      && layout.tiles[r][c] === kind);

    if (same(row, col - 1) || same(row, col + 1)) return 'horizontal';
    if (same(row - 1, col) || same(row + 1, col)) return 'vertical';
    return null;
  }

  /** A value in 0..1 as a wash of the accent over the cell. */
  function valueTint(share) {
    const alpha = 0.06 + 0.34 * Math.max(0, Math.min(1, share));
    return 'rgba(56, 217, 255, ' + alpha.toFixed(3) + ')';
  }

  function drawArrow(row, col, action, alpha) {
    const box = cellRect(row, col);
    const centreX = box.x + box.size / 2;
    const centreY = box.y + box.size / 2;
    const reach = box.size * 0.26;
    const deltas = [[0, -1], [0, 1], [-1, 0], [1, 0]];
    const [dx, dy] = deltas[action];

    ctx.save();
    ctx.strokeStyle = colours.muted;
    ctx.globalAlpha = alpha;
    ctx.lineWidth = Math.max(1, box.size * 0.045);
    ctx.lineCap = 'round';
    ctx.beginPath();
    ctx.moveTo(centreX - dx * reach, centreY - dy * reach);
    ctx.lineTo(centreX + dx * reach, centreY + dy * reach);
    ctx.stroke();
    // The head, so direction is readable and not just orientation.
    const headX = centreX + dx * reach;
    const headY = centreY + dy * reach;
    const wing = box.size * 0.11;
    ctx.beginPath();
    ctx.moveTo(headX, headY);
    ctx.lineTo(headX - dx * wing + dy * wing, headY - dy * wing + dx * wing);
    ctx.moveTo(headX, headY);
    ctx.lineTo(headX - dx * wing - dy * wing, headY - dy * wing - dx * wing);
    ctx.stroke();
    ctx.restore();
  }

  return {
    attach(element, config) {
      canvas = element;
      ctx = canvas.getContext('2d');
      entities = config.entities;
      settings = {
        cellSize: config.render.cell_size,
        cellMin: config.render.cell_size_min,
        padding: config.render.grid_padding,
      };
      resolveColours();
    },

    /** Called on resize, on sidebar toggle, and when the palette changes. */
    resize(layout, inset) {
      const ratio = window.devicePixelRatio || 1;
      view.width = canvas.clientWidth;
      view.height = canvas.clientHeight;
      view.inset = inset || 0;
      canvas.width = Math.round(view.width * ratio);
      canvas.height = Math.round(view.height * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
      if (layout) fit(layout.rows, layout.cols);
    },

    /** Which entities belong in the legend, in definition order. */
    /**
     * The key — for the chamber on screen, not for the whole game.
     *
     * It used to list every entity in `config.ENTITIES` that asks to be in a
     * legend, which is all thirty of them: room 1's key named the storage
     * shelves, the security drones, the blast door, the turbine housings and
     * the ventilation fans, none of which are anywhere in room 1. Thirty rows
     * down the side of a ten-cell chamber, most of them describing other
     * rooms.
     *
     * `layout` is the grid this screen is drawing, so the kinds actually
     * present can be counted off it. The agent is added because it is never a
     * tile and is always on screen. Given no layout it falls back to the old
     * behaviour rather than showing an empty key.
     */
    legend(layout) {
      let wanted = null;
      if (layout && layout.tiles) {
        wanted = new Set();
        layout.tiles.forEach(row => row.forEach(kind => wanted.add(kind)));
        wanted.add('agent');
      }
      return Object.keys(entities)
        .filter(kind => entities[kind].in_legend)
        .filter(kind => wanted === null || wanted.has(kind))
        .map(kind => ({
          kind: kind,
          label: entities[kind].label,
        }));
    },

    /**
     * Draw one kind into a small canvas, for the legend.
     *
     * The same recipes as the grid, so the key cannot show something the
     * chamber does not.
     */
    swatch(element, kind) {
      const definition = entities[kind];
      if (!definition) return;

      const ratio = window.devicePixelRatio || 1;
      const size = element.clientWidth || 16;
      element.width = Math.round(size * ratio);
      element.height = Math.round(size * ratio);

      const surface = element.getContext('2d');
      surface.setTransform(ratio, 0, 0, ratio, 0, 0);
      surface.clearRect(0, 0, size, size);

      // Floor draws nothing on its own, so the swatch shows the surface
      // it actually is.
      if (definition.shape === 'floor') {
        surface.fillStyle = colours[kind];
        surface.fillRect(0, 0, size, size);
        return;
      }
      window.Shapes.draw(surface, definition.shape,
                         { left: 0, top: 0, width: size, height: size },
                         { color: colours[kind] }, size, colours);
    },

    /**
     * Draw one frame.
     *
     *   layout    the static grid
     *   learned   values and policy, or null
     *   agent     {row, col} in fractional cells, already interpolated
     *   trail     cells the current replay has walked, or null
     */
    draw(scene) {
      if (!ctx || !scene.layout) return;
      const layout = scene.layout;
      const learned = scene.learned || {};
      const values = learned.values || null;
      const policy = scene.showPolicy ? (learned.policy || null) : null;

      ctx.fillStyle = colours.base;
      ctx.fillRect(0, 0, view.width, view.height);

      // The range the value tint is scaled against, recomputed per frame so
      // the picture is readable from the first sweep rather than only once
      // the numbers have grown.
      let lowest = 0;
      let highest = 1;
      if (values) {
        const numbers = Object.keys(values).map(key => values[key]);
        lowest = Math.min.apply(null, numbers);
        highest = Math.max.apply(null, numbers);
        if (highest - lowest < 1e-9) highest = lowest + 1;
      }

      for (let row = 0; row < layout.rows; row += 1) {
        for (let col = 0; col < layout.cols; col += 1) {
          const kind = layout.tiles[row][col];
          const box = cellRect(row, col);

          // The floor goes down first for every cell, so the value wash
          // has something to sit on and the furniture has something to
          // sit on top of. A wall or a laser is painted over it whole.
          ctx.fillStyle = colours.floor || colours[kind];
          ctx.fillRect(box.x, box.y, box.size - 1, box.size - 1);

          if (values && kind !== 'wall') {
            const value = values[row + ',' + col];
            if (value !== undefined) {
              const share = (value - lowest) / (highest - lowest);
              ctx.fillStyle = valueTint(share);
              ctx.fillRect(box.x, box.y, box.size - 1, box.size - 1);
            }
          }

          // Drawn after the wash rather than under it, so a beam stays a
          // beam however high the value of the cell it crosses.
          drawShape(kind, row, col, layout);

          if (policy && kind !== 'wall' && kind !== 'hazard' && kind !== 'goal') {
            const action = policy[row + ',' + col];
            if (action !== undefined) drawArrow(row, col, action, 0.55);
          }
        }
      }

      /* Where a replay has already been.

         BROKEN AT EVERY JUMP, NOT DRAWN AS ONE POLYLINE
         Two things in this chamber move the agent somewhere it did not walk:
         a teleport pad, which sends it to the other pad, and a beam, which
         throws it back to the door it came in by. Joining those two cells with
         a straight line drew a long diagonal streak across a dozen cells the
         agent never entered — which is the trajectory "spilling into
         neighbouring cells" that was reported. It was not a rounding problem;
         the line was describing a walk that never happened.

         So a segment is only drawn between cells that are orthogonally
         adjacent — which is the only kind of move a step can make. A jump
         leaves a gap in the trail, which is the truth about it. */
      if (scene.trail && scene.trail.length > 1) {
        ctx.save();
        ctx.strokeStyle = colours.accent;
        ctx.globalAlpha = 0.35;
        ctx.lineWidth = Math.max(1.5, view.cell * 0.08);
        ctx.lineJoin = 'round';
        ctx.lineCap = 'round';

        const centre = cell => {
          const box = cellRect(cell[0], cell[1]);
          return { x: box.x + box.size / 2, y: box.y + box.size / 2 };
        };
        const walked = (one, two) => {
          const rows = Math.abs(one[0] - two[0]);
          const cols = Math.abs(one[1] - two[1]);
          // One cell, orthogonally — or standing still, which a wall bump is.
          return rows + cols <= 1;
        };

        ctx.beginPath();
        for (let index = 1; index < scene.trail.length; index += 1) {
          const from = scene.trail[index - 1];
          const to = scene.trail[index];
          if (!walked(from, to)) continue;      // a jump: no line for it
          const a = centre(from);
          const b = centre(to);
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
        }
        ctx.stroke();
        ctx.restore();
      }

      // The agent, at whatever fractional position it was handed.
      if (scene.agent) {
        const box = cellRect(scene.agent.row, scene.agent.col);
        const centre = { x: box.x + box.size / 2, y: box.y + box.size / 2 };

        /* A skid, on the one step where the floor overruled the plan.
           The policy arrow in this cell points where the action was aimed and
           R-5 has gone somewhere else, and without saying so that reads as a
           drawing fault rather than as ice. Drawn under the agent, in the
           hazard tone, as two short arcs — the mark a slide leaves. */
        if (scene.slipped) {
          ctx.save();
          ctx.strokeStyle = colours.hazard;
          ctx.globalAlpha = 0.85;
          ctx.lineWidth = Math.max(1.4, box.size * 0.055);
          ctx.lineCap = 'round';
          const reach = box.size * 0.34;
          [-1, 1].forEach(side => {
            ctx.beginPath();
            ctx.arc(centre.x, centre.y, reach,
                    side > 0 ? -0.55 : Math.PI - 0.55,
                    side > 0 ? 0.55 : Math.PI + 0.55);
            ctx.stroke();
          });
          ctx.restore();
        }

        window.Shapes.agent(ctx, centre, box.size * 0.62, null, colours);
      }
    },

  };
})();
