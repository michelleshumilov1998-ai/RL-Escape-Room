'use strict';

/* =====================================================================
   The room screen's renderer.

   `draw` is a pure function of one view snapshot: it reads the snapshot,
   puts pixels on a canvas, and mutates nothing it is given.  Everything
   about time has already been decided by the playback controller and
   arrives as a position that may sit between two steps.

   `Renderer.create(canvas)` makes an independent view.  There are two on
   this screen — the world, and the small still in the step inspector —
   and they share no state beyond the palette.

   What a wall or a laser actually looks like is not here: that is
   `shapes.js`, which room 1's renderer draws from as well, so the two
   screens cannot disagree about what a thing looks like.  This file
   decides only *where* each shape goes.

   One transform converts world units to pixels, which is what lets a
   10x10 grid and a 10x10-metre wind tunnel share the whole drawing path.
   Nothing here indexes a cell.
   ===================================================================== */

window.Renderer = (function () {

  const C = window.ROOM_CONFIG;

  /* Read once, from the stylesheet, so no colour is written down twice. */
  let palette = {};

  function readPalette() {
    const computed = window.getComputedStyle(document.documentElement);
    palette = {};
    Object.keys(C.colors).forEach(name => {
      palette[name] = computed.getPropertyValue(C.colors[name]).trim();
    });
  }

  /** Resolve a colour that may be a custom-property name or a literal. */
  function colour(value) {
    if (!value) return palette.muted;
    if (value.charAt(0) !== '-') return value;
    const computed = window.getComputedStyle(document.documentElement);
    return computed.getPropertyValue(value).trim() || palette.muted;
  }

  /* ---------------------------------------------------------------------
     A view. One canvas, one transform.
     ------------------------------------------------------------------- */

  function create(canvas) {
    const ctx = canvas.getContext('2d');

    /* Pixels per world unit, and where the world's top-left lands. In CSS
       pixels: the device pixel ratio is applied to the context in `fit`. */
    const view = { scale: 1, offsetX: 0, offsetY: 0, width: 0, height: 0 };

    function screenX(worldX) { return view.offsetX + worldX * view.scale; }
    function screenY(worldY) { return view.offsetY + worldY * view.scale; }
    function screenLength(length) { return length * view.scale; }

    /**
     * Fit the world into what is left, and size the backing store.
     *
     * `inset` is how many pixels are covered on the right by the sidebar.
     * The aspect ratio is never touched: whichever axis runs out first
     * sets the scale for both.
     */
    function fit(room, inset, padding) {
      const ratio = window.devicePixelRatio || 1;
      const cssWidth = canvas.clientWidth;
      const cssHeight = canvas.clientHeight;

      canvas.width = Math.round(cssWidth * ratio);
      canvas.height = Math.round(cssHeight * ratio);
      ctx.setTransform(ratio, 0, 0, ratio, 0, 0);

      view.width = cssWidth;
      view.height = cssHeight;
      if (!room) return;

      const pad = padding === undefined ? C.render.padding : padding;
      const available = {
        width: Math.max(1, cssWidth - (inset || 0) - pad * 2),
        height: Math.max(1, cssHeight - pad * 2),
      };

      view.scale = Math.min(available.width / room.worldSize.width,
                            available.height / room.worldSize.height);

      // Centred in what is left after the sidebar, not in the viewport,
      // so opening it slides the world rather than hiding half of it.
      const drawnWidth = room.worldSize.width * view.scale;
      const drawnHeight = room.worldSize.height * view.scale;
      view.offsetX = (cssWidth - (inset || 0) - drawnWidth) / 2;
      view.offsetY = (cssHeight - drawnHeight) / 2;
    }

    function drawField(room) {
      ctx.fillStyle = palette.field;
      ctx.fillRect(screenX(0), screenY(0),
                   screenLength(room.worldSize.width),
                   screenLength(room.worldSize.height));
    }

    function drawCellBoundaries(room) {
      const cellPx = screenLength(room.cellSize);
      if (cellPx < C.render.minCellPx) return;

      ctx.strokeStyle = palette.hairlineFaint;
      ctx.lineWidth = C.render.gridLineWidth;
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      for (let x = 0; x <= room.worldSize.width + 1e-9; x += room.cellSize) {
        const at = Math.round(screenX(x)) + 0.5;
        ctx.moveTo(at, screenY(0));
        ctx.lineTo(at, screenY(room.worldSize.height));
      }
      for (let y = 0; y <= room.worldSize.height + 1e-9; y += room.cellSize) {
        const at = Math.round(screenY(y)) + 0.5;
        ctx.moveTo(screenX(0), at);
        ctx.lineTo(screenX(room.worldSize.width), at);
      }
      ctx.stroke();
      ctx.globalAlpha = 1;
    }

    /**
     * One entity, through its type's recipe.
     *
     * `state` is whatever this step reported for it. The string means
     * nothing here — it is a key into the type's own `states` table, so
     * the room decides what a collapsed bridge looks like.
     */
    function drawEntity(room, entity, state, moved) {
      const type = room.entityTypes[entity.type];
      // The type, then anything this entity says about itself, then
      // whatever state it is in — each overriding the last.
      let look = entity.appearance
        ? Object.assign({}, type, entity.appearance)
        : type;
      if (state && type.states && type.states[state]) {
        look = Object.assign({}, look, type.states[state]);
      }
      if (look.hidden) return;

      // Where it is now, which is where the episode last put it if it is
      // something that moves, and otherwise where the room declared it.
      const where = moved || entity.position;

      /* The whole of `look`, with only the colour swapped for its resolved
         value. Building a fresh object here instead — which is what this
         used to do — quietly threw away every other field a type, an
         entity or a state had set: which way a vent blows, whether a
         generator is lit, whether a door is open. The recipes went on
         reading them and getting undefined. */
      window.Shapes.draw(ctx, look.shape, {
        left: screenX(where.x - entity.size.width / 2),
        top: screenY(where.y - entity.size.height / 2),
        width: screenLength(entity.size.width),
        height: screenLength(entity.size.height),
      }, Object.assign({}, look, { color: colour(look.color) }),
         view.scale, palette);
    }

    function drawHeatmap(room, cells) {
      let largest = 0;
      cells.forEach(entry => {
        const size = Math.abs(entry.value);
        if (size > largest) largest = size;
      });
      if (largest === 0) return;

      cells.forEach(entry => {
        const share = Math.abs(entry.value) / largest;
        ctx.globalAlpha = share * C.render.heatmapMaxOpacity;
        ctx.fillStyle = entry.value < 0 ? palette.hazard : palette.accent;
        ctx.fillRect(screenX(entry.cell[1] * room.cellSize),
                     screenY(entry.cell[0] * room.cellSize),
                     screenLength(room.cellSize),
                     screenLength(room.cellSize));
      });
      ctx.globalAlpha = 1;
    }

    const HEADINGS = { up: -Math.PI / 2, down: Math.PI / 2,
                       left: Math.PI, right: 0 };

    function drawArrows(room, cells) {
      const cellPx = screenLength(room.cellSize);
      if (cellPx < C.render.minCellPx * 2) return;

      const length = cellPx * 0.3;
      ctx.strokeStyle = palette.accent;
      ctx.globalAlpha = C.render.arrowOpacity;
      ctx.lineWidth = Math.max(1, cellPx * 0.05);
      ctx.lineCap = 'round';

      cells.forEach(entry => {
        const heading = HEADINGS[entry.direction];
        if (heading === undefined) return;
        const centreX = screenX((entry.cell[1] + 0.5) * room.cellSize);
        const centreY = screenY((entry.cell[0] + 0.5) * room.cellSize);
        const dx = Math.cos(heading) * length;
        const dy = Math.sin(heading) * length;

        ctx.beginPath();
        ctx.moveTo(centreX - dx, centreY - dy);
        ctx.lineTo(centreX + dx, centreY + dy);
        ctx.moveTo(centreX + dx, centreY + dy);
        ctx.lineTo(centreX + dx - Math.cos(heading - 0.6) * length * 0.5,
                   centreY + dy - Math.sin(heading - 0.6) * length * 0.5);
        ctx.moveTo(centreX + dx, centreY + dy);
        ctx.lineTo(centreX + dx - Math.cos(heading + 0.6) * length * 0.5,
                   centreY + dy - Math.sin(heading + 0.6) * length * 0.5);
        ctx.stroke();
      });
      ctx.globalAlpha = 1;
    }

    function drawObservation(position, observation) {
      const radius = screenLength(observation.radius);
      const heading = observation.heading || 0;
      const spread = observation.spread === undefined ? Math.PI / 3
                                                     : observation.spread;
      ctx.fillStyle = palette.accent;
      ctx.globalAlpha = C.render.observationOpacity;
      ctx.beginPath();
      ctx.moveTo(screenX(position.x), screenY(position.y));
      ctx.arc(screenX(position.x), screenY(position.y), radius,
              heading - spread, heading + spread);
      ctx.closePath();
      ctx.fill();
      ctx.globalAlpha = 1;
    }

    function drawTrail(points) {
      if (points.length < 2) return;
      ctx.lineWidth = C.trail.width;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      // Segment by segment, because each fades by a different amount and
      // one stroked path can only have one alpha.
      for (let index = 1; index < points.length; index += 1) {
        const share = index / points.length;
        ctx.globalAlpha = C.trail.tailOpacity
          + (C.trail.headOpacity - C.trail.tailOpacity) * share;
        ctx.strokeStyle = palette.accent;
        ctx.beginPath();
        ctx.moveTo(screenX(points[index - 1].x), screenY(points[index - 1].y));
        ctx.lineTo(screenX(points[index].x), screenY(points[index].y));
        ctx.stroke();
      }
      ctx.globalAlpha = 1;
    }

    /**
     * Draw one snapshot. Nothing in it is modified.
     *
     *   room, position, trail, entityStates, overlays, observation, facing
     */
    function draw(snapshot) {
      if (!snapshot || !snapshot.room) return;
      const room = snapshot.room;
      const states = snapshot.entityStates || null;
      const moved = snapshot.entityPositions || null;
      const overlays = snapshot.overlays || null;

      ctx.clearRect(0, 0, view.width, view.height);
      drawField(room);
      if (room.isGrid) drawCellBoundaries(room);

      // Under the entities: an overlay is about the room, not on top of it.
      if (room.isGrid && overlays && overlays.heatmap) {
        drawHeatmap(room, overlays.heatmap);
      }

      // Statics first, then the things that end an episode, so a hazard is
      // never buried under something drawn after it.
      const order = { static: 0, goal: 1, hazard: 2 };
      const entities = (room.entities || []).filter(
        entity => room.entityTypes[entity.type].role !== 'agent');
      // A copy: sorting in place would reorder the caller's array.
      entities.slice()
        .sort((one, two) => {
          const a = order[room.entityTypes[one.type].role] || 0;
          const b = order[room.entityTypes[two.type].role] || 0;
          return a - b;
        })
        .forEach(entity => {
          drawEntity(room, entity, states ? states[entity.id] : null,
                     moved ? moved[entity.id] : null);
        });

      if (room.isGrid && overlays && overlays.arrows) {
        drawArrows(room, overlays.arrows);
      }

      if (!snapshot.position) return;
      if (snapshot.observation) {
        drawObservation(snapshot.position, snapshot.observation);
      }
      if (snapshot.trail) drawTrail(snapshot.trail);

      const unit = room.isGrid ? room.cellSize : 1;
      window.Shapes.agent(
        ctx,
        { x: screenX(snapshot.position.x), y: screenY(snapshot.position.y) },
        screenLength(unit * C.render.agentSize),
        snapshot.facing === undefined ? null : snapshot.facing,
        palette);
    }

    return {
      fit: fit,
      draw: draw,
      toScreen: function (x, y) { return { x: screenX(x), y: screenY(y) }; },
    };
  }

  /* ---------------------------------------------------------------------
     The legend, generated from the room's entity types
     ------------------------------------------------------------------- */

  function legend(room) {
    return Object.keys(room.entityTypes)
      .filter(type => room.entityTypes[type].inLegend)
      .map(type => ({ type: type, label: room.entityTypes[type].label }));
  }

  /**
   * Draw one entity type into a small canvas, for the legend.
   *
   * The same recipes as the world, so the key cannot show something the
   * grid does not.
   */
  function swatch(canvas, room, type) {
    const look = room.entityTypes[type];
    if (!look) return;

    const ratio = window.devicePixelRatio || 1;
    const size = canvas.clientWidth || 16;
    canvas.width = Math.round(size * ratio);
    canvas.height = Math.round(size * ratio);

    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, size, size);

    window.Shapes.draw(ctx, look.shape,
                       { left: 0, top: 0, width: size, height: size },
                       Object.assign({}, look, { color: colour(look.color) }),
                       size, palette);
  }

  return {
    create: create,
    legend: legend,
    swatch: swatch,
    readPalette: readPalette,
    // Exposed so a screen can check a recipe name it is about to use.
    shapes: window.Shapes ? window.Shapes.names : [],
  };
})();
