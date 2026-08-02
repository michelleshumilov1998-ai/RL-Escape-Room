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

      /* A chamber with a wall around it has to be scaled to fit the *building*
         rather than the floor, or the masonry is cut off at the canvas edge.
         Rooms 1 to 3 need nothing here because their walls are cells of the
         grid and so are already inside `worldSize`; rooms 4 and 5 state a ring
         of wall tiles outside theirs, and declare how wide it is. */
      const margin = room.wallMargin || 0;
      const extent = {
        width: room.worldSize.width + margin * 2,
        height: room.worldSize.height + margin * 2,
      };

      view.scale = Math.min(available.width / extent.width,
                            available.height / extent.height);

      // Centred in what is left after the sidebar, not in the viewport,
      // so opening it slides the world rather than hiding half of it.
      //
      // The *building* is centred, not the floor — so a chamber with a wall
      // round it sits in the middle of the canvas rather than being pushed
      // off-centre by the thickness of its own masonry.
      const drawnWidth = extent.width * view.scale;
      const drawnHeight = extent.height * view.scale;
      view.offsetX = (cssWidth - (inset || 0) - drawnWidth) / 2
                   + margin * view.scale;
      view.offsetY = (cssHeight - drawnHeight) / 2 + margin * view.scale;
    }

    function drawField(room) {
      ctx.fillStyle = palette.field;
      ctx.fillRect(screenX(0), screenY(0),
                   screenLength(room.worldSize.width),
                   screenLength(room.worldSize.height));
    }

    /**
     * The floor, tiled.
     *
     * `cellSize` in a grid room and `tileSize` in a continuous one. They are
     * the same one-metre pitch and the same hairline at the same opacity,
     * because it is the same laboratory floor: what differs between the rooms
     * is whether a tile is also a *state*, and that is nothing to do with how
     * the floor looks. Rooms 4 and 5 drew no tiling at all, which is most of
     * why they read as somewhere else.
     */
    function drawCellBoundaries(room) {
      const tile = room.cellSize || room.tileSize;
      if (!(tile > 0)) return;
      const cellPx = screenLength(tile);
      if (cellPx < C.render.minCellPx) return;

      ctx.strokeStyle = palette.hairlineFaint;
      ctx.lineWidth = C.render.gridLineWidth;
      ctx.globalAlpha = 0.7;
      ctx.beginPath();
      for (let x = 0; x <= room.worldSize.width + 1e-9; x += tile) {
        const at = Math.round(screenX(x)) + 0.5;
        ctx.moveTo(at, screenY(0));
        ctx.lineTo(at, screenY(room.worldSize.height));
      }
      for (let y = 0; y <= room.worldSize.height + 1e-9; y += tile) {
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
    function drawEntity(room, entity, state, moved, phase, mission) {
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

      /* The animation clock, offset by whatever this entity asked for, so a
         bank of fans does not turn as one object. It is a number and nothing
         else: a recipe may use it to choose an angle or an opacity, and there
         is no way for it to reach the simulation. Rooms 1 to 3 declare no
         phase and no recipe of theirs reads one, so they are unaffected. */
      look = Object.assign({}, look,
                           { phase: (phase || 0) + (look.phase || 0) });

      /* How many recorded frames ago the mission stage changed, for the one
         recipe that draws a flourish when it does. Derived from the recording
         — see `Playback.sinceActivation` — and never from a clock here, so a
         terminal cannot pulse on a frame where the environment did not report
         an activation. Undefined in every room that has no mission stage. */
      if (mission && typeof mission.sinceActivation === 'number') {
        look = Object.assign({}, look,
                             { sinceActivation: mission.sinceActivation });
      }

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

    /**
     * The objective marker: a ring and four chevrons on the current target.
     *
     * THE TARGET IS THE ENVIRONMENT'S, NOT A GUESS
     * `detail.target` is the very point `WarehouseWorld.target_of` returns for
     * this state — the same point the progress reward is measured against and
     * the same one the observation's bearing is computed from. So the marker
     * cannot point somewhere the agent is not actually being sent, and it
     * moves from the terminal to the door at exactly the frame the mission
     * stage changes, because that is when `target_of` starts answering
     * differently. Nothing here knows what a terminal or a door is.
     */
    function drawObjective(target, phase, stage) {
      const M = C.mission;
      const at = { x: screenX(target.x), y: screenY(target.y) };
      const radius = screenLength(M.markerRadius);
      // Stage 0 is the terminal, which is a system to be reached; stage 1 is
      // the way out. Cyan then green, matching the thing being pointed at.
      const tone = stage ? palette.goal : palette.accent;
      const breathe = 0.5 + 0.5 * Math.sin(phase * M.markerPulse);

      ctx.save();
      ctx.strokeStyle = tone;
      ctx.globalAlpha = 0.35 + 0.35 * breathe;
      ctx.lineWidth = M.markerWidth;
      ctx.beginPath();
      ctx.arc(at.x, at.y, radius + breathe * M.markerBreath, 0, Math.PI * 2);
      ctx.stroke();

      // Four chevrons pointing inwards at it, which is what makes it read as
      // "go here" rather than as one more circle on the floor.
      ctx.globalAlpha = 0.85;
      ctx.lineWidth = M.markerWidth * 1.2;
      const reach = radius + M.markerBreath * 2.2 + breathe * M.markerBreath;
      const arm = radius * 0.34;
      ctx.beginPath();
      for (let index = 0; index < 4; index += 1) {
        const angle = index * Math.PI / 2 + Math.PI / 4;
        const tip = { x: at.x + Math.cos(angle) * reach,
                      y: at.y + Math.sin(angle) * reach };
        // Two strokes back from the tip, forming an arrowhead aimed inwards.
        [angle + 2.5, angle - 2.5].forEach(back => {
          ctx.moveTo(tip.x, tip.y);
          ctx.lineTo(tip.x + Math.cos(back) * arm,
                     tip.y + Math.sin(back) * arm);
        });
      }
      ctx.stroke();
      ctx.restore();
    }

    /**
     * A pulse running from the terminal to the door, just after activation.
     *
     * Shown only while `sinceActivation` is within the flourish window, and
     * that number is counted off the recorded frames rather than off a clock,
     * so it appears in a replay at exactly the step it appeared live. It is
     * the one piece of drawing that says *why* the door opened: the terminal
     * did it.
     */
    function drawMissionLink(entities, since) {
      const M = C.mission;
      if (!(since >= 0 && since < M.linkFrames)) return;

      const find = id => entities.filter(entity => entity.id === id)[0];
      const terminal = find('terminal');
      const door = find('exit');
      if (!terminal || !door) return;

      const through = since / M.linkFrames;
      const from = { x: screenX(terminal.position.x),
                     y: screenY(terminal.position.y) };
      const to = { x: screenX(door.position.x), y: screenY(door.position.y) };

      ctx.save();
      // The whole run, fading out.
      ctx.globalAlpha = (1 - through) * 0.35;
      ctx.strokeStyle = palette.goal;
      ctx.lineWidth = M.linkWidth;
      ctx.setLineDash([6, 6]);
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
      ctx.setLineDash([]);

      // And a bright head travelling along it, so the direction is legible:
      // the signal goes from the terminal to the door and not the other way.
      const head = { x: from.x + (to.x - from.x) * through,
                     y: from.y + (to.y - from.y) * through };
      ctx.globalAlpha = 1 - through;
      ctx.fillStyle = palette.goal;
      ctx.beginPath();
      ctx.arc(head.x, head.y, M.linkHeadRadius, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    /**
     * The sensor fan: what R-5 can actually see, at this instant.
     *
     * THIS IS THE SENSOR, NOT A PICTURE OF IT
     * Every number here arrives on the frame, computed by the environment in
     * the same call that produced the agent's observation — the heading, the
     * cone half-angle, the range, the three free distances and the list of
     * obstacles that passed the visibility rule. Nothing is recomputed and
     * nothing is guessed, so what is on screen cannot claim a sensor the
     * agent does not have. That mattered particularly for the heading: it is
     * the velocity direction while moving and the bearing to the objective
     * while nearly still, and a drawing that assumed the first would point
     * the cone the wrong way every time the drone slowed down.
     *
     * The cone drawn is exactly the region the visibility rule admits: radius
     * `sensorRange` metres, half-angle `sensorSpread` radians, with the outer
     * two rays lying along its edges. An obstacle outside it is never marked
     * as detected, however near it looks.
     */
    function drawSensors(position, detail, sensorRange, sensorSpread) {
      const S = C.sensors;
      const range = sensorRange;
      const spread = sensorSpread;
      if (!(range > 0)) return;

      const heading = detail.heading || 0;
      const rays = detail.sensors || {};
      const origin = { x: screenX(position.x), y: screenY(position.y) };
      const radius = screenLength(range);

      /* The colour of one ray, by how much room is left along it. Read
         downwards: 1.0 is nothing in range at all. */
      function rayColour(free) {
        if (free <= S.critical) return palette.hazard;
        if (free <= S.near) return palette.warn;
        return palette.accent;
      }

      // The cone: a faint wash over the whole region the rule admits, and a
      // brighter arc on its outer edge so the range itself is readable.
      ctx.fillStyle = palette.accent;
      ctx.globalAlpha = S.coneOpacity;
      ctx.beginPath();
      ctx.moveTo(origin.x, origin.y);
      ctx.arc(origin.x, origin.y, radius, heading - spread, heading + spread);
      ctx.closePath();
      ctx.fill();

      ctx.globalAlpha = S.arcOpacity;
      ctx.strokeStyle = palette.accent;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.arc(origin.x, origin.y, radius, heading - spread, heading + spread);
      ctx.stroke();
      ctx.globalAlpha = 1;

      // The three rays, at the angles the environment used and the lengths it
      // measured. `left` and `right` sit on the cone's two edges.
      const fan = [
        { key: 'left', angle: heading - spread },
        { key: 'centre', angle: heading },
        { key: 'right', angle: heading + spread },
      ];
      fan.forEach(ray => {
        const free = typeof rays[ray.key] === 'number' ? rays[ray.key] : 1;
        const length = screenLength(free * range);
        const tip = {
          x: origin.x + Math.cos(ray.angle) * length,
          y: origin.y + Math.sin(ray.angle) * length,
        };
        ctx.strokeStyle = rayColour(free);
        ctx.lineWidth = S.rayWidth;
        ctx.globalAlpha = 0.85;
        ctx.beginPath();
        ctx.moveTo(origin.x, origin.y);
        ctx.lineTo(tip.x, tip.y);
        ctx.stroke();

        // A dot where the ray stopped, but only when it stopped on
        // something. A ray that ran to its full range met nothing, and
        // marking its end would draw a detection that did not happen.
        if (free < 0.999) {
          ctx.globalAlpha = 1;
          ctx.fillStyle = rayColour(free);
          ctx.beginPath();
          ctx.arc(tip.x, tip.y, S.hitRadius, 0, Math.PI * 2);
          ctx.fill();
        }
      });
      ctx.globalAlpha = 1;
    }

    /**
     * Ring the obstacles the agent can currently see.
     *
     * Drawn from `detail.visible`, which is the environment's own answer to
     * the visibility rule — so an obstacle three metres away and slightly
     * outside the cone gets no ring, which is the honest picture of what the
     * agent was working from when it flew into one.
     */
    function drawDetections(detail, moved) {
      const S = C.sensors;
      (detail.visible || []).forEach(seen => {
        const at = moved ? moved[seen.id] : null;
        if (!at) return;
        ctx.strokeStyle = palette.warn;
        ctx.lineWidth = S.markWidth;
        ctx.globalAlpha = 0.9;
        ctx.beginPath();
        ctx.arc(screenX(at.x), screenY(at.y), screenLength(S.markRadius),
                0, Math.PI * 2);
        ctx.stroke();
        ctx.globalAlpha = 1;
      });
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

      // The decorative clock. Deterministic: it comes in on the snapshot,
      // worked out from the step being drawn, so scrubbing a replay to the
      // same step draws the identical frame and nothing here needs a loop of
      // its own competing with the room's.
      const phase = snapshot.phase || 0;

      ctx.clearRect(0, 0, view.width, view.height);

      /* THE SHAKE
         A bridge section has just given way, so the whole view is knocked
         about for a couple of frames. Applied as a canvas translate around
         everything, so nothing downstream has to know it is happening and no
         world coordinate is touched — the shake is in the camera, not in the
         simulation.

         `collapseAgo` is counted in recorded frames (see `Playback`), so the
         shake occupies the same steps of the episode at every playback speed
         and a replay reproduces it exactly. It decays to nothing across the
         window, and both offsets come from `Math.sin` of the frame rather than
         from a random number, so scrubbing to the same step shakes the same
         way. Reduced motion skips it entirely. */
      const ago = snapshot.collapseAgo;
      let shaken = false;
      if (typeof ago === 'number' && ago >= 0 && ago < C.impact.frames
          && !window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        const decay = 1 - ago / C.impact.frames;
        const swing = C.impact.amplitude * decay * decay;
        const beat = ago * C.impact.frequency;
        ctx.save();
        ctx.translate(Math.sin(beat) * swing,
                      Math.sin(beat * 1.7 + 1.1) * swing * 0.7);
        shaken = true;
      }

      drawField(room);
      // Any room with a tiled floor, which is now every room.
      drawCellBoundaries(room);

      // Under the entities: an overlay is about the room, not on top of it.
      if (room.isGrid && overlays && overlays.heatmap) {
        drawHeatmap(room, overlays.heatmap);
      }

      // Statics first, then the things that end an episode, so a hazard is
      // never buried under something drawn after it.
      const order = { static: 0, goal: 1, hazard: 2 };
      /* The snapshot may replace the room's entity list.
         Most rooms are furnished once and for all, and `room.entities` is
         right for the whole session. Room 5 is a different warehouse every
         episode, so it sends the current one with the moment — and drawing
         the cached list there meant drawing shelves that had moved and
         omitting the ones the drone was about to die on. The types never
         change, only which things exist and where. */
      const entities = (snapshot.entities || room.entities || []).filter(
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
                     moved ? moved[entity.id] : null, phase, snapshot.detail);
        });

      /* The mission, for a room that has one. Both of these read the state the
         environment recorded — the objective marker sits on the very point the
         reward is measured against, and the link only appears on the frames
         where the recording says the stage changed. Drawn over the furniture
         so the current objective is never buried under a shelf, and under the
         drone so R-5 is never buried under the marker. */
      if (snapshot.detail && snapshot.detail.target) {
        drawObjective(snapshot.detail.target, phase, snapshot.detail.stage);
        if (typeof snapshot.detail.sinceActivation === 'number') {
          drawMissionLink(entities, snapshot.detail.sinceActivation);
        }
      }

      if (room.isGrid && overlays && overlays.arrows) {
        drawArrows(room, overlays.arrows);
      }

      if (!snapshot.position) {
        if (shaken) ctx.restore();
        return;
      }
      if (snapshot.observation) {
        drawObservation(snapshot.position, snapshot.observation);
      }

      /* Room 5's sensors. Under the trail and the drone, over the warehouse,
         because it is a thing the drone is emitting rather than a thing lying
         on the floor. `detail` is absent in every other room, which is what
         keeps this branch from touching them. */
      if (snapshot.detail && snapshot.detail.sensors) {
        drawSensors(snapshot.position, snapshot.detail,
                    room.sensorRange, room.sensorSpread);
        drawDetections(snapshot.detail, moved);
      }

      if (snapshot.trail) drawTrail(snapshot.trail);

      const unit = room.isGrid ? room.cellSize : 1;

      /* A room whose state carries a velocity gets the drone form: the same
         R-5, in a frame, tilted the way it is travelling. Everything below is
         read off the velocity the *episode recorded* — nothing is integrated
         here and no state is invented, which is what keeps a replay a replay.
         Rooms 1 to 3 report no velocity and take the branch they always did. */
      if (snapshot.velocity) {
        window.Shapes.drone(
          ctx,
          { x: screenX(snapshot.position.x), y: screenY(snapshot.position.y) },
          screenLength(unit * C.render.agentSize),
          snapshot.velocity,
          {
            phase: phase,
            // What counts as fast here, so the drawing can show braking
            // without knowing the room's rules. Undefined elsewhere.
            landingSpeed: snapshot.landingSpeed,
            speedLimit: snapshot.speedLimit,
            facing: snapshot.facing,
          },
          palette);
        if (shaken) ctx.restore();
        return;
      }

      window.Shapes.agent(
        ctx,
        { x: screenX(snapshot.position.x), y: screenY(snapshot.position.y) },
        screenLength(unit * C.render.agentSize),
        snapshot.facing === undefined ? null : snapshot.facing,
        palette);

      if (shaken) ctx.restore();
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
