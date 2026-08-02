'use strict';

/* =====================================================================
   The shape recipes.

   A thing on a grid is not a coloured square.  A wall is masonry, ice is
   a puddle with a shine on it, an abyss is a hole with a lip, a bridge is
   planks with gaps you can see the dark through, a laser is a beam with
   emitters at each end.  This file knows how to draw each of those, and
   nothing else.

   It is shared by both screens deliberately.  Room 1 is a planner drawing
   a value table and room 2 is a learner drawing a trajectory, and their
   renderers have almost nothing else in common — but a wall has to look
   like the same wall in both, and the legend on each screen is drawn with
   these same recipes, so a key can never show something its grid does
   not.

   Deliberately ignorant of everything around it:

     * no DOM.  Colours arrive already resolved to whatever the stylesheet
       says they are; nothing here calls getComputedStyle.
     * no entity types.  A recipe is named by the room — `shape: 'abyss'`
       — and which things in a room are abysses is never this file's
       business.
     * no geometry beyond the box it is handed.  Where that box came from,
       and whether the room is a grid at all, is the renderer's problem.

   `box`     { left, top, width, height } in pixels
   `look`    { color }  already resolved, e.g. '#3E4650'
   `unit`    pixels per world unit, so line weights scale with the room
             rather than with the screen
   `palette` { base, muted, accent, goal, hazard, hairlineFaint } resolved
   ===================================================================== */

window.Shapes = (function () {

  /** Glow strengths, if the page carries a config. Modest defaults if not. */
  function glowFor(name) {
    const config = window.ROOM_CONFIG;
    const glow = config && config.render && config.render.glow;
    if (!glow) return { laser: 14, exit: 10, agent: 9 }[name] || 0;
    return glow[name] || 0;
  }

  function withGlow(ctx, colour, strength, draw) {
    if (!strength) { draw(); return; }
    ctx.save();
    ctx.shadowColor = colour;
    ctx.shadowBlur = strength;
    draw();
    ctx.restore();
  }

  /**
   * The deck of a steel bridge: girders, plating, rivets.
   *
   * One function for the sound sections and the failing ones, because they are
   * the same structure and differ in what is painted on it. `weak` adds the
   * hazard chevrons and the stress crack.
   */
  function drawDeck(ctx, box, look, unit, palette, weak) {
    const girder = Math.max(1.8, unit * 0.13);
    const wallTone = palette.cellWall || palette.muted;
    ctx.save();

    // 1. Industrial supports underneath, showing against the shaft below. Drawn
    //    first so the deck sits on top of them.
    if (unit >= 14) {
      ctx.globalAlpha = 0.5;
      ctx.fillStyle = wallTone;
      const strut = Math.max(1, unit * 0.045);
      [0.22, 0.5, 0.78].forEach(at => {
        ctx.fillRect(box.left + box.width * at - strut / 2,
                     box.top + box.height * 0.72, strut, box.height * 0.3);
      });
      // A tie bar across the struts.
      ctx.fillRect(box.left, box.top + box.height * 0.9, box.width,
                   Math.max(0.8, unit * 0.025));
    }

    // 2. The deck plates, bolted, with the joints showing dark between them.
    const plates = 3;
    const pitch = (box.height - girder * 2) / plates;
    const joint = Math.max(0.7, unit * 0.04);
    ctx.globalAlpha = 1;
    ctx.fillStyle = look.color;
    for (let index = 0; index < plates; index += 1) {
      ctx.fillRect(box.left, box.top + girder + index * pitch + joint / 2,
                   box.width, Math.max(0.5, pitch - joint));
    }
    // A brushed-metal sheen across the plating.
    const sheen = ctx.createLinearGradient(box.left, box.top, box.left,
                                           box.top + box.height);
    sheen.addColorStop(0, '#FFFFFF');
    sheen.addColorStop(0.5, 'transparent');
    sheen.addColorStop(1, palette.base);
    ctx.globalAlpha = 0.12;
    ctx.fillStyle = sheen;
    ctx.fillRect(box.left, box.top + girder, box.width,
                 box.height - girder * 2);

    // 3. The two edge girders — the structure — and safety rails above them.
    ctx.globalAlpha = 1;
    ctx.fillStyle = wallTone;
    ctx.fillRect(box.left, box.top, box.width, girder);
    ctx.fillRect(box.left, box.top + box.height - girder, box.width, girder);

    if (unit >= 16) {
      // Side safety rails: a top rail on stanchions, along both edges.
      ctx.globalAlpha = 0.8;
      ctx.strokeStyle = palette.muted;
      ctx.lineWidth = Math.max(0.7, unit * 0.022);
      ctx.beginPath();
      [box.top - girder * 0.5, box.top + box.height + girder * 0.5]
        .forEach(y => {
          ctx.moveTo(box.left, y);
          ctx.lineTo(box.left + box.width, y);
          for (let index = 0; index < 3; index += 1) {
            const x = box.left + box.width * (0.18 + index * 0.32);
            ctx.moveTo(x, y);
            ctx.lineTo(x, y > box.top + box.height / 2
                            ? y - girder * 0.9 : y + girder * 0.9);
          }
        });
      ctx.stroke();
    }

    // 4. Bolts along both girders.
    if (unit >= 14) {
      ctx.fillStyle = palette.base;
      ctx.globalAlpha = 0.6;
      for (let index = 0; index < 4; index += 1) {
        const x = box.left + box.width * (0.14 + index * 0.24);
        const r = Math.max(0.7, unit * 0.024);
        ctx.beginPath();
        ctx.arc(x, box.top + girder / 2, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.beginPath();
        ctx.arc(x, box.top + box.height - girder / 2, r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.globalAlpha = 1;
    }

    if (weak) {
      // 5. Hazard stripes on a section that may give way. Diagonal bands in
      //    the hazard tone across the deck — the standard industrial marking,
      //    and unmistakable next to a plain plated section.
      ctx.save();
      ctx.beginPath();
      ctx.rect(box.left, box.top + girder, box.width,
               box.height - girder * 2);
      ctx.clip();
      ctx.globalAlpha = 0.42;
      ctx.strokeStyle = palette.hazard;
      ctx.lineWidth = Math.max(1.4, unit * 0.075);
      ctx.beginPath();
      for (let index = -1; index < 5; index += 1) {
        const x = box.left + box.width * (index * 0.26);
        ctx.moveTo(x, box.top + box.height);
        ctx.lineTo(x + box.width * 0.3, box.top);
      }
      ctx.stroke();
      ctx.restore();

      // And a stress crack along the span, which is where it will part.
      ctx.globalAlpha = 0.85;
      ctx.strokeStyle = palette.base;
      ctx.lineWidth = Math.max(0.8, unit * 0.03);
      ctx.beginPath();
      ctx.moveTo(box.left, box.top + box.height * 0.5);
      ctx.lineTo(box.left + box.width * 0.35, box.top + box.height * 0.44);
      ctx.lineTo(box.left + box.width * 0.62, box.top + box.height * 0.56);
      ctx.lineTo(box.left + box.width, box.top + box.height * 0.48);
      ctx.stroke();
    }
    ctx.restore();
  }

  /**
   * Laboratory stonework: three courses of brick on a mortar bed.
   *
   * Extracted from the `wall` recipe so that everything in the building which
   * is *masonry* is the same masonry. Rooms 4 and 5 are enclosed by the same
   * tiles as rooms 1 to 3, and the three pieces of room 5's architecture that
   * are built into the fabric — the doorway's jambs, the pier the control
   * terminal is mounted on, and the stanchions the beam emitters sit on — are
   * bricked with this rather than each inventing its own stone.
   *
   * `courses` and `split` let a narrow pier use fewer, larger bricks so the
   * coursing stays legible when the box is a fraction of a metre wide.
   */
  function layBricks(ctx, box, colour, unit, palette, courses, split) {
    const rows = courses || 3;
    const columns = split || 2;

    // The mortar bed. Everything else is bricks laid on top of it.
    ctx.fillStyle = palette.base;
    ctx.fillRect(box.left, box.top, box.width, box.height);

    const courseHeight = box.height / rows;
    const brickWidth = box.width / columns;
    // Below a few pixels the gaps swallow the bricks, so the mortar thins to
    // nothing and it goes back to reading as a solid block.
    const gap = Math.min(courseHeight * 0.22, Math.max(0.5, unit * 0.045));

    ctx.fillStyle = colour;
    for (let course = 0; course < rows; course += 1) {
      const top = box.top + course * courseHeight;
      const offset = course % 2 === 0 ? 0 : -brickWidth / 2;
      for (let x = box.left + offset;
           x < box.left + box.width;
           x += brickWidth) {
        // Clipped to the box, so half bricks at the edges line up with the
        // half bricks of the tile next door.
        const left = Math.max(x, box.left);
        const right = Math.min(x + brickWidth - gap, box.left + box.width);
        if (right > left) {
          ctx.fillRect(left, top + gap / 2, right - left, courseHeight - gap);
        }
      }
    }
  }

  /**
   * A small caps label under a thing, on its own dark plate.
   *
   * The two objectives in room 5 have to be identifiable without reading any
   * documentation — which mostly means saying what they are, in the room. The
   * plate is there because the warehouse floor behind the text varies and
   * unbacked type on it is unreadable at some layouts and fine at others.
   *
   * Skipped entirely below `minUnit` pixels per world unit: type that has
   * become three pixels tall is noise on the drawing rather than information,
   * and the legend still names everything.
   */
  function drawLabel(ctx, box, text, colour, unit, palette, options) {
    const settings = options || {};
    const minUnit = settings.minUnit === undefined ? 26 : settings.minUnit;
    if (unit < minUnit || !text) return;

    const size = Math.max(7, Math.min(11, unit * 0.115));
    ctx.save();
    ctx.font = '600 ' + size.toFixed(1) + 'px ' + (window.ROOM_CONFIG
      && window.ROOM_CONFIG.labelFont || 'system-ui, sans-serif');
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';

    const metrics = ctx.measureText(text);
    const padX = size * 0.5;
    const padY = size * 0.34;
    const centreX = box.left + box.width / 2;
    // Under the box by default; above it when the caller says the room runs
    // out of floor below.
    const centreY = settings.above
      ? box.top - size - padY * 2
      : box.top + box.height + size * 0.9;

    ctx.globalAlpha = 0.78;
    ctx.fillStyle = palette.base;
    ctx.fillRect(centreX - metrics.width / 2 - padX,
                 centreY - size / 2 - padY,
                 metrics.width + padX * 2, size + padY * 2);

    ctx.globalAlpha = 0.95;
    ctx.fillStyle = colour;
    ctx.fillText(text, centreX, centreY);
    ctx.restore();
  }

  const RECIPES = {

    /** A plain filled cell. The fallback. */
    block(ctx, box, look) {
      ctx.fillStyle = look.color;
      ctx.fillRect(box.left, box.top, box.width, box.height);
    },

    /**
     * Nothing at all.
     *
     * Floor is the surface everything else sits on, and the renderer has
     * already laid it down — along with, on the planner's screen, the
     * value wash over it. Painting it again here would rub that out.
     */
    floor() {},

    /**
     * Brickwork: courses of individual bricks with mortar between them.
     *
     * The mortar is the page colour showing through the gaps rather than a
     * line drawn over the top, which is what makes it read as bricks
     * instead of as a square with a grid on it. The stagger runs half a
     * brick per course, and bricks are clipped at the cell edge so a run
     * of walls courses continuously across the join.
     */
    wall(ctx, box, look, unit, palette) {
      ctx.save();
      layBricks(ctx, box, look.color, unit, palette);
      ctx.restore();
    },

    /**
     * Ice: a slick of meltwater with a crystal sitting on it.
     *
     * The slick alone was being read as "a pale tile". The six-armed
     * crystal is the part that says ice, so it is drawn at full strength
     * over a slick that has been dimmed to make room for it.
     */
    /**
     * Ice: the WHOLE TILE, not a snowflake on a floor.
     *
     * It used to be a six-armed crystal drawn in the middle of an otherwise
     * ordinary floor tile, and at cell size that is a grey smudge — the tile
     * still read as floor, so the one mechanic room 1 is built around was
     * invisible until R-5 slid on it.
     *
     * Now the tile *is* frozen: a sheet of frosted blue glass over the floor,
     * with a frozen rim, thin cracks through it, a couple of specular
     * reflections and a soft cyan bloom. Nothing on it is a symbol; it is a
     * surface, and the player reads "slippery" from the material.
     *
     * Deterministic. Every crack and glint comes from the tile's own position,
     * so a tile looks the same on every frame and two tiles look different
     * from each other. `look.phase` is not used at all — ice does not animate,
     * and a shimmering floor would pull the eye off the agent.
     */
    ice(ctx, box, look, unit, palette) {
      const seed = Math.round(box.left * 7.3 + box.top * 3.1);
      const random = index => {
        const value = Math.sin((seed + index * 37) * 12.9898) * 43758.5453;
        return value - Math.floor(value);
      };
      const inset = Math.min(box.width, box.height) * 0.06;
      const sheet = {
        left: box.left + inset, top: box.top + inset,
        width: box.width - inset * 2, height: box.height - inset * 2,
      };

      ctx.save();

      // 1. The bloom under it, so the tile glows rather than merely being
      //    pale. Drawn first and wider than the tile, clipped by the caller's
      //    cell, which is what makes a run of ice read as one frozen sheet.
      const glow = ctx.createRadialGradient(
        box.left + box.width / 2, box.top + box.height / 2, 0,
        box.left + box.width / 2, box.top + box.height / 2,
        Math.max(box.width, box.height) * 0.72);
      // The accent, so the bloom is cyan. `look.color` is the pale ice tone
      // and glowing with it made the tile read white rather than frozen.
      glow.addColorStop(0, palette.accent);
      glow.addColorStop(1, 'transparent');
      ctx.globalAlpha = 0.3;
      ctx.fillStyle = glow;
      ctx.fillRect(box.left, box.top, box.width, box.height);

      // 2. The frozen body: a translucent blue sheet over the floor beneath.
      ctx.globalAlpha = 0.62;
      ctx.fillStyle = look.color;
      ctx.fillRect(sheet.left, sheet.top, sheet.width, sheet.height);
      // A cyan cast through the body: frosted *blue* glass rather than frosted
      // grey glass, which is what the pale tone alone came out as.
      ctx.globalAlpha = 0.2;
      ctx.fillStyle = palette.accent;
      ctx.fillRect(sheet.left, sheet.top, sheet.width, sheet.height);

      // 3. Depth. Lighter towards the top-left, as though lit from there, so
      //    the surface has a direction and does not read as flat paint.
      const sheen = ctx.createLinearGradient(
        sheet.left, sheet.top, sheet.left + sheet.width,
        sheet.top + sheet.height);
      sheen.addColorStop(0, '#FFFFFF');
      sheen.addColorStop(0.45, 'transparent');
      sheen.addColorStop(1, palette.base);
      ctx.globalAlpha = 0.16;
      ctx.fillStyle = sheen;
      ctx.fillRect(sheet.left, sheet.top, sheet.width, sheet.height);

      // 4. The frozen rim: frost gathers at the edges of a sheet of ice, and
      //    it is what separates one tile from the floor beside it.
      // Frost gathered at the edge — a soft icy band, not a white picture
      // frame. The bright white outline it used to have was the loudest thing
      // in the chamber and read as a pane of glass rather than as ice.
      ctx.globalAlpha = 0.5;
      ctx.strokeStyle = look.color;
      ctx.lineWidth = Math.max(1.4, unit * 0.09);
      ctx.strokeRect(sheet.left, sheet.top, sheet.width, sheet.height);
      ctx.globalAlpha = 0.4;
      ctx.strokeStyle = palette.accent;
      ctx.lineWidth = Math.max(0.7, unit * 0.025);
      ctx.strokeRect(sheet.left, sheet.top, sheet.width, sheet.height);

      // 5. Cracks: three thin fractures across the sheet, each a shallow
      //    dog-leg rather than a straight line, because ice does not crack
      //    straight. Skipped when there are not the pixels to see them.
      if (unit >= 14) {
        ctx.globalAlpha = 0.42;
        ctx.strokeStyle = '#DFF4FF';
        ctx.lineWidth = Math.max(0.5, unit * 0.016);
        ctx.lineCap = 'round';
        for (let index = 0; index < 3; index += 1) {
          const fromEdge = random(index * 3);
          const toEdge = random(index * 3 + 1);
          const bend = 0.3 + random(index * 3 + 2) * 0.4;
          const from = { x: sheet.left + sheet.width * fromEdge,
                         y: sheet.top };
          const to = { x: sheet.left + sheet.width * toEdge,
                       y: sheet.top + sheet.height };
          const mid = { x: from.x + (to.x - from.x) * bend
                             + sheet.width * (random(index) - 0.5) * 0.3,
                        y: sheet.top + sheet.height * bend };
          ctx.beginPath();
          ctx.moveTo(from.x, from.y);
          ctx.lineTo(mid.x, mid.y);
          ctx.lineTo(to.x, to.y);
          ctx.stroke();
        }
      }

      // 6. Two specular reflections — the giveaway that a surface is wet or
      //    glassy. Long, thin, and at a shallow angle.
      ctx.globalAlpha = 0.4;
      ctx.fillStyle = '#EAF8FF';
      [[0.22, 0.3, 0.34, 0.055], [0.55, 0.66, 0.22, 0.04]].forEach(spec => {
        ctx.save();
        ctx.translate(sheet.left + sheet.width * spec[0],
                      sheet.top + sheet.height * spec[1]);
        ctx.rotate(-0.42);
        ctx.beginPath();
        ctx.ellipse(0, 0, sheet.width * spec[2], sheet.height * spec[3],
                    0, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
      });

      ctx.restore();
    },

    /**
     * A hole in the floor, seen from above.
     *
     * The depth is done with two inset rings rather than a gradient: the
     * floor's broken edge, a shelf below it, and then blackness. Each ring
     * is drawn a little smaller and a little darker, which is what reads
     * as "this goes down" while staying flat and inside the palette.
     */
    /**
     * A deep industrial maintenance shaft. One cell of it.
     *
     * WHAT IT WAS AND WHY THAT DID NOT WORK
     * A dark rectangle with a broken lip around it. Read as a black hole in
     * the floor with no indication of what it was or why it mattered — players
     * did not know what the big black block in the middle of room 2 *was*.
     *
     * It is now a maintenance shaft dropping away below the laboratory: metal
     * lining plates on the walls, service pipes and a rail running down it,
     * cross beams at intervals, machinery far below, warning lamps on the lip
     * and dust drifting up out of it. The depth comes from shading — the walls
     * are lit at the lip and black at the bottom of the frame — so it reads as
     * something with a bottom a long way down rather than as a flat shape.
     *
     * HOW IT JOINS UP ACROSS TILES
     * A tile only knows its own box, and the shaft is twenty of them. Two
     * things make it continuous. `look.rims` says which sides face something
     * that is not shaft, so only the outer edge gets a lip and the interior
     * opens straight through (see `definition.py`). And every vertical feature
     * is placed at a fraction of the tile's own width — all the tiles in a
     * column share a left edge, so the pipes and the rail line up down the
     * whole shaft without any tile being told where it is.
     */
    abyss(ctx, box, look, unit, palette) {
      const rims = look.rims || { top: true, bottom: true,
                                  left: true, right: true };
      const phase = look.phase || 0;
      // Which row of the shaft this is, for the features that repeat down it.
      // Offset by a constant the whole frame shares, which is all that matters
      // for keeping beams on the same rows as each other.
      const row = Math.round(box.top / Math.max(1, unit));
      const lip = Math.min(box.width, box.height) * 0.12;

      ctx.save();

      // 1. The void. Everything else is lining and structure inside it.
      ctx.fillStyle = palette.base;
      ctx.fillRect(box.left, box.top, box.width, box.height);

      // 2. Depth: a wash that is lighter at the top of the cell and black at
      //    the bottom, so a column of these reads as falling away.
      const depth = ctx.createLinearGradient(0, box.top, 0,
                                             box.top + box.height);
      depth.addColorStop(0, palette.cellWall || palette.muted);
      // A mid stop at the base tone, so the fall-off is steep near the lip
      // and then flat: light dies fast at the mouth of a shaft, and what is
      // below that is all equally dark.
      depth.addColorStop(0.55, palette.base);
      depth.addColorStop(1, palette.base);
      ctx.globalAlpha = rims.top ? 0.55 : 0.3;
      ctx.fillStyle = depth;
      ctx.fillRect(box.left, box.top, box.width, box.height);

      // Corner shadowing: the sides of the shaft are darker than the middle,
      // which is what gives a flat cell the roundness of a real opening.
      const across = ctx.createLinearGradient(box.left, 0,
                                              box.left + box.width, 0);
      across.addColorStop(0, palette.base);
      across.addColorStop(0.5, 'transparent');
      across.addColorStop(1, palette.base);
      ctx.globalAlpha = 0.5;
      ctx.fillStyle = across;
      ctx.fillRect(box.left, box.top, box.width, box.height);

      // 3. The metal lining of the side walls, in receding plates. Only where
      //    there is actually a wall — an interior tile has shaft either side.
      ctx.globalAlpha = 0.5;
      ctx.fillStyle = palette.cellWall || palette.muted;
      const wall = box.width * 0.14;
      if (rims.left) ctx.fillRect(box.left, box.top, wall, box.height);
      if (rims.right) {
        ctx.fillRect(box.left + box.width - wall, box.top, wall, box.height);
      }
      // Plate seams across the lining, so the walls have a scale.
      if (unit >= 14 && (rims.left || rims.right)) {
        ctx.globalAlpha = 0.5;
        ctx.strokeStyle = palette.base;
        ctx.lineWidth = Math.max(0.5, unit * 0.02);
        ctx.beginPath();
        for (let index = 1; index < 3; index += 1) {
          const y = box.top + (box.height / 3) * index;
          if (rims.left) {
            ctx.moveTo(box.left, y);
            ctx.lineTo(box.left + wall, y);
          }
          if (rims.right) {
            ctx.moveTo(box.left + box.width - wall, y);
            ctx.lineTo(box.left + box.width, y);
          }
        }
        ctx.stroke();
      }

      // 4. Service pipes and a maintenance rail running down the shaft. At
      //    fractions of the tile, so they are continuous between rows.
      if (unit >= 12) {
        const runs = [
          { at: 0.3, width: 0.055, tone: palette.cellWall || palette.muted },
          { at: 0.42, width: 0.03, tone: palette.muted },
          { at: 0.72, width: 0.045, tone: palette.cellWall || palette.muted },
        ];
        runs.forEach(run => {
          const x = box.left + box.width * run.at;
          const w = Math.max(1, box.width * run.width);
          ctx.globalAlpha = 0.55;
          ctx.fillStyle = run.tone;
          ctx.fillRect(x, box.top, w, box.height);
          // A highlight down one side, so a pipe is round rather than a stripe.
          ctx.globalAlpha = 0.3;
          ctx.fillStyle = '#FFFFFF';
          ctx.fillRect(x, box.top, Math.max(0.5, w * 0.3), box.height);
        });
      }

      // 5. A cross beam every third row: the structure that holds the lining
      //    apart, and the strongest single cue that this has depth.
      if (unit >= 12 && row % 3 === 1) {
        ctx.globalAlpha = 0.75;
        ctx.fillStyle = palette.cellWall || palette.muted;
        const beam = Math.max(1.5, box.height * 0.12);
        const y = box.top + box.height * 0.34;
        ctx.fillRect(box.left, y, box.width, beam);
        ctx.globalAlpha = 0.4;
        ctx.fillStyle = palette.base;
        ctx.fillRect(box.left, y + beam * 0.6, box.width, beam * 0.4);
        // Bolts along it.
        if (unit >= 18) {
          ctx.globalAlpha = 0.5;
          ctx.fillStyle = palette.base;
          for (let index = 0; index < 3; index += 1) {
            const bx = box.left + box.width * (0.2 + index * 0.3);
            ctx.beginPath();
            ctx.arc(bx, y + beam / 2, Math.max(0.6, unit * 0.02), 0,
                    Math.PI * 2);
            ctx.fill();
          }
        }
      }

      // 6. A red warning lamp on the lip, blinking. Only on the top edge —
      //    that is the edge a player stands next to.
      if (rims.top && unit >= 14) {
        const lit = 0.45 + 0.55 * Math.abs(Math.sin(phase * 1.6 + row));
        const lampX = box.left + box.width * 0.86;
        const lampY = box.top + lip * 1.6;
        ctx.globalAlpha = lit;
        withGlow(ctx, palette.hazard, glowFor('laser') * 0.5 * lit, () => {
          ctx.fillStyle = palette.hazard;
          ctx.beginPath();
          ctx.arc(lampX, lampY, Math.max(1, unit * 0.035), 0, Math.PI * 2);
          ctx.fill();
        });
      }

      // 7. Machinery a long way down, on the last row of the shaft: a couple
      //    of dim blocks, so the bottom is a place rather than an absence.
      if (rims.bottom && unit >= 14) {
        ctx.globalAlpha = 0.3;
        ctx.fillStyle = palette.cellWall || palette.muted;
        ctx.fillRect(box.left + box.width * 0.2,
                     box.top + box.height * 0.6,
                     box.width * 0.24, box.height * 0.4);
        ctx.fillRect(box.left + box.width * 0.56,
                     box.top + box.height * 0.72,
                     box.width * 0.3, box.height * 0.28);
        // One green service LED down there, very small.
        ctx.globalAlpha = 0.5 + 0.5 * Math.abs(Math.sin(phase * 0.9));
        ctx.fillStyle = palette.goal;
        ctx.beginPath();
        ctx.arc(box.left + box.width * 0.7, box.top + box.height * 0.78,
                Math.max(0.7, unit * 0.02), 0, Math.PI * 2);
        ctx.fill();
      }

      // 8. Dust drifting up out of it. Three motes per cell, rising and
      //    wrapping, all off the shared clock so a replay reproduces them.
      if (unit >= 16) {
        ctx.fillStyle = '#FFFFFF';
        for (let index = 0; index < 3; index += 1) {
          const drift = ((phase * 0.05 + index * 0.37 + row * 0.11) % 1.0);
          const y = box.top + box.height * (1 - drift);
          const x = box.left + box.width
                  * (0.2 + 0.6 * ((index * 0.41 + row * 0.23) % 1.0));
          ctx.globalAlpha = 0.16 * Math.sin(drift * Math.PI);
          ctx.beginPath();
          ctx.arc(x, y, Math.max(0.5, unit * 0.016), 0, Math.PI * 2);
          ctx.fill();
        }
      }

      // 8b. Fog, hanging in the shaft.
      /* The one thing the shaft had no cue for was AIR. Pipes, beams and
         machinery all say "there is structure down there"; none of them say
         "and it is a long way through a lot of atmosphere". Two slow bands of
         haze drifting across the cell do that, and they are what stop the
         lower half reading as flat black.

         Both are driven off `phase`, the same shared clock as the dust and the
         lamps, so a replay of the same step draws the same fog. Nothing here
         is random per frame. */
      if (unit >= 14) {
        for (let index = 0; index < 2; index += 1) {
          const drift = ((phase * 0.012 + index * 0.5 + row * 0.13) % 1.0);
          // Across and back rather than round: a wrap would be a visible jump.
          const sway = Math.sin(drift * Math.PI * 2) * box.width * 0.16;
          const bandY = box.top + box.height * (0.34 + index * 0.36);
          const bandH = box.height * 0.3;
          const haze = ctx.createLinearGradient(0, bandY, 0, bandY + bandH);
          haze.addColorStop(0, 'transparent');
          haze.addColorStop(0.5, palette.muted);
          haze.addColorStop(1, 'transparent');
          // Thicker further down, where there is more air to see through.
          ctx.globalAlpha = 0.05 + 0.05 * index;
          ctx.fillStyle = haze;
          ctx.fillRect(box.left + sway - box.width * 0.2, bandY,
                       box.width * 1.4, bandH);
        }
      }

      // 9. The broken lip of the floor the shaft was cut through, on the sides
      //    that have one. Drawn last, over the lining, because it is nearer.
      /* The lip is BROKEN STONE, not a red band.
         `look.color` for a shaft is `--hazard`, and filling the whole lip with
         it turned a twenty-cell shaft into a bright red rectangle that read as
         a framed window and dominated the room. The lip is the cut edge of the
         laboratory floor, so it is drawn in the floor's own stone with a thin
         hazard line painted just inboard of it — which is what a real edge
         protection marking looks like, and it leaves the shaft reading as a
         hole in a stone floor. */
      const edge = (x, y, w, h, side) => {
        if (w <= 0 || h <= 0) return;
        // The stone itself.
        ctx.globalAlpha = 0.9;
        ctx.fillStyle = palette.cellWall || palette.muted;
        ctx.fillRect(x, y, w, h);
        // Shaded towards the opening, so the edge has a thickness.
        ctx.globalAlpha = 0.45;
        ctx.fillStyle = palette.base;
        const bevel = Math.max(1, Math.min(w, h) * 0.4);
        if (side === 'top') ctx.fillRect(x, y + h - bevel, w, bevel);
        else if (side === 'bottom') ctx.fillRect(x, y, w, bevel);
        else if (side === 'left') ctx.fillRect(x + w - bevel, y, bevel, h);
        else ctx.fillRect(x, y, bevel, h);

        // The hazard marking, a thin line just inboard of the edge.
        ctx.globalAlpha = 0.55;
        ctx.fillStyle = look.color;
        const mark = Math.max(0.8, Math.min(w, h) * 0.18);
        if (side === 'top') ctx.fillRect(x, y + h - mark, w, mark);
        else if (side === 'bottom') ctx.fillRect(x, y, w, mark);
        else if (side === 'left') ctx.fillRect(x + w - mark, y, mark, h);
        else ctx.fillRect(x, y, mark, h);
      };
      if (rims.top) edge(box.left, box.top, box.width, lip, 'top');
      if (rims.bottom) {
        edge(box.left, box.top + box.height - lip, box.width, lip, 'bottom');
      }
      if (rims.left) edge(box.left, box.top, lip, box.height, 'left');
      if (rims.right) {
        edge(box.left + box.width - lip, box.top, lip, box.height, 'right');
      }

      // A ragged bite out of the top lip, so it is broken stone rather than a
      // cut tile. Only where there is a top lip to break.
      if (unit >= 12 && rims.top) {
        ctx.globalAlpha = 1;
        ctx.fillStyle = palette.base;
        for (let index = 0; index < 3; index += 1) {
          const width = box.width / 6.6;
          const x = box.left + box.width * (0.18 + index * 0.32);
          ctx.beginPath();
          ctx.moveTo(x, box.top + lip);
          ctx.lineTo(x + width / 2, box.top + lip * 0.2);
          ctx.lineTo(x + width, box.top + lip);
          ctx.closePath();
          ctx.fill();
        }
      }
      ctx.restore();
    },

    /**
     * Planks over a hole, with the dark showing through the gaps.
     *
     * The hole is drawn first on purpose: a bridge is a thing over a void,
     * and the void has to be visible for that to be legible.
     */
    /**
     * A steel bridge over the shaft.
     *
     * It was a rope bridge — slats hung from two ropes between posts — and in
     * an industrial laboratory that read as a stripey floor rather than as a
     * structure over a drop. Now it is what a service gantry over a shaft
     * actually is: two edge girders, plate decking between them with the dark
     * showing through the joints, and rivets along the girders.
     *
     * The void is drawn first, at full depth, because a bridge only reads as a
     * bridge if you can see what is under it.
     */
    bridge(ctx, box, look, unit, palette) {
      RECIPES.abyss(ctx, box, { color: palette.hairlineFaint,
                                rims: look.rims }, unit, palette);
      drawDeck(ctx, box, look, unit, palette, false);
    },

    /**
     * The same gantry, but a section that may give way under the next step.
     *
     * Marked rather than merely tinted: hazard chevrons across the deck and a
     * stress crack down the middle. The player has to be able to tell a sound
     * section from a failing one at a glance, because which is which is the
     * whole decision this chamber is about.
     */
    bridgeWeak(ctx, box, look, unit, palette) {
      RECIPES.abyss(ctx, box, { color: palette.hairlineFaint,
                                rims: look.rims }, unit, palette);
      drawDeck(ctx, box, look, unit, palette, true);
    },

    /** What is left after a plank gives way: the hole, and two stubs. */
    /**
     * A section that has given way. The most violent thing in the game, and it
     * used to be two small angled stubs.
     *
     * A heavy plated deck failing: the girders sheared and tilted, the plates
     * broken off and tumbling into the shaft, sparks off the torn supports and
     * dust hanging in the gap. Every piece is placed from `look.phase` and the
     * tile's own position, so it is deterministic — the same frame of a replay
     * draws the same wreckage, which is the rule everything in this project
     * follows.
     */
    bridgeBroken(ctx, box, look, unit, palette) {
      const phase = look.phase || 0;
      const wallTone = palette.cellWall || palette.muted;
      const seed = Math.round(box.left * 5.7 + box.top * 2.3);
      const random = index => {
        const value = Math.sin((seed + index * 53) * 12.9898) * 43758.5453;
        return value - Math.floor(value);
      };

      // The shaft, now open where the deck was.
      RECIPES.abyss(ctx, box, { color: palette.hazard, rims: look.rims,
                                phase: phase }, unit, palette);

      ctx.save();

      // 1. The two sheared ends, still bolted to what is left, tilted down
      //    into the gap. Heavy — these are girders, not planks.
      const girder = Math.max(2, unit * 0.15);
      const stub = box.width * 0.3;
      [[box.left, 1, 0.3], [box.left + box.width, -1, -0.34]]
        .forEach(([anchor, direction, tilt]) => {
          ctx.save();
          ctx.translate(anchor, box.top + box.height * 0.42);
          ctx.rotate(tilt);
          ctx.globalAlpha = 0.95;
          ctx.fillStyle = wallTone;
          ctx.fillRect(direction > 0 ? 0 : -stub, 0, stub, girder);
          // The deck plate still attached to it, thinner and darker.
          ctx.globalAlpha = 0.7;
          ctx.fillStyle = look.color;
          ctx.fillRect(direction > 0 ? 0 : -stub * 0.8, girder,
                       stub * 0.8, girder * 0.7);
          // A torn edge: three teeth where the metal parted.
          ctx.globalAlpha = 0.9;
          ctx.fillStyle = palette.base;
          for (let index = 0; index < 3; index += 1) {
            const at = direction > 0 ? stub - girder * 0.5 : -stub + girder * 0.2;
            ctx.beginPath();
            ctx.moveTo(at, girder * (index * 0.5));
            ctx.lineTo(at + direction * girder * 0.5, girder * (index * 0.5 + 0.25));
            ctx.lineTo(at, girder * (index * 0.5 + 0.5));
            ctx.closePath();
            ctx.fill();
          }
          ctx.restore();
        });

      // 2. Plates tumbling into the shaft. Each falls and spins on its own
      //    schedule and wraps, so the wreckage keeps dropping rather than
      //    freezing after a second.
      if (unit >= 12) {
        for (let index = 0; index < 4; index += 1) {
          const speed = 0.35 + random(index) * 0.4;
          const drop = ((phase * speed + random(index + 9)) % 1.0);
          const x = box.left + box.width * (0.18 + random(index + 3) * 0.64);
          const y = box.top + box.height * (0.35 + drop * 0.75);
          const size = box.width * (0.1 + random(index + 6) * 0.1);
          ctx.save();
          ctx.translate(x, y);
          ctx.rotate(drop * 7 + index);
          // Fading as it falls away into the dark.
          ctx.globalAlpha = 0.75 * (1 - drop * 0.8);
          ctx.fillStyle = index % 2 ? wallTone : look.color;
          ctx.fillRect(-size / 2, -size * 0.18, size, size * 0.36);
          ctx.restore();
        }
      }

      // 3. Sparks off the torn supports, at the two sheared ends. Short-lived
      //    and bright, on a faster cycle than the debris.
      if (unit >= 14) {
        for (let index = 0; index < 6; index += 1) {
          const life = ((phase * 1.6 + index * 0.17) % 1.0);
          if (life > 0.55) continue;              // dark most of the time
          const side = index % 2 ? 0.08 : 0.92;
          const angle = -1.1 + random(index + 12) * 2.2;
          const reach = box.width * 0.3 * life;
          const x = box.left + box.width * side + Math.cos(angle) * reach;
          const y = box.top + box.height * 0.42 + Math.sin(angle) * reach
                  + life * life * box.height * 0.5;
          ctx.globalAlpha = (1 - life / 0.55) * 0.95;
          withGlow(ctx, palette.warn || palette.hazard, glowFor('laser') * 0.6,
                   () => {
            ctx.fillStyle = palette.warn || palette.hazard;
            ctx.beginPath();
            ctx.arc(x, y, Math.max(0.6, unit * 0.018), 0, Math.PI * 2);
            ctx.fill();
          });
        }
      }

      // 4. Dust hanging in the gap, thrown up by the failure.
      ctx.globalAlpha = 0.1;
      ctx.fillStyle = '#FFFFFF';
      for (let index = 0; index < 3; index += 1) {
        const puff = ((phase * 0.12 + index * 0.4) % 1.0);
        ctx.beginPath();
        ctx.arc(box.left + box.width * (0.25 + index * 0.26),
                box.top + box.height * (0.5 - puff * 0.3),
                box.width * (0.12 + puff * 0.14), 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    },

    /**
     * A beam with an emitter at each end.
     *
     * `look.orientation` decides which way it runs, so a line of beam
     * cells joins into one continuous beam instead of a picket fence of
     * separate strokes. A caller that knows its neighbours sets it; one
     * that does not falls back to the long axis of the box, which is
     * right for a beam declared as a single wide or tall entity.
     */
    laser(ctx, box, look, unit, palette) {
      const horizontal = look.orientation
        ? look.orientation === 'horizontal'
        : box.width > box.height;
      const thickness = Math.max(1, unit * 0.09);

      const from = horizontal
        ? { x: box.left, y: box.top + box.height / 2 }
        : { x: box.left + box.width / 2, y: box.top };
      const to = horizontal
        ? { x: box.left + box.width, y: box.top + box.height / 2 }
        : { x: box.left + box.width / 2, y: box.top + box.height };

      ctx.save();
      withGlow(ctx, look.color, glowFor('laser'), () => {
        ctx.strokeStyle = look.color;
        ctx.lineWidth = thickness;
        ctx.lineCap = 'butt';
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
      });

      // The emitters the beam comes out of.
      ctx.fillStyle = palette.muted;
      const housing = unit * 0.22;
      if (horizontal) {
        ctx.fillRect(box.left - housing * 0.2, from.y - housing / 2,
                     housing, housing);
        ctx.fillRect(box.left + box.width - housing * 0.8, to.y - housing / 2,
                     housing, housing);
      } else {
        ctx.fillRect(from.x - housing / 2, box.top - housing * 0.2,
                     housing, housing);
        ctx.fillRect(to.x - housing / 2, box.top + box.height - housing * 0.8,
                     housing, housing);
      }
      ctx.restore();
    },

    /** A door, lit from inside. */
    exit(ctx, box, look, unit, palette) {
      ctx.save();
      ctx.fillStyle = palette.base;
      ctx.fillRect(box.left, box.top, box.width, box.height);

      const inset = box.width * 0.16;
      withGlow(ctx, look.color, glowFor('exit'), () => {
        ctx.fillStyle = look.color;
        ctx.fillRect(box.left + inset, box.top + inset * 0.6,
                     box.width - inset * 2, box.height - inset * 0.9);
      });

      // The frame, and a dark seam: a door, not a tile.
      ctx.strokeStyle = palette.muted;
      ctx.lineWidth = Math.max(0.6, unit * 0.04);
      ctx.strokeRect(box.left + inset, box.top + inset * 0.6,
                     box.width - inset * 2, box.height - inset * 0.9);
      ctx.strokeStyle = palette.base;
      ctx.globalAlpha = 0.5;
      ctx.beginPath();
      ctx.moveTo(box.left + box.width / 2, box.top + inset);
      ctx.lineTo(box.left + box.width / 2, box.top + box.height - inset * 0.5);
      ctx.stroke();
      ctx.restore();
    },

    /** A marked pad on the floor: corner ticks, nothing filled. */
    start(ctx, box, look, unit) {
      ctx.save();
      ctx.strokeStyle = look.color;
      ctx.lineWidth = Math.max(0.8, unit * 0.05);
      const arm = box.width * 0.26;
      const inset = box.width * 0.16;
      [
        [box.left + inset, box.top + inset, 1, 1],
        [box.left + box.width - inset, box.top + inset, -1, 1],
        [box.left + inset, box.top + box.height - inset, 1, -1],
        [box.left + box.width - inset, box.top + box.height - inset, -1, -1],
      ].forEach(corner => {
        const x = corner[0];
        const y = corner[1];
        ctx.beginPath();
        ctx.moveTo(x, y);
        ctx.lineTo(x + arm * corner[2], y);
        ctx.moveTo(x, y);
        ctx.lineTo(x, y + arm * corner[3]);
        ctx.stroke();
      });
      ctx.restore();
    },

    /**
     * A generator: a squat machine with a core and cooling fins.
     *
     * Dormant it is drawn in whatever colour the room gives it; a room
     * that wants it lit says so with a state, and the glow comes from the
     * colour that state carries rather than from anything decided here.
     */
    generator(ctx, box, look, unit, palette) {
      const inset = Math.min(box.width, box.height) * 0.16;
      const body = {
        left: box.left + inset, top: box.top + inset * 1.3,
        width: box.width - inset * 2, height: box.height - inset * 2.2,
      };

      ctx.save();
      withGlow(ctx, look.color, look.lit ? glowFor('exit') : 0, () => {
        ctx.fillStyle = look.color;
        ctx.fillRect(body.left, body.top, body.width, body.height);
      });

      // The core: a dark window into the machine, with a bar across it
      // that brightens when the room says the thing is running.
      ctx.fillStyle = palette.base;
      ctx.fillRect(body.left + body.width * 0.22, body.top + body.height * 0.22,
                   body.width * 0.56, body.height * 0.44);
      ctx.fillStyle = look.lit ? look.color : palette.hairlineFaint;
      ctx.fillRect(body.left + body.width * 0.3, body.top + body.height * 0.36,
                   body.width * 0.4, body.height * 0.16);

      // Cooling fins along the foot, which is what stops it reading as a
      // crate with a stripe on it.
      if (unit >= 14) {
        ctx.fillStyle = look.color;
        const fins = 3;
        for (let index = 0; index < fins; index += 1) {
          ctx.fillRect(body.left + body.width * (0.14 + index * 0.3),
                       body.top + body.height,
                       body.width * 0.18, inset * 0.7);
        }
      }
      ctx.restore();
    },

    /**
     * Oil: a slick with no grip at all.
     *
     * Deliberately unlike ice, because they behave differently and telling
     * them apart matters: ice throws you sideways, oil carries you on. So
     * this is dark rather than pale, and has a sheen running the way it
     * would carry you rather than a crystal sitting on it.
     */
    oil(ctx, box, look, unit, palette) {
      const centreX = box.left + box.width / 2;
      const centreY = box.top + box.height / 2;

      ctx.save();
      ctx.fillStyle = look.color;
      ctx.globalAlpha = 0.85;
      ctx.beginPath();
      ctx.ellipse(centreX, centreY, box.width * 0.44, box.height * 0.38,
                  0, 0, Math.PI * 2);
      ctx.fill();

      // A darker eye in the middle, so it reads as depth rather than paint.
      ctx.fillStyle = palette.base;
      ctx.globalAlpha = 0.55;
      ctx.beginPath();
      ctx.ellipse(centreX, centreY, box.width * 0.26, box.height * 0.2,
                  0, 0, Math.PI * 2);
      ctx.fill();

      // The sheen: two long highlights lying flat, which is what says
      // "this will carry you" rather than "this will trip you".
      ctx.strokeStyle = palette.accent;
      ctx.globalAlpha = 0.3;
      ctx.lineWidth = Math.max(0.7, unit * 0.035);
      ctx.lineCap = 'round';
      ctx.beginPath();
      ctx.moveTo(centreX - box.width * 0.28, centreY - box.height * 0.12);
      ctx.lineTo(centreX + box.width * 0.06, centreY - box.height * 0.12);
      ctx.moveTo(centreX - box.width * 0.1, centreY + box.height * 0.14);
      ctx.lineTo(centreX + box.width * 0.24, centreY + box.height * 0.14);
      ctx.stroke();
      ctx.restore();
    },

    /** A battery: a cell with a terminal and a charge showing. */
    battery(ctx, box, look, unit, palette) {
      const width = box.width * 0.44;
      const height = box.height * 0.62;
      const left = box.left + (box.width - width) / 2;
      const top = box.top + (box.height - height) / 2;

      ctx.save();
      withGlow(ctx, look.color, glowFor('exit'), () => {
        ctx.fillStyle = look.color;
        ctx.fillRect(left, top, width, height);
        // The terminal on the cap.
        ctx.fillRect(left + width * 0.3, top - height * 0.12,
                     width * 0.4, height * 0.12);
      });

      // Charge bars cut out of it, so it reads as a cell and not a brick.
      if (unit >= 12) {
        ctx.fillStyle = palette.base;
        for (let bar = 0; bar < 3; bar += 1) {
          ctx.fillRect(left + width * 0.2, top + height * (0.18 + bar * 0.25),
                       width * 0.6, height * 0.1);
        }
      }
      ctx.restore();
    },

    /** A teleport pad: rings on the floor, with a bright centre. */
    teleport(ctx, box, look, unit, palette) {
      const centreX = box.left + box.width / 2;
      const centreY = box.top + box.height / 2;
      const radius = Math.min(box.width, box.height) * 0.42;

      ctx.save();
      ctx.strokeStyle = look.color;
      ctx.lineWidth = Math.max(0.8, unit * 0.045);
      [1, 0.66, 0.34].forEach((scale, index) => {
        ctx.globalAlpha = 0.35 + index * 0.2;
        ctx.beginPath();
        ctx.ellipse(centreX, centreY, radius * scale, radius * scale * 0.62,
                    0, 0, Math.PI * 2);
        ctx.stroke();
      });

      withGlow(ctx, look.color, glowFor('exit'), () => {
        ctx.globalAlpha = 1;
        ctx.fillStyle = look.color;
        ctx.beginPath();
        ctx.ellipse(centreX, centreY, radius * 0.16, radius * 0.1,
                    0, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.restore();
      void palette;
    },

    /**
     * A one-way door: a hatch with the only direction it opens marked.
     *
     * Drawn pointing down because that is the way it works. If a room ever
     * wants one that opens some other way, it says so with `look.opens`
     * and this follows it.
     */
    oneway(ctx, box, look, unit, palette) {
      const towards = { right: 0, down: Math.PI / 2, left: Math.PI,
                        up: -Math.PI / 2 }[look.opens || 'down'];
      const centreX = box.left + box.width / 2;
      const centreY = box.top + box.height / 2;
      const reach = Math.min(box.width, box.height) * 0.3;

      ctx.save();
      ctx.fillStyle = look.color;
      ctx.fillRect(box.left, box.top, box.width, box.height);

      // The jambs, so it is a doorway rather than a tile.
      ctx.fillStyle = palette.base;
      ctx.globalAlpha = 0.5;
      ctx.fillRect(box.left, box.top, box.width * 0.12, box.height);
      ctx.fillRect(box.left + box.width * 0.88, box.top,
                   box.width * 0.12, box.height);
      ctx.globalAlpha = 1;

      // Two chevrons, the way it lets you through.
      ctx.strokeStyle = palette.accent;
      ctx.lineWidth = Math.max(1, unit * 0.055);
      ctx.lineCap = 'round';
      ctx.beginPath();
      [-0.34, 0.24].forEach(offset => {
        const tipX = centreX + Math.cos(towards) * (reach + offset * reach * 2);
        const tipY = centreY + Math.sin(towards) * (reach + offset * reach * 2);
        [2.4, -2.4].forEach(spread => {
          ctx.moveTo(tipX, tipY);
          ctx.lineTo(tipX + Math.cos(towards + spread) * reach * 0.7,
                     tipY + Math.sin(towards + spread) * reach * 0.7);
        });
      });
      ctx.stroke();
      ctx.restore();
    },

    /** A brake zone: bars across the flow, like a rumble strip. */
    brake(ctx, box, look, unit, palette) {
      ctx.save();
      ctx.fillStyle = look.color;
      ctx.globalAlpha = 0.07;
      ctx.fillRect(box.left, box.top, box.width, box.height);

      ctx.strokeStyle = look.color;
      ctx.globalAlpha = 0.45;
      ctx.lineWidth = Math.max(1, unit * 0.05);
      ctx.lineCap = 'round';
      ctx.beginPath();
      const bars = 4;
      for (let index = 1; index <= bars; index += 1) {
        const x = box.left + (box.width * index) / (bars + 1);
        ctx.moveTo(x, box.top + box.height * 0.18);
        ctx.lineTo(x, box.top + box.height * 0.82);
      }
      ctx.stroke();

      ctx.globalAlpha = 0.25;
      ctx.setLineDash([unit * 0.2, unit * 0.25]);
      ctx.strokeRect(box.left, box.top, box.width, box.height);
      ctx.setLineDash([]);
      ctx.restore();
      void palette;
    },

    /** A charging station: a plate with a bolt struck through it. */
    charger(ctx, box, look, unit, palette) {
      const centreX = box.left + box.width / 2;
      const centreY = box.top + box.height / 2;
      const span = Math.min(box.width, box.height);

      ctx.save();
      ctx.fillStyle = palette.base;
      ctx.fillRect(box.left + box.width * 0.14, box.top + box.height * 0.14,
                   box.width * 0.72, box.height * 0.72);
      ctx.strokeStyle = look.color;
      ctx.lineWidth = Math.max(0.8, unit * 0.04);
      ctx.strokeRect(box.left + box.width * 0.14, box.top + box.height * 0.14,
                     box.width * 0.72, box.height * 0.72);

      withGlow(ctx, look.color, glowFor('exit'), () => {
        ctx.fillStyle = look.color;
        ctx.beginPath();
        ctx.moveTo(centreX + span * 0.08, centreY - span * 0.26);
        ctx.lineTo(centreX - span * 0.12, centreY + span * 0.02);
        ctx.lineTo(centreX + span * 0.01, centreY + span * 0.02);
        ctx.lineTo(centreX - span * 0.07, centreY + span * 0.26);
        ctx.lineTo(centreX + span * 0.14, centreY - span * 0.03);
        ctx.lineTo(centreX + span * 0.01, centreY - span * 0.03);
        ctx.closePath();
        ctx.fill();
      });
      ctx.restore();
    },

    /**
     * A live cable run: two rails with an arc between them.
     *
     * Drawn the same whether it is live or not, because a trap that
     * announces its timing is not a trap — what changes is the colour the
     * room gives it, and the room only gives it one on the step it bites.
     */
    trap(ctx, box, look, unit, palette) {
      ctx.save();
      ctx.strokeStyle = look.color;
      ctx.lineWidth = Math.max(1, unit * 0.06);
      ctx.lineCap = 'round';

      // The rails.
      ctx.beginPath();
      ctx.moveTo(box.left + box.width * 0.12, box.top + box.height * 0.28);
      ctx.lineTo(box.left + box.width * 0.88, box.top + box.height * 0.28);
      ctx.moveTo(box.left + box.width * 0.12, box.top + box.height * 0.72);
      ctx.lineTo(box.left + box.width * 0.88, box.top + box.height * 0.72);
      ctx.stroke();

      // The arc jumping between them.
      withGlow(ctx, look.color, glowFor('laser'), () => {
        ctx.beginPath();
        ctx.moveTo(box.left + box.width * 0.36, box.top + box.height * 0.28);
        ctx.lineTo(box.left + box.width * 0.54, box.top + box.height * 0.46);
        ctx.lineTo(box.left + box.width * 0.4, box.top + box.height * 0.52);
        ctx.lineTo(box.left + box.width * 0.6, box.top + box.height * 0.72);
        ctx.stroke();
      });
      ctx.restore();
      void palette;
    },

    /**
     * A repair bay: a cross on a plate, the universal "get patched up".
     */
    repair(ctx, box, look, unit, palette) {
      const centreX = box.left + box.width / 2;
      const centreY = box.top + box.height / 2;
      const arm = Math.min(box.width, box.height) * 0.3;
      const thickness = arm * 0.42;

      ctx.save();
      ctx.fillStyle = palette.base;
      ctx.fillRect(box.left + box.width * 0.12, box.top + box.height * 0.12,
                   box.width * 0.76, box.height * 0.76);

      withGlow(ctx, look.color, glowFor('exit'), () => {
        ctx.fillStyle = look.color;
        ctx.fillRect(centreX - thickness / 2, centreY - arm, thickness, arm * 2);
        ctx.fillRect(centreX - arm, centreY - thickness / 2, arm * 2, thickness);
      });
      ctx.restore();
      void unit;
    },

    /**
     * A key: a bow, a shank and two teeth.
     *
     * Drawn small and lit, because it is a thing to be picked up rather
     * than a part of the building, and it has to be tellable from the
     * machinery around it at a glance.
     */
    key(ctx, box, look, unit, palette) {
      const centreX = box.left + box.width * 0.5;
      const centreY = box.top + box.height * 0.5;
      const span = Math.min(box.width, box.height);

      ctx.save();
      withGlow(ctx, look.color, glowFor('exit'), () => {
        ctx.strokeStyle = look.color;
        ctx.lineWidth = Math.max(1, unit * 0.075);
        ctx.lineCap = 'round';

        // The bow, hollow, so it reads as something you hold.
        ctx.beginPath();
        ctx.arc(centreX - span * 0.22, centreY, span * 0.15, 0, Math.PI * 2);
        ctx.stroke();

        // The shank, and the teeth on the end of it.
        ctx.beginPath();
        ctx.moveTo(centreX - span * 0.07, centreY);
        ctx.lineTo(centreX + span * 0.3, centreY);
        if (span >= 12) {
          ctx.moveTo(centreX + span * 0.16, centreY);
          ctx.lineTo(centreX + span * 0.16, centreY + span * 0.13);
          ctx.moveTo(centreX + span * 0.3, centreY);
          ctx.lineTo(centreX + span * 0.3, centreY + span * 0.18);
        }
        ctx.stroke();
      });
      ctx.restore();
      void palette;
    },

    /**
     * A blast door: two heavy leaves meeting in the middle.
     *
     * Sealed, the leaves meet and the seam is hazard-striped. A room that
     * opens it says so with a state that sets `open`, and the leaves pull
     * back to the jambs leaving the way through.
     */
    door(ctx, box, look, unit, palette) {
      ctx.save();

      // The opening itself: dark, and visible only once the leaves move.
      ctx.fillStyle = palette.base;
      ctx.fillRect(box.left, box.top, box.width, box.height);

      const leaf = look.open ? box.width * 0.16 : box.width * 0.5;
      ctx.fillStyle = look.color;
      ctx.fillRect(box.left, box.top, leaf, box.height);
      ctx.fillRect(box.left + box.width - leaf, box.top, leaf, box.height);

      // Hazard stripes on the leading edges, angled, so a shut door reads
      // as something that is deliberately in the way.
      if (!look.open && unit >= 12) {
        ctx.fillStyle = palette.base;
        ctx.globalAlpha = 0.55;
        const stripes = 3;
        for (let index = 0; index < stripes; index += 1) {
          const y = box.top + box.height * (0.12 + index * 0.3);
          ctx.fillRect(box.left + box.width * 0.28, y,
                       box.width * 0.44, box.height * 0.1);
        }
        ctx.globalAlpha = 1;
      }

      // The jambs, so it reads as a doorway rather than as two blocks.
      ctx.fillStyle = palette.muted;
      const jamb = Math.max(1, unit * 0.06);
      ctx.fillRect(box.left, box.top, jamb, box.height);
      ctx.fillRect(box.left + box.width - jamb, box.top, jamb, box.height);
      ctx.restore();
    },

    /**
     * A sentry: a dome on a track, with one eye.
     *
     * Deliberately unlike the agent's silhouette — no head, no arms, wider
     * than it is tall — because the one thing that must never happen is
     * mistaking the guard for R-5 at a glance.
     */
    guard(ctx, box, look, unit, palette) {
      const centreX = box.left + box.width / 2;
      const centreY = box.top + box.height / 2;
      const radius = Math.min(box.width, box.height) * 0.34;

      ctx.save();
      withGlow(ctx, look.color, glowFor('agent'), () => {
        ctx.fillStyle = look.color;
        // The dome.
        ctx.beginPath();
        ctx.arc(centreX, centreY - radius * 0.1, radius, Math.PI, 0);
        ctx.closePath();
        ctx.fill();
        // The track it runs on.
        ctx.fillRect(centreX - radius * 1.15, centreY + radius * 0.1,
                     radius * 2.3, radius * 0.55);
      });

      // One eye, dead centre, which is the whole character of the thing.
      if (unit >= 10) {
        ctx.fillStyle = palette.base;
        ctx.beginPath();
        ctx.arc(centreX, centreY - radius * 0.35, radius * 0.3, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();
    },

    /**
     * The draught from a vent: the lane it blows down, and which way.
     *
     * Drawn under everything else, as a wash with chevrons repeated along
     * it. Wind is the one thing in any of these rooms that acts on the
     * agent from a distance without being anywhere near it, so leaving it
     * invisible means the drone is shoved by nothing at all.
     */
    stream(ctx, box, look, unit, palette) {
      const towards = { right: 0, down: Math.PI / 2, left: Math.PI,
                        up: -Math.PI / 2 }[look.blows] || 0;
      const sideways = look.blows === 'left' || look.blows === 'right';

      ctx.save();

      // The lane itself, barely there: it is a hint, not a wall.
      ctx.fillStyle = look.color;
      ctx.globalAlpha = 0.05;
      ctx.fillRect(box.left, box.top, box.width, box.height);

      // Its edges, so where the wind stops is visible.
      ctx.strokeStyle = look.color;
      ctx.globalAlpha = 0.22;
      ctx.lineWidth = Math.max(0.6, unit * 0.02);
      ctx.setLineDash([unit * 0.25, unit * 0.3]);
      ctx.beginPath();
      if (sideways) {
        ctx.moveTo(box.left, box.top);
        ctx.lineTo(box.left + box.width, box.top);
        ctx.moveTo(box.left, box.top + box.height);
        ctx.lineTo(box.left + box.width, box.top + box.height);
      } else {
        ctx.moveTo(box.left, box.top);
        ctx.lineTo(box.left, box.top + box.height);
        ctx.moveTo(box.left + box.width, box.top);
        ctx.lineTo(box.left + box.width, box.top + box.height);
      }
      ctx.stroke();
      ctx.setLineDash([]);

      // Chevrons down the lane, pointing downwind. Two files of them, off
      // centre, so the lane reads as moving air rather than as a corridor.
      const length = sideways ? box.width : box.height;
      const spacing = Math.max(unit * 0.9, 18);
      const size = Math.min(unit * 0.28, spacing * 0.4);
      ctx.globalAlpha = 0.4;
      ctx.lineWidth = Math.max(1, unit * 0.035);
      ctx.lineCap = 'round';
      ctx.beginPath();
      [0.32, 0.68].forEach(offset => {
        for (let at = spacing * 0.5; at < length; at += spacing) {
          const x = sideways ? box.left + at : box.left + box.width * offset;
          const y = sideways ? box.top + box.height * offset : box.top + at;
          [2.4, -2.4].forEach(spread => {
            ctx.moveTo(x, y);
            ctx.lineTo(x + Math.cos(towards + spread) * size,
                       y + Math.sin(towards + spread) * size);
          });
        }
      });
      ctx.stroke();
      ctx.restore();
      void palette;
    },

    /**
     * A vent: a grille with chevrons showing which way it blows.
     *
     * `look.blows` is 'right', 'left', 'up' or 'down'. The chevrons point
     * that way, because a fan that does not say which way it pushes is
     * just a box on a wall.
     */
    vent(ctx, box, look, unit, palette) {
      const towards = { right: 0, down: Math.PI / 2, left: Math.PI,
                        up: -Math.PI / 2 }[look.blows] || 0;

      ctx.save();
      ctx.fillStyle = look.color;
      ctx.fillRect(box.left, box.top, box.width, box.height);

      // The grille: slats across the short axis of the housing.
      ctx.fillStyle = palette.base;
      ctx.globalAlpha = 0.6;
      const across = box.width > box.height;
      const slats = 4;
      for (let index = 0; index < slats; index += 1) {
        if (across) {
          ctx.fillRect(box.left + box.width * (0.12 + index * 0.22), box.top,
                       box.width * 0.09, box.height);
        } else {
          ctx.fillRect(box.left, box.top + box.height * (0.12 + index * 0.22),
                       box.width, box.height * 0.09);
        }
      }
      ctx.globalAlpha = 1;

      // Two chevrons in the draught, pointing downwind.
      if (unit >= 10) {
        const centreX = box.left + box.width / 2;
        const centreY = box.top + box.height / 2;
        const reach = Math.min(box.width, box.height) * 0.42;
        ctx.strokeStyle = palette.accent;
        ctx.globalAlpha = 0.75;
        ctx.lineWidth = Math.max(1, unit * 0.05);
        ctx.lineCap = 'round';
        ctx.beginPath();
        [0.15, 0.62].forEach(offset => {
          const tipX = centreX + Math.cos(towards) * reach * (offset + 0.5);
          const tipY = centreY + Math.sin(towards) * reach * (offset + 0.5);
          [2.5, -2.5].forEach(spread => {
            ctx.moveTo(tipX, tipY);
            ctx.lineTo(tipX + Math.cos(towards + spread) * reach * 0.42,
                       tipY + Math.sin(towards + spread) * reach * 0.42);
          });
        });
        ctx.stroke();
        ctx.globalAlpha = 1;
      }
      ctx.restore();
    },

    /** A crate: a block with a cross-brace. */
    obstacle(ctx, box, look, unit, palette) {
      ctx.save();
      ctx.fillStyle = look.color;
      ctx.fillRect(box.left, box.top, box.width, box.height);
      ctx.strokeStyle = palette.base;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = Math.max(0.5, unit * 0.03);
      ctx.beginPath();
      ctx.moveTo(box.left, box.top);
      ctx.lineTo(box.left + box.width, box.top + box.height);
      ctx.moveTo(box.left + box.width, box.top);
      ctx.lineTo(box.left, box.top + box.height);
      ctx.stroke();
      ctx.restore();
    },

    /**
     * The agent, filling a box.
     *
     * The positional form below is what the renderers use while drawing a
     * frame, because an agent is placed by its centre and may sit between
     * two cells. This is for the times when it is one item among others in
     * a box, which in practice means the legend.
     */
    agent(ctx, box, look, unit, palette) {
      drawAgent(ctx,
                { x: box.left + box.width / 2, y: box.top + box.height / 2 },
                Math.min(box.width, box.height) * 0.8, null, palette);
    },

    /** A landing pad: a bar with two markers on it. */
    pad(ctx, box, look, unit, palette) {
      ctx.save();
      withGlow(ctx, look.color, glowFor('exit'), () => {
        ctx.fillStyle = look.color;
        ctx.fillRect(box.left, box.top, box.width, box.height);
      });
      ctx.fillStyle = palette.base;
      const mark = box.width * 0.12;
      ctx.fillRect(box.left + box.width * 0.24, box.top, mark, box.height);
      ctx.fillRect(box.left + box.width * 0.64, box.top, mark, box.height);
      ctx.restore();
    },

    /* -----------------------------------------------------------------
       Room 4 — the wind tunnel.

       Five of these are the chamber's real furniture and two are drawn
       explanations of rules that already exist. None of them decides
       anything: `look.blows` and `look.strength` arrive worked out from the
       wind vector on the Python side, and `look.phase` is the animation
       clock the renderer hands down. A recipe that computed its own wind
       direction could disagree with the model about which way the air goes,
       which is the one thing this split exists to prevent.
       ----------------------------------------------------------------- */

    /* No `tunnelWall` recipe any more. It drew the shell of rooms 4 and 5
       as a thin stroked frame with corner brackets and measurement ticks,
       which is why those chambers read as a pressure vessel rather than as
       part of the same stone laboratory as rooms 1 to 3. Both rooms now
       state a ring of ordinary `wall` tiles outside their boundary, so the
       masonry above draws them and there is nothing separate to keep in
       step. See `chamber_wall_entities` in `game/drone.py`. */

    /**
     * An industrial ventilation fan: housing, blades, hub, and a draught.
     *
     * The blades turn with `look.phase`, which is the renderer's clock. The
     * rotation is decoration in the strict sense — the wind zone in front of
     * the fan is what applies the force, and this would still be a fan with
     * the animation switched off. Reduced motion pins the phase, and then the
     * blades simply stand still.
     */
    fan(ctx, box, look, unit, palette) {
      const towards = BLOWS[look.blows] === undefined ? Math.PI / 2
                                                      : BLOWS[look.blows];
      const centreX = box.left + box.width / 2;
      const centreY = box.top + box.height / 2;
      const radius = Math.min(box.width, box.height) / 2;
      const strength = look.strength === undefined ? 1 : look.strength;
      // A stopped fan is a fan on a chamber with the wind turned down to zero,
      // and it should look stopped.
      const spin = (look.phase || 0) * (1.4 + strength * 2.2);

      ctx.save();

      // Housing: a dark ring, with the duct mouth behind it.
      ctx.fillStyle = palette.base;
      ctx.globalAlpha = 0.9;
      ctx.beginPath();
      ctx.arc(centreX, centreY, radius, 0, Math.PI * 2);
      ctx.fill();

      ctx.globalAlpha = 1;
      ctx.strokeStyle = look.color;
      ctx.lineWidth = Math.max(1.4, radius * 0.16);
      ctx.beginPath();
      ctx.arc(centreX, centreY, radius * 0.9, 0, Math.PI * 2);
      ctx.stroke();

      // Blades. Below about eight pixels of radius they are a smudge, so the
      // fan becomes a plain ringed disc instead of a grey mess.
      if (radius >= 8) {
        const blades = 5;
        ctx.fillStyle = look.color;
        ctx.globalAlpha = 0.75;
        for (let index = 0; index < blades; index += 1) {
          const angle = spin + index * (Math.PI * 2 / blades);
          ctx.beginPath();
          ctx.moveTo(centreX, centreY);
          ctx.arc(centreX, centreY, radius * 0.78,
                  angle, angle + Math.PI * 2 / blades * 0.52);
          ctx.closePath();
          ctx.fill();
        }
        ctx.globalAlpha = 1;
      }

      // The hub, lit: the one accent on the machine.
      withGlow(ctx, palette.accent, glowFor('agent') * 0.5, () => {
        ctx.fillStyle = palette.accent;
        ctx.beginPath();
        ctx.arc(centreX, centreY, Math.max(1.5, radius * 0.17), 0, Math.PI * 2);
        ctx.fill();
      });

      // The draught leaving it, so which way it blows is readable without
      // reference to the lane it feeds.
      if (radius >= 7) {
        ctx.strokeStyle = palette.accent;
        ctx.globalAlpha = 0.5;
        ctx.lineWidth = Math.max(1, radius * 0.1);
        ctx.lineCap = 'round';
        ctx.beginPath();
        [-0.42, 0, 0.42].forEach(across => {
          const sideX = Math.cos(towards + Math.PI / 2) * radius * across;
          const sideY = Math.sin(towards + Math.PI / 2) * radius * across;
          const from = radius * 1.0;
          const to = radius * (1.15 + 0.5 * strength);
          ctx.moveTo(centreX + sideX + Math.cos(towards) * from,
                     centreY + sideY + Math.sin(towards) * from);
          ctx.lineTo(centreX + sideX + Math.cos(towards) * to,
                     centreY + sideY + Math.sin(towards) * to);
        });
        ctx.stroke();
      }
      ctx.restore();
    },

    /**
     * A turbine housing: the pillars, which are solid and fatal.
     *
     * Drawn as a disc filling the width of its box, because the box handed
     * over is the square that bounds the collision circle. Anything drawn
     * inside that is smaller than the thing the environment tests, which is
     * the one error this shape must not make: a drone that looks clear of the
     * housing and crashes anyway reads as a bug in the physics.
     */
    turbine(ctx, box, look, unit, palette) {
      const centreX = box.left + box.width / 2;
      const centreY = box.top + box.height / 2;
      const radius = box.width / 2;

      ctx.save();

      // The shadow under it, which is what makes it read as standing in the
      // chamber rather than painted on the floor.
      ctx.fillStyle = palette.base;
      ctx.globalAlpha = 0.45;
      ctx.beginPath();
      ctx.arc(centreX + radius * 0.1, centreY + radius * 0.12, radius,
              0, Math.PI * 2);
      ctx.fill();

      // The body, out to the full collision radius.
      ctx.globalAlpha = 1;
      ctx.fillStyle = look.color;
      ctx.beginPath();
      ctx.arc(centreX, centreY, radius, 0, Math.PI * 2);
      ctx.fill();

      // A darker inner ring and panel seams: machinery, not a dot.
      ctx.strokeStyle = palette.base;
      ctx.globalAlpha = 0.5;
      ctx.lineWidth = Math.max(1, radius * 0.1);
      ctx.beginPath();
      ctx.arc(centreX, centreY, radius * 0.62, 0, Math.PI * 2);
      ctx.stroke();

      if (radius >= 9) {
        ctx.lineWidth = Math.max(0.8, radius * 0.06);
        ctx.beginPath();
        for (let index = 0; index < 6; index += 1) {
          const angle = index * Math.PI / 3;
          ctx.moveTo(centreX + Math.cos(angle) * radius * 0.62,
                     centreY + Math.sin(angle) * radius * 0.62);
          ctx.lineTo(centreX + Math.cos(angle) * radius * 0.98,
                     centreY + Math.sin(angle) * radius * 0.98);
        }
        ctx.stroke();

        // Hazard stripes across the hub, and a lamp on it.
        ctx.globalAlpha = 0.32;
        ctx.strokeStyle = palette.hazard;
        ctx.lineWidth = Math.max(1, radius * 0.14);
        ctx.beginPath();
        [-0.3, 0.1].forEach(offset => {
          ctx.moveTo(centreX - radius * 0.42, centreY + radius * offset);
          ctx.lineTo(centreX + radius * 0.42, centreY + radius * offset);
        });
        ctx.stroke();
      }

      ctx.globalAlpha = 1;
      ctx.fillStyle = palette.hazard;
      ctx.beginPath();
      ctx.arc(centreX, centreY, Math.max(1.2, radius * 0.13), 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
      void unit;
    },

    /**
     * The stabilisation field: dense air that costs speed, not life.
     *
     * Deliberately the calmest thing in the chamber. It is the one region that
     * *helps* — it is what makes a gentle arrival easy — so it must not be
     * dressed as a hazard. Concentric marks and short slow streaks, and no red
     * anywhere near it.
     */
    stabiliser(ctx, box, look, unit, palette) {
      ctx.save();
      ctx.fillStyle = look.color;
      ctx.globalAlpha = 0.07;
      ctx.fillRect(box.left, box.top, box.width, box.height);

      // A soft boundary, so where the help starts is visible.
      ctx.strokeStyle = look.color;
      ctx.globalAlpha = 0.26;
      ctx.lineWidth = Math.max(0.8, unit * 0.025);
      ctx.setLineDash([unit * 0.18, unit * 0.22]);
      ctx.strokeRect(box.left, box.top, box.width, box.height);
      ctx.setLineDash([]);

      const centreX = box.left + box.width / 2;
      const centreY = box.top + box.height / 2;
      const reach = Math.min(box.width, box.height) * 0.42;

      // Concentric rings: the stabiliser mark.
      ctx.globalAlpha = 0.3;
      ctx.lineWidth = Math.max(0.8, unit * 0.03);
      ctx.beginPath();
      [0.45, 0.75, 1.0].forEach(share => {
        ctx.moveTo(centreX + reach * share, centreY);
        ctx.arc(centreX, centreY, reach * share, 0, Math.PI * 2);
      });
      ctx.stroke();

      // Short streaks being damped: they shorten towards the middle, which is
      // the whole idea of the field in one mark.
      if (unit >= 12) {
        ctx.globalAlpha = 0.34;
        ctx.lineCap = 'round';
        ctx.lineWidth = Math.max(1, unit * 0.035);
        ctx.beginPath();
        for (let index = 0; index < 8; index += 1) {
          const angle = index * Math.PI / 4;
          const from = reach * 1.15;
          const to = reach * (1.15 - 0.3);
          ctx.moveTo(centreX + Math.cos(angle) * from,
                     centreY + Math.sin(angle) * from);
          ctx.lineTo(centreX + Math.cos(angle) * to,
                     centreY + Math.sin(angle) * to);
        }
        ctx.stroke();
      }
      ctx.restore();
      void palette;
    },

    /**
     * The thruster overcharge: doubles the push, and charges for the privilege.
     *
     * Read as risk rather than as death — it is not a wall and does not end a
     * run, it just makes the drone much harder to slow down. Amber-red field
     * with acceleration streaks running outwards, plainly a different kind of
     * thing from the calm blue of the stabiliser.
     */
    overcharge(ctx, box, look, unit, palette) {
      ctx.save();
      ctx.fillStyle = look.color;
      ctx.globalAlpha = 0.09;
      ctx.fillRect(box.left, box.top, box.width, box.height);

      // Hazard striping along the edges, which is how every other room in this
      // project says "this one is on you".
      ctx.strokeStyle = look.color;
      ctx.globalAlpha = 0.45;
      ctx.lineWidth = Math.max(1, unit * 0.05);
      ctx.beginPath();
      ctx.moveTo(box.left, box.top);
      ctx.lineTo(box.left + box.width, box.top);
      ctx.moveTo(box.left, box.top + box.height);
      ctx.lineTo(box.left + box.width, box.top + box.height);
      ctx.stroke();

      // Diagonal stripes, clipped to the field.
      ctx.save();
      ctx.beginPath();
      ctx.rect(box.left, box.top, box.width, box.height);
      ctx.clip();
      ctx.globalAlpha = 0.13;
      ctx.lineWidth = Math.max(1.5, unit * 0.1);
      const step = Math.max(unit * 0.42, 9);
      ctx.beginPath();
      for (let at = -box.height; at < box.width + box.height; at += step) {
        ctx.moveTo(box.left + at, box.top);
        ctx.lineTo(box.left + at + box.height, box.top + box.height);
      }
      ctx.stroke();
      ctx.restore();

      // Acceleration marks: pairs of chevrons opening outwards, pulsing on the
      // renderer's clock.
      const centreX = box.left + box.width / 2;
      const centreY = box.top + box.height / 2;
      const pulse = 0.5 + 0.5 * Math.sin((look.phase || 0) * 2.2);
      if (unit >= 12) {
        ctx.strokeStyle = look.color;
        ctx.globalAlpha = 0.35 + 0.25 * pulse;
        ctx.lineWidth = Math.max(1, unit * 0.045);
        ctx.lineCap = 'round';
        const size = Math.min(box.width, box.height) * 0.16;
        ctx.beginPath();
        [-1, 1].forEach(side => {
          const y = centreY + side * Math.min(box.height * 0.26, unit * 0.7);
          [-0.5, 0.5].forEach(offset => {
            const tipY = y + side * size * (0.4 + pulse * 0.3);
            const x = centreX + offset * size * 2.2;
            ctx.moveTo(x - size * 0.7, tipY - side * size * 0.7);
            ctx.lineTo(x, tipY);
            ctx.lineTo(x + size * 0.7, tipY - side * size * 0.7);
          });
        });
        ctx.stroke();
      }
      ctx.restore();
      void palette;
    },

    /**
     * A status lamp. Small, and the only thing in the room that blinks.
     *
     * `look.color` is overridden per entity, so the same recipe is the green
     * lamp beside the platform and the red one over the overcharge.
     */
    warningLight(ctx, box, look, unit, palette) {
      const centreX = box.left + box.width / 2;
      const centreY = box.top + box.height / 2;
      const radius = Math.min(box.width, box.height) / 2;
      const pulse = 0.45 + 0.55 * Math.abs(Math.sin((look.phase || 0) * 1.6));

      ctx.save();
      // The fitting, so it is a lamp on something and not a floating dot.
      ctx.fillStyle = palette.base;
      ctx.globalAlpha = 0.8;
      ctx.beginPath();
      ctx.arc(centreX, centreY, radius, 0, Math.PI * 2);
      ctx.fill();

      ctx.globalAlpha = 1;
      withGlow(ctx, look.color, glowFor('exit') * pulse, () => {
        ctx.fillStyle = look.color;
        ctx.globalAlpha = 0.5 + 0.5 * pulse;
        ctx.beginPath();
        ctx.arc(centreX, centreY, radius * 0.6, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.restore();
      void unit;
    },

    /* ---- the landing platform, in its four states ------------------- */

    platform(ctx, box, look, unit, palette) {
      drawPlatform(ctx, box, look, unit, palette, 'clear');
    },
    platformWarning(ctx, box, look, unit, palette) {
      drawPlatform(ctx, box, look, unit, palette, 'fast');
    },
    platformLanded(ctx, box, look, unit, palette) {
      drawPlatform(ctx, box, look, unit, palette, 'landed');
    },
    platformCrashed(ctx, box, look, unit, palette) {
      drawPlatform(ctx, box, look, unit, palette, 'crashed');
    },

    /* -----------------------------------------------------------------
       Room 5 — the adaptive warehouse.

       The colour language is fixed and every recipe obeys it: the wall grey
       is architecture, cyan is R-5 and the systems on its side, red is
       something that can end the run, green is a system that has been
       disarmed or unlocked. Nothing here changes colour except by changing
       *state*, and only the environment sets a state — so a green light
       always means the terminal was reached, never that a timer elapsed.
       ----------------------------------------------------------------- */

    /**
     * A storage shelf: uprights, cross-braces and stacked cargo.
     *
     * Drawn to the exact rectangle the collision test uses, so nothing looks
     * passable that is not. The bays are laid out along whichever axis is
     * longer, which is what makes a tall rack and a long rack read as the same
     * object seen two ways rather than as two different things.
     */
    shelf(ctx, box, look, unit, palette) {
      const along = box.width >= box.height;
      const span = along ? box.width : box.height;
      const bays = Math.max(2, Math.round(span / Math.max(unit * 0.55, 12)));

      ctx.save();
      // The shadow, so the rack stands on the floor.
      ctx.globalAlpha = 0.4;
      ctx.fillStyle = palette.base;
      ctx.fillRect(box.left + unit * 0.05, box.top + unit * 0.06,
                   box.width, box.height);

      // The frame.
      ctx.globalAlpha = 1;
      ctx.fillStyle = look.color;
      ctx.fillRect(box.left, box.top, box.width, box.height);

      // Bays: dark gaps with cargo sitting in them, so it reads as a rack
      // full of boxes rather than a solid block.
      ctx.fillStyle = palette.base;
      ctx.globalAlpha = 0.62;
      for (let index = 1; index < bays; index += 1) {
        const at = span * (index / bays);
        if (along) ctx.fillRect(box.left + at - unit * 0.02, box.top,
                                Math.max(1, unit * 0.045), box.height);
        else ctx.fillRect(box.left, box.top + at - unit * 0.02,
                          box.width, Math.max(1, unit * 0.045));
      }

      if (unit >= 14) {
        // Cargo in about two bays in three, inset so the frame still shows.
        ctx.globalAlpha = 0.5;
        ctx.fillStyle = palette.muted;
        for (let index = 0; index < bays; index += 1) {
          if (index % 3 === 2) continue;
          const from = span * (index / bays);
          const size = span / bays;
          const inset = Math.min(box.width, box.height) * 0.26;
          if (along) {
            ctx.fillRect(box.left + from + size * 0.2, box.top + inset,
                         size * 0.6, box.height - inset * 2);
          } else {
            ctx.fillRect(box.left + inset, box.top + from + size * 0.2,
                         box.width - inset * 2, size * 0.6);
          }
        }
      }

      // A hazard stripe along the foot of the rack: warehouse floor marking.
      ctx.globalAlpha = 0.3;
      ctx.strokeStyle = palette.muted;
      ctx.lineWidth = Math.max(1, unit * 0.04);
      ctx.strokeRect(box.left, box.top, box.width, box.height);
      ctx.restore();
    },

    /** A cargo unit on a conveyor. Small, and plainly not a hazard. */
    crate(ctx, box, look, unit, palette) {
      ctx.save();
      ctx.globalAlpha = 0.85;
      ctx.fillStyle = look.color;
      ctx.fillRect(box.left, box.top, box.width, box.height);
      ctx.globalAlpha = 0.45;
      ctx.strokeStyle = palette.base;
      ctx.lineWidth = Math.max(0.8, unit * 0.025);
      ctx.beginPath();
      ctx.moveTo(box.left, box.top);
      ctx.lineTo(box.left + box.width, box.top + box.height);
      ctx.moveTo(box.left + box.width, box.top);
      ctx.lineTo(box.left, box.top + box.height);
      ctx.stroke();
      ctx.restore();
    },

    /**
     * A conveyor lane: a recessed channel with rollers across it.
     *
     * Deliberately quiet. It carries cargo past and it cannot hurt anything,
     * so it is drawn below everything else in the palest tone the room has.
     */
    conveyor(ctx, box, look, unit, palette) {
      const horizontal = look.horizontal !== false;
      ctx.save();
      ctx.globalAlpha = 0.14;
      ctx.fillStyle = look.color;
      ctx.fillRect(box.left, box.top, box.width, box.height);

      ctx.globalAlpha = 0.3;
      ctx.strokeStyle = look.color;
      ctx.lineWidth = Math.max(0.6, unit * 0.018);
      ctx.beginPath();
      const span = horizontal ? box.width : box.height;
      const step = Math.max(unit * 0.3, 7);
      for (let at = step / 2; at < span; at += step) {
        if (horizontal) {
          ctx.moveTo(box.left + at, box.top);
          ctx.lineTo(box.left + at, box.top + box.height);
        } else {
          ctx.moveTo(box.left, box.top + at);
          ctx.lineTo(box.left + box.width, box.top + at);
        }
      }
      ctx.stroke();
      ctx.restore();
      void palette;
    },

    /**
     * A security drone: the one thing in the room that hunts.
     *
     * Read as a machine on a circuit, not as a red dot — a dark housing, a
     * rotor ring, and a single red eye that pulses. Drawn to the full 0.5 m
     * collision width, so what is on screen is what kills.
     */
    cart(ctx, box, look, unit, palette) {
      const centreX = box.left + box.width / 2;
      const centreY = box.top + box.height / 2;
      const radius = Math.min(box.width, box.height) / 2;
      const phase = look.phase || 0;
      const pulse = 0.5 + 0.5 * Math.sin(phase * 3.2);

      ctx.save();
      // Shadow under it.
      ctx.globalAlpha = 0.4;
      ctx.fillStyle = palette.base;
      ctx.beginPath();
      ctx.ellipse(centreX + radius * 0.12, centreY + radius * 0.5,
                  radius * 0.85, radius * 0.3, 0, 0, Math.PI * 2);
      ctx.fill();

      // Housing.
      ctx.globalAlpha = 1;
      ctx.fillStyle = palette.muted;
      ctx.beginPath();
      ctx.arc(centreX, centreY, radius, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = palette.base;
      ctx.globalAlpha = 0.55;
      ctx.lineWidth = Math.max(1, radius * 0.24);
      ctx.beginPath();
      ctx.arc(centreX, centreY, radius * 0.72, 0, Math.PI * 2);
      ctx.stroke();

      // The eye, and its warning glow.
      ctx.globalAlpha = 1;
      withGlow(ctx, look.color, glowFor('laser') * (0.4 + 0.5 * pulse), () => {
        ctx.fillStyle = look.color;
        ctx.beginPath();
        ctx.arc(centreX, centreY, Math.max(1.2, radius * 0.36), 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.restore();
    },

    /**
     * A lit security beam, along the diagonal of the box it is given.
     *
     * `look.downhill` says which diagonal, and it comes from the endpoints the
     * collision test uses — so the line drawn is the line that kills. Getting
     * that wrong is the one mistake this shape must not make.
     */
    laser(ctx, box, look, unit, palette) {
      drawBeam(ctx, box, look, unit, palette, true);
    },
    laserIdle(ctx, box, look, unit, palette) {
      drawBeam(ctx, box, look, unit, palette, false);
    },

    /** An alarm beacon: a small dome that throws a wash of its own colour. */
    beacon(ctx, box, look, unit, palette) {
      drawBeacon(ctx, box, look, unit, palette, true);
    },
    beaconDim(ctx, box, look, unit, palette) {
      drawBeacon(ctx, box, look, unit, palette, false);
    },

    /* ---- the control terminal, armed and spent ---------------------- */

    terminal(ctx, box, look, unit, palette) {
      drawTerminal(ctx, box, look, unit, palette, false);
    },
    terminalUsed(ctx, box, look, unit, palette) {
      drawTerminal(ctx, box, look, unit, palette, true);
    },

    /* ---- the blast door, locked and open ---------------------------- */

    blastdoor(ctx, box, look, unit, palette) {
      drawBlastDoor(ctx, box, look, unit, palette, true);
    },
    blastdoorLocked(ctx, box, look, unit, palette) {
      drawBlastDoor(ctx, box, look, unit, palette, false);
    },
  };

  /** Which way a compass word points, in radians. y grows downwards. */
  const BLOWS = { right: 0, down: Math.PI / 2, left: Math.PI,
                  up: -Math.PI / 2 };

  /**
   * A security beam and its two emitters.
   *
   * `lit` is the state the *room* reported, never a timer this file keeps.
   * A dark beam still draws its emitters and a hairline between them, because
   * the timing is only learnable if you can see where a beam is about to be —
   * a beam that vanished completely would make the room a memory test.
   */
  function drawBeam(ctx, box, look, unit, palette, lit) {
    const downhill = look.downhill !== false;
    const from = { x: box.left, y: downhill ? box.top : box.top + box.height };
    const to = { x: box.left + box.width,
                 y: downhill ? box.top + box.height : box.top };
    const phase = look.phase || 0;

    ctx.save();
    if (lit) {
      // The beam, with a soft flare along it.
      const flicker = 0.82 + 0.18 * Math.sin(phase * 9);
      withGlow(ctx, look.color, glowFor('laser') * flicker, () => {
        ctx.strokeStyle = look.color;
        ctx.globalAlpha = flicker;
        ctx.lineWidth = Math.max(1.4, unit * 0.055);
        ctx.beginPath();
        ctx.moveTo(from.x, from.y);
        ctx.lineTo(to.x, to.y);
        ctx.stroke();
      });
    } else {
      // Unpowered: the rail the beam will run along, and nothing more.
      ctx.strokeStyle = look.color;
      ctx.globalAlpha = 0.28;
      ctx.lineWidth = Math.max(0.6, unit * 0.02);
      ctx.setLineDash([unit * 0.16, unit * 0.2]);
      ctx.beginPath();
      ctx.moveTo(from.x, from.y);
      ctx.lineTo(to.x, to.y);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    /* The emitters, which are there whether the beam is or not — each on a
       short stone stanchion bricked from the same masonry as the laboratory
       wall, so a beam reads as bolted to the architecture rather than
       floating in the air. The housing was already drawn at both ends; this
       is the same footprint, textured as stonework and given a cap. */
    ctx.globalAlpha = 1;
    const housing = Math.max(3, unit * 0.16);
    [from, to].forEach(end => {
      const pillar = {
        left: end.x - housing / 2,
        top: end.y - housing / 2,
        width: housing,
        height: housing,
      };
      // Two courses of one brick: any finer and the coursing is mush at this
      // size, and the point is only that it is plainly the same stone.
      layBricks(ctx, pillar, palette.cellWall || palette.muted, unit,
                palette, 2, 1);

      // The cap the emitter head sits in.
      ctx.globalAlpha = 0.9;
      ctx.strokeStyle = palette.muted;
      ctx.lineWidth = Math.max(0.8, unit * 0.018);
      ctx.strokeRect(pillar.left, pillar.top, pillar.width, pillar.height);

      // The head itself: lit with the beam, dark without it.
      ctx.globalAlpha = lit ? 1 : 0.55;
      ctx.fillStyle = lit ? look.color : palette.base;
      withGlow(ctx, look.color, lit ? glowFor('laser') * 0.5 : 0, () => {
        ctx.beginPath();
        ctx.arc(end.x, end.y, housing * 0.24, 0, Math.PI * 2);
        ctx.fill();
      });
      ctx.globalAlpha = 1;
    });
    ctx.restore();
  }

  /** An alarm beacon. `lit` comes from the room, and its colour with it. */
  function drawBeacon(ctx, box, look, unit, palette, lit) {
    const centreX = box.left + box.width / 2;
    const centreY = box.top + box.height / 2;
    const radius = Math.min(box.width, box.height) / 2;
    const pulse = 0.55 + 0.45 * Math.abs(Math.sin((look.phase || 0) * 2.2));

    ctx.save();
    ctx.fillStyle = palette.base;
    ctx.globalAlpha = 0.85;
    ctx.beginPath();
    ctx.arc(centreX, centreY, radius, 0, Math.PI * 2);
    ctx.fill();

    if (lit) {
      // A wash of its own colour on the floor around it, which is what makes a
      // lamp read as lighting the room rather than as a coloured dot.
      ctx.globalAlpha = 0.1 + 0.1 * pulse;
      ctx.fillStyle = look.color;
      ctx.beginPath();
      ctx.arc(centreX, centreY, radius * 2.6, 0, Math.PI * 2);
      ctx.fill();
    }

    ctx.globalAlpha = lit ? 0.6 + 0.4 * pulse : 0.4;
    withGlow(ctx, look.color, lit ? glowFor('exit') * pulse : 0, () => {
      ctx.fillStyle = look.color;
      ctx.beginPath();
      ctx.arc(centreX, centreY, radius * 0.55, 0, Math.PI * 2);
      ctx.fill();
    });
    ctx.restore();
    void unit;
  }

  /**
   * The control terminal: a console with a screen.
   *
   * Armed it is cyan and its screen scrolls; spent it is green and steady with
   * a tick on it. `used` is the room's word — this file never decides that the
   * security system has been disarmed.
   */
  function drawTerminal(ctx, box, look, unit, palette, used) {
    const phase = look.phase || 0;
    const pulse = 0.5 + 0.5 * Math.sin(phase * (used ? 1.2 : 2.6));

    ctx.save();
    // Shadow, then the console body: a plinth with a raked screen on it.
    ctx.globalAlpha = 0.4;
    ctx.fillStyle = palette.base;
    ctx.fillRect(box.left + unit * 0.04, box.top + unit * 0.05,
                 box.width, box.height);

    /* The pier the console is mounted on, bricked from the laboratory's own
       masonry — so the terminal reads as a wall-mounted console in a stone
       building rather than as a free-standing kiosk. Same footprint the
       plinth already occupied; only the texture changed, so nothing here
       claims any geometry the physics does not have. */
    ctx.globalAlpha = 1;
    const plinth = box.height * 0.34;
    const pier = {
      left: box.left, top: box.top + box.height - plinth,
      width: box.width, height: plinth,
    };
    layBricks(ctx, pier, palette.cellWall || palette.muted, unit, palette,
              2, 3);
    // A capping course along the top of the pier, which is what the console
    // is bolted to.
    ctx.globalAlpha = 0.85;
    ctx.fillStyle = palette.muted;
    ctx.fillRect(pier.left, pier.top, pier.width,
                 Math.max(1.2, unit * 0.03));

    // The screen.
    const screen = {
      left: box.left + box.width * 0.12,
      top: box.top + box.height * 0.1,
      width: box.width * 0.76,
      height: box.height * 0.52,
    };
    ctx.fillStyle = palette.base;
    ctx.fillRect(screen.left, screen.top, screen.width, screen.height);

    withGlow(ctx, look.color, glowFor('exit') * (0.6 + 0.6 * pulse), () => {
      ctx.strokeStyle = look.color;
      ctx.lineWidth = Math.max(1.2, unit * 0.045);
      ctx.strokeRect(screen.left, screen.top, screen.width, screen.height);
    });

    if (unit >= 12) {
      ctx.globalAlpha = 0.75;
      ctx.strokeStyle = look.color;
      ctx.lineWidth = Math.max(1, unit * 0.028);
      ctx.beginPath();
      if (used) {
        // A tick: the system is down.
        ctx.moveTo(screen.left + screen.width * 0.24,
                   screen.top + screen.height * 0.52);
        ctx.lineTo(screen.left + screen.width * 0.44,
                   screen.top + screen.height * 0.74);
        ctx.lineTo(screen.left + screen.width * 0.78,
                   screen.top + screen.height * 0.26);
      } else {
        // Scrolling readout lines, running on the room's clock.
        const rows = 3;
        for (let index = 0; index < rows; index += 1) {
          const drift = ((phase * 0.25 + index * 0.33) % 1.0);
          const y = screen.top + screen.height * (0.2 + 0.28 * index);
          const width = screen.width * (0.3 + 0.45 * drift);
          ctx.moveTo(screen.left + screen.width * 0.12, y);
          ctx.lineTo(screen.left + screen.width * 0.12 + width, y);
        }
      }
      ctx.stroke();
    }

    // Status lamps on the plinth.
    ctx.globalAlpha = used ? 1 : 0.4 + 0.5 * pulse;
    ctx.fillStyle = look.color;
    const lamp = Math.max(1, unit * 0.05);
    [0.3, 0.5, 0.7].forEach(share => {
      ctx.fillRect(box.left + box.width * share - lamp / 2,
                   box.top + box.height - plinth * 0.6, lamp, lamp);
    });

    // The activation pulse: a ring thrown outwards for a moment after the
    // terminal is reached.
    //
    // `look.sinceActivation` is how many recorded frames ago the *environment*
    // reported the stage change — counted from the recording, never from a
    // clock of this file's own. Absent, or long past, and nothing is drawn.
    // That is what stops the flourish from ever appearing on a terminal that
    // was not actually activated.
    const since = look.sinceActivation;
    if (used && typeof since === 'number' && since >= 0 && since < PULSE_FRAMES) {
      const through = since / PULSE_FRAMES;
      ctx.globalAlpha = (1 - through) * 0.8;
      ctx.strokeStyle = look.color;
      ctx.lineWidth = Math.max(1.5, unit * 0.05);
      withGlow(ctx, look.color, glowFor('exit') * 2 * (1 - through), () => {
        ctx.beginPath();
        ctx.arc(box.left + box.width / 2, box.top + box.height / 2,
                Math.max(box.width, box.height) * (0.5 + through * 0.9),
                0, Math.PI * 2);
        ctx.stroke();
      });
    }
    ctx.restore();

    // What it is, and what state the security system is in. Two lines, so the
    // objective and its consequence are both stated rather than implied by a
    // colour the player has to have been told the meaning of.
    drawLabel(ctx, box, used ? 'SECURITY DISABLED' : 'CONTROL TERMINAL',
              look.color, unit, palette);
    if (!used) {
      drawLabel(ctx, {left: box.left, top: box.top + unit * 0.42,
                      width: box.width, height: box.height},
                'SECURITY SYSTEM ACTIVE', palette.muted, unit, palette,
                {minUnit: 40});
    }
  }

  /** How many recorded frames the activation flourish lasts. */
  const PULSE_FRAMES = 14;

  /**
   * The blast door. Shut and red until the terminal is reached, then green and
   * drawn open with the outside showing through.
   *
   * `open` is the room's state. The door opening is the single clearest signal
   * the mission has advanced, so it must never be drawn open on a guess.
   */
  function drawBlastDoor(ctx, box, look, unit, palette, open) {
    const phase = look.phase || 0;
    const pulse = 0.5 + 0.5 * Math.sin(phase * (open ? 1.4 : 3.4));

    ctx.save();
    /* The surround, bricked from the laboratory's own masonry: the blast door
       is a doorway cut through the wall of the building, and the jambs are the
       stone either side of it. Heavier than anything else in the room, because
       it is the way out.

       Only the texture of the surround changed — it was a flat fill of
       `--muted` over exactly this footprint. The opening inside it is where
       the drone flies, and reaching the door is a distance test against the
       centre rather than a test against these bricks, so nothing here implies
       geometry the environment does not have. */
    ctx.globalAlpha = 0.95;
    layBricks(ctx, box, palette.cellWall || palette.muted, unit, palette, 4, 4);

    const inset = Math.min(box.width, box.height) * 0.16;
    const inner = {
      left: box.left + inset, top: box.top + inset,
      width: box.width - inset * 2, height: box.height - inset * 2,
    };

    if (open) {
      // Daylight through the doorway, and the two leaves drawn right back
      // against the frame.
      //
      // THE LEAVES HAVE TO MOVE, NOT MERELY CHANGE COLOUR
      // This used to draw them at 16% of the opening whether shut or open, so
      // the only difference between a locked door and an unlocked one was the
      // colour of the same closed rectangle — and a player who did not already
      // know that green meant open could not see that anything had happened.
      // Shut, each leaf covers half the opening and they meet in the middle;
      // open, they are 12% and pushed to the jambs, with the exterior blazing
      // between them.
      const leaf = inner.width * 0.12;

      ctx.globalAlpha = 1;
      withGlow(ctx, look.color, glowFor('exit') * 1.8, () => {
        ctx.fillStyle = look.color;
        ctx.fillRect(inner.left + leaf, inner.top,
                     inner.width - leaf * 2, inner.height);
      });

      // A brighter core to the daylight, so the opening reads as light coming
      // through rather than as a green panel.
      ctx.globalAlpha = 0.55;
      ctx.fillStyle = '#FFFFFF';
      const core = inner.width * 0.16;
      ctx.fillRect(inner.left + inner.width / 2 - core / 2,
                   inner.top + inner.height * 0.06,
                   core, inner.height * 0.88);

      // The two leaves, retracted into the jambs.
      ctx.globalAlpha = 1;
      ctx.fillStyle = palette.muted;
      ctx.fillRect(inner.left, inner.top, leaf, inner.height);
      ctx.fillRect(inner.left + inner.width - leaf, inner.top, leaf,
                   inner.height);

      // Guide rails, showing how far the leaves have travelled.
      ctx.globalAlpha = 0.5;
      ctx.strokeStyle = palette.base;
      ctx.lineWidth = Math.max(1, unit * 0.02);
      ctx.beginPath();
      ctx.moveTo(inner.left + leaf, inner.top + inner.height * 0.5);
      ctx.lineTo(inner.left + inner.width - leaf,
                 inner.top + inner.height * 0.5);
      ctx.stroke();
    } else {
      // Shut: two leaves meeting in the middle, with hazard chevrons.
      ctx.globalAlpha = 1;
      ctx.fillStyle = palette.base;
      ctx.fillRect(inner.left, inner.top, inner.width, inner.height);

      ctx.globalAlpha = 0.5;
      ctx.strokeStyle = look.color;
      ctx.lineWidth = Math.max(1.2, unit * 0.05);
      ctx.beginPath();
      ctx.moveTo(inner.left + inner.width / 2, inner.top);
      ctx.lineTo(inner.left + inner.width / 2, inner.top + inner.height);
      ctx.stroke();

      if (unit >= 12) {
        ctx.globalAlpha = 0.32;
        ctx.lineWidth = Math.max(1, unit * 0.06);
        ctx.beginPath();
        for (let index = 0; index < 3; index += 1) {
          const y = inner.top + inner.height * (0.22 + index * 0.28);
          ctx.moveTo(inner.left + inner.width * 0.16, y);
          ctx.lineTo(inner.left + inner.width * 0.42,
                     y + inner.height * 0.12);
          ctx.moveTo(inner.left + inner.width * 0.58,
                     y + inner.height * 0.12);
          ctx.lineTo(inner.left + inner.width * 0.84, y);
        }
        ctx.stroke();
      }
    }

    // The padlock, dead centre, while it is shut. The one symbol that needs no
    // explaining and no colour vocabulary — a door with a lock on it is shut
    // whatever the palette is doing, and it is simply absent once the door
    // opens rather than being recoloured.
    if (!open && unit >= 20) {
      const size = Math.min(inner.width, inner.height) * 0.3;
      const centreX = inner.left + inner.width / 2;
      const centreY = inner.top + inner.height / 2;
      const body = { width: size, height: size * 0.72 };

      ctx.globalAlpha = 0.65 + 0.3 * pulse;
      withGlow(ctx, look.color, glowFor('exit') * 0.8 * pulse, () => {
        // The shackle, above the body.
        ctx.strokeStyle = look.color;
        ctx.lineWidth = Math.max(1.4, size * 0.15);
        ctx.beginPath();
        ctx.arc(centreX, centreY - body.height * 0.34, size * 0.29,
                Math.PI, 0);
        ctx.stroke();
        // The body.
        ctx.fillStyle = look.color;
        ctx.fillRect(centreX - body.width / 2,
                     centreY - body.height * 0.28,
                     body.width, body.height);
      });
      // The keyhole, punched out of the body so it reads as a lock and not a
      // brick.
      ctx.globalAlpha = 1;
      ctx.fillStyle = palette.base;
      ctx.beginPath();
      ctx.arc(centreX, centreY + body.height * 0.08, size * 0.1, 0,
              Math.PI * 2);
      ctx.fill();
    }

    // The status lamps, on the frame. One while shut and three while open, so
    // the change is legible in the shape as well as the colour.
    const lamp = Math.max(1.5, unit * 0.07);
    const shares = open ? [0.32, 0.5, 0.68] : [0.5];
    shares.forEach(share => {
      ctx.globalAlpha = open ? 1 : 0.5 + 0.5 * pulse;
      withGlow(ctx, look.color, glowFor('exit') * (open ? 1.2 : pulse), () => {
        ctx.fillStyle = look.color;
        ctx.fillRect(box.left + box.width * share - lamp / 2,
                     box.top + inset * 0.2, lamp, lamp);
      });
    });
    ctx.restore();

    drawLabel(ctx, box,
              open ? 'FINAL EXIT — UNLOCKED' : 'FINAL EXIT — LOCKED',
              look.color, unit, palette);
  }

  /**
   * The landing platform.
   *
   * One function behind four recipe names, because the four differ in colour
   * and in one flourish and in nothing else. `mode` is the state the *room*
   * reported — never anything this file worked out. A platform that decided
   * for itself when a landing had succeeded would be able to contradict the
   * environment, and getting that wrong is worse than any amount of dullness.
   */
  function drawPlatform(ctx, box, look, unit, palette, mode) {
    const centreX = box.left + box.width / 2;
    const centreY = box.top + box.height / 2;
    const landed = mode === 'landed';
    const crashed = mode === 'crashed';
    const warning = mode === 'fast';
    const pulse = 0.5 + 0.5 * Math.sin((look.phase || 0) * (warning ? 4.5 : 1.5));

    ctx.save();

    // The deck: a dark plate, so the marks on it read.
    ctx.fillStyle = palette.base;
    ctx.globalAlpha = 0.72;
    ctx.fillRect(box.left, box.top, box.width, box.height);

    // The footprint, exactly the rectangle the environment tests. Anything
    // larger would promise a landing where there is none.
    ctx.globalAlpha = 1;
    ctx.strokeStyle = look.color;
    ctx.lineWidth = Math.max(1.5, unit * 0.055);
    const glow = glowFor('exit') * (landed ? 2.2 : crashed ? 1.6 : 0.7 + pulse * 0.5);
    withGlow(ctx, look.color, glow, () => {
      ctx.strokeRect(box.left, box.top, box.width, box.height);
    });

    // Corner alignment marks.
    const arm = Math.min(box.width, box.height) * 0.3;
    ctx.globalAlpha = 0.9;
    ctx.lineWidth = Math.max(1, unit * 0.04);
    ctx.beginPath();
    [[0, 0, 1, 1], [1, 0, -1, 1], [0, 1, 1, -1], [1, 1, -1, -1]]
      .forEach(([cx, cy, sx, sy]) => {
        const x = box.left + box.width * cx;
        const y = box.top + box.height * cy;
        ctx.moveTo(x + sx * arm, y);
        ctx.lineTo(x, y);
        ctx.lineTo(x, y + sy * arm);
      });
    ctx.stroke();

    // The centre mark: concentric rings and a cross, the landing target.
    const reach = Math.min(box.width, box.height) * 0.32;
    if (reach >= 4) {
      ctx.globalAlpha = landed ? 0.95 : 0.5 + pulse * 0.3;
      ctx.lineWidth = Math.max(0.8, unit * 0.03);
      ctx.beginPath();
      [0.5, 1.0].forEach(share => {
        ctx.moveTo(centreX + reach * share, centreY);
        ctx.arc(centreX, centreY, reach * share, 0, Math.PI * 2);
      });
      ctx.moveTo(centreX - reach * 1.25, centreY);
      ctx.lineTo(centreX + reach * 1.25, centreY);
      ctx.moveTo(centreX, centreY - reach * 1.25);
      ctx.lineTo(centreX, centreY + reach * 1.25);
      ctx.stroke();
    }

    // Deck lights along the long edges, brighter as the state gets louder.
    const lamps = Math.max(2, Math.round(box.width / Math.max(unit * 0.5, 10)));
    ctx.globalAlpha = landed ? 1 : 0.35 + pulse * 0.45;
    ctx.fillStyle = look.color;
    for (let index = 0; index < lamps; index += 1) {
      const x = box.left + box.width * ((index + 0.5) / lamps);
      const size = Math.max(1, unit * 0.045);
      ctx.fillRect(x - size / 2, box.top + size * 0.4, size, size);
      ctx.fillRect(x - size / 2, box.top + box.height - size * 1.4, size, size);
    }

    // A landing settles; a crash rings outwards. Both derived from the phase,
    // so a replay scrubbed to the same step draws the same flourish.
    if (landed || crashed) {
      ctx.globalAlpha = crashed ? 0.5 : 0.4;
      ctx.strokeStyle = look.color;
      ctx.lineWidth = Math.max(1.2, unit * 0.05);
      const ring = reach * (crashed ? 1.6 + pulse * 1.1 : 1.4);
      ctx.beginPath();
      ctx.arc(centreX, centreY, ring, 0, Math.PI * 2);
      ctx.stroke();

      if (crashed && unit >= 10) {
        // Sparks: eight short spurs, deterministic in the phase.
        ctx.globalAlpha = 0.7;
        ctx.lineCap = 'round';
        ctx.lineWidth = Math.max(1, unit * 0.04);
        ctx.beginPath();
        for (let index = 0; index < 8; index += 1) {
          const angle = index * Math.PI / 4 + (look.phase || 0) * 0.5;
          const from = reach * 0.9;
          const to = from + reach * (0.5 + 0.4 * ((index % 3) / 2));
          ctx.moveTo(centreX + Math.cos(angle) * from,
                     centreY + Math.sin(angle) * from);
          ctx.lineTo(centreX + Math.cos(angle) * to,
                     centreY + Math.sin(angle) * to);
        }
        ctx.stroke();
      }
    }
    ctx.restore();
  }

  /**
   * The agent: a machine, not a square. Drawn from its centre.
   *
   * `facing` is in radians, or null when which way it is pointing is not
   * known — a grid agent between two cells has a direction, a planner's
   * replay does not always.
   */
  function drawAgent(ctx, centre, size, facing, palette) {
    const left = centre.x - size / 2;
    const top = centre.y - size / 2;
    const at = (x, y) => ({ x: left + size * x, y: top + size * y });

    ctx.save();

    // Under about ten pixels every one of the parts below is a smudge, so
    // it becomes a plain lit block instead of a bad robot.
    if (size < 10) {
      withGlow(ctx, palette.accent, glowFor('agent'), () => {
        ctx.fillStyle = palette.accent;
        ctx.fillRect(left + size * 0.15, top + size * 0.15,
                     size * 0.7, size * 0.7);
      });
      ctx.restore();
      return;
    }

    withGlow(ctx, palette.accent, glowFor('agent'), () => {
      ctx.fillStyle = palette.accent;
      ctx.strokeStyle = palette.accent;
      ctx.lineCap = 'round';

      // The aerial, and the bead on the end of it.
      ctx.lineWidth = Math.max(1, size * 0.05);
      ctx.beginPath();
      const mast = at(0.5, 0.16);
      ctx.moveTo(mast.x, mast.y);
      ctx.lineTo(at(0.5, 0.04).x, at(0.5, 0.04).y);
      ctx.stroke();
      ctx.beginPath();
      ctx.arc(at(0.5, 0.03).x, at(0.5, 0.03).y, size * 0.06, 0, Math.PI * 2);
      ctx.fill();

      // Head, body, and an arm on each side.
      ctx.fillRect(at(0.24, 0.16).x, at(0.24, 0.16).y, size * 0.52, size * 0.26);
      ctx.fillRect(at(0.28, 0.46).x, at(0.28, 0.46).y, size * 0.44, size * 0.32);
      ctx.fillRect(at(0.12, 0.5).x, at(0.12, 0.5).y, size * 0.13, size * 0.2);
      ctx.fillRect(at(0.75, 0.5).x, at(0.75, 0.5).y, size * 0.13, size * 0.2);

      // Two treads it runs on.
      ctx.fillRect(at(0.2, 0.82).x, at(0.2, 0.82).y, size * 0.24, size * 0.13);
      ctx.fillRect(at(0.56, 0.82).x, at(0.56, 0.82).y, size * 0.24, size * 0.13);
    });

    // The features are cut out in the page colour, so they read as part of
    // the silhouette rather than as paint on top of it.
    ctx.fillStyle = palette.base;

    // Two eyes.
    const eye = size * 0.09;
    ctx.fillRect(at(0.33, 0.24).x, at(0.33, 0.24).y, eye, eye);
    ctx.fillRect(at(0.58, 0.24).x, at(0.58, 0.24).y, eye, eye);

    // A grille across the chest.
    if (size >= 16) {
      for (let line = 0; line < 3; line += 1) {
        ctx.fillRect(at(0.36, 0.54 + line * 0.07).x,
                     at(0.36, 0.54 + line * 0.07).y,
                     size * 0.28, size * 0.03);
      }
    } else {
      ctx.fillRect(at(0.38, 0.58).x, at(0.38, 0.58).y, size * 0.24, size * 0.05);
    }

    // Which way it is going, when that is known: a wedge on the leading
    // edge rather than turning the whole machine, which at this size would
    // just make it unreadable.
    if (facing !== null && facing !== undefined) {
      ctx.fillStyle = palette.accent;
      const reach = size * 0.62;
      const tipX = centre.x + Math.cos(facing) * reach;
      const tipY = centre.y + Math.sin(facing) * reach;
      const base = size * 0.46;
      ctx.beginPath();
      ctx.moveTo(tipX, tipY);
      ctx.lineTo(centre.x + Math.cos(facing + 2.6) * base,
                 centre.y + Math.sin(facing + 2.6) * base);
      ctx.lineTo(centre.x + Math.cos(facing - 2.6) * base,
                 centre.y + Math.sin(facing - 2.6) * base);
      ctx.closePath();
      ctx.fill();
    }
    ctx.restore();
  }

  /**
   * Draw one shape by name.
   *
   * An unknown name falls back to a plain block rather than throwing: a
   * room that names a recipe this file has not got yet should look dull,
   * not break the screen.
   */
  function draw(ctx, name, box, look, unit, palette) {
    const recipe = RECIPES[name] || RECIPES.block;
    ctx.save();
    recipe(ctx, box, look, unit, palette);
    ctx.restore();
  }

  /**
   * R-5 in the drone frame. Drawn from its centre, like `drawAgent`.
   *
   * The same machine as in rooms 1 to 3 — the lit cyan core, the aerial bead,
   * the metallic body — carried in a four-rotor frame. It is a separate
   * function rather than a state of `drawAgent` because a hovering frame and a
   * walking chassis share a palette and nothing else.
   *
   * EVERYTHING HERE IS READ, NOTHING IS DERIVED
   * `velocity` is the velocity the episode recorded, in world units. The tilt,
   * the trail intensity, the rotor blur and the braking marks are all
   * presentation squeezed out of that one number. No position is integrated,
   * no state is invented, and the drone's orientation is not part of the
   * simulation — the assignment fixes the state at four numbers and this adds
   * nothing to them.
   *
   * When the drone is nearly stationary there is no meaningful direction to
   * point in, so it holds a neutral hover rather than snapping to an angle
   * that a rounding error picked.
   */
  function drawDrone(ctx, centre, size, velocity, options, palette) {
    const settings = options || {};
    const phase = settings.phase || 0;
    const vx = velocity.x || 0;
    const vy = velocity.y || 0;
    const speed = Math.hypot(vx, vy);
    const limit = settings.speedLimit || 1;
    const share = Math.min(1, speed / limit);

    // Below this there is no direction worth showing; hovering is the honest
    // pose. `facing` is the last meaningful heading the caller knew about, so
    // a drone that has stopped keeps pointing where it was going.
    const moving = speed > 0.04;
    const heading = moving ? Math.atan2(vy, vx)
      : (settings.facing === null || settings.facing === undefined
         ? -Math.PI / 2 : settings.facing);

    // Tilt into the direction of travel, the way a real quadrotor leans to
    // accelerate. Capped well short of anything acrobatic.
    const tilt = Math.max(-0.34, Math.min(0.34, share * 0.34));
    const hover = Math.sin(phase * 2.1) * size * 0.03;

    ctx.save();
    ctx.translate(centre.x, centre.y + hover);

    // The shadow, which is what puts the drone *in* the chamber. It shrinks as
    // the frame lifts on the hover, so the two read as one movement.
    ctx.save();
    ctx.globalAlpha = 0.34;
    ctx.fillStyle = palette.base;
    ctx.beginPath();
    ctx.ellipse(size * 0.06, size * 0.5 - hover * 0.6,
                size * (0.42 - hover / size * 0.1), size * 0.13, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Under about ten pixels the frame is a smudge, so it becomes the same lit
    // block `drawAgent` falls back to — consistent rather than a bad drone.
    if (size < 10) {
      withGlow(ctx, palette.accent, glowFor('agent'), () => {
        ctx.fillStyle = palette.accent;
        ctx.fillRect(-size * 0.35, -size * 0.35, size * 0.7, size * 0.7);
      });
      ctx.restore();
      return;
    }

    ctx.rotate(Math.cos(heading) * tilt * 0.6);

    const arm = size * 0.4;
    const rotor = size * 0.17;

    // Arms and rotor housings, in the wall grey so the frame reads as metal
    // and the core stays the only lit thing on it.
    ctx.strokeStyle = palette.muted;
    ctx.globalAlpha = 0.85;
    ctx.lineWidth = Math.max(1.2, size * 0.055);
    ctx.lineCap = 'round';
    ctx.beginPath();
    [[-1, -1], [1, -1], [-1, 1], [1, 1]].forEach(([sx, sy]) => {
      ctx.moveTo(0, 0);
      ctx.lineTo(sx * arm * 0.72, sy * arm * 0.62);
    });
    ctx.stroke();

    // The rotors. They blur with speed rather than spinning visibly, which is
    // both what a real rotor looks like and cheaper than drawing blades.
    [[-1, -1], [1, -1], [-1, 1], [1, 1]].forEach(([sx, sy], index) => {
      const x = sx * arm * 0.72;
      const y = sy * arm * 0.62;
      const wobble = Math.sin(phase * 6 + index * 1.7) * rotor * 0.06;

      ctx.globalAlpha = 0.9;
      ctx.strokeStyle = palette.muted;
      ctx.lineWidth = Math.max(1, size * 0.04);
      ctx.beginPath();
      ctx.arc(x, y, rotor, 0, Math.PI * 2);
      ctx.stroke();

      // The disc, faint: it is moving air, not a solid.
      ctx.globalAlpha = 0.16 + 0.16 * share;
      ctx.fillStyle = palette.accent;
      ctx.beginPath();
      ctx.ellipse(x, y, rotor * 0.88 + wobble, rotor * 0.3, 0, 0, Math.PI * 2);
      ctx.fill();
    });

    // The body: a small metallic shell with the lit core in it.
    ctx.globalAlpha = 1;
    ctx.fillStyle = palette.muted;
    ctx.beginPath();
    ctx.ellipse(0, 0, size * 0.24, size * 0.18, 0, 0, Math.PI * 2);
    ctx.fill();

    withGlow(ctx, palette.accent, glowFor('agent'), () => {
      ctx.fillStyle = palette.accent;
      ctx.beginPath();
      ctx.arc(0, 0, size * 0.105, 0, Math.PI * 2);
      ctx.fill();
    });

    // Which way is forward: a short lit spur plus two navigation lights on the
    // leading arms. Without this a symmetrical frame gives nothing away.
    ctx.save();
    ctx.rotate(heading);
    ctx.strokeStyle = palette.accent;
    ctx.globalAlpha = 0.9;
    ctx.lineWidth = Math.max(1, size * 0.05);
    ctx.beginPath();
    ctx.moveTo(size * 0.16, 0);
    ctx.lineTo(size * 0.34, 0);
    ctx.stroke();

    ctx.fillStyle = palette.accent;
    ctx.globalAlpha = 0.55 + 0.45 * Math.abs(Math.sin(phase * 3));
    [-0.5, 0.5].forEach(side => {
      ctx.beginPath();
      ctx.arc(size * 0.22, side * size * 0.2, Math.max(1, size * 0.035),
              0, Math.PI * 2);
      ctx.fill();
    });
    ctx.restore();

    // Braking: when it is travelling slowly enough to land, say so. This is
    // the one piece of feedback about the room's actual rule, and the
    // threshold is handed in rather than assumed.
    if (settings.landingSpeed && moving
        && Math.abs(vx) <= settings.landingSpeed
        && Math.abs(vy) <= settings.landingSpeed) {
      ctx.globalAlpha = 0.4 + 0.25 * Math.sin(phase * 3.4);
      ctx.strokeStyle = palette.goal || palette.accent;
      ctx.lineWidth = Math.max(1, size * 0.04);
      ctx.beginPath();
      ctx.arc(0, 0, size * 0.5, 0, Math.PI * 2);
      ctx.stroke();
    }

    ctx.restore();
  }

  return {
    draw: draw,
    agent: drawAgent,
    drone: drawDrone,
    has: function (name) { return Object.prototype.hasOwnProperty.call(RECIPES, name); },
    names: Object.keys(RECIPES),
  };
})();
