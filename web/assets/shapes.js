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
      // The mortar bed. Everything else is bricks laid on top of it.
      ctx.fillStyle = palette.base;
      ctx.fillRect(box.left, box.top, box.width, box.height);

      const courses = 3;
      const courseHeight = box.height / courses;
      const brickWidth = box.width / 2;
      // Below a few pixels the gaps swallow the bricks, so the mortar
      // thins to nothing and it goes back to reading as a solid block.
      const gap = Math.min(courseHeight * 0.22, Math.max(0.5, unit * 0.045));

      ctx.fillStyle = look.color;
      for (let course = 0; course < courses; course += 1) {
        const top = box.top + course * courseHeight;
        const offset = course % 2 === 0 ? 0 : -brickWidth / 2;
        for (let x = box.left + offset;
             x < box.left + box.width;
             x += brickWidth) {
          // Clipped to the cell, so half bricks at the edges line up with
          // the half bricks of the cell next door.
          const left = Math.max(x, box.left);
          const right = Math.min(x + brickWidth - gap, box.left + box.width);
          if (right > left) {
            ctx.fillRect(left, top + gap / 2, right - left,
                         courseHeight - gap);
          }
        }
      }
      ctx.restore();
    },

    /**
     * Ice: a slick of meltwater with a crystal sitting on it.
     *
     * The slick alone was being read as "a pale tile". The six-armed
     * crystal is the part that says ice, so it is drawn at full strength
     * over a slick that has been dimmed to make room for it.
     */
    ice(ctx, box, look, unit, palette) {
      const centreX = box.left + box.width / 2;
      const centreY = box.top + box.height / 2;
      const radius = Math.min(box.width, box.height) * 0.3;

      ctx.save();

      // The meltwater underneath.
      ctx.fillStyle = look.color;
      ctx.globalAlpha = 0.4;
      ctx.beginPath();
      ctx.ellipse(centreX, centreY, box.width * 0.42, box.height * 0.36,
                  0, 0, Math.PI * 2);
      ctx.fill();

      // The crystal: six arms, each with a pair of branches. Below about a
      // dozen pixels the branches are a smudge, so it drops to a plain
      // six-pointed star.
      const branched = radius > 5;
      ctx.globalAlpha = 1;
      ctx.strokeStyle = look.color;
      ctx.lineWidth = Math.max(0.7, unit * 0.045);
      ctx.lineCap = 'round';
      ctx.beginPath();
      for (let arm = 0; arm < 6; arm += 1) {
        const angle = (Math.PI / 3) * arm;
        const tipX = centreX + Math.cos(angle) * radius;
        const tipY = centreY + Math.sin(angle) * radius;
        ctx.moveTo(centreX, centreY);
        ctx.lineTo(tipX, tipY);

        if (!branched) continue;
        const at = 0.58;                       // where the branches sit
        const jointX = centreX + Math.cos(angle) * radius * at;
        const jointY = centreY + Math.sin(angle) * radius * at;
        const branch = radius * 0.32;
        [angle + 0.9, angle - 0.9].forEach(direction => {
          ctx.moveTo(jointX, jointY);
          ctx.lineTo(jointX + Math.cos(direction) * branch,
                     jointY + Math.sin(direction) * branch);
        });
      }
      ctx.stroke();

      // One glint, off centre, so the surface reads as wet.
      ctx.strokeStyle = palette.base;
      ctx.globalAlpha = 0.45;
      ctx.lineWidth = Math.max(0.6, unit * 0.03);
      ctx.beginPath();
      ctx.ellipse(centreX - box.width * 0.14, centreY + box.height * 0.2,
                  box.width * 0.12, box.height * 0.05, 0.4, 0, Math.PI * 2);
      ctx.stroke();
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
    abyss(ctx, box, look, unit, palette) {
      ctx.save();

      // The rim: the broken edge of the floor the hole was made in.
      ctx.fillStyle = look.color;
      ctx.globalAlpha = 0.55;
      ctx.fillRect(box.left, box.top, box.width, box.height);

      // The shelf, a little way in, and then the dark. Insetting all four
      // sides is what makes it a hole rather than a dark tile.
      const shelf = Math.min(box.width, box.height) * 0.12;
      ctx.globalAlpha = 1;
      ctx.fillStyle = palette.hairlineFaint;
      ctx.fillRect(box.left + shelf, box.top + shelf,
                   box.width - shelf * 2, box.height - shelf * 2);
      ctx.fillStyle = palette.base;
      ctx.fillRect(box.left + shelf * 2.1, box.top + shelf * 2.1,
                   box.width - shelf * 4.2, box.height - shelf * 4.2);

      // A ragged bite out of the near edge, so the rim is broken stone and
      // not a rounded tile. Skipped when there are not the pixels for it.
      if (unit >= 12) {
        ctx.fillStyle = palette.base;
        const teeth = 3;
        for (let index = 0; index < teeth; index += 1) {
          const width = box.width / (teeth * 2.2);
          const x = box.left + box.width * (0.18 + index * 0.32);
          ctx.beginPath();
          ctx.moveTo(x, box.top + shelf);
          ctx.lineTo(x + width / 2, box.top + shelf * 0.2);
          ctx.lineTo(x + width, box.top + shelf);
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
    bridge(ctx, box, look, unit, palette) {
      // The hole first, at full depth: a bridge is a thing over a void, and
      // the void has to be visible through it for that to read.
      RECIPES.abyss(ctx, box, { color: palette.hairlineFaint }, unit, palette);

      ctx.save();

      // Slats laid across the span, with the dark showing between them.
      // Four is enough to read as decking and few enough that the gaps
      // survive at cell size.
      const slats = 4;
      const pitch = box.height / slats;
      const gap = Math.max(1, Math.min(pitch * 0.34, unit * 0.07));
      ctx.fillStyle = look.color;
      for (let index = 0; index < slats; index += 1) {
        ctx.fillRect(box.left, box.top + index * pitch + gap / 2,
                     box.width, pitch - gap);
      }

      // The two ropes the slats hang from, running the length of the span
      // and standing slightly proud of the decking at each edge. This is
      // what makes it a bridge rather than a striped floor.
      const rope = Math.max(1, unit * 0.06);
      ctx.fillStyle = palette.muted;
      ctx.fillRect(box.left, box.top - rope * 0.2, box.width, rope);
      ctx.fillRect(box.left, box.top + box.height - rope * 0.8,
                   box.width, rope);

      // The posts the ropes are strung between, at the ends of the span.
      if (unit >= 12) {
        const post = Math.max(1.4, unit * 0.09);
        ctx.fillRect(box.left, box.top - rope * 0.2, post,
                     box.height + rope * 0.6);
        ctx.fillRect(box.left + box.width - post, box.top - rope * 0.2,
                     post, box.height + rope * 0.6);
      }
      ctx.restore();
    },

    /** What is left after a plank gives way: the hole, and two stubs. */
    bridgeBroken(ctx, box, look, unit, palette) {
      RECIPES.abyss(ctx, box, { color: palette.hazard }, unit, palette);

      ctx.save();
      ctx.fillStyle = look.color;
      ctx.globalAlpha = 0.8;
      const thickness = box.height * 0.14;
      const middle = box.top + box.height * 0.42;
      // Stubs at each side, angled, so the span reads as snapped.
      ctx.save();
      ctx.translate(box.left, middle);
      ctx.rotate(0.22);
      ctx.fillRect(0, 0, box.width * 0.26, thickness);
      ctx.restore();
      ctx.save();
      ctx.translate(box.left + box.width, middle + box.height * 0.1);
      ctx.rotate(-0.26);
      ctx.fillRect(-box.width * 0.26, 0, box.width * 0.26, thickness);
      ctx.restore();
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
  };

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

  return {
    draw: draw,
    agent: drawAgent,
    has: function (name) { return Object.prototype.hasOwnProperty.call(RECIPES, name); },
    names: Object.keys(RECIPES),
  };
})();
