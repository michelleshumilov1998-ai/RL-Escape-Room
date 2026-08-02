'use strict';

/* =====================================================================
   THE DATA CONTRACT.

   This file is the whole agreement between the interface and whatever
   produces its data.  Nothing below knows where the data comes from: it
   arrives from the Python side over HTTP, assembled by `producer.js`, and
   before that it came from a mock producer in the browser.  Neither the
   renderer, the playback controller, the charts nor the shell changed when
   it was swapped, which is the property this file exists to protect.

   Read this before reading anything else in this directory.

   ---------------------------------------------------------------------
   RoomDefinition — what a room *is*.  Static for the room's lifetime.
   ---------------------------------------------------------------------

     id            string, stable, e.g. 'room2'
     name          string, shown as the title
     sector        string, the small label above the title
     worldSize     { width, height }   in world units
     isGrid        boolean
     cellSize      number, world units per cell.  Grid rooms only.
     entities      [ Entity ]
     entityTypes   { type: EntityType }
     info          RoomInfo
     parameterSchema [ Parameter ]
     metric        { key, label, format }   which single number the status
                   strip shows.  `key` names a field on EpisodeMetrics.
     playback      optional { stepsPerSecond }.  How fast a replay of this
                   room should be shown.  A room whose time step is a
                   fiftieth of a second takes several hundred of them to
                   cross itself, and playing those at the rate that suits
                   a grid of ten cells would take minutes.  This is a
                   presentation hint and nothing else: it changes how long
                   you look at an episode, never what happened in it.

     Entity
       id          string, unique within the room
       type        string, a key of entityTypes
       position    { x, y }   the entity's CENTRE, in world units
       size        { width, height }   in world units
       appearance  optional partial EntityType, merged over the type for
                   this entity alone.  For the things that share a type
                   but differ in one detail — three vents that blow three
                   different ways — where making a type per detail would
                   put three near-identical rows in the legend.

     EntityType
       label       one word, for the legend
       color       a CSS custom property name, e.g. '--cell-wall'.  The
                   colour itself lives in the stylesheet, never here.
       shape       which drawing recipe the renderer should use: 'wall',
                   'ice', 'abyss', 'maintenanceVoid', 'bridge',
                   'bridgeBroken', 'laser', 'exit', 'start', 'obstacle',
                   'pad', or 'block' for a plain filled cell.  A thing on the grid is meant to
                   look like what it is rather than like a coloured
                   square, and this is how a room says which it is.  The
                   renderer knows how to draw an abyss; which entities
                   are abysses stays the room's business.
       inLegend    boolean.  Floor is drawn but not listed.
       role        'static' | 'hazard' | 'goal' | 'agent'.  What the
                   renderer draws it as; keeps the drawing code from
                   having to recognise particular type names.
       states      optional { state: partial EntityType }.  How this type
                   looks when a step reports that state for one of its
                   entities — a collapsed bridge, an opened door.  Any of
                   color, glyph, opacity or hidden may be overridden.
                   This is what keeps `state` opaque to the renderer: the
                   room says what a state looks like, so the drawing code
                   never has to know what 'collapsed' means.

     RoomInfo
       objective   one sentence
       obstacles   [ string ]
       actionSet   string
       rewards     [ [ label, value ] ]
       terminal    string
       note        string, optional
       routes      optional [ { name, steps, best, risk } ].  When a room
                   is *about* a choice between routes, this is that
                   choice stated plainly, so the screen can show what is
                   being traded off instead of leaving it to be inferred
                   from the layout.

     Parameter
       key         string
       label       string
       symbol      string, may be empty
       type        'number'
       min, max, step, default
       description one line of plain English
       appliesLive boolean.  False means changing it invalidates a run,
                   so the run is marked stale until Reset is pressed.
       choices     optional [ number ].  When present the control steps
                   between these values instead of by `step`.

   ---------------------------------------------------------------------
   POSITIONS ARE ALWAYS CONTINUOUS WORLD UNITS.
   ---------------------------------------------------------------------

   There is one coordinate system and the renderer has one transform.  A
   grid room is not a special case of drawing; it is a room whose agent
   happens to move between lattice points and which asks for cell
   boundaries to be drawn.

   Concretely: cell (row r, col c) of a grid with cellSize 1 has its
   centre at { x: c + 0.5, y: r + 0.5 }.  Centres rather than corners,
   because `position` means the same thing for a wall, a drone and an
   agent, and only a centre can.  So a grid agent's positions land on
   half-integers — the lattice is what is regular about them, not the
   parity of the numbers.  Nothing indexes cells; nothing rounds.

   y increases downwards, matching both the canvas and the way the room
   layouts are written down as rows of text.

   ---------------------------------------------------------------------
   TrajectoryBatch — what *happened*.  Handed over whole.
   ---------------------------------------------------------------------

     episodes      [ Episode ]
     metrics       [ EpisodeMetrics ]   one entry per episode, same order

     Episode
       index       0-based
       steps       [ Step ]
       totalReward number
       outcome     'success' | 'failure' | 'timeout'
       epsilon     number, the exploration rate this episode ran at

     Step
       position    { x, y }   world units
       velocity    { x, y } or null.  Null in rooms that have no velocity.
       reward      number, for this step alone
       action      string, for display only
       entityStates [ { id, state } ] or null.  Anything about a static
                   entity that changed this step — a door opening, a
                   bridge collapsing.  The renderer reads `state` as an
                   opaque string and only uses it to pick an appearance.
       entityPositions [ { id, position } ] or null.  Where an entity that
                   moves is at this step: a patrolling guard, a lift, a
                   conveyor.  The entity is still declared once in the
                   room with a position of its own; this overrides it for
                   the length of the episode, and the last value seen
                   stands until another arrives.  Kept separate from
                   `entityStates` because moving and changing appearance
                   are different things and a thing may do either, both,
                   or neither.

     EpisodeMetrics
       episode     number, matching Episode.index
       reward      number
       steps       number
       epsilon     number
       convergence number.  Some measure that ought to fall towards zero.
                   The interface does not care which one.

   ---------------------------------------------------------------------
   OVERLAYS — optional, and deliberately meaningless here.
   ---------------------------------------------------------------------

   A grid room may be handed a heatmap and an arrow layer:

     heatmap       [ { cell: [row, col], value } ]
     arrows        [ { cell: [row, col], direction } ]  direction is one
                   of 'up' | 'down' | 'left' | 'right'

   The numbers are drawn relative to the largest of them and nothing
   more.  Whether they are state values, visit counts or something else
   is not the interface's business.

   A continuous room may be handed an observation range:

     observation   { radius, heading, spread } in world units and radians,
                   or null.  Drawn around the agent.
   ===================================================================== */

window.Contract = (function () {

  /* These checks exist to make a malformed producer fail loudly at the
     boundary rather than quietly half-draw.  They are cheap and run once
     per load, so they stay on. */

  function fail(what) {
    throw new Error('contract: ' + what);
  }

  function checkRoom(room) {
    if (!room || typeof room !== 'object') fail('the room is not an object');
    ['id', 'name', 'worldSize', 'entityTypes', 'info'].forEach(field => {
      if (room[field] === undefined) fail('the room has no ' + field);
    });
    if (!(room.worldSize.width > 0) || !(room.worldSize.height > 0)) {
      fail('the world has no size');
    }
    if (room.isGrid && !(room.cellSize > 0)) {
      fail('a grid room needs a cellSize');
    }

    (room.entities || []).forEach(entity => {
      if (!room.entityTypes[entity.type]) {
        fail('entity ' + entity.id + ' has unknown type ' + entity.type);
      }
      if (!entity.position || typeof entity.position.x !== 'number'
          || typeof entity.position.y !== 'number') {
        fail('entity ' + entity.id + ' has no numeric position');
      }
    });

    (room.parameterSchema || []).forEach(parameter => {
      ['key', 'label', 'min', 'max', 'default'].forEach(field => {
        if (parameter[field] === undefined) {
          fail('parameter ' + parameter.key + ' has no ' + field);
        }
      });
      if (parameter.appliesLive === undefined) {
        fail('parameter ' + parameter.key + ' does not say if it applies live');
      }
    });

    return room;
  }

  function checkBatch(batch) {
    if (!batch || !Array.isArray(batch.episodes)) {
      fail('the batch has no episodes array');
    }
    if (!Array.isArray(batch.metrics)) {
      fail('the batch has no metrics array');
    }
    // An empty batch is legal — the interface has to cope with one, and
    // the charts are tested against it.
    batch.episodes.forEach(episode => {
      if (!Array.isArray(episode.steps) || !episode.steps.length) {
        fail('episode ' + episode.index + ' has no steps');
      }
      if (!episode.outcome) fail('episode ' + episode.index + ' has no outcome');
    });
    return batch;
  }

  return { checkRoom: checkRoom, checkBatch: checkBatch };
})();
