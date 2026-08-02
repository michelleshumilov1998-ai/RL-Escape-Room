'use strict';

/* =====================================================================
   Every tunable number on this screen, in one object.

   If a value governs how something looks or how fast it happens, it is
   here and not in the file that uses it.  Colours are the exception in
   the same way they are everywhere else in this project: the names of
   the CSS custom properties are here, the colours themselves are in
   base.css, and nothing writes a hex value down twice.
   ===================================================================== */

window.ROOM_CONFIG = {

  /* ---- playback ---------------------------------------------------- *
     `stepsPerSecond` is how fast the agent walks; `animated: false` is
     different in kind — it consumes the batch without drawing the way
     through it, which is what makes a thousand-episode run watchable at
     all.                                                              */
  speeds: {
    slow: { label: 'Slow', stepsPerSecond: 3, animated: true },
    normal: { label: 'Normal', stepsPerSecond: 9, animated: true },
    fast: { label: 'Fast', stepsPerSecond: 28, animated: true },
    turbo: { label: 'Turbo', stepsPerSecond: null, animated: false },
  },
  speedDefault: 'normal',

  /* How often the recorded episodes are pulled in while training is being
     watched, so the graphs and the episode list fill in as it learns rather
     than only once it has finished. The recording arrives whole, so this is
     a compromise between a live curve and a pointless amount of traffic. */
  liveBatchRefreshMs: 2000,

  /* Turbo's work is time-budgeted rather than counted, so a fast machine
     does not run away with the episode counter between two frames. Kept
     well under a frame so the interface never stops answering. */
  turboBudgetMs: 10,
  turboMaxEpisodesPerFrame: 40,

  /* How long the agent takes to slide from one step to the next, as a
     share of the time between steps. 1.0 means it is always moving. */
  moveTween: 0.9,

  /* Replaying one episode is paced for reading, not for progress, so it
     has its own rate and a beat on the final frame before it loops. */
  replayStepsPerSecond: 4,
  replayPauseSeconds: 1.1,

  /* Replaying one episode with Turbo selected. Turbo is not a rate — it
     consumes a whole batch per frame — so a single replay uses this multiple
     of the room's own rate instead. Deliberately modest: the point of the
     ceiling is that even the fastest chip still draws every recorded frame,
     because dropping frames is exactly what makes a drone look as though it
     teleported. */
  turboReplayScale: 6,

  /* ---- the trail --------------------------------------------------- */
  trail: {
    length: 26,          // steps kept behind the agent
    width: 2,            // px, before devicePixelRatio
    headOpacity: 0.55,
    tailOpacity: 0.0,
  },

  /* ---- drawing ----------------------------------------------------- */
  render: {
    padding: 34,         // px of margin around the world, before scaling
    minCellPx: 6,        // below this a grid's cell boundaries are dropped
    agentSize: 0.62,     // share of a cell (or world unit) the agent fills
    gridLineWidth: 1,
    heatmapMaxOpacity: 0.5,
    arrowOpacity: 0.75,
    observationOpacity: 0.13,

    /* Shadow blur, in px, for the few things that emit light rather than
       merely being coloured. Anything else is flat. */
    glow: { laser: 14, exit: 10, agent: 9 },
  },

  /* ---- impact feedback ---------------------------------------------- *
     The shake when a bridge section fails. Counted in RECORDED FRAMES rather
     than in seconds, so it occupies the same steps of the episode however fast
     it is being played back and lands identically in a replay.

     At room 2's replay rate of 4 frames a second, `frames: 2` is about a fifth
     of a second — the brief's 0.2-0.3s. `amplitude` is in CSS pixels, and it is
     small on purpose: enough to register as an impact, not enough to make the
     chamber hard to look at.                                              */
  impact: {
    frames: 2,
    amplitude: 5,
    // How fast it rattles within that window, in radians per frame.
    frequency: 26,
  },

  /* ---- the mission marker (room 5) ---------------------------------- *
     The ring and chevrons that sit on the current objective, and the pulse
     that runs from the terminal to the door when the door unlocks. Both are
     drawn from state the environment reported; these are only sizes.

     `linkFrames` is counted in *recorded frames* rather than in seconds, so
     the flourish occupies the same steps of the episode however fast it is
     being played back — and lands on the same steps in a replay as it did
     live.                                                                */
  mission: {
    markerRadius: 0.95,      // world units
    markerBreath: 0.09,      // how far the ring breathes, world units
    markerPulse: 1.6,        // radians per unit of the room's clock
    markerWidth: 1.6,        // px
    linkFrames: 14,
    linkWidth: 2,
    linkHeadRadius: 4,
  },

  /* ---- the sensor fan (room 5) -------------------------------------- *
     What R-5 can see, drawn from the very numbers it was given. The three
     rays and the cone come over on the frame; nothing here is invented.

     `near` and `critical` are the free-distance fractions at which a ray
     turns amber and then red. A ray reports 1.0 for "nothing within range"
     and 0.0 for "touching", so these are read downwards: below 0.55 of the
     range something is in the way, below 0.28 it is close enough to be a
     collision risk at the drone's top speed.                            */
  sensors: {
    near: 0.55,
    critical: 0.28,
    coneOpacity: 0.07,
    rayWidth: 1.6,
    arcOpacity: 0.3,
    hitRadius: 3.0,
    /* The ring drawn round an obstacle that passed the visibility rule. */
    markRadius: 0.42,
    markWidth: 1.4,
  },

  /* ---- the step inspector ------------------------------------------ *
     The small static grid in the sidebar. Tighter margins than the world
     view, because it has a sidebar's width to live in.                 */
  stepReplay: {
    padding: 6,
  },

  /* The semantic palette, as custom-property names. Every room reuses
     these five; a room never introduces a colour of its own. */
  colors: {
    base: '--base',
    muted: '--muted',
    accent: '--accent',
    goal: '--goal',
    hazard: '--hazard',
    warn: '--warn',
    hairline: '--hairline',
    hairlineFaint: '--hairline-faint',
    field: '--cell-floor',
    /* The masonry colour, so anything bricked with `layBricks` is the
       same stone as the laboratory wall. */
    cellWall: '--cell-wall',
  },

  /* ---- charts ------------------------------------------------------ */
  charts: {
    height: 74,               // css px
    smoothingWindow: 20,      // rolling average, in episodes
    maxPointsDrawn: 900,      // beyond this the series is bucketed
    rawOpacity: 0.34,
    padding: { top: 5, right: 3, bottom: 5, left: 3 },
  },

  /* ---- the episode browser ----------------------------------------- *
     Newest first, and capped: a thousand-episode run would otherwise put
     a thousand rows in the document and rebuild them all every frame.
     What is dropped is said so, never silently.                        */
  episodes: {
    maxRows: 120,
  },

  /* ---- transitions ------------------------------------------------- */
  timing: {
    sidebarMs: 200,
    hoverMs: 170,
  },
};
