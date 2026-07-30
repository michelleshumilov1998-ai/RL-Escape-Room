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
    hairline: '--hairline',
    hairlineFaint: '--hairline-faint',
    field: '--cell-floor',
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
