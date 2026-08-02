'use strict';

/* =====================================================================
   The real producer.  Replaces `mock.js`.

   Same two questions as the mock answered — what is this room, and what
   happened — but the answers come from the Python side over HTTP instead
   of being invented here.  Nothing downstream changed to make this work:
   the renderer, the playback controller, the charts and the shell all
   read `contract.js` and have never heard of either module.

   ---------------------------------------------------------------------
   THE ONE DIFFERENCE FROM THE MOCK, AND IT IS UNAVOIDABLE
   ---------------------------------------------------------------------

   The mock could invent a finished run instantly, so `room()` and
   `batch()` were plain function calls.  A real run has to be trained
   before there is anything to watch, and training happens on the far end
   of a network, so both are promises here.

   Training is driven from this side in slices: each request runs as much
   as fits in a time budget and returns.  A single "train it all" call
   would be simpler and would also mean one HTTP request that takes
   several seconds, during which the page could report nothing — so
   instead the caller is handed progress as it goes and can draw it.

   ===================================================================== */

window.Producer = (function () {

  let definition = null;
  let batch = null;

  /* The session lives in `sim.js`, which owns the request plumbing and the
     session's lifetime. This module only knows the sequence. */
  const S = window.Sim;

  /**
   * Open the room. Nothing is trained here.
   *
   * The definition arrives in the very first response, so the chamber can be
   * drawn at once and the player can look at it, read what it is and change
   * a parameter before anything runs. Training is what Play is for.
   */
  async function open(roomId) {
    await S.open(roomId);
    definition = S.describe.definition;
    batch = null;
    return definition;
  }

  /**
   * Run until the algorithm says it is finished.
   *
   * The guard is a runaway-loop backstop, not a policy: the run ends when
   * the session leaves TRAINING. Without it a server that never reported
   * TRAINED would spin here forever.
   */
  /**
   * One slice of training, drawn by the caller.
   *
   * This is the whole of what Play does now. The screen asks for a small
   * amount of work per frame and draws the world that comes back, so the
   * agent is watched *while* it learns rather than afterwards. `steps` is
   * the animated speeds; `budgetMs` is Turbo, which does as much as fits in
   * the time and is not worth drawing every step of.
   *
   * Resolves to null when a request is already outstanding, so a slow frame
   * is skipped rather than stacking work up.
   */
  function slice(options) {
    return S.advance(options);
  }

  /** Begin training. Idempotent: pressing Play twice is not two runs. */
  function begin() {
    return S.play();
  }

  function hold() {
    return S.pause();
  }

  /** Fetch what has been recorded so far, for the replay panels. */
  async function episodes() {
    batch = await S.episodes();
    return batch;
  }

  /**
   * One run of the learned policy, with the exploration taken out.
   *
   * Comes back shaped as an Episode, so it goes wherever a recorded episode
   * goes. This is the route the agent settled on, which no recorded episode
   * is: ε stops at its floor rather than reaching zero, so every episode that
   * was trained on still has random steps in it.
   */
  async function replay() {
    const snapshot = await S.replay();
    return snapshot ? snapshot.replay : null;
  }

  /**
   * Back to episode zero with the parameters as they now stand.
   *
   * Reset-scope parameters are applied by the server on Reset, which is why
   * the values are sent first and the reset follows. Nothing is trained here
   * either: Reset returns the chamber to the state it opened in.
   */
  async function restart(values) {
    if (values) await S.setParameters(values);
    await S.reset();
    definition = S.describe.definition;
    batch = null;
    return definition;
  }

  /**
   * The learned policy measured on training, validation and unseen layouts,
   * with the random baseline beside it. Weights are frozen throughout.
   */
  function evaluate(layouts) {
    return S.evaluate(layouts);
  }

  /**
   * A warehouse the agent has never trained on, flown by the learned policy
   * and recorded frame by frame. Called again, it picks a different one.
   */
  function testRoom(seed) {
    return S.testRoom(seed);
  }

  return {
    open: open,
    begin: begin,
    hold: hold,
    slice: slice,
    episodes: episodes,
    replay: replay,
    restart: restart,
    evaluate: evaluate,
    testRoom: testRoom,

    /* Read back without a round trip. The shell asks for these while
       painting, which must not be async. */
    get room() { return definition; },
    get batch() { return batch || { episodes: [], metrics: [] }; },
    get snapshot() { return S.snapshot; },
    get algorithms() { return S.describe ? S.describe.algorithms : []; },
    get algorithmDefault() {
      return S.describe ? S.describe.algorithmDefault : null;
    },

    /** Which single number the status strip shows, and its live value. */
    metric() {
      const snapshot = S.snapshot;
      return snapshot ? snapshot.metric : null;
    },

    setAlgorithm(key) { return S.setAlgorithm(key); },
    setParameters(values) { return S.setParameters(values); },
    release() { S.release(); },
  };
})();
