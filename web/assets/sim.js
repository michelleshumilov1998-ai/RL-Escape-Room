'use strict';

/* =====================================================================
   The simulation, as the page sees it.

   This is the whole of the boundary.  The page writes intents in and reads
   snapshots out; it never holds an environment, a value table or an
   algorithm, and it cannot advance the simulation by drawing.  Everything
   below is either a request or the last snapshot that came back.

   The simulation itself lives in `game/` on the Python side, which is what
   lets it be tested without a browser and run thousands of steps without
   rendering once.  Swapping that for an in-page engine would mean
   rewriting this file and nothing else.
   ===================================================================== */

window.Sim = (function () {

  let sessionId = null;
  let describe = null;
  let snapshot = null;
  let config = null;
  // One advance in flight at a time. Without this a slow frame would queue
  // requests behind each other and the episode counter would run away.
  let advancing = false;

  async function request(method, path, body) {
    const options = { method: method, headers: {} };
    if (body !== undefined) {
      options.headers['Content-Type'] = 'application/json';
      options.body = JSON.stringify(body);
    }
    const response = await fetch(path, options);

    // A body that will not parse is a real failure and has to say so. An
    // earlier version substituted an empty object here, which turned a
    // malformed response into a page that sat on "Loading" forever with
    // nothing to go on.
    const text = await response.text();
    let payload = null;
    try {
      payload = text ? JSON.parse(text) : {};
    } catch (problem) {
      throw new Error(method + ' ' + path + ': the response is not JSON ('
                      + response.status + ')');
    }

    if (!response.ok) {
      throw new Error(payload.error || (method + ' ' + path + ' failed'));
    }
    return payload;
  }

  function command(name, body) {
    if (!sessionId) return Promise.reject(new Error('no session'));
    return request('POST', '/api/session/' + sessionId + '/' + name,
                   body || {}).then(next => {
      snapshot = next;
      return snapshot;
    });
  }

  return {
    /** Build a session for one room and fetch everything static about it. */
    async open(room) {
      config = await request('GET', '/api/config');
      const created = await request('POST', '/api/session', { room: room });
      sessionId = created.session;
      describe = created.describe;
      snapshot = created.snapshot;
      return snapshot;
    },

    get config() { return config; },
    get describe() { return describe; },
    get snapshot() { return snapshot; },
    get state() { return snapshot ? snapshot.state : 'IDLE'; },
    get playing() { return Boolean(snapshot && snapshot.playing); },
    get busy() { return advancing; },

    play() { return command('play'); },
    pause() { return command('pause'); },
    step() { return command('step'); },
    reset() { return command('reset'); },
    replay() { return command('replay'); },

    /**
     * The episodes recorded during training, whole.
     *
     * The only request that does not answer with a snapshot, so it must not
     * go through `command` — that would overwrite the last snapshot with a
     * batch and every reader of `Sim.snapshot` would then be looking at the
     * wrong kind of object.
     */
    episodes() {
      if (!sessionId) return Promise.reject(new Error('no session'));
      return request('POST', '/api/session/' + sessionId + '/episodes', {});
    },
    setParameters(values) { return command('parameters', { values: values }); },
    resetParameter(name) { return command('parameter-default', { name: name }); },
    setAlgorithm(key) { return command('algorithm', { key: key }); },

    /**
     * Do some work. `steps` for the animated tiers, `budgetMs` for Turbo.
     *
     * Resolves to null when a request is already outstanding, so the caller
     * can simply skip that frame rather than stacking work up.
     */
    async advance(options) {
      if (advancing) return null;
      advancing = true;
      try {
        return await command('advance', options);
      } finally {
        advancing = false;
      }
    },

    /** Let the server drop the session. Called on the way out. */
    release() {
      if (!sessionId) return;
      const path = '/api/session/' + sessionId;
      sessionId = null;
      // keepalive, so it still goes out while the page is being replaced.
      fetch(path, { method: 'DELETE', keepalive: true }).catch(() => {});
    },
  };
})();
