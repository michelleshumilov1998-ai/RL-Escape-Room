'use strict';

/* =====================================================================
   Playback, and the run state machine.

   THE STATE MACHINE
   -----------------
   Four states and one transition table.  Nothing outside this file
   decides what may happen next, and an illegal transition is not
   something the interface has to remember to prevent — `apply` simply
   has no entry for it and returns the state unchanged.

     IDLE        not advancing.  Either nothing has been played yet, or
                 playback is held part-way through.
     RUNNING     advancing through the batch, episode after episode.
     FINISHED    the last episode of the batch has been played out.
     REPLAYING   advancing through one chosen episode, on a loop.

   Paused is IDLE.  Where playback is up to is *data* — a cursor — not a
   control flag, so holding a run and resuming it needs no fifth state
   and no boolean anywhere.  What Play resumes is likewise read from the
   data: an episode has been chosen, or it has not.

   THE LOOP
   --------
   One requestAnimationFrame loop, owned here.  The three animated tiers
   walk the steps at a rate and interpolate between them.  Turbo is
   different in kind: it consumes whole episodes inside a time budget and
   draws only where it ended up, which is the only way a thousand-episode
   batch is watchable.  The budget is what keeps Pause, Reset and Exit
   answering at every speed.
   ===================================================================== */

window.Playback = (function () {

  const C = window.ROOM_CONFIG;

  const IDLE = 'IDLE';
  const RUNNING = 'RUNNING';
  const FINISHED = 'FINISHED';
  const REPLAYING = 'REPLAYING';

  /* Read once at load. A batch is not data the interface may edit, so
     every transition below either returns a new state name or the one it
     was given. */
  const TRANSITIONS = {
    IDLE: { play: 'resume', step: IDLE, reset: IDLE, select: REPLAYING,
            clear: IDLE, exhaust: FINISHED },
    RUNNING: { pause: IDLE, step: IDLE, reset: IDLE, select: REPLAYING,
               exhaust: FINISHED },
    FINISHED: { reset: IDLE, select: REPLAYING },
    REPLAYING: { pause: IDLE, step: IDLE, reset: IDLE, select: REPLAYING,
                 clear: IDLE },
  };

  function create(handlers) {
    const on = handlers || {};

    const self = {
      state: IDLE,
      room: null,
      batch: null,
      speed: C.speedDefault,

      /* Where playback is up to. `episode` indexes the batch, `step`
         indexes that episode's steps, `part` is the fraction of the way
         to the next step. */
      cursor: { episode: 0, step: 0, part: 0 },

      /* Which episode the replay browser has chosen, or null. */
      selected: null,

      /* Metrics for the episodes played so far, so the charts fill in as
         the run goes rather than all at once at the end. */
      metrics: [],

      frame: null,
      lastTime: 0,
      hold: 0,          // the beat on a replay's final frame
    };

    /* -------------------------------------------------------------------
       The machine
       ----------------------------------------------------------------- */

    function apply(action) {
      const allowed = TRANSITIONS[self.state];
      const next = allowed ? allowed[action] : undefined;
      if (next === undefined) return false;      // illegal: nothing happens

      if (next === 'resume') {
        self.state = self.selected === null ? RUNNING : REPLAYING;
      } else {
        self.state = next;
      }
      if (on.state) on.state(self.state);
      return true;
    }

    /* -------------------------------------------------------------------
       Loading
       ----------------------------------------------------------------- */

    function load(room, batch) {
      self.room = room;
      self.batch = batch;
      self.state = IDLE;
      self.cursor = { episode: 0, step: 0, part: 0 };
      self.selected = null;
      self.metrics = [];
      self.hold = 0;
      if (on.state) on.state(self.state);
    }

    /**
     * The episode with a given episode *number* — not array position.
     *
     * The batch is a sample of the run: episodes 0, 1, 2, 43, 86, … 1499 sit
     * at positions 0…39. So a number is not a position, and treating one as
     * the other quietly resolves to nothing for every episode whose number
     * runs past the length of the batch. That is what left the step inspector
     * offering three episodes out of forty.
     */
    function episodeNumbered(number) {
      if (!self.batch) return null;
      const found = self.batch.episodes.filter(
        episode => episode.index === number);
      return found.length ? found[0] : null;
    }

    /** The episode playback is currently walking, batch or replay. */
    function currentEpisode() {
      if (!self.batch || !self.batch.episodes.length) return null;
      // A selection is an episode number; the cursor is a position, because
      // it is counting its way along the batch.
      if (self.selected !== null) return episodeNumbered(self.selected);
      const at = Math.min(self.cursor.episode,
                          self.batch.episodes.length - 1);
      return self.batch.episodes[at];
    }

    /* -------------------------------------------------------------------
       Moving through the batch
       ----------------------------------------------------------------- */

    /** Record an episode's metrics once, as it completes. */
    function completeEpisode(index) {
      const entry = self.batch.metrics[index];
      if (!entry) return;
      if (self.metrics.length && self.metrics[self.metrics.length - 1]
          .episode === entry.episode) {
        return;
      }
      self.metrics.push(entry);
      if (on.episode) on.episode(entry);
    }

    /**
     * Advance the batch cursor by `count` whole steps.
     *
     * Returns false when the batch runs out, which is the only way
     * FINISHED is reached.
     */
    function advanceSteps(count) {
      let left = count;
      while (left > 0) {
        const episode = self.batch.episodes[self.cursor.episode];
        if (!episode) return false;

        const remaining = episode.steps.length - 1 - self.cursor.step;
        if (remaining > left) {
          self.cursor.step += left;
          return true;
        }

        // This episode is done; bank it and move to the next.
        left -= remaining;
        completeEpisode(self.cursor.episode);
        if (self.cursor.episode >= self.batch.episodes.length - 1) {
          self.cursor.step = episode.steps.length - 1;
          self.cursor.part = 0;
          return false;
        }
        self.cursor.episode += 1;
        self.cursor.step = 0;
        left -= 1;                    // the move into the next episode
      }
      return true;
    }

    /**
     * How much faster this room runs than a grid of ten cells.
     *
     * One number scales both the training tiers and the replay, so a room
     * that declares a quick pace is quick everywhere and the speed chips
     * keep their relative meaning.
     */
    function paceScale() {
      const rate = self.room && self.room.playback
        && self.room.playback.stepsPerSecond;
      return rate ? rate / C.replayStepsPerSecond : 1;
    }

    function runAnimated(dt) {
      const tier = C.speeds[self.speed];
      self.cursor.part += dt * tier.stepsPerSecond * paceScale();
      if (self.cursor.part < 1) return;

      const whole = Math.floor(self.cursor.part);
      self.cursor.part -= whole;
      if (!advanceSteps(whole)) {
        self.cursor.part = 0;
        apply('exhaust');
      }
    }

    function runTurbo() {
      // Time-budgeted, and capped, so neither a slow machine nor a fast
      // one can turn one frame into a stall.
      const deadline = performance.now() + C.turboBudgetMs;
      let episodes = 0;

      while (performance.now() < deadline
             && episodes < C.turboMaxEpisodesPerFrame) {
        const episode = self.batch.episodes[self.cursor.episode];
        if (!episode) break;

        completeEpisode(self.cursor.episode);
        episodes += 1;

        if (self.cursor.episode >= self.batch.episodes.length - 1) {
          self.cursor.step = episode.steps.length - 1;
          self.cursor.part = 0;
          apply('exhaust');
          return;
        }
        self.cursor.episode += 1;
        self.cursor.step = 0;
        self.cursor.part = 0;
      }
    }

    function runReplay(dt) {
      const episode = currentEpisode();
      if (!episode) return;

      if (self.hold > 0) {
        self.hold -= dt;
        if (self.hold > 0) return;
        self.cursor.step = 0;
        self.cursor.part = 0;
      }

      // A room may set its own replay pace: a continuous room takes
      // hundreds of small steps to cross itself, and the rate that suits
      // ten cells would take minutes to get through one flight.
      const rate = (self.room && self.room.playback
                    && self.room.playback.stepsPerSecond)
        || C.replayStepsPerSecond;
      self.cursor.part += dt * rate;
      while (self.cursor.part >= 1) {
        self.cursor.part -= 1;
        self.cursor.step += 1;
        if (self.cursor.step >= episode.steps.length - 1) {
          // Hold on the last frame before going round again.
          self.cursor.step = episode.steps.length - 1;
          self.cursor.part = 0;
          self.hold = C.replayPauseSeconds;
          break;
        }
      }
    }

    /* -------------------------------------------------------------------
       The snapshot handed to the renderer
       ----------------------------------------------------------------- */

    function reducedMotion() {
      return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    }

    /**
     * Where the agent is now, in world units.
     *
     * Between two steps when it is moving, exactly on a step when reduced
     * motion is asked for or when playback is held. `moveTween` shortens
     * the slide so the agent settles before the next step rather than
     * gliding continuously.
     */
    function position(episode) {
      const steps = episode.steps;
      const index = Math.min(self.cursor.step, steps.length - 1);
      const current = steps[index].position;

      // A replay is paced by `replayStepsPerSecond` rather than by the
      // speed tier, so it interpolates whatever that tier is. Without
      // this, asking to see the final route while Turbo is still selected
      // would snap the agent from cell to cell.
      const animated = self.state === REPLAYING
                    || C.speeds[self.speed].animated;
      if (reducedMotion() || !animated || self.state === IDLE
          || index >= steps.length - 1) {
        return { x: current.x, y: current.y };
      }

      const next = steps[index + 1].position;
      const share = Math.min(1, self.cursor.part / C.moveTween);
      return {
        x: current.x + (next.x - current.x) * share,
        y: current.y + (next.y - current.y) * share,
      };
    }

    /* Both of these take the step to work up to, rather than reading the
       cursor, so the step inspector can build any frame of any episode
       without disturbing playback. */

    function trailFor(episode, upTo) {
      const from = Math.max(0, upTo - C.trail.length);
      const points = [];
      for (let index = from; index <= upTo; index += 1) {
        const step = episode.steps[index];
        if (step) points.push(step.position);
      }
      return points;
    }

    /**
     * Where the things that move are, at this step.
     *
     * Accumulated the same way as the states: the last position seen
     * stands until another arrives, so a guard that reported its cell ten
     * steps ago has not vanished.
     */
    function positionsFor(episode, upTo) {
      const positions = {};
      for (let index = 0; index <= upTo; index += 1) {
        const step = episode.steps[index];
        if (!step || !step.entityPositions) continue;
        step.entityPositions.forEach(entry => {
          positions[entry.id] = entry.position;
        });
      }
      return positions;
    }

    function statesFor(episode, upTo) {
      // Accumulated from the start of the episode: a plank that gave way
      // twenty steps ago has not come back.
      const states = {};
      for (let index = 0; index <= upTo; index += 1) {
        const step = episode.steps[index];
        if (!step || !step.entityStates) continue;
        step.entityStates.forEach(entry => { states[entry.id] = entry.state; });
      }
      return states;
    }

    /** Everything the renderer needs, and nothing it does not. */
    function snapshot() {
      const episode = currentEpisode();
      if (!episode) return { room: self.room, position: null };

      const index = Math.min(self.cursor.step, episode.steps.length - 1);
      const step = episode.steps[index];

      return {
        room: self.room,
        position: position(episode),
        velocity: step.velocity || null,
        facing: step.velocity ? headingOf(step) : null,
        trail: trailFor(episode, self.cursor.step),
        entityStates: statesFor(episode, self.cursor.step),
        entityPositions: positionsFor(episode, self.cursor.step),
        overlays: self.overlays || null,
        observation: self.room.observation
          ? Object.assign({}, self.room.observation,
                          { heading: headingOf(step) })
          : null,
      };
    }

    /** Which way the agent is facing, for the observation cone. */
    function headingOf(step) {
      if (!step.velocity) return 0;
      if (step.velocity.x === 0 && step.velocity.y === 0) return 0;
      return Math.atan2(step.velocity.y, step.velocity.x);
    }

    /* -------------------------------------------------------------------
       The frame loop
       ----------------------------------------------------------------- */

    function tick(now) {
      const dt = Math.min(0.05, (now - self.lastTime) / 1000) || 0;
      self.lastTime = now;

      if (self.batch && self.batch.episodes.length) {
        if (self.state === RUNNING) {
          if (C.speeds[self.speed].animated) runAnimated(dt);
          else runTurbo();
        } else if (self.state === REPLAYING) {
          runReplay(dt);
        }
      }

      if (on.frame) on.frame(snapshot());
      self.frame = window.requestAnimationFrame(tick);
    }

    function start() {
      if (self.frame !== null) return;
      self.lastTime = performance.now();
      self.frame = window.requestAnimationFrame(tick);
    }

    function stop() {
      if (self.frame === null) return;
      window.cancelAnimationFrame(self.frame);
      self.frame = null;
    }

    /* -------------------------------------------------------------------
       The controls. Each one is a transition and then its consequence.
       ----------------------------------------------------------------- */

    return {
      load: load,
      start: start,
      stop: stop,
      snapshot: snapshot,

      get state() { return self.state; },
      get metrics() { return self.metrics; },
      get cursor() { return self.cursor; },
      get selected() { return self.selected; },
      get speed() { return self.speed; },
      get episode() { return currentEpisode(); },

      /** How many episodes the batch holds at all. */
      get total() {
        return self.batch ? self.batch.episodes.length : 0;
      },

      /** One episode by its episode number, for the replay browser. */
      episodeAt: function (number) {
        return episodeNumbered(number);
      },

      /**
       * Reconstruct one exact frame of one episode.
       *
       * Presentation only: it reads the batch and touches neither the
       * cursor nor the state, so the step inspector can be scrubbed while
       * a run is going and change nothing about it. The position is the
       * step's own — never interpolated, because this is a still.
       */
      frameAt: function (episodeNumber, stepIndex) {
        const episode = episodeNumbered(episodeNumber);
        if (!episode) return null;

        const last = episode.steps.length - 1;
        const at = Math.max(0, Math.min(last, stepIndex));
        const step = episode.steps[at];

        // Everything paid for on the way to this step, this step included.
        let total = 0;
        for (let index = 0; index <= at; index += 1) {
          total += episode.steps[index].reward;
        }

        return {
          snapshot: {
            room: self.room,
            position: { x: step.position.x, y: step.position.y },
            trail: trailFor(episode, at),
            entityStates: statesFor(episode, at),
            entityPositions: positionsFor(episode, at),
            observation: self.room.observation
              ? Object.assign({}, self.room.observation,
                              { heading: headingOf(step) })
              : null,
            facing: step.velocity ? headingOf(step) : null,
          },
          episode: episode,
          step: step,
          index: at,
          lastIndex: last,
          cumulativeReward: total,
        };
      },

      /** How far through the current episode, as a share. For the scrubber. */
      get progress() {
        const episode = currentEpisode();
        if (!episode || episode.steps.length < 2) return 0;
        return self.cursor.step / (episode.steps.length - 1);
      },

      /** How many episodes have been played out. For the status strip. */
      get episodesPlayed() { return self.metrics.length; },

      play: function () { return apply('play'); },
      pause: function () { return apply('pause'); },

      step: function () {
        // One entry forward, then held. Legal from IDLE, RUNNING and
        // REPLAYING; the table refuses it once FINISHED.
        if (!apply('step')) return false;
        self.cursor.part = 0;
        self.hold = 0;
        if (self.selected === null) {
          if (!advanceSteps(1)) apply('exhaust');
          return true;
        }
        const episode = currentEpisode();
        if (self.cursor.step < episode.steps.length - 1) self.cursor.step += 1;
        return true;
      },

      reset: function () {
        if (!apply('reset')) return false;
        self.cursor = { episode: 0, step: 0, part: 0 };
        self.selected = null;
        self.metrics = [];
        self.hold = 0;
        return true;
      },

      selectEpisode: function (index) {
        if (!apply('select')) return false;
        self.selected = index;
        self.cursor.step = 0;
        self.cursor.part = 0;
        self.hold = 0;
        return true;
      },

      /** Leave the replay and go back to holding the batch where it was. */
      clearEpisode: function () {
        if (!apply('clear')) return false;
        self.selected = null;
        self.cursor.part = 0;
        self.hold = 0;
        return true;
      },

      /** Jump within the episode being shown. Presentation only. */
      scrubTo: function (share) {
        const episode = currentEpisode();
        if (!episode) return;
        const last = episode.steps.length - 1;
        self.cursor.step = Math.max(0, Math.min(last, Math.round(share * last)));
        self.cursor.part = 0;
        self.hold = 0;
      },

      setSpeed: function (key) {
        if (!C.speeds[key]) return;
        self.speed = key;
        self.cursor.part = 0;
      },

      setOverlays: function (overlays) { self.overlays = overlays; },

      /** True when a run is part-way through and would be lost. */
      get inProgress() {
        if (self.state === FINISHED) return false;
        return self.metrics.length > 0 || self.cursor.step > 0
               || self.cursor.episode > 0;
      },
    };
  }

  return {
    create: create,
    IDLE: IDLE, RUNNING: RUNNING, FINISHED: FINISHED, REPLAYING: REPLAYING,
  };
})();
