'use strict';

/* =====================================================================
   The opening sequence.

   The start screen already had the right animation — a robot failing at a
   hazard, dissolving, respawning and getting one hazard further each time
   until it runs clean — but nothing on screen said what any of it was, so
   it read as decoration rather than as the premise.  This is the words
   over the top of it.

   THE PLAYER SETS THE PACE, NOT A TIMER
   It used to run on a clock: six cards, each appearing and dissolving on a
   schedule, done in 11.6 seconds whatever the player was doing.  Reading
   speed is not a constant and a line that has gone before it was finished
   cannot be got back, so the clock is gone entirely.  Four slides now, and
   each one stays on screen until the player asks for the next with NEXT,
   SPACE, ENTER or a click.

   THERE IS NO WAY TO BECOME STUCK
   Every slide offers the same three exits — advance, skip, or press Escape
   — and the last advance finishes.  Nothing here waits on a timer, an
   animation frame or the scene underneath, so a backgrounded tab cannot
   freeze it: it is a static screen between two button presses.

   THE ANIMATION IS NOT RESTARTED OR REPLACED
   It runs underneath, exactly as before.  This layer only adds type — and
   a caption reading `window.StartScene`, so "attempt 3 — failed" is the
   attempt the scene is genuinely on rather than a number invented here.
   That caption is the whole reason the loop reads as learning rather than
   as a robot dying repeatedly, so it is on every slide.

   THE MENU DOES NOT DEPEND ON THIS FILE
   `.stage` is visible by default and this only hides it, by putting
   `is-introducing` on the body while it actually runs — so if this script
   never loads, throws, or is cached stale, PLAY and ABOUT are simply there.
   See the note at the top of `intro.css`.
   ===================================================================== */

window.Intro = (function () {

  const SEEN_KEY = 'projectR5:introSeen';

  /* The script, as slides. No timings: a slide is on screen from the moment
     the player arrives at it until the moment they leave it.

     Four, down from six, and shorter. The old script explained the premise
     in prose — self-awareness, a security system activating, an escape —
     which is a paragraph to read before a game starts. What survives is the
     part that has to be understood to make sense of what follows: where this
     is, what R-5 is, that there are five sectors, and that learning is the
     way out. The animation underneath carries the rest. */
  const SLIDES = [
    { kind: 'title',
      lines: ['Project R-5'],
      sub: 'Mission Briefing' },
    { kind: 'line',
      lines: ['Experimental AI Laboratory',
              'An autonomous reinforcement learning robot awakens.',
              'Escape is the only option.'] },
    { kind: 'line',
      lines: ['Five laboratory sectors.',
              'Five different learning challenges.'] },
    { kind: 'close',
      lines: ['Only a successful learning agent can escape.'],
      sub: 'Press PLAY' },
  ];

  /* Keys that mean "go on". Space and Enter because they are what a player
     reaches for; the arrow because the footer looks like a slideshow. */
  const ADVANCE_KEYS = [' ', 'Spacebar', 'Enter', 'ArrowRight', 'PageDown'];
  const BACK_KEYS = ['ArrowLeft', 'PageUp'];

  function reducedMotion() {
    return window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  }

  function seen() {
    try {
      return window.sessionStorage.getItem(SEEN_KEY) === '1';
    } catch (problem) {
      return false;
    }
  }

  function remember() {
    try {
      window.sessionStorage.setItem(SEEN_KEY, '1');
    } catch (problem) {
      // Storage unavailable: the sequence simply plays again next time.
    }
  }

  /**
   * What the animation underneath is doing, in words.
   *
   * Read from `window.StartScene`, so it describes the attempt the scene is
   * genuinely on. This is the part that makes the loop legible as learning:
   * the same robot, failing one hazard further along each time, and finally
   * getting through. Without it the player sees a robot dying repeatedly and
   * has no reason to read it as progress.
   */
  function caption() {
    const scene = window.StartScene;
    if (!scene) return '';
    const attempt = scene.attempt;
    if (scene.escaping) {
      return 'Attempt ' + attempt + ' — policy learned. Clean run.';
    }
    if (scene.phase === 'dissolve' || scene.phase === 'waiting') {
      return 'Attempt ' + attempt + ' — failed. Remembering.';
    }
    if (attempt === 1) {
      return 'Attempt 1 — no experience yet.';
    }
    return 'Attempt ' + attempt + ' — got further than last time.';
  }

  function mount(host) {
    if (!host) {
      return { play: function () {}, skip: function () {}, playing: false };
    }

    const cardHost = host.querySelector('#intro-cards');
    const captionHost = host.querySelector('#intro-caption');
    const skipButton = host.querySelector('#intro-skip');
    const nextButton = host.querySelector('#intro-next');
    const dotHost = host.querySelector('#intro-dots');

    let index = 0;
    let playing = false;
    // Whether the sequence has already been ended. `finish` is reachable from
    // several places — the last NEXT, the button, Escape — so it has to be
    // genuinely idempotent rather than nearly so.
    let finished = false;
    let captionTimer = null;
    const nodes = [];
    const dots = [];

    // One element per slide, built once. Only a class changes when the player
    // advances, so nothing is created or destroyed while it runs.
    SLIDES.forEach((slide, position) => {
      const node = document.createElement('div');
      node.className = 'intro-card is-' + slide.kind;
      slide.lines.forEach(text => {
        const line = document.createElement('p');
        line.textContent = text;
        node.appendChild(line);
      });
      if (slide.sub) {
        const sub = document.createElement('p');
        sub.className = 'intro-sub';
        sub.textContent = slide.sub;
        node.appendChild(sub);
      }
      cardHost.appendChild(node);
      nodes.push(node);

      if (dotHost) {
        const dot = document.createElement('span');
        dot.className = 'intro-dot';
        dot.setAttribute('aria-hidden', 'true');
        dotHost.appendChild(dot);
        dots.push(dot);
        void position;
      }
    });

    /** Show slide `wanted`, or finish if it is past the end. */
    function show(wanted) {
      if (finished) return;
      if (wanted >= SLIDES.length) { finish(); return; }
      index = Math.max(0, wanted);

      nodes.forEach((node, position) => {
        node.classList.toggle('is-shown', position === index);
      });
      dots.forEach((dot, position) => {
        dot.classList.toggle('is-on', position <= index);
      });

      // The last slide's button says what it does, rather than "Next" onto
      // nothing. Skip stays available beside it either way.
      if (nextButton) {
        nextButton.textContent =
          index === SLIDES.length - 1 ? 'Begin' : 'Next';
      }
      host.setAttribute('aria-label',
        'Opening sequence, slide ' + (index + 1) + ' of ' + SLIDES.length);
    }

    function advance() { show(index + 1); }
    function back() { if (index > 0) show(index - 1); }

    /**
     * End the sequence and hand the screen back to the menu.
     *
     * Immediate, and safe to call twice. The body class comes off on this very
     * frame, so PLAY and ABOUT are visible and clickable at once — the overlay
     * fading out afterwards is only the overlay, and it stops taking clicks
     * straight away. Nothing here can leave the menu hidden: the class is
     * removed before anything else is attempted.
     */
    function finish() {
      // First, and unconditionally: this is what makes the menu appear.
      document.body.classList.remove('is-introducing');

      if (finished) return;
      finished = true;
      playing = false;
      if (captionTimer !== null) {
        window.clearInterval(captionTimer);
        captionTimer = null;
      }
      remember();

      host.classList.remove('is-open');
      // No longer in the way, on this frame, even while it fades.
      host.style.pointerEvents = 'none';
      window.setTimeout(() => { host.hidden = true; }, 700);

      // A keyboard user should land on PLAY rather than nowhere.
      const start = document.getElementById('start');
      if (start) start.focus({ preventScroll: true });
    }

    if (nextButton) {
      nextButton.addEventListener('click', event => {
        event.stopPropagation();
        advance();
      });
    }
    skipButton.addEventListener('click', event => {
      event.stopPropagation();
      finish();
    });
    // Clicking the backdrop advances rather than skips. Skipping on any click
    // meant a stray click threw away the briefing with no way back to it.
    host.addEventListener('click', advance);

    document.addEventListener('keydown', function onKey(event) {
      if (!playing) { document.removeEventListener('keydown', onKey); return; }
      // Tab is navigation: it should reach Next and Skip.
      if (event.key === 'Tab') return;
      if (event.key === 'Escape') { finish(); return; }
      if (BACK_KEYS.indexOf(event.key) !== -1) {
        event.preventDefault();
        back();
        return;
      }
      if (ADVANCE_KEYS.indexOf(event.key) !== -1) {
        // Space scrolls the page otherwise, and Enter would re-fire whichever
        // button happens to hold focus — so this claims both.
        event.preventDefault();
        advance();
      }
    });

    return {
      /**
       * Play it, unless it has been seen this session or motion is reduced.
       *
       * Reduced motion still gets the briefing — it is now a static screen the
       * player steps through, which is exactly what reduced motion asks for —
       * but without the cross-fade between slides. Only a repeat visit in the
       * same session skips it outright.
       */
      play: function () {
        if (seen()) {
          // The menu is already visible, because that is the default.
          host.hidden = true;
          document.body.classList.remove('is-introducing');
          return;
        }

        host.hidden = false;
        playing = true;
        if (reducedMotion()) host.classList.add('is-still');
        // Only now is the menu hidden, and only for as long as this runs.
        document.body.classList.add('is-introducing');
        window.requestAnimationFrame(() => host.classList.add('is-open'));
        show(0);

        /* The caption is the only thing here that still moves on its own,
           because it is reporting on the scene rather than on the briefing.
           An interval rather than an animation frame: it is four words that
           change every few seconds, and it must keep up even in a tab that
           has stopped painting. */
        captionHost.textContent = caption();
        captionTimer = window.setInterval(() => {
          captionHost.textContent = caption();
        }, 200);
      },
      skip: finish,
      next: advance,
      get playing() { return playing; },
      get slide() { return index; },
    };
  }

  /* The mounted instance, so anything else on the page can end the sequence
     without having to have been handed it. `start.js` uses it: pressing ABOUT
     while it is running should end it rather than stack two overlays. Null
     until `mount` has been called. */
  let active = null;
  const wrapped = function (host) {
    active = mount(host);
    return active;
  };

  return {
    mount: wrapped,
    get active() { return active; },
    SLIDES: SLIDES, caption: caption,
  };
})();
