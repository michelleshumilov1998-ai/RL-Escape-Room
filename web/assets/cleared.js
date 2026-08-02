'use strict';

/* =====================================================================
   The sector-cleared flash.

   Shown the moment a chamber is genuinely beaten, on both screens: the
   planner screen for room 1 and the room screen for rooms 2 to 5.  One
   component rather than two, because "you cleared a sector" has to look
   the same wherever it happens — that is most of what makes five rooms
   feel like one game.

   IT DOES NOT INTERRUPT ANYTHING
   It is a `pointer-events: none` layer over the top, it clears itself
   after a few seconds, and it touches no control and no run state.  A run
   carries on training behind it.  That is deliberate: the brief is to
   celebrate the moment, not to make the player dismiss a dialog.

   Shown once per clearing.  The screens latch that themselves — see
   `everCleared` in the room shell — so pressing Reset and clearing again
   shows it again, and repainting does not.
   ===================================================================== */

window.Cleared = (function () {

  /* How long the flash lasts, in milliseconds. Five seconds, as briefed:
     long enough to read three lines, short enough not to be in the way. The
     fade out is the tail of it rather than an addition to it. */
  const VISIBLE_MS = 5000;
  const FADE_MS = 700;

  function mount(host) {
    if (!host) return { show: function () {}, hide: function () {} };

    const title = host.querySelector('.cleared-title');
    const sector = host.querySelector('.cleared-sector');
    const next = host.querySelector('.cleared-next');
    let timer = null;
    let fading = null;

    function hide() {
      if (timer) { window.clearTimeout(timer); timer = null; }
      if (fading) { window.clearTimeout(fading); fading = null; }
      host.classList.remove('is-open');
      host.hidden = true;
    }

    return {
      /**
       * Flash it.
       *
       * `options.sector` is what the chamber is called, so the middle line
       * names the sector actually cleared rather than saying "a sector".
       * `options.last` swaps the third line: there is no next sector after
       * the fifth, and promising one would be a small lie on the one screen
       * that should not tell any.
       */
      show: function (options) {
        const settings = options || {};
        hide();

        title.textContent = 'Mission Complete';
        sector.textContent = settings.sector
          ? settings.sector + ' cleared' : 'Sector cleared';
        next.textContent = settings.last
          ? 'Final sector — mission data secured'
          : 'Preparing next sector…';

        host.hidden = false;
        // Next frame, so the transition has a value to animate from.
        window.requestAnimationFrame(function () {
          host.classList.add('is-open');
        });

        timer = window.setTimeout(function () {
          host.classList.remove('is-open');
          fading = window.setTimeout(hide, FADE_MS);
        }, VISIBLE_MS);
      },

      hide: hide,
      get isOpen() { return !host.hidden; },
    };
  }

  return { mount: mount, VISIBLE_MS: VISIBLE_MS };
})();
