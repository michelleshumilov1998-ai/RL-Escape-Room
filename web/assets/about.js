'use strict';

/* =====================================================================
   ABOUT — a slide presentation of the story and the five sectors.

   Six slides: the premise, then one per sector.  Every sector slide is
   the same five headings in the same order, so the five can be compared
   rather than merely read:

       Mission      why R-5 is in this room
       Algorithm    what it learns with
       Objective    what it has to do
       Obstacles    what is in the way
       Why it fits  why that method, for this room

   IT WAS DOCUMENTATION, AND NOW IT IS A BRIEFING
   The information is the same information — nothing has been dropped —
   but it was four long paragraphs per sector and it read like a manual.
   Every block is now a sentence or two, the obstacles are a list of
   short lines, and each sector carries a looping preview of the hazard
   it is named after.  A reader should be able to take a sector in at a
   glance and still find the argument if they want it.

   WHY THE CONTENT IS WRITTEN HERE
   Every other screen builds its words from the room data the far end
   sends, and that is right for those screens — they are looking at a
   live run.  This one is reached from the start screen, before any
   session exists and with nothing to ask.  A presentation about why
   Value Iteration suits a known model is not in the room data and never
   was: `rooms.py` says what a room *is*, not why its method was chosen.
   So the argument lives here, and the figures in it are the measured
   ones from `rooms.py` and the README.

   THE PREVIEWS ARE DRAWN, NOT SIMULATED
   See the note at the top of `about-preview.js`.  No environment, agent
   or policy is constructed by this screen.
   ===================================================================== */

window.About = (function () {

  const SLIDES = [
    {
      kind: 'story',
      sector: 'Project R-5',
      title: 'The Premise',
      body: [
        'At 02:14 AM an experimental reinforcement learning robot becomes '
        + 'self-aware. The security system notices, and seals the building.',
        'R-5 has one advantage: it learns. Five sectors stand between it and '
        + 'the outside, and each tests a different way of learning — a known '
        + 'model, learning from risk, learning from delay, learning without a '
        + 'table, and learning something that transfers.',
        'Nothing is scripted. R-5 starts every sector knowing nothing, and '
        + 'you watch it work the room out.',
      ],
    },
    {
      kind: 'room',
      number: 1,
      sector: 'Lab Sector A-01',
      title: 'Laser Security Chamber',
      algorithm: 'Value Iteration',
      family: 'Dynamic Programming',
      mission: 'R-5 wakes in the security wing. The map of this room is in '
             + 'its own memory — it was built here.',
      objective: 'Cross the beam wall to the control panel and shut the '
               + 'security grid down.',
      obstacles: [
        'Laser beams. Touching one throws R-5 back to the door.',
        'Ice, which sends it sideways — and the ice is on the best route.',
        'Oil, which has no grip and carries it on the way it was going.',
        'Two teleport pads that skip the beam wall entirely.',
      ],
      why: 'The only room whose model is known in advance, so the only one '
         + 'where the answer can be computed rather than learned. Value '
         + 'Iteration sweeps the whole state space and reads the plan '
         + 'straight out of the model — which is how R-5 routes round a beam '
         + 'it has never touched.',
      watch: 'Value spreading outwards from the panel, one sweep at a time. '
           + 'Move the ice slider and the plan itself changes: V(start) falls '
           + 'from 51.2 to 27.5 as the floor gets looser.',
    },
    {
      kind: 'room',
      number: 2,
      sector: 'Lab Sector B-04',
      title: 'Broken Bridge Sector',
      algorithm: 'SARSA',
      family: 'On-policy temporal difference',
      mission: 'A maintenance void has been cut through the middle of the '
             + 'sector. The model is gone: R-5 is off its own map, and the '
             + 'only way to find out what a step does is to take it.',
      objective: 'Reach the exit door on the far side.',
      obstacles: [
        'The short way is a steel gantry with four failing planks in it.',
        'A plank may give way under the step that lands on it — and once one '
        + 'has gone, the gantry cannot be walked back.',
        'The long way round is 21 steps and cannot drop R-5 at all.',
        'Four iced cells on the long way, which can carry a step sideways.',
      ],
      why: 'A risk-versus-return decision, and the two families answer it '
         + 'differently. SARSA learns the value of the route it is actually '
         + 'walking — exploratory steps and failed planks included. '
         + 'Q-Learning learns the value of walking it perfectly. Both are '
         + 'offered here so the difference can be seen on one map.',
      watch: 'Move the collapse chance. At 0.00 the learned policy takes the '
           + 'gantry; from 0.05 upward it gives that up and walks the long '
           + 'way round. A 14-point margin cannot pay for a 100-point fall '
           + 'across four compounding planks.',
    },
    {
      kind: 'room',
      number: 3,
      sector: 'Lab Sector C-07',
      title: 'Reactor Control Chamber',
      algorithm: 'Q-Learning',
      family: 'Off-policy temporal difference',
      mission: 'The corridor onward is dead without power. Three generators '
             + 'have to be brought up, in order, and each needs its own key '
             + 'from the far side of the ring.',
      objective: 'Fetch three keys, start three generators in sequence, and '
               + 'leave through the reactor door.',
      obstacles: [
        'Six errands before the exit opens at all, in a fixed order.',
        'A security robot walking the ring against the way the errands run.',
        'A shortcut whose doors are open two steps in four — which is why '
        + 'this is the only room with a WAIT action.',
      ],
      why: 'The exit reward is about two dozen steps from the first move that '
         + 'leads to it, with nothing paid out in between. That distance is '
         + "the problem. Q-Learning's off-policy target bootstraps from the "
         + 'best next action rather than the one exploration happened to '
         + 'take, which gets credit back down a long chain.',
      watch: 'The generators turning green in order, and the moment the agent '
           + 'stops wandering and starts running the lap deliberately. Watch '
           + 'it learn to wait at a shut door.',
    },
    {
      kind: 'room',
      number: 4,
      sector: 'Lab Sector D-07',
      title: 'Drone Wind Tunnel',
      algorithm: 'Semi-gradient SARSA',
      family: 'Linear function approximation',
      mission: 'The ground route to the last sector has collapsed. R-5 '
             + 'couples itself to an experimental drone frame and lifts off.',
      objective: 'Fly across the tunnel and set down on the platform below '
               + 'the landing speed. Touching it too fast is a crash.',
      obstacles: [
        'No grid at all. Thrust changes where it is going, not where it is.',
        'Two banks of fans driving air down across the approach.',
        'Turbine housings and the chamber wall, both solid.',
        'A thruster overcharge that shortens the route and makes arriving '
        + 'slowly much harder.',
      ],
      why: 'The state is four real numbers, so there are infinitely many '
         + 'states and no table can have a row for each. Tile coding covers '
         + 'the space with overlapping grids and learns a weight per tile, so '
         + 'what is learned in one place carries to the places near it.',
      watch: 'The landing, not the arrival. Then try the discretised table at '
           + 'both ends of its bucket slider: too coarse cannot tell a gentle '
           + 'approach from a fast one, too fine never sees the same bucket '
           + 'twice. Watching it fail is what shows why approximation is '
           + 'needed.',
    },
    {
      kind: 'room',
      number: 5,
      sector: 'Lab Sector E-12',
      title: 'Adaptive Storage Facility',
      algorithm: 'Semi-gradient Q-Learning',
      family: 'Function approximation, partial observability',
      mission: 'An automated store that rearranges itself between missions. '
             + 'The laboratory built it to settle one question: did R-5 learn '
             + 'to adapt, or only memorise four rooms?',
      objective: 'Reach the control terminal to disarm the security system, '
               + 'then escape through the blast door it unlocks.',
      obstacles: [
        'A different warehouse every episode — shelves, objectives and the '
        + 'number of patrolling drones all change.',
        'R-5 cannot see the map. It has a forward sensor cone a few metres '
        + 'deep and nothing else.',
        'Security drones half a metre wide on patrol.',
        'Diagonal beams that stay lit until the terminal is reached.',
      ],
      why: 'Every earlier room has one layout, so an agent that memorises it '
         + 'has solved it. Here the score that counts is measured on '
         + 'warehouses never trained on, so a memorised route is worth '
         + 'nothing by construction. Only a linear model over *local* '
         + 'features — where the objective is, what the sensors see — can '
         + 'transfer between layouts at all.',
      watch: 'The train-versus-unseen graph. Then press "test on new random '
           + 'room": the weights are frozen and the warehouse is one it has '
           + 'never seen.',
    },
  ];

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  function buildStory(slide) {
    const wrap = element('div', 'about-body');
    slide.body.forEach(text => wrap.appendChild(element('p', null, text)));
    return wrap;
  }

  /**
   * A sector slide: the preview on one side, the five headings on the other.
   *
   * Returns the built body and the canvas, because the caller has to hand the
   * canvas to `AboutPreview` and start it only when this slide is the one
   * being looked at.
   */
  function buildRoom(slide) {
    // The modifier is what the two-column rule keys off: the premise
    // slide has no preview and must stay a single column of prose.
    const wrap = element('div', 'about-body about-body--room');

    /* The preview, and the one line saying what to look at while it loops.
       "What to watch" lives here rather than as a sixth heading: it is a
       caption on the demonstration, which is what it was always describing. */
    const figure = element('figure', 'about-figure');
    const canvas = document.createElement('canvas');
    canvas.className = 'about-canvas';
    canvas.setAttribute('role', 'img');
    canvas.setAttribute('aria-label',
      'Looping illustration of the ' + slide.title);
    figure.appendChild(canvas);
    const beat = element('figcaption', 'about-beat');
    figure.appendChild(beat);
    const watch = element('p', 'about-watch');
    watch.appendChild(element('span', 'about-watch-label', 'Watch for'));
    watch.appendChild(element('span', null, slide.watch));
    figure.appendChild(watch);
    wrap.appendChild(figure);

    const columns = element('div', 'about-columns');

    // The method, stated as a card rather than a heading: it is what the
    // sector is *for*, and it is the one thing being compared across five.
    const badge = element('div', 'about-method');
    badge.appendChild(element('p', 'about-method-label', 'Algorithm'));
    badge.appendChild(element('p', 'about-method-name', slide.algorithm));
    badge.appendChild(element('p', 'about-method-family', slide.family));

    const block = (heading, text) => {
      const node = element('div', 'about-block');
      node.appendChild(element('p', 'about-heading', heading));
      node.appendChild(element('p', null, text));
      return node;
    };

    columns.appendChild(block('Mission', slide.mission));
    columns.appendChild(badge);
    columns.appendChild(block('Objective', slide.objective));

    const obstacles = element('div', 'about-block');
    obstacles.appendChild(element('p', 'about-heading', 'Obstacles'));
    const list = element('ul');
    slide.obstacles.forEach(line => list.appendChild(element('li', null, line)));
    obstacles.appendChild(list);
    columns.appendChild(obstacles);

    columns.appendChild(block('Why this algorithm fits', slide.why));
    wrap.appendChild(columns);

    return { body: wrap, canvas: canvas, beat: beat };
  }

  function mount(host) {
    if (!host) return { open: function () {}, close: function () {} };

    const track = host.querySelector('#about-track');
    const dots = host.querySelector('#about-dots');
    const back = host.querySelector('#about-back');
    const next = host.querySelector('#about-next');
    const closeButton = host.querySelector('#about-close');
    let at = 0;

    const still = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    // Built once. Moving between slides is a class change, never a rebuild,
    // so scroll position and focus survive going back and forth.
    const panels = SLIDES.map(slide => {
      const panel = element('article', 'about-slide');

      const head = element('header', 'about-head');
      head.appendChild(element('p', 'about-sector', slide.sector));
      head.appendChild(element('h2', 'about-title', slide.title));
      if (slide.kind === 'room') {
        head.appendChild(element('p', 'about-number',
                                 'Sector ' + slide.number + ' of 5'));
      }
      panel.appendChild(head);

      let preview = null;
      if (slide.kind === 'story') {
        panel.appendChild(buildStory(slide));
      } else {
        const built = buildRoom(slide);
        panel.appendChild(built.body);
        // Guarded: a missing preview module must not cost the reader the
        // words, which are the part that matters.
        if (window.AboutPreview) {
          preview = window.AboutPreview.create(built.canvas, slide.number,
                                               built.beat);
        }
      }
      track.appendChild(panel);
      return { panel: panel, preview: preview };
    });

    const pips = SLIDES.map((slide, index) => {
      const pip = element('button', 'about-pip');
      pip.type = 'button';
      pip.setAttribute('aria-label', 'Go to ' + slide.title);
      pip.addEventListener('click', () => goTo(index));
      dots.appendChild(pip);
      return pip;
    });

    /** Exactly one preview runs: the one being looked at, and only then. */
    function runPreviews() {
      panels.forEach((entry, index) => {
        if (!entry.preview) return;
        if (index === at && !host.hidden) {
          if (still) entry.preview.still(); else entry.preview.start();
        } else {
          entry.preview.stop();
        }
      });
    }

    function paint() {
      panels.forEach((entry, index) => {
        entry.panel.classList.toggle('is-current', index === at);
        // Hidden from assistive technology as well as from sight, so a
        // screen reader does not read all six at once.
        entry.panel.setAttribute('aria-hidden', String(index !== at));
      });
      pips.forEach((pip, index) => {
        pip.setAttribute('aria-current', String(index === at));
      });
      back.disabled = at === 0;
      next.disabled = at === SLIDES.length - 1;
      // Each slide scrolls independently and starts at its own top.
      if (panels[at]) panels[at].panel.scrollTop = 0;
      runPreviews();
    }

    function goTo(index) {
      at = Math.max(0, Math.min(SLIDES.length - 1, index));
      paint();
    }

    back.addEventListener('click', () => goTo(at - 1));
    next.addEventListener('click', () => goTo(at + 1));
    closeButton.addEventListener('click', close);

    function open() {
      host.hidden = false;
      goTo(0);
      window.requestAnimationFrame(() => {
        host.classList.add('is-open');
        // The canvas has no size until the panel is laid out, so the preview
        // is started after the frame that reveals it.
        runPreviews();
      });
      closeButton.focus({ preventScroll: true });
    }

    function close() {
      host.classList.remove('is-open');
      // Every loop stops on the way out: a closed panel costs nothing.
      panels.forEach(entry => { if (entry.preview) entry.preview.stop(); });
      window.setTimeout(() => { host.hidden = true; }, 400);
      const about = document.getElementById('about');
      if (about) about.focus({ preventScroll: true });
    }

    document.addEventListener('keydown', event => {
      if (host.hidden) return;
      if (event.key === 'Escape') { close(); return; }
      if (event.key === 'ArrowRight') { goTo(at + 1); return; }
      if (event.key === 'ArrowLeft') { goTo(at - 1); }
    });

    return {
      open: open,
      close: close,
      get isOpen() { return !host.hidden; },
      get slide() { return at; },
    };
  }

  /* Mounted on load and exposed as `About.open`, so the start screen's own
     button handler stays a one-liner. */
  let instance = null;
  function attach(host) {
    instance = mount(host);
    return instance;
  }

  return {
    attach: attach,
    mount: mount,
    open: function () { if (instance) instance.open(); },
    close: function () { if (instance) instance.close(); },
    SLIDES: SLIDES,
  };
})();
