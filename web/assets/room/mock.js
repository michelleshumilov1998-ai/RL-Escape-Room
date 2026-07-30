'use strict';

/* =====================================================================
   ***  TEMPORARY.  THROW THIS FILE AWAY.  ***

   The mock producer.  The only place in the interface where data is
   invented, so the screen can be built and judged before any
   reinforcement learning is wired up behind it.

   Nothing here is a learning algorithm and nothing here is an
   environment.  The agent below is a random walk biased towards one of
   two routes, and which route it settles on is worked out from the
   collapse chance with a line of arithmetic — not learned.  That is
   deliberate: it is enough to make the screen show the right thing, and
   obviously not enough to be mistaken for the real thing.

   When the Python side starts producing RoomDefinitions and
   TrajectoryBatches, delete this file and point the shell at that.
   Nothing else changes: everything downstream reads `contract.js` and
   has never heard of this module.
   ===================================================================== */

window.Mock = (function () {

  /* Seeded, so reloading gives the same batch back and a problem does not
     vanish when you go to look at it. */
  function random(seed) {
    let state = seed >>> 0;
    return function () {
      state = (state + 0x6D2B79F5) >>> 0;
      let t = state;
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  /* ---------------------------------------------------------------------
     Entity types.  Each names a drawing recipe rather than a colour and a
     glyph, so a wall looks like masonry and an abyss looks like a hole.
     ------------------------------------------------------------------- */

  const TYPES = {
    wall: {
      label: 'Wall', color: '--cell-wall', shape: 'wall',
      inLegend: true, role: 'static',
    },
    abyss: {
      label: 'Abyss', color: '--hazard', shape: 'abyss',
      inLegend: true, role: 'hazard',
    },
    bridge: {
      label: 'Bridge', color: '--cell-start', shape: 'bridge',
      inLegend: true, role: 'static',
      // What a plank looks like once it has given way. The renderer reads
      // 'collapsed' as an opaque key; this is where it gets its meaning.
      states: {
        collapsed: { shape: 'bridgeBroken', color: '--hazard' },
        // Crossed, and gone behind whatever crossed it.
        spent: { shape: 'bridgeBroken', color: '--cell-wall' },
      },
    },
    start: {
      label: 'Start', color: '--muted', shape: 'start',
      inLegend: true, role: 'static',
    },
    exit: {
      label: 'Exit', color: '--goal', shape: 'exit',
      inLegend: true, role: 'goal',
    },
    agent: {
      label: 'R-5', color: '--accent', shape: 'agent',
      inLegend: true, role: 'agent',
    },
    laser: {
      label: 'Laser', color: '--hazard', shape: 'laser',
      inLegend: true, role: 'hazard',
    },

    /* Room 3's furniture. The reactor's casing is walls by another name,
       but the label is what the legend shows, and in this room it is not
       a wall — it is the thing the beams are containing. */
    casing: {
      label: 'Casing', color: '--cell-wall', shape: 'wall',
      inLegend: true, role: 'static',
    },
    beam: {
      label: 'Containment', color: '--hazard', shape: 'laser',
      inLegend: true, role: 'hazard',
    },
    generator: {
      label: 'Generator', color: '--muted', shape: 'generator',
      inLegend: true, role: 'static',
      states: {
        // Its key is in hand: powered up and waiting, but not yet running.
        armed: { color: '--accent', lit: true },
        // Online: lit, and in the colour of something that has gone right.
        live: { color: '--goal', lit: true },
      },
    },
    key: {
      label: 'Key', color: '--accent', shape: 'key',
      inLegend: true, role: 'static',
      states: { taken: { hidden: true } },
    },
    door: {
      label: 'Blast door', color: '--cell-wall', shape: 'door',
      inLegend: true, role: 'static',
      states: { open: { open: true, color: '--cell-start' } },
    },
    guard: {
      label: 'Sentry', color: '--hazard', shape: 'guard',
      inLegend: true, role: 'hazard',
    },

    /* Room 2's additions. */
    card: {
      label: 'Access card', color: '--accent', shape: 'key',
      inLegend: true, role: 'static',
      states: { taken: { hidden: true } },
    },
    lockedDoor: {
      label: 'Exit door', color: '--cell-wall', shape: 'door',
      inLegend: true, role: 'goal',
      // Shut until the card is in hand, and then plainly not shut.
      states: { open: { open: true, color: '--goal' } },
    },
    repair: {
      label: 'Repair bay', color: '--goal', shape: 'repair',
      inLegend: true, role: 'static',
      states: { used: { color: '--accent' }, spent: { opacity: 0.3 } },
    },

    /* Room 3's additions. */
    cell: {
      label: 'Battery', color: '--goal', shape: 'battery',
      inLegend: true, role: 'static',
      states: { taken: { hidden: true } },
    },
    charger: {
      label: 'Charger', color: '--accent', shape: 'charger',
      inLegend: true, role: 'static',
      states: { used: { opacity: 0.35 } },
    },
    trap: {
      label: 'Live cable', color: '--hazard', shape: 'trap',
      inLegend: true, role: 'hazard',
      states: { arcing: { color: '--hazard' } },
    },
    shortcut: {
      label: 'Ring gate', color: '--cell-wall', shape: 'door',
      inLegend: true, role: 'static',
      // A charged robot opens both of them at once.
      states: { open: { open: true, color: '--accent' } },
    },
    ice: {
      label: 'Ice', color: '--cell-slippery', shape: 'ice',
      inLegend: true, role: 'static',
    },
    obstacle: {
      label: 'Obstacle', color: '--cell-wall', shape: 'obstacle',
      inLegend: true, role: 'static',
    },
    pad: {
      label: 'Pad', color: '--goal', shape: 'pad',
      inLegend: true, role: 'goal',
    },

    /* Rooms 4 and 5. A vent's direction is the vent's own business, not
       its type's, so each one carries it — otherwise three near-identical
       rows would appear in the legend for one idea. */
    vent: {
      label: 'Vent', color: '--cell-start', shape: 'vent',
      inLegend: true, role: 'static',
    },
    draught: {
      label: 'Draught', color: '--accent', shape: 'stream',
      inLegend: true, role: 'static',
    },
    brake: {
      label: 'Brake zone', color: '--cell-slippery', shape: 'brake',
      inLegend: true, role: 'static',
    },
    boost: {
      label: 'Boost zone', color: '--accent', shape: 'stream',
      inLegend: true, role: 'static',
    },
    gate: {
      label: 'Moving gate', color: '--cell-wall', shape: 'block',
      inLegend: true, role: 'hazard',
    },
    shelf: {
      label: 'Shelving', color: '--cell-wall', shape: 'wall',
      inLegend: true, role: 'static',
    },
    conveyor: {
      label: 'Conveyor', color: '--cell-start', shape: 'stream',
      inLegend: true, role: 'static',
    },
    mover: {
      label: 'Maintenance', color: '--hazard', shape: 'guard',
      inLegend: true, role: 'hazard',
    },
    terminal: {
      label: 'Terminal', color: '--accent', shape: 'charger',
      inLegend: true, role: 'goal',
      states: { down: { color: '--goal' } },
    },
    exitGate: {
      label: 'Main gate', color: '--cell-wall', shape: 'door',
      inLegend: true, role: 'goal',
      states: { open: { open: true, color: '--goal' } },
    },
    crate: {
      label: 'Crate', color: '--cell-wall', shape: 'obstacle',
      inLegend: true, role: 'hazard',
      // The building declares every crate it might hold; a run stows the
      // ones it does not use rather than the room changing shape.
      states: { present: {}, stowed: { hidden: true } },
    },
  };

  function pick(names) {
    const out = {};
    names.forEach(name => { out[name] = TYPES[name]; });
    return out;
  }

  /* ---------------------------------------------------------------------
     Room 2 — the choice this room exists to pose
     ------------------------------------------------------------------- */

  /* A chasm cuts the sector in two.  A bridge crosses it, and the bridge
     may give way underfoot.

         the bridge   7 steps, straight across, worth +93 if it holds
         the lap      21 steps up the west wall, along the roof and down
                      the east wall.  Dry, solid, and worth +79 every time

     Both routes fetch the access card, so both are paid for it, and the
     long way also passes the repair bay.  That leaves the crossing worth
     seven more than the lap — and the question the room asks is how much
     collapse risk seven points is worth.  With a fall costing 100 the
     answer is about 0.6% per plank: under that, cross; over it, walk
     round.  The slider runs to 4% for exactly that reason, and is there to
     be pushed through the boundary.

     THE MIDDLE IS A VAULT, NOT A BLOCK
     The centre used to be one solid rectangle, which is the dullest thing
     a sealed middle can be.  It is now a sealed vault with a walkable ring
     around it, open at the top — and the ring's far side overlooks the
     chasm.  That makes it a trap rather than a shortcut: an agent that
     wanders in can only come back out the way it went in, or fall.  The
     route lengths are unchanged at 7 and 21, checked by breadth-first
     search over this very grid, so none of the arithmetic above moves.

         #  wall    A  the chasm    B  a plank of the bridge
         S  start   G  the exit     .  floor                            */
  const ROOM2_GRID = [
    '##########',
    '#...R....#',   // r1  the roof, and a repair bay on the long way round
    '#.##..##.#',
    '#.#....#.#',
    '#.#.##.#.#',   // the vault, sealed
    '#.#.##.#.#',
    '#.#....#.#',
    '#CAAAAAA.#',   // r7  the access card, and the chasm
    '#SBBBBB~D#',   // r8  start, the planks, wet steel, the locked door
    '##########',
  ];

  const TILE_TYPES = {
    '#': 'wall', A: 'abyss', B: 'bridge', S: 'start', D: 'lockedDoor',
    C: 'card', R: 'repair', '~': 'ice',
  };

  const PLANKS = 5;              // how many bridge cells there are
  const BRIDGE_STEPS = 9;        // two of them are the detour for the card
  const LAP_STEPS = 21;          // which passes the card on its first step
  const REWARDS = { step: -1, wall: -2, fall: -100, card: 20, repair: 5,
                    lockedOut: -5, exit: 120 };

  /** What each route is worth, given a per-plank collapse chance. */
  function routeValues(collapse) {
    const holds = Math.pow(1 - collapse, PLANKS);
    // Both routes fetch the card, so both are paid for it.
    const crossed = REWARDS.exit + REWARDS.card - BRIDGE_STEPS;   // +131
    // A fall happens part way over, so a few steps have been paid for.
    const fell = REWARDS.fall - 3;                                // -103
    return {
      bridge: holds * crossed + (1 - holds) * fell,
      // The long way round passes the repair bay, which pays a little.
      lap: REWARDS.exit + REWARDS.card + REWARDS.repair - LAP_STEPS,  // +124
      holds: holds,
    };
  }

  function gridRoom(parameters) {
    const collapse = (parameters && parameters.bridgeCollapse !== undefined)
      ? parameters.bridgeCollapse : 0.01;
    const value = routeValues(collapse);
    const rows = ROOM2_GRID.length;
    const cols = ROOM2_GRID[0].length;
    const entities = [];

    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const character = ROOM2_GRID[row].charAt(col);
        const type = TILE_TYPES[character];
        if (!type) continue;                   // floor is the background
        // The three things an episode talks about by name.
        const named = { C: 'card', D: 'door', R: 'repair' }[character];
        entities.push({
          id: named || ('cell-' + row + '-' + col),
          type: type,
          // Centres, in world units: cell (r,c) sits at (c + 0.5, r + 0.5).
          position: { x: col + 0.5, y: row + 0.5 },
          size: { width: 1, height: 1 },
        });
      }
    }

    return {
      id: 'room2',
      name: 'Broken Bridge Sector',
      sector: 'LAB SECTOR B-04',
      worldSize: { width: cols, height: rows },
      isGrid: true,
      cellSize: 1,
      entities: entities,
      entityTypes: pick(['wall', 'abyss', 'bridge', 'ice', 'card', 'repair',
                         'lockedDoor', 'start', 'agent']),
      metric: { key: 'reward', label: 'Mean reward', format: '%.1f' },
      parameterSchema: ROOM2_PARAMETERS,
      algorithm: {
        label: 'SARSA',
        summary: 'Learns the value of the policy it is actually following, '
               + 'including the exploratory steps that policy takes.',
        updateRule: 'Q(s,a) ← Q(s,a) + α[ r + γ Q(s′,a′) − Q(s,a) ]',
        watchFor: 'Awaiting the algorithm layer. Until it is wired up the '
                + 'trajectories here come from the mock producer and '
                + 'nothing is learned.',
      },
      info: {
        objective: 'Find the access card, then cross the sector to the '
                 + 'exit door it opens. The model is not given: the only '
                 + 'way to find out what a step does is to take it.',
        obstacles: [
          'The exit door is locked. Nothing opens it but the access card, '
          + 'and trying it without one costs a moment and tells you so.',
          'A chasm splits the sector. Stepping into it ends the run at once.',
          'A bridge of five planks crosses it. Each plank may give way as '
          + 'it is stepped on — and every plank that is crossed gives way '
          + 'behind whatever crossed it, so the way back is not there.',
          'The plating at the far end of the crossing is wet. A step taken '
          + 'on it may go sideways instead, and sideways there is down.',
          'A sealed vault fills the middle, so going round is a full lap of '
          + 'the outer wall — past the repair bay, which is good for one '
          + 'fall and one only.',
        ],
        actionSet: 'Up, down, left, right. Nothing else.',
        rewards: [
          ['Each step', String(REWARDS.step)],
          ['Walking into a wall', '-1 + (-2) = -3'],
          ['Falling into the chasm', '-1 + (-100) = -101'],
          ['Collecting the access card', '-1 + 20 = +19'],
          ['Using the repair bay', '-1 + 5 = +4'],
          ['Trying the door without the card', '-1 + (-5) = -6'],
          ['Reaching the exit with it', '-1 + 120 = +119'],
        ],
        // The trade-off, stated rather than left to be inferred from the
        // shape of the room.
        routes: [
          { name: 'The bridge', steps: BRIDGE_STEPS,
            best: '+' + (REWARDS.exit + REWARDS.card - BRIDGE_STEPS),
            risk: (value.holds * 100).toFixed(1) + '% chance of crossing '
                + 'at ' + (collapse * 100).toFixed(1) + '% per plank, and '
                + 'wet steel at the far end' },
          { name: 'The lap', steps: LAP_STEPS,
            best: '+' + (REWARDS.exit + REWARDS.card + REWARDS.repair
                         - LAP_STEPS),
            risk: 'none, and it passes the repair bay' },
        ],
        terminal: 'The run ends at the exit door with the card in hand, or '
                + 'in the chasm, and otherwise when the step limit is '
                + 'reached.',
        note: 'The bridge is worth 14 more than the lap when it holds. So '
            + 'the room is one question asked over and over: how much '
            + 'collapse risk is 14 points worth? Below about 1.9% per plank '
            + 'the crossing is the better bet and the agent settles on it; '
            + 'above that it walks round. Move the collapse slider through '
            + 'that boundary and the route it ends up taking changes.',
      },
    };
  }

  /* Room 2's parameters. Model-free, and one of its own: the thing this
     room is about is worth a slider. */
  const ROOM2_PARAMETERS = [
    { key: 'alpha', label: 'Learning rate', symbol: 'α', type: 'number',
      min: 0.01, max: 1.0, step: 0.01, default: 0.10, appliesLive: true,
      description: 'How much of each new estimate to keep.' },
    { key: 'gamma', label: 'Discount factor', symbol: 'γ', type: 'number',
      min: 0.50, max: 0.999, step: 0.005, default: 0.95, appliesLive: false,
      description: 'How much a reward in the future is worth now.' },
    { key: 'epsilon', label: 'Exploration rate', symbol: 'ε', type: 'number',
      min: 0.0, max: 1.0, step: 0.01, default: 1.00, appliesLive: true,
      description: 'How often a random action is taken instead of the best one.' },
    { key: 'epsilonMin', label: 'Minimum exploration', symbol: '', type: 'number',
      min: 0.0, max: 0.5, step: 0.01, default: 0.10, appliesLive: true,
      description: 'Exploration never decays below this.' },
    { key: 'epsilonDecay', label: 'Exploration decay', symbol: '', type: 'number',
      min: 0.90, max: 1.0, step: 0.001, default: 0.965, appliesLive: true,
      description: 'Exploration is multiplied by this after every episode.' },
    { key: 'slip', label: 'Wet steel', symbol: '', type: 'number',
      min: 0.0, max: 0.5, step: 0.01, default: 0.12, appliesLive: false,
      description: 'Chance that a step taken on the wet plating beside the '
                 + 'chasm goes sideways instead. Sideways, there, is down.' },
    { key: 'bridgeCollapse', label: 'Bridge collapse chance', symbol: '',
      type: 'number', min: 0.0, max: 0.04, step: 0.002, default: 0.002,
      appliesLive: false,
      description: 'Chance that a plank gives way as it is stepped on. Past '
                 + 'about 0.6% the lap becomes the better bet and the route '
                 + 'the agent settles on changes — a finer scale than it '
                 + 'used to be, because a fall now costs 100 rather than 60.' },
    { key: 'qInit', label: 'Initial Q value', symbol: '', type: 'number',
      min: 0.0, max: 150.0, step: 5.0, default: 90.0, appliesLive: false,
      description: 'What every untried action is worth to begin with.' },
    { key: 'episodes', label: 'Episodes to train', symbol: '', type: 'number',
      min: 100, max: 8000, step: 100, default: 1500, appliesLive: false,
      description: 'How many attempts to learn from before the run is finished.' },
  ];

  /* ---- walking the sector -------------------------------------------- */

  function tileAt(row, col) {
    if (row < 0 || col < 0 || row >= ROOM2_GRID.length) return '#';
    if (col >= ROOM2_GRID[0].length) return '#';
    return ROOM2_GRID[row].charAt(col);
  }

  function findTile(target) {
    for (let row = 0; row < ROOM2_GRID.length; row += 1) {
      for (let col = 0; col < ROOM2_GRID[0].length; col += 1) {
        if (ROOM2_GRID[row].charAt(col) === target) return { row: row, col: col };
      }
    }
    return { row: 1, col: 1 };
  }

  const MOVES = [
    { name: 'UP', row: -1, col: 0 },
    { name: 'DOWN', row: 1, col: 0 },
    { name: 'LEFT', row: 0, col: -1 },
    { name: 'RIGHT', row: 0, col: 1 },
  ];

  /**
   * Distance to a cell, as a breadth-first sweep from it.
   *
   * `useBridge` says whether the planks count as floor. Two fields to the
   * door — one that may cross and one that may not — is how the mock
   * settles on a route: it is a pair of maps, not a value function.
   */
  function distanceTo(target, useBridge) {
    const field = {};
    const key = (row, col) => row + ',' + col;
    const queue = [[target.row, target.col, 0]];
    field[key(target.row, target.col)] = 0;

    while (queue.length) {
      const [row, col, depth] = queue.shift();
      MOVES.forEach(move => {
        const nextRow = row + move.row;
        const nextCol = col + move.col;
        const tile = tileAt(nextRow, nextCol);
        if (tile === '#' || tile === 'A') return;
        if (tile === 'B' && !useBridge) return;
        if (field[key(nextRow, nextCol)] !== undefined) return;
        field[key(nextRow, nextCol)] = depth + 1;
        queue.push([nextRow, nextCol, depth + 1]);
      });
    }
    return field;
  }

  const CARD_CELL = findTile('C');
  const DOOR_CELL = findTile('D');
  const TO_CARD = distanceTo(CARD_CELL, true);
  const VIA_BRIDGE = distanceTo(DOOR_CELL, true);
  const VIA_LAP = distanceTo(DOOR_CELL, false);

  function bestMove(row, col, field, next, blocked) {
    let best = null;
    let shortest = Infinity;
    MOVES.forEach(move => {
      const nextRow = row + move.row;
      const nextCol = col + move.col;
      const tile = tileAt(nextRow, nextCol);
      if (tile === '#') return;
      if (blocked && blocked[nextRow + ',' + nextCol]) return;
      const at = field[nextRow + ',' + nextCol];
      // The chasm has no distance, so it is never chosen deliberately.
      if (at === undefined) return;
      // Ties broken at random, so a corridor is not always walked the
      // same way round.
      const jitter = next() * 0.1;
      if (at + jitter < shortest) {
        shortest = at + jitter;
        best = move;
      }
    });
    return best || MOVES[Math.floor(next() * MOVES.length)];
  }

  /**
   * One attempt at the sector.
   *
   * The card first, then the door, and the door does not open without it.
   * Three things can go wrong on the way: a plank giving way underfoot,
   * wet steel throwing a sideways step into the chasm, and a plank that
   * has already been crossed being gone when it is needed again.
   */
  function walk(next, epsilon, collapse, field, maxSteps, slip) {
    const start = findTile('S');
    let row = start.row;
    let col = start.col;
    let hasCard = false;
    let shielded = false;
    // Planks give way behind whatever crosses them; a route that doubles
    // back finds them missing.
    const spent = {};

    const steps = [{
      position: { x: col + 0.5, y: row + 0.5 },
      velocity: null, reward: 0, action: '—', entityStates: null,
      entityPositions: null,
    }];

    let total = 0;
    let outcome = 'timeout';

    for (let count = 0; count < maxSteps; count += 1) {
      const aiming = hasCard ? field : TO_CARD;
      let move = next() < epsilon
        ? MOVES[Math.floor(next() * MOVES.length)]
        : bestMove(row, col, aiming, next, spent);

      // Wet steel: a sideways step on it goes sideways, and beside the
      // chasm that is the whole danger.
      const standingOn = tileAt(row, col);
      if (standingOn === '~' && next() < slip) {
        const sideways = move.row === 0
          ? [{ name: 'UP', row: -1, col: 0 }, { name: 'DOWN', row: 1, col: 0 }]
          : [{ name: 'LEFT', row: 0, col: -1 },
             { name: 'RIGHT', row: 0, col: 1 }];
        move = sideways[next() < 0.5 ? 0 : 1];
      }

      const targetRow = row + move.row;
      const targetCol = col + move.col;
      const tile = tileAt(targetRow, targetCol);
      const targetKey = targetRow + ',' + targetCol;

      // A wall, or a plank that is no longer there.
      if (tile === '#' || spent[targetKey]) {
        const reward = REWARDS.step + REWARDS.wall;
        total += reward;
        steps.push({
          position: { x: col + 0.5, y: row + 0.5 },
          velocity: null, reward: reward, action: move.name,
          entityStates: null, entityPositions: null,
        });
        continue;
      }

      const leftBehind = tileAt(row, col) === 'B'
        ? [{ id: 'cell-' + row + '-' + col, state: 'spent' }]
        : [];
      if (leftBehind.length) spent[row + ',' + col] = true;

      const fromRow = row;
      const fromCol = col;
      row = targetRow;
      col = targetCol;

      let reward = REWARDS.step;
      let states = leftBehind;
      let fell = false;

      if (tile === 'A') {
        fell = true;
      } else if (tile === 'B' && next() < collapse) {
        // The plank gives way underfoot.
        fell = true;
        states = states.concat([
          { id: 'cell-' + row + '-' + col, state: 'collapsed' }]);
      } else if (tile === 'C' && !hasCard) {
        hasCard = true;
        reward += REWARDS.card;
        states = states.concat([{ id: 'card', state: 'taken' },
                                { id: 'door', state: 'open' }]);
      } else if (tile === 'R' && !shielded) {
        shielded = true;
        reward += REWARDS.repair;
        states = states.concat([{ id: 'repair', state: 'used' }]);
      } else if (tile === 'D') {
        if (hasCard) {
          reward += REWARDS.exit;
          outcome = 'success';
        } else {
          // Shut, and it costs something to find that out.
          reward += REWARDS.lockedOut;
          row = fromRow;
          col = fromCol;
        }
      }

      if (fell) {
        if (shielded) {
          // The repair bay's one save: hauled back to the start, shaken.
          shielded = false;
          reward += REWARDS.fall / 4;
          row = start.row;
          col = start.col;
          states = states.concat([{ id: 'repair', state: 'spent' }]);
        } else {
          reward += REWARDS.fall;
          outcome = 'failure';
        }
      }

      total += reward;
      steps.push({
        position: { x: col + 0.5, y: row + 0.5 },
        velocity: null, reward: reward, action: move.name,
        entityStates: states.length ? states : null, entityPositions: null,
      });

      if (outcome !== 'timeout') break;
    }

    return { steps: steps, totalReward: total, outcome: outcome };
  }

  /* ---------------------------------------------------------------------
     Room 3 — the Reactor Control Chamber
     ------------------------------------------------------------------- */

  /* Three generators, three keys, and a sentry walking the only approach
     to the door.

     This room is a sequence rather than a choice, and the keys are what
     make the sequence long.  Each generator is locked, each key is cut
     for one machine and useless at the other two, and both are strewn
     across the chamber from each other — so the run is six errands and
     then a door:

         key a → generator 1 → key b → generator 2 → key c → generator 3

     Nothing counts out of turn.  A key picked up early opens nothing, a
     generator reached before its key does not run, and the blast door
     stays shut until all three are running, so the way out cannot be
     stumbled into.  Seventy-four steps if every one of them is right,
     against room 2's seven.

     That is what makes it harder rather than merely bigger: what has to
     be carried from one part of the room to the next is no longer where
     the agent is, but how much of the errand list is already done.

     The containment ring is the reactor's, and it is fatal.  The sentry
     walks row 7 from end to end, which is the row the door is approached
     across, so the last leg has to be timed rather than merely walked.

         #  casing   H  containment beam   D  the blast door
         a b c  the keys      1 2 3  the generators they open
         S  start    X  the way out        .  floor                     */
  const ROOM3_GRID = [
    'S...b....1',   // r0  the start, a key, and the first generator
    'c........,',   // r1  the third key, and a trap that comes and goes
    '.......,..',   // r2
    '...HHVH...',   // r3  a gap in the containment ring, sealed by a gate
    '...H#.H#V#',   // r4  the passage through it, and one down the east side
    '...H#PH...',   // r5  the charging station, sealed inside the ring
    '...HHVH...',   // r6  the far gate: only a charged robot opens either
    '..,.......',   // r7  the sentry's beat, with a trap on it
    '2...#D#..3',   // r8  two more generators, and the door between them
    '.aE.#X#...',   // r9  the first key, and the way out behind the door
  ];

  const ROOM3_TRAP_PHASE = 2;    // live on even steps, dead on odd

  const ROOM3_TILES = {
    '#': 'casing', H: 'beam', D: 'door', S: 'start', X: 'exit',
    1: 'generator', 2: 'generator', 3: 'generator',
    a: 'key', b: 'key', c: 'key',
    E: 'cell', P: 'charger', ',': 'trap', V: 'shortcut',
  };

  /* Which generator each key turns over. Every key is cut for one machine
     and does nothing at the other two, which is why they are worth
     crossing the chamber for separately. */
  const KEY_OPENS = { a: '1', b: '2', c: '3' };

  /* The order the whole thing has to happen in: fetch a key, run its
     generator, and again, and again. Nothing counts out of turn. */
  const ROOM3_CHAIN = ['a', '1', 'b', '2', 'c', '3', 'X'];

  /* Where the sentry walks, and the order the generators come in. */
  const PATROL_ROW = 7;
  const PATROL_FROM = 1;
  const PATROL_TO = 8;
  const ROOM3_REWARDS = { step: -1, wall: -2, beam: -80, caught: -30,
                          trap: -20, cell: 15, charge: 10, key: 10,
                          generator: 25, unfinished: -10, exit: 150 };

  function room3TileAt(row, col) {
    if (row < 0 || col < 0 || row >= ROOM3_GRID.length) return '#';
    if (col >= ROOM3_GRID[0].length) return '#';
    return ROOM3_GRID[row].charAt(col);
  }

  function room3Find(target) {
    for (let row = 0; row < ROOM3_GRID.length; row += 1) {
      for (let col = 0; col < ROOM3_GRID[0].length; col += 1) {
        if (ROOM3_GRID[row].charAt(col) === target) return { row: row, col: col };
      }
    }
    return { row: 0, col: 0 };
  }

  /** Distance to one cell, with the door shut or open. Biases the walk. */
  function room3Field(goal, doorOpen) {
    const field = {};
    const key = (row, col) => row + ',' + col;
    const queue = [[goal.row, goal.col, 0]];
    field[key(goal.row, goal.col)] = 0;

    while (queue.length) {
      const [row, col, depth] = queue.shift();
      MOVES.forEach(move => {
        const nextRow = row + move.row;
        const nextCol = col + move.col;
        const tile = room3TileAt(nextRow, nextCol);
        if (tile === '#' || tile === 'H') return;
        if (tile === 'D' && !doorOpen) return;
        // The ring gates need a charged robot; the field that plans the
        // ordinary route pretends they are solid.
        if (tile === 'V') return;
        if (field[key(nextRow, nextCol)] !== undefined) return;
        field[key(nextRow, nextCol)] = depth + 1;
        queue.push([nextRow, nextCol, depth + 1]);
      });
    }
    return field;
  }

  /* One field per leg of the chain. The last is the only one that may
     pass the door, because the door is only open by then. */
  const ROOM3_LEGS = ROOM3_CHAIN.map((tile, index) => {
    const cell = room3Find(tile);
    return {
      tile: tile,
      cell: cell,
      field: room3Field(cell, index === ROOM3_CHAIN.length - 1),
    };
  });
  const LAST_LEG = ROOM3_CHAIN.length - 1;      // the way out

  /* Every gate into the containment ring, so one battery opens them all. */
  const ROOM3_GATES = [];
  ROOM3_GRID.forEach((row, rowIndex) => {
    for (let col = 0; col < row.length; col += 1) {
      if (row.charAt(col) === 'V') {
        ROOM3_GATES.push('gate-' + rowIndex + '-' + col);
      }
    }
  });

  /** Where the sentry is on a given step. Up the row, then back down it. */
  function patrolCell(step) {
    const span = PATROL_TO - PATROL_FROM;
    const phase = step % (span * 2);
    const col = phase <= span ? PATROL_FROM + phase
                              : PATROL_TO - (phase - span);
    return { row: PATROL_ROW, col: col };
  }

  function gridRoom3() {
    const rows = ROOM3_GRID.length;
    const cols = ROOM3_GRID[0].length;
    const entities = [];

    for (let row = 0; row < rows; row += 1) {
      for (let col = 0; col < cols; col += 1) {
        const character = ROOM3_GRID[row].charAt(col);
        const type = ROOM3_TILES[character];
        if (!type) continue;
        let id = 'cell-' + row + '-' + col;
        if (type === 'generator') id = 'generator-' + character;
        if (type === 'key') id = 'key-' + character;
        if (type === 'cell') id = 'battery';
        if (type === 'charger') id = 'charger';
        if (type === 'trap') id = 'trap-' + row + '-' + col;
        if (type === 'shortcut') id = 'gate-' + row + '-' + col;
        entities.push({
          id: id,
          type: type,
          position: { x: col + 0.5, y: row + 0.5 },
          size: { width: 1, height: 1 },
        });
      }
    }

    // The sentry is declared once, where its beat begins. Every step then
    // says where it has got to.
    const first = patrolCell(0);
    entities.push({
      id: 'sentry', type: 'guard',
      position: { x: first.col + 0.5, y: first.row + 0.5 },
      size: { width: 1, height: 1 },
    });

    return {
      id: 'room3',
      name: 'Reactor Control Chamber',
      sector: 'LAB SECTOR C-07',
      worldSize: { width: cols, height: rows },
      isGrid: true,
      cellSize: 1,
      entities: entities,
      entityTypes: pick(['casing', 'beam', 'generator', 'key', 'door',
                         'cell', 'charger', 'trap', 'shortcut',
                         'start', 'exit', 'guard', 'agent']),
      metric: { key: 'reward', label: 'Mean reward', format: '%.1f' },
      parameterSchema: ROOM3_PARAMETERS,
      // Seventy-four steps at best, so it is shown a good deal quicker
      // than a room that is over in seven.
      playback: { stepsPerSecond: 18 },
      algorithm: {
        label: 'Q-Learning',
        summary: 'Learns the value of behaving perfectly from here on, '
               + 'whatever it actually did next — so exploring costs it '
               + 'nothing in what it believes.',
        updateRule: 'Q(s,a) ← Q(s,a) + α[ r + γ max Q(s′,a′) − Q(s,a) ]',
        watchFor: 'Awaiting the algorithm layer. Until it is wired up the '
                + 'trajectories here come from the mock producer and '
                + 'nothing is learned.',
      },
      info: {
        objective: 'Find each generator’s key, bring the three of them '
                 + 'online in order, and leave through the blast door they '
                 + 'unseal. A battery and a charging station are there for '
                 + 'the taking, and neither is on the way.',
        obstacles: [
          'Every generator is locked, and every key is cut for one machine '
          + 'and does nothing at the other two.',
          'Keys and generators are strewn apart, so each one is a crossing '
          + 'of the chamber in its own right.',
          'Nothing counts out of turn: a key picked up early opens '
          + 'nothing, and a generator reached before its key does not run.',
          'The blast door stays shut until all three are running, so the '
          + 'way out cannot be stumbled into early.',
          'The reactor’s containment ring is fatal to touch.',
          'A sentry walks the row the door is approached across. It does '
          + 'not stop, and meeting it costs 30 and a shove back the way '
          + 'you came.',
          'Two runs of cable are live on even steps and dead on odd, so '
          + 'whether they can be crossed depends on when you arrive as '
          + 'much as on where you are.',
          'A battery lies beside the first key. It is not needed for the '
          + 'job, but it opens the gates into the containment ring, and '
          + 'the charging station is sealed inside them.',
        ],
        actionSet: 'Up, down, left, right, and wait. Waiting costs a step '
                 + 'like any other move, and it is the only way past a '
                 + 'sentry that is standing where you need to be.',
        rewards: [
          ['Each step, waiting included', '-1'],
          ['Walking into the casing', '-1 + (-2) = -3'],
          ['Touching the containment ring', '-1 + (-80) = -81'],
          ['Meeting the sentry', '-1 + (-80) = -81'],
          ['Picking up the right key', '-1 + 10 = +9'],
          ['Meeting the sentry', '-1 + (-30) = -31, and shoved back'],
          ['Touching a live cable', '-1 + (-20) = -21'],
          ['Collecting the battery', '-1 + 15 = +14'],
          ['Reaching the charging station', '-1 + 10 = +9'],
          ['Trying the way out unfinished', '-1 + (-10) = -11'],
          ['Bringing a generator online', '-1 + 25 = +24'],
          ['Reaching the way out', '-1 + 150 = +149'],
        ],
        routes: [
          { name: 'Key, generator, three times over', steps: 74,
            best: '+181', risk: 'the sentry and two live cables' },
        ],
        terminal: 'The run ends at the way out or in the containment ring, '
                + 'and otherwise when the step limit is reached. The sentry '
                + 'and the cables hurt without stopping anything.',
        note: 'This room is a sequence rather than a choice, and the keys '
            + 'are what make the sequence long. What has to be carried '
            + 'from one part of it to the next is not where the agent is '
            + 'but how much of the errand list is already done — which is '
            + 'why it needs a method that can hold a value for being '
            + 'part-way through. A generator lights up the moment its key '
            + 'is in hand, so which key belongs to which machine is '
            + 'something you can see rather than something to remember.',
      },
    };
  }

  const ROOM3_PARAMETERS = [
    { key: 'alpha', label: 'Learning rate', symbol: 'α', type: 'number',
      min: 0.01, max: 1.0, step: 0.01, default: 0.15, appliesLive: true,
      description: 'How much of each new estimate to keep.' },
    { key: 'gamma', label: 'Discount factor', symbol: 'γ', type: 'number',
      min: 0.50, max: 0.999, step: 0.005, default: 0.98, appliesLive: false,
      description: 'How much a reward in the future is worth now. This room '
                 + 'is long, so it needs a patient one.' },
    { key: 'epsilon', label: 'Exploration rate', symbol: 'ε', type: 'number',
      min: 0.0, max: 1.0, step: 0.01, default: 1.00, appliesLive: true,
      description: 'How often a random action is taken instead of the best one.' },
    { key: 'epsilonMin', label: 'Minimum exploration', symbol: '', type: 'number',
      min: 0.0, max: 0.5, step: 0.01, default: 0.05, appliesLive: true,
      description: 'Exploration never decays below this.' },
    { key: 'epsilonDecay', label: 'Exploration decay', symbol: '', type: 'number',
      min: 0.90, max: 1.0, step: 0.001, default: 0.975, appliesLive: true,
      description: 'Exploration is multiplied by this after every episode.' },
    { key: 'qInit', label: 'Initial Q value', symbol: '', type: 'number',
      min: 0.0, max: 150.0, step: 5.0, default: 0.0, appliesLive: false,
      description: 'What every untried action is worth to begin with.' },
    { key: 'episodes', label: 'Episodes to train', symbol: '', type: 'number',
      min: 100, max: 8000, step: 100, default: 3000, appliesLive: false,
      description: 'How many attempts to learn from before the run is finished.' },
  ];

  /**
   * The move a settled policy would make: towards the next objective, but
   * never into the sentry.
   *
   * Returns null when every step that helps is unsafe, which the caller
   * turns into a wait. That is the whole of the timing problem this room
   * poses, standing in for a policy that has learned it.
   */
  function reactorMove(row, col, field, leg, sentryWas, sentryNow, next) {
    let best = null;
    let shortest = Infinity;

    MOVES.forEach(move => {
      const nextRow = row + move.row;
      const nextCol = col + move.col;
      const tile = room3TileAt(nextRow, nextCol);
      if (tile === '#' || tile === 'H') return;
      if (tile === 'D' && leg < LAST_LEG) return;

      // Walking onto it, or trading places with it, are the same mistake.
      if (nextRow === sentryNow.row && nextCol === sentryNow.col) return;
      if (nextRow === sentryWas.row && nextCol === sentryWas.col
          && row === sentryNow.row && col === sentryNow.col) return;

      const distance = field[nextRow + ',' + nextCol];
      if (distance === undefined) return;
      const jitter = next() * 0.1;
      if (distance + jitter < shortest) {
        shortest = distance + jitter;
        best = move;
      }
    });

    /* Standing still beats walking away. When the sentry is across the
       only way forward, every remaining move is a step backwards, and the
       thing to do is let it go past — which is the timing problem this
       room poses and the reason it has a fourth action at all.

       But only when standing still is survivable. The sentry walks over
       whatever is in its way, so a drone in the cell it is about to enter
       has to be somewhere else by then, even if somewhere else is
       backwards. */
    const trampled = sentryNow.row === row && sentryNow.col === col;
    const here = field[row + ',' + col];
    if (!trampled && here !== undefined && shortest >= here) return null;
    return best;
  }

  /**
   * Whether the agent and the sentry met this step.
   *
   * Three ways, and the third is the one that was missed for a while: the
   * agent standing still — waiting, or shoved back by a wall — while the
   * sentry walks onto it. Standing on the same cell counts however either
   * of them got there, so `to` being equal to `from` is not a special
   * case, it is simply covered by the first test.
   */
  function meetsSentry(from, to, was, now) {
    if (to.row === now.row && to.col === now.col) return true;
    // Trading places is meeting it just as much as landing on it.
    return to.row === was.row && to.col === was.col
        && from.row === now.row && from.col === now.col;
  }

  /**
   * One run at the reactor.
   *
   * The sentry moves every step whatever the agent does, so the two can
   * meet head on, swap places, or the sentry can simply walk into a drone
   * that is not moving at all. All three count as being caught.
   */
  function reactorRun(next, epsilon, maxSteps) {
    const start = room3Find('S');
    let row = start.row;
    let col = start.col;
    let leg = 0;                       // which generator is wanted next

    const first = patrolCell(0);
    const steps = [{
      position: { x: col + 0.5, y: row + 0.5 },
      velocity: null, reward: 0, action: '—', entityStates: null,
      entityPositions: [{ id: 'sentry',
                          position: { x: first.col + 0.5,
                                      y: first.row + 0.5 } }],
    }];

    let total = 0;
    let outcome = 'timeout';
    // Whether the special battery is aboard, which is what opens the gates
    // into the containment ring.
    let charged = false;
    let recharged = false;

    for (let count = 0; count < maxSteps; count += 1) {
      const field = ROOM3_LEGS[leg].field;
      const sentryWas = patrolCell(count);
      const sentryNow = patrolCell(count + 1);
      const sentryAt = { id: 'sentry',
                         position: { x: sentryNow.col + 0.5,
                                     y: sentryNow.row + 0.5 } };

      // The trap on the cable run is live on even steps and dead on odd,
      // so what is safe depends on when you arrive as much as where.
      const trapLive = count % ROOM3_TRAP_PHASE === 0;

      const move = next() < epsilon
        ? MOVES[Math.floor(next() * MOVES.length)]
        : reactorMove(row, col, field, leg, sentryWas, sentryNow, next);

      const here = { row: row, col: col };

      // Nowhere safe to go: stand still and let the sentry pass. This is
      // the room's fourth action, and the reason it has one. Standing
      // still is not safety, though — the sentry is still walking, and it
      // will walk over anything in its way.
      if (!move) {
        const caught = meetsSentry(here, here, sentryWas, sentryNow);
        const reward = ROOM3_REWARDS.step
                     + (caught ? ROOM3_REWARDS.caught : 0);
        total += reward;
        steps.push({
          position: { x: col + 0.5, y: row + 0.5 },
          velocity: null, reward: reward, action: 'WAIT',
          entityStates: null, entityPositions: [sentryAt],
        });
        if (caught) { outcome = 'failure'; break; }
        continue;
      }

      const targetRow = row + move.row;
      const targetCol = col + move.col;
      const tile = room3TileAt(targetRow, targetCol);

      // The casing, the door while it is still shut, and the ring gates
      // without a battery to open them, are all simply solid. Being shoved
      // back by one leaves the agent exactly where it was, which is no
      // protection from something walking towards it.
      if (tile === '#' || (tile === 'D' && leg < LAST_LEG)
          || (tile === 'V' && !charged)) {
        const caught = meetsSentry(here, here, sentryWas, sentryNow);
        const reward = ROOM3_REWARDS.step + ROOM3_REWARDS.wall
                     + (caught ? ROOM3_REWARDS.caught : 0);
        total += reward;
        steps.push({
          position: { x: col + 0.5, y: row + 0.5 },
          velocity: null, reward: reward, action: move.name,
          entityStates: null, entityPositions: [sentryAt],
        });
        if (caught) { outcome = 'failure'; break; }
        continue;
      }

      const fromRow = here.row;
      const fromCol = here.col;
      row = targetRow;
      col = targetCol;

      let reward = ROOM3_REWARDS.step;
      let states = null;

      const met = meetsSentry(here, { row: row, col: col },
                              sentryWas, sentryNow);

      if (tile === 'H') {
        reward += ROOM3_REWARDS.beam;
        outcome = 'failure';
      } else if (met) {
        // The sentry does not end the trial: it shoves R-5 back the way it
        // came and charges it for the trouble.
        reward += ROOM3_REWARDS.caught;
        row = fromRow;
        col = fromCol;
      } else if (tile === ',' && trapLive) {
        // Live cable. It hurts, and it does not stop the run.
        reward += ROOM3_REWARDS.trap;
        states = [{ id: 'trap-' + row + '-' + col, state: 'arcing' }];
      } else if (tile === 'E' && !charged) {
        charged = true;
        reward += ROOM3_REWARDS.cell;
        states = [{ id: 'battery', state: 'taken' }].concat(
          ROOM3_GATES.map(id => ({ id: id, state: 'open' })));
      } else if (tile === 'P' && !recharged) {
        recharged = true;
        reward += ROOM3_REWARDS.charge;
        states = [{ id: 'charger', state: 'used' }];
      } else if (tile === 'X' && leg < LAST_LEG) {
        // The way out, reached with the job unfinished. It does not open,
        // and finding that out costs something.
        reward += ROOM3_REWARDS.unfinished;
        row = fromRow;
        col = fromCol;
      } else if (leg < LAST_LEG && tile === ROOM3_LEGS[leg].tile) {
        // Whatever was next in the chain. Everything else is scenery: a
        // key picked up early opens nothing, and a generator reached out
        // of turn does not run.
        if (KEY_OPENS[tile]) {
          reward += ROOM3_REWARDS.key;
          states = [
            { id: 'key-' + tile, state: 'taken' },
            // Its generator lights up as soon as the key is in hand, so
            // which key belongs to which machine is visible rather than
            // something to be remembered.
            { id: 'generator-' + KEY_OPENS[tile], state: 'armed' },
          ];
        } else {
          reward += ROOM3_REWARDS.generator;
          states = [{ id: 'generator-' + tile, state: 'live' }];
          if (tile === '3') {
            // The last one unseals the door.
            const door = room3Find('D');
            states.push({ id: 'cell-' + door.row + '-' + door.col,
                          state: 'open' });
          }
        }
        leg += 1;
      } else if (tile === 'X' && leg === LAST_LEG) {
        reward += ROOM3_REWARDS.exit;
        outcome = 'success';
      }

      total += reward;
      steps.push({
        position: { x: col + 0.5, y: row + 0.5 },
        velocity: null, reward: reward, action: move.name,
        entityStates: states, entityPositions: [sentryAt],
      });

      if (outcome !== 'timeout') break;
    }

    return { steps: steps, totalReward: total, outcome: outcome };
  }

  /* ---------------------------------------------------------------------
     Room 4 — the Drone Wind Tunnel
     ------------------------------------------------------------------- */

  /* The first room with no grid in it at all.

     The assignment fixes this room's dynamics, and they are followed here:
     the hall is ten metres square, the state is (x, y, vx, vy), a heading
     is chosen every 0.02 s, and each velocity component is one of -1, 0
     or 1 metres per second.  Nothing is rounded to a cell and nothing is
     indexed by one; the renderer draws it through the same transform as
     every grid so far, which is the point of having had one transform.

     A consequence worth knowing: at a fiftieth of a second a step, and a
     metre a second at most, crossing the hall takes several hundred steps.
     That is why the room carries a playback hint.                       */

  const TUNNEL_SIZE = 10;
  const TUNNEL_DT = 0.02;
  const TUNNEL_START = { x: 1.0, y: 1.2 };
  const TUNNEL_PAD = { x: 8.4, y: 8.4 };
  /* The brief asks for a continuous reward here rather than a flat cost
     per tick, and it is right to: a drone that is paid only for arriving
     has no idea whether the last hundredth of a second helped. So the bulk
     of what it earns is the distance it closed on the pad, and the flat
     costs are small beside it. */
  const TUNNEL_REWARDS = { tick: -0.01, closing: 1.0, touched: -100,
                           zone: -5, hard: -30, landed: 200 };

  /* The two zones, and the gate that comes and goes. */
  const BRAKE = { x: 5.6, y: 2.4, width: 2.4, height: 1.4 };
  const BOOST = { x: 2.6, y: 7.4, width: 2.6, height: 1.4 };

  /* A barrier that rises and falls across the middle of the hall. It is
     the one thing here that has to be timed rather than merely steered
     round, and where it is depends only on the clock. */
  const GATE = { x: 6.4, span: 3.0, top: 3.6, travel: 2.6, period: 260 };

  function gateAt(tick) {
    // Up and back down, forever.
    const phase = (tick % GATE.period) / GATE.period;
    const swing = phase < 0.5 ? phase * 2 : (1 - phase) * 2;
    return GATE.top + swing * GATE.travel;
  }

  function inZone(zone, x, y) {
    return Math.abs(x - zone.x) < zone.width / 2
        && Math.abs(y - zone.y) < zone.height / 2;
  }

  /* Where the fans are, which way each one blows, and the lane each one
     blows down.

     Four vents, four directions, and every lane lines up with the vent it
     comes from. It did not used to: the south vent was drawn between five
     and eight metres and its wind acted between eight and ten, so the
     drone was shoved by a fan that was nowhere near it. A lane is now
     derived from the vent's own extent, which is the only way the two can
     stay honest with each other.

     The wind is still a function of position and nothing else: which lane
     the drone is in decides what is pushing it, and its own speed and
     heading make no difference at all. */
  const VENT_PUSH = 0.5;

  const VENTS = [
    { id: 'vent-north', blows: 'down',
      at: { x: 2.75, y: 0.35 }, size: { width: 2.5, height: 0.5 } },
    { id: 'vent-east', blows: 'left',
      at: { x: 9.65, y: 2.1 }, size: { width: 0.5, height: 2.2 } },
    { id: 'vent-west', blows: 'right',
      at: { x: 0.35, y: 6.75 }, size: { width: 0.5, height: 2.5 } },
    { id: 'vent-south', blows: 'up',
      at: { x: 7.75, y: 9.65 }, size: { width: 2.5, height: 0.5 } },
  ];

  /* Derived, never written down twice: a vent that blows up or down owns
     the band of x it spans, and one that blows left or right owns the band
     of y. Move the vent and its draught moves with it. */
  VENTS.forEach(vent => {
    const sideways = vent.blows === 'left' || vent.blows === 'right';
    const centre = sideways ? vent.at.y : vent.at.x;
    const half = (sideways ? vent.size.height : vent.size.width) / 2;
    vent.lane = { axis: sideways ? 'y' : 'x',
                  from: centre - half, to: centre + half };
    vent.push = {
      x: { left: -VENT_PUSH, right: VENT_PUSH }[vent.blows] || 0,
      y: { up: -VENT_PUSH, down: VENT_PUSH }[vent.blows] || 0,
    };
  });

  /** What the air is doing at a point. Position only, by design. */
  function windAt(x, y) {
    let pushX = 0;
    let pushY = 0;
    VENTS.forEach(vent => {
      const along = vent.lane.axis === 'x' ? x : y;
      if (along >= vent.lane.from && along <= vent.lane.to) {
        pushX += vent.push.x;
        pushY += vent.push.y;
      }
    });
    return { x: pushX, y: pushY };
  }

  function tunnelRoom() {
    const entities = [
      { id: 'pad', type: 'pad', position: { x: TUNNEL_PAD.x, y: TUNNEL_PAD.y },
        size: { width: 1.6, height: 0.4 } },
      { id: 'release', type: 'start',
        position: { x: TUNNEL_START.x, y: TUNNEL_START.y },
        size: { width: 0.9, height: 0.9 } },
    ];
    entities.push({
      id: 'brake', type: 'brake',
      position: { x: BRAKE.x, y: BRAKE.y },
      size: { width: BRAKE.width, height: BRAKE.height },
    });
    entities.push({
      id: 'boost', type: 'boost',
      position: { x: BOOST.x, y: BOOST.y },
      size: { width: BOOST.width, height: BOOST.height },
      appearance: { blows: 'right' },
    });
    entities.push({
      id: 'gate', type: 'gate',
      position: { x: GATE.x, y: gateAt(0) },
      size: { width: 0.45, height: GATE.span },
    });

    VENTS.forEach(vent => {
      entities.push({ id: vent.id, type: 'vent', position: vent.at,
                      size: vent.size,
                      appearance: { blows: vent.blows } });

      // The lane it blows down, across the whole hall. Unshifted at the
      // front of the list so it is drawn under everything.
      const sideways = vent.lane.axis === 'y';
      const middle = (vent.lane.from + vent.lane.to) / 2;
      const width = vent.lane.to - vent.lane.from;
      entities.unshift({
        id: vent.id + '-draught', type: 'draught',
        position: sideways ? { x: TUNNEL_SIZE / 2, y: middle }
                           : { x: middle, y: TUNNEL_SIZE / 2 },
        size: sideways ? { width: TUNNEL_SIZE, height: width }
                       : { width: width, height: TUNNEL_SIZE },
        appearance: { blows: vent.blows },
      });
    });

    return {
      id: 'room4',
      name: 'Drone Wind Tunnel',
      sector: 'LAB SECTOR D-07',
      worldSize: { width: TUNNEL_SIZE, height: TUNNEL_SIZE },
      isGrid: false,
      entities: entities,
      entityTypes: pick(['pad', 'start', 'vent', 'draught', 'brake', 'boost',
                         'gate', 'agent']),
      metric: { key: 'reward', label: 'Mean reward', format: '%.1f' },
      parameterSchema: ROOM4_PARAMETERS,
      // Hundreds of small steps per flight, so it is shown quickly.
      playback: { stepsPerSecond: 90 },
      algorithm: {
        label: 'Semi-Gradient SARSA',
        summary: 'There is no table to fill in: position and velocity are '
               + 'real numbers, so the value of a state is computed from '
               + 'features of it and the weights are nudged instead.',
        updateRule: 'w ← w + α[ r + γ q̂(s′,a′,w) − q̂(s,a,w) ] ∇q̂(s,a,w)',
        watchFor: 'Awaiting the algorithm layer. Until it is wired up the '
                + 'flights here come from the mock producer and nothing is '
                + 'learned.',
      },
      info: {
        objective: 'Fly to the landing pad and arrive slowly. Position and '
                 + 'velocity are real numbers, so there is no table that '
                 + 'fits them and the value has to be approximated.',
        obstacles: [
          'Four fans blow across the hall, each down its own lane and each '
          + 'in a different direction: one south, one north, one east and '
          + 'one west. The lanes are drawn, so where the air is moving and '
          + 'which way is something you can see.',
          'The wind is a function of where the drone is and of nothing '
          + 'else — its own speed and heading make no difference to what '
          + 'is pushing it. Where two lanes cross, both do.',
          'The walls stop the drone rather than ending the flight, but a '
          + 'drone held against one is not getting anywhere.',
          'A barrier rises and falls across the middle of the hall. Where '
          + 'it is depends only on the clock, so it has to be timed rather '
          + 'than steered around.',
          'Two zones change what a moment of thrust is worth: one takes '
          + 'the speed out of it, the other adds half again. Both cost a '
          + 'little to be in.',
          'Each velocity component is -1, 0 or +1, so there are only '
          + 'three speeds to arrive at: nothing, a metre a second along '
          + 'one axis, or √2 across the diagonal. The diagonal is a '
          + 'crash. Straightening up before the pad is the skill.',
        ],
        actionSet: 'A heading every 0.02 s. Each velocity component is one '
                 + 'of -1, 0 or +1 metres per second, so the drone has nine '
                 + 'ways to be moving and no more.',
        rewards: [
          ['Each time unit', '-0.01'],
          ['Ground closed on the pad', '+1 per metre, and -1 per metre lost'],
          ['Being inside a zone', '-5'],
          ['Touching a wall or the barrier', '-100'],
          ['Arriving across the diagonal', '-30'],
          ['Landing safely', '+200'],
        ],
        terminal: 'The flight ends on the pad, whether it lands or crashes, '
                + 'and otherwise when the time limit runs out.',
        note: 'This is the first room whose state cannot be written down '
            + 'as a cell. Everything before it could be kept in a table; '
            + 'from here on the value of a state has to be computed from '
            + 'features of it, which is the whole reason the room exists. '
            + 'The reward is continuous for the same reason: most of what '
            + 'a flight earns is the ground it closed on the pad, so every '
            + 'hundredth of a second has something to say about itself '
            + 'rather than waiting for the landing to find out.',
      },
    };
  }

  const ROOM4_PARAMETERS = [
    { key: 'alpha', label: 'Learning rate', symbol: 'α', type: 'number',
      min: 0.01, max: 1.0, step: 0.01, default: 0.20, appliesLive: true,
      description: 'Divided across the tilings, so more tilings means '
                 + 'smaller steps each.' },
    { key: 'gamma', label: 'Discount factor', symbol: 'γ', type: 'number',
      min: 0.50, max: 1.0, step: 0.005, default: 1.0, appliesLive: false,
      description: 'Undiscounted by default: every flight ends, so there is '
                 + 'nothing to discount against.' },
    { key: 'epsilon', label: 'Exploration rate', symbol: 'ε', type: 'number',
      min: 0.0, max: 1.0, step: 0.01, default: 1.00, appliesLive: true,
      description: 'How often a random heading is taken instead of the best.' },
    { key: 'epsilonMin', label: 'Minimum exploration', symbol: '', type: 'number',
      min: 0.0, max: 0.5, step: 0.01, default: 0.05, appliesLive: true,
      description: 'Exploration never decays below this.' },
    { key: 'epsilonDecay', label: 'Exploration decay', symbol: '', type: 'number',
      min: 0.90, max: 1.0, step: 0.001, default: 0.94, appliesLive: true,
      description: 'Exploration is multiplied by this after every flight.' },
    { key: 'tilings', label: 'Number of tilings', symbol: '', type: 'number',
      min: 1, max: 32, step: 1, default: 8, appliesLive: false,
      description: 'How many offset grids the four dimensions are covered '
                 + 'with. More generalises more smoothly and costs more.' },
    { key: 'tilesPerDimension', label: 'Tiles per dimension', symbol: '',
      type: 'number', min: 2, max: 16, step: 1, default: 8, appliesLive: false,
      description: 'How finely each of x, y, vx and vy is cut up.' },
    { key: 'episodes', label: 'Flights to train', symbol: '', type: 'number',
      min: 100, max: 8000, step: 100, default: 2000, appliesLive: false,
      description: 'How many flights to learn from before the run is done.' },
  ];

  /** The nine ways the drone may be moving. */
  const HEADINGS = [];
  [-1, 0, 1].forEach(vx => {
    [-1, 0, 1].forEach(vy => { HEADINGS.push({ x: vx, y: vy }); });
  });

  /**
   * One flight.
   *
   * `skill` stands in for how well the weights have been learned: at zero
   * the heading is close to random, at one it is chosen to close on the
   * pad and to arrive slow. Nothing is learned; this is a path, not a
   * policy.
   */
  function flight(next, skill, epsilon, maxSteps) {
    let x = TUNNEL_START.x;
    let y = TUNNEL_START.y;
    let velocity = { x: 0, y: 0 };

    const gateSpot = tick => ({ id: 'gate',
                               position: { x: GATE.x, y: gateAt(tick) } });

    const steps = [{
      position: { x: x, y: y }, velocity: { x: 0, y: 0 },
      reward: 0, action: '—', entityStates: null,
      entityPositions: [gateSpot(0)],
    }];

    let total = 0;
    let outcome = 'timeout';
    let wasAway = Math.hypot(TUNNEL_PAD.x - x, TUNNEL_PAD.y - y);

    for (let tick = 0; tick < maxSteps; tick += 1) {
      // Choose a heading: at random while exploring, otherwise the one
      // that points at the pad, slowed as it gets close.
      if (next() < epsilon || skill < next() * 0.4) {
        velocity = HEADINGS[Math.floor(next() * HEADINGS.length)];
      } else {
        const toX = TUNNEL_PAD.x - x;
        const toY = TUNNEL_PAD.y - y;
        const gateY = gateAt(tick + 1);
        const atTheGate = Math.abs(x - GATE.x) < 1.1
          && Math.abs(y - gateY) < GATE.span / 2 + 0.5;

        if (atTheGate) {
          // Wait it out rather than fly into it: the barrier is the one
          // thing here that has to be timed.
          velocity = { x: 0, y: y < gateY ? -1 : 1 };
        } else if (Math.hypot(toX, toY) > 1.0) {
          // Still crossing the hall: both components, which is the
          // quickest way about and also the fastest way to arrive.
          velocity = { x: Math.sign(toX), y: Math.sign(toY) };
        } else {
          // The approach. Straightening up onto one axis is what makes
          // the difference between landing and arriving at √2 m/s, and it
          // is the whole skill this room asks for.
          velocity = Math.abs(toX) > Math.abs(toY)
            ? { x: Math.sign(toX), y: 0 }
            : { x: 0, y: Math.sign(toY) };
        }
      }

      const wind = windAt(x, y);
      // The zones scale what a moment of thrust is worth: one takes the
      // speed out of it, the other adds to it. Both are a function of
      // where the drone is and nothing else, like the wind.
      let scale = 1;
      if (inZone(BRAKE, x, y)) scale = 0.35;
      if (inZone(BOOST, x, y)) scale = 1.9;

      x += (velocity.x * scale + wind.x) * TUNNEL_DT;
      y += (velocity.y * scale + wind.y) * TUNNEL_DT;

      let reward = TUNNEL_REWARDS.tick;

      // Most of what a flight earns is the ground it closed on the pad,
      // so every hundredth of a second has something to say.
      const away = Math.hypot(TUNNEL_PAD.x - x, TUNNEL_PAD.y - y);
      reward += (wasAway - away) * TUNNEL_REWARDS.closing;
      wasAway = away;

      if (inZone(BRAKE, x, y) || inZone(BOOST, x, y)) {
        reward += TUNNEL_REWARDS.zone;
      }

      // The walls hold the drone rather than ending the flight.
      if (x < 0.2 || x > TUNNEL_SIZE - 0.2 || y < 0.2
          || y > TUNNEL_SIZE - 0.2) {
        x = Math.max(0.2, Math.min(TUNNEL_SIZE - 0.2, x));
        y = Math.max(0.2, Math.min(TUNNEL_SIZE - 0.2, y));
        reward += TUNNEL_REWARDS.touched;
      }

      // The gate, which is somewhere else every moment.
      const gateY = gateAt(tick + 1);
      if (Math.abs(x - GATE.x) < 0.3
          && Math.abs(y - gateY) < GATE.span / 2) {
        reward += TUNNEL_REWARDS.touched;
        outcome = 'failure';
      }

      const reached = Math.abs(x - TUNNEL_PAD.x) < 0.55
                   && Math.abs(y - TUNNEL_PAD.y) < 0.45;
      if (reached) {
        /* With each component one of -1, 0 or +1, there are only three
           speeds to arrive at: nothing, one metre a second along an axis,
           and √2 across the diagonal. So the rule is simply whether the
           drone straightened up before it got here. */
        const speed = Math.hypot(velocity.x, velocity.y);
        if (speed < 1.2) { reward += TUNNEL_REWARDS.landed; outcome = 'success'; }
        else { reward += TUNNEL_REWARDS.hard; outcome = 'failure'; }
      }

      total += reward;
      steps.push({
        position: { x: x, y: y },
        velocity: { x: velocity.x, y: velocity.y },
        reward: reward,
        action: velocity.x + ',' + velocity.y,
        entityStates: null,
        entityPositions: [gateSpot(tick + 1)],
      });
      if (outcome !== 'timeout') break;
    }

    return { steps: steps, totalReward: total, outcome: outcome };
  }

  /* ---------------------------------------------------------------------
     Room 5 — the Adaptive Storage Facility
     ------------------------------------------------------------------- */

  /* The building is different every time.

     This is the room the assignment marks optional, and the one that asks
     the hardest question: a policy that has memorised one warehouse is
     worth nothing here, because the next episode is not that warehouse.
     The count and the positions of the obstacles are drawn fresh for every
     run, and the last few runs use layouts the agent has never been near.

     What the agent may see is a parameter rather than a fact: the look-
     ahead decides how far in front of itself it can make anything out, and
     the cone the renderer draws around it is exactly that distance.       */

  const STORE_SIZE = 10;
  const STORE_DT = 0.02;
  const STORE_START = { x: 0.9, y: 5.0 };
  const STORE_TERMINAL = { x: 8.6, y: 1.6 };
  const STORE_GATE = { x: 9.2, y: 8.4 };
  const OBSTACLE_MAX = 8;
  const OBSTACLE_WIDTH = 0.5;              // the brief's half a metre
  const STORE_REWARDS = { tick: -0.02, closing: 1.0, hit: -70,
                          terminal: 80, escaped: 160 };

  /* The shelving. Fixed, and the same in every trial: it is the part of
     the building the evaluation system does not rearrange, which is what
     stops the whole thing being noise. */
  const SHELVES = [
    { x: 3.4, y: 2.2, width: 0.6, height: 3.4 },
    { x: 3.4, y: 7.6, width: 0.6, height: 3.0 },
    { x: 6.2, y: 4.6, width: 0.6, height: 3.6 },
    { x: 6.2, y: 0.9, width: 0.6, height: 1.4 },
  ];

  /* Two maintenance robots, each on a fixed run. They do not chase and
     they do not stop; meeting one ends the trial. */
  const MAINTENANCE = [
    // Neither run spans the building. One that did would be a moving wall
    // with no way past, which is a different room from this one.
    { id: 'mover-a', axis: 'y', at: 5.0, from: 1.0, to: 5.0, speed: 0.9,
      phase: 0 },
    { id: 'mover-b', axis: 'x', at: 7.4, from: 1.2, to: 5.4, speed: 0.7,
      phase: 0.5 },
  ];

  /** Where a maintenance robot is at a given moment. Up and back, forever. */
  function moverAt(mover, tick) {
    const span = mover.to - mover.from;
    const cycle = (2 * span) / mover.speed;
    const time = (tick * STORE_DT) / 1 + mover.phase * cycle;
    const phase = (time % cycle) / cycle;
    const along = phase < 0.5 ? mover.from + phase * 2 * span
                              : mover.to - (phase - 0.5) * 2 * span;
    return mover.axis === 'y' ? { x: mover.at, y: along }
                              : { x: along, y: mover.at };
  }

  /* A conveyor. Standing on it, the robot is carried whether it likes it
     or not — and it runs towards the shelving, so it is a shortcut that
     has to be got off in time. */
  const CONVEYOR = { x: 5.0, y: 9.0, width: 4.4, height: 0.8,
                     push: { x: 1.4, y: 0 } };

  function onConveyor(x, y) {
    return Math.abs(x - CONVEYOR.x) < CONVEYOR.width / 2
        && Math.abs(y - CONVEYOR.y) < CONVEYOR.height / 2;
  }

  /* A coarse map of the fixed shelving, so the mock can find its way round
     a three-metre rack instead of bouncing off it. Only the shelving is in
     it — the crates move every trial and the robots move every step, and
     both are dealt with as they are met. This is a route, not a policy. */
  const STORE_CELL = 0.5;
  const STORE_CELLS = Math.round(STORE_SIZE / STORE_CELL);

  function storeCellCentre(row, col) {
    return { x: (col + 0.5) * STORE_CELL, y: (row + 0.5) * STORE_CELL };
  }

  /* The route keeps a good deal further off than a collision needs: it is
     walked diagonally and in continuous space, so a path that merely
     clears a rack on paper clips its corner in practice. */
  const ROUTE_CLEARANCE = 0.45;

  function storeBlocked(row, col, crates) {
    if (row < 0 || col < 0 || row >= STORE_CELLS || col >= STORE_CELLS) {
      return true;
    }
    const at = storeCellCentre(row, col);
    const nearShelf = SHELVES.some(shelf =>
      Math.abs(at.x - shelf.x) < shelf.width / 2 + ROUTE_CLEARANCE
      && Math.abs(at.y - shelf.y) < shelf.height / 2 + ROUTE_CLEARANCE);
    if (nearShelf) return true;
    // The crates are different every trial, so the route is worked out
    // for the trial rather than once for the building.
    return (crates || []).some(crate =>
      Math.abs(at.x - crate.x) < OBSTACLE_WIDTH / 2 + ROUTE_CLEARANCE
      && Math.abs(at.y - crate.y) < OBSTACLE_WIDTH / 2 + ROUTE_CLEARANCE);
  }

  function storeField(target, crates) {
    const field = {};
    const startRow = Math.floor(target.y / STORE_CELL);
    const startCol = Math.floor(target.x / STORE_CELL);
    const queue = [[startRow, startCol, 0]];
    field[startRow + ',' + startCol] = 0;

    while (queue.length) {
      const [row, col, depth] = queue.shift();
      [[-1, 0], [1, 0], [0, -1], [0, 1]].forEach(([dr, dc]) => {
        const nextRow = row + dr;
        const nextCol = col + dc;
        if (storeBlocked(nextRow, nextCol, crates)) return;
        if (field[nextRow + ',' + nextCol] !== undefined) return;
        field[nextRow + ',' + nextCol] = depth + 1;
        queue.push([nextRow, nextCol, depth + 1]);
      });
    }
    return field;
  }

  /**
   * The whole way round the racks, as a list of places to be.
   *
   * Worked out once and then followed, rather than recomputed every tick:
   * a heading picked afresh each hundredth of a second flips the moment
   * the robot drifts over a cell boundary, and the robot spends the trial
   * shuffling between two cells instead of crossing the building.
   */
  function storeRoute(field, x, y) {
    let row = Math.floor(y / STORE_CELL);
    let col = Math.floor(x / STORE_CELL);
    const route = [];

    for (let guard = 0; guard < STORE_CELLS * STORE_CELLS; guard += 1) {
      const here = field[row + ',' + col];
      if (here === undefined || here === 0) break;

      let step = null;
      let shortest = here;
      [[-1, 0], [1, 0], [0, -1], [0, 1]].forEach(([dr, dc]) => {
        const at = field[(row + dr) + ',' + (col + dc)];
        if (at === undefined || at >= shortest) return;
        shortest = at;
        step = [row + dr, col + dc];
      });
      if (!step) break;
      row = step[0];
      col = step[1];
      route.push(storeCellCentre(row, col));
    }
    return route;
  }

  function hitsShelf(x, y) {
    return SHELVES.some(shelf =>
      Math.abs(x - shelf.x) < shelf.width / 2 + 0.12
      && Math.abs(y - shelf.y) < shelf.height / 2 + 0.12);
  }

  /** A warehouse: how many crates there are, and where they stand. */
  function warehouse(next, crateCount) {
    const count = crateCount || (3 + Math.floor(next() * (OBSTACLE_MAX - 2)));
    const crates = [];
    for (let index = 0; index < count; index += 1) {
      crates.push({
        // Kept out of the two metres around the start and the exit, so a
        // layout is never impossible before it begins.
        x: 2.6 + next() * 5.0,
        y: 0.8 + next() * 8.4,
      });
    }
    return crates;
  }

  function storeRoom(parameters) {
    const lookAhead = (parameters && parameters.lookAhead !== undefined)
      ? parameters.lookAhead : 2.5;

    const entities = [
      // The conveyor first, so everything else is drawn on top of it.
      { id: 'conveyor', type: 'conveyor',
        position: { x: CONVEYOR.x, y: CONVEYOR.y },
        size: { width: CONVEYOR.width, height: CONVEYOR.height },
        appearance: { blows: 'right' } },
      { id: 'terminal', type: 'terminal',
        position: { x: STORE_TERMINAL.x, y: STORE_TERMINAL.y },
        size: { width: 0.9, height: 0.9 },
      },
      { id: 'gate', type: 'exitGate',
        position: { x: STORE_GATE.x, y: STORE_GATE.y },
        size: { width: 0.5, height: 1.8 } },
      { id: 'release', type: 'start',
        position: { x: STORE_START.x, y: STORE_START.y },
        size: { width: 0.9, height: 0.9 } },
    ];
    SHELVES.forEach((shelf, index) => {
      entities.push({
        id: 'shelf-' + index, type: 'shelf',
        position: { x: shelf.x, y: shelf.y },
        size: { width: shelf.width, height: shelf.height },
      });
    });
    MAINTENANCE.forEach(mover => {
      const at = moverAt(mover, 0);
      entities.push({
        id: mover.id, type: 'mover',
        position: at, size: { width: 0.7, height: 0.7 },
      });
    });
    // Every crate the building may contain is declared once. Each run says
    // where its own are and hides the rest.
    for (let index = 0; index < OBSTACLE_MAX; index += 1) {
      entities.push({
        id: 'crate-' + index, type: 'crate',
        position: { x: -5, y: -5 },
        size: { width: OBSTACLE_WIDTH, height: OBSTACLE_WIDTH },
      });
    }

    return {
      id: 'room5',
      name: 'Adaptive Storage Facility',
      sector: 'LAB SECTOR E-12',
      worldSize: { width: STORE_SIZE, height: STORE_SIZE },
      isGrid: false,
      entities: entities,
      entityTypes: pick(['shelf', 'crate', 'conveyor', 'mover', 'terminal',
                         'exitGate', 'start', 'agent']),
      metric: { key: 'reward', label: 'Mean reward', format: '%.1f' },
      parameterSchema: ROOM5_PARAMETERS,
      playback: { stepsPerSecond: 90 },
      // What the agent can make out ahead of it, drawn where it is looking.
      observation: { radius: lookAhead, spread: Math.PI / 3.4 },
      algorithm: {
        label: 'Semi-Gradient Q-Learning',
        summary: 'Approximates the value of behaving perfectly from here '
               + 'on, from features of the state rather than from a table '
               + '— which is the only way anything carries over to a '
               + 'building it has not seen.',
        updateRule: 'w ← w + α[ r + γ max q̂(s′,a′,w) − q̂(s,a,w) ] ∇q̂(s,a,w)',
        watchFor: 'Awaiting the algorithm layer. Until it is wired up the '
                + 'runs here come from the mock producer and nothing is '
                + 'learned.',
      },
      info: {
        objective: 'Reach the control terminal to shut the evaluation '
                 + 'system down, which unseals the main gate — then get to '
                 + 'the gate. The building is rearranged between trials, so '
                 + 'the way through has to be worked out rather than '
                 + 'remembered.',
        obstacles: [
          'Shelving, which is the same in every trial and is the only part '
          + 'of the building that is.',
          'Between three and eight crates, half a metre across, in fresh '
          + 'positions every single trial.',
          'Two maintenance robots running fixed routes. They do not chase '
          + 'and they do not stop.',
          'A conveyor along the south wall, which carries whatever is '
          + 'standing on it whether it meant to be carried or not.',
          'Touching any of them ends the trial.',
          'The agent sees only what is within its look-ahead, and only in '
          + 'front of it. The cone drawn around it is exactly what it can '
          + 'make out — never the map, never where the gate is.',
        ],
        actionSet: 'A heading every 0.02 s, as in the wind tunnel.',
        rewards: [
          ['Each time unit', '-0.02'],
          ['Ground closed on what it is heading for', '+1 per metre'],
          ['Touching shelving, a crate or a robot', '-70'],
          ['Reaching the control terminal', '+80'],
          ['Reaching the main gate after it', '+160'],
        ],
        terminal: 'The trial ends at the main gate, or against anything '
                + 'solid, and otherwise when the time limit runs out. '
                + 'Reaching the terminal does not end it — it is halfway.',
        note: 'This is the lab’s final assessment, and the building is '
            + 'rebuilt for every trial to find out whether R-5 has learned '
            + 'to think or merely to repeat. Nothing here can be '
            + 'memorised. What the observation carries '
            + 'is the agent’s own motion and whatever is inside the look-'
            + 'ahead — never the map, never where it is in the building — '
            + 'so what is learned has to be a way of getting through '
            + 'warehouses rather than a way through this one.',
      },
    };
  }

  const ROOM5_PARAMETERS = [
    { key: 'alpha', label: 'Learning rate', symbol: 'α', type: 'number',
      min: 0.01, max: 1.0, step: 0.01, default: 0.20, appliesLive: true,
      description: 'Divided across the tilings, so more tilings means '
                 + 'smaller steps each.' },
    { key: 'gamma', label: 'Discount factor', symbol: 'γ', type: 'number',
      min: 0.50, max: 1.0, step: 0.005, default: 0.99, appliesLive: false,
      description: 'How much a reward in the future is worth now.' },
    { key: 'epsilon', label: 'Exploration rate', symbol: 'ε', type: 'number',
      min: 0.0, max: 1.0, step: 0.01, default: 1.00, appliesLive: true,
      description: 'How often a random heading is taken instead of the best.' },
    { key: 'epsilonMin', label: 'Minimum exploration', symbol: '', type: 'number',
      min: 0.0, max: 0.5, step: 0.01, default: 0.05, appliesLive: true,
      description: 'Exploration never decays below this.' },
    { key: 'epsilonDecay', label: 'Exploration decay', symbol: '', type: 'number',
      min: 0.90, max: 1.0, step: 0.001, default: 0.94, appliesLive: true,
      description: 'Exploration is multiplied by this after every run.' },
    { key: 'crates', label: 'Crates per trial', symbol: '', type: 'number',
      min: 0, max: 8, step: 1, default: 0, appliesLive: false,
      description: 'How many crates the system puts out. Left at zero it '
                 + 'draws a fresh number between three and eight for every '
                 + 'trial, which is the setting the room is about.' },
    { key: 'lookAhead', label: 'Look-ahead', symbol: '', type: 'number',
      min: 0.5, max: 5.0, step: 0.1, default: 2.5, appliesLive: false,
      description: 'How far ahead the agent can make anything out, in '
                 + 'metres, measured centre to centre. Everything beyond it '
                 + 'may as well not be there.' },
    { key: 'tilings', label: 'Number of tilings', symbol: '', type: 'number',
      min: 1, max: 32, step: 1, default: 8, appliesLive: false,
      description: 'How many offset grids the features are covered with.' },
    { key: 'episodes', label: 'Runs to train', symbol: '', type: 'number',
      min: 100, max: 8000, step: 100, default: 2500, appliesLive: false,
      description: 'How many runs to learn from before it is done.' },
  ];

  /**
   * One run across the facility, through a building drawn for it.
   *
   * The crates are reported at the first step, which is what puts them
   * where this run has them and hides the ones it does not use.
   */
  /**
   * One trial in the facility.
   *
   * Two halves, and the second does not exist until the first is done:
   * find the control terminal, which shuts the evaluation system down and
   * opens the main gate, then get to the gate. Crates move between trials,
   * the shelving does not, and two maintenance robots run their fixed
   * routes throughout.
   */
  function storeRun(next, skill, epsilon, lookAhead, maxSteps, crateCount) {
    const crates = warehouse(next, crateCount);
    let x = STORE_START.x;
    let y = STORE_START.y;
    let velocity = { x: 0, y: 0 };
    let shutDown = false;

    const placed = [];
    const stowed = [];
    for (let index = 0; index < OBSTACLE_MAX; index += 1) {
      if (index < crates.length) {
        placed.push({ id: 'crate-' + index, position: crates[index] });
        stowed.push({ id: 'crate-' + index, state: 'present' });
      } else {
        stowed.push({ id: 'crate-' + index, state: 'stowed' });
      }
    }

    const movers = tick => MAINTENANCE.map(
      mover => ({ id: mover.id, position: moverAt(mover, tick) }));

    const steps = [{
      position: { x: x, y: y }, velocity: { x: 0, y: 0 },
      reward: 0, action: '—',
      entityStates: stowed, entityPositions: placed.concat(movers(0)),
    }];

    let total = 0;
    let outcome = 'timeout';
    let target = STORE_TERMINAL;
    let route = storeRoute(storeField(STORE_TERMINAL, crates), x, y);
    let wasAway = Math.hypot(target.x - x, target.y - y);

    for (let tick = 0; tick < maxSteps; tick += 1) {
      if (next() < epsilon || skill < next() * 0.35) {
        velocity = HEADINGS[Math.floor(next() * HEADINGS.length)];
      } else {
        // Head for whatever is wanted next, and swing round anything close
        // enough to make out. Only what is inside the look-ahead may be
        // reacted to, which is the whole point of the parameter.
        // Walk the route worked out for this half of the trial, and fall
        // back to the straight line once it has run out.
        while (route.length && Math.hypot(route[0].x - x, route[0].y - y)
               < STORE_CELL * 0.7) {
          route.shift();
        }
        const waypoint = route.length ? route[0] : target;
        let steer = {
          x: Math.abs(waypoint.x - x) < 0.08 ? 0 : Math.sign(waypoint.x - x),
          y: Math.abs(waypoint.y - y) < 0.08 ? 0 : Math.sign(waypoint.y - y),
        };
        /* A maintenance robot is judged on where it will be next rather
           than where it is, and only at close quarters. Giving it a wide
           berth sounds safer and is not: the robot patrols the line the
           route has to cross, so a robot that is avoided from two metres
           away is never crossed at all, and the trial ends in a standoff
           rather than at the terminal. Close in, the thing to do is stand
           still and let it go past. */
        MAINTENANCE.forEach(mover => {
          const at = moverAt(mover, tick + 1);
          const wouldX = x + steer.x * STORE_DT;
          const wouldY = y + steer.y * STORE_DT;
          if (Math.hypot(at.x - wouldX, at.y - wouldY) < 0.7) {
            steer = { x: 0, y: 0 };
          }
        });
        velocity = steer;
      }

      x += velocity.x * STORE_DT;
      y += velocity.y * STORE_DT;

      // The conveyor carries whatever is standing on it.
      if (onConveyor(x, y)) {
        x += CONVEYOR.push.x * STORE_DT;
        y += CONVEYOR.push.y * STORE_DT;
      }

      x = Math.max(0.2, Math.min(STORE_SIZE - 0.2, x));
      y = Math.max(0.2, Math.min(STORE_SIZE - 0.2, y));

      let reward = STORE_REWARDS.tick;
      const away = Math.hypot(target.x - x, target.y - y);
      reward += (wasAway - away) * STORE_REWARDS.closing;
      wasAway = away;

      let states = null;

      const struck = crates.some(
        crate => Math.abs(crate.x - x) < OBSTACLE_WIDTH * 0.8
              && Math.abs(crate.y - y) < OBSTACLE_WIDTH * 0.8)
        || hitsShelf(x, y)
        || MAINTENANCE.some(mover => {
          const at = moverAt(mover, tick + 1);
          return Math.abs(at.x - x) < 0.45 && Math.abs(at.y - y) < 0.45;
        });

      if (struck) {
        reward += STORE_REWARDS.hit;
        outcome = 'failure';
      } else if (!shutDown
                 && Math.hypot(STORE_TERMINAL.x - x, STORE_TERMINAL.y - y)
                    < 0.6) {
        // The evaluation system stops here: the crates settle, the lights
        // go green, and the main gate unseals.
        shutDown = true;
        reward += STORE_REWARDS.terminal;
        states = [{ id: 'terminal', state: 'down' },
                  { id: 'gate', state: 'open' }];
        target = STORE_GATE;
        route = storeRoute(storeField(STORE_GATE, crates), x, y);
        wasAway = Math.hypot(target.x - x, target.y - y);
      } else if (shutDown
                 && Math.hypot(STORE_GATE.x - x, STORE_GATE.y - y) < 0.7) {
        reward += STORE_REWARDS.escaped;
        outcome = 'success';
      }

      total += reward;
      steps.push({
        position: { x: x, y: y },
        velocity: { x: velocity.x, y: velocity.y },
        reward: reward,
        action: velocity.x + ',' + velocity.y,
        entityStates: states,
        entityPositions: movers(tick + 1),
      });
      if (outcome !== 'timeout') break;
    }

    return { steps: steps, totalReward: total, outcome: outcome };
  }

  /* ---------------------------------------------------------------------
     Batches
     ------------------------------------------------------------------- */

  /**
   * A batch that gets better as it goes.
   *
   * Exploration decays, so early episodes wander and fail and later ones
   * go more or less straight there. Which route "straight there" means is
   * decided once, from the collapse chance: this is the mock standing in
   * for the thing the room is about.
   */
  function batch(room, options) {
    const settings = Object.assign({
      episodes: 140, seed: 7, epsilonStart: 1.0, epsilonMin: 0.10,
      epsilonDecay: 0.965, bridgeCollapse: 0.01,
      lookAhead: 2.5,
      slip: 0.12,
      // A fiftieth of a second a step means several hundred of them to
      // cross a ten-metre hall, so the continuous rooms get a far longer
      // leash and fewer runs to hold in memory.
      // Room 5 is two journeys rather than one, at a fiftieth of a second
      // a step, so it needs half again what the wind tunnel does.
      maxSteps: { room3: 420, room4: 900, room5: 1500 }[room.id] || 140,
    }, options || {});
    if (room.id === 'room4') settings.episodes = Math.min(settings.episodes, 90);
    // Longer trials, so fewer of them held in memory at once.
    if (room.id === 'room5') settings.episodes = Math.min(settings.episodes, 60);

    const next = random(settings.seed);
    const value = routeValues(settings.bridgeCollapse);
    const field = value.bridge >= value.lap ? VIA_BRIDGE : VIA_LAP;

    const episodes = [];
    const metrics = [];
    let epsilon = settings.epsilonStart;

    for (let index = 0; index < settings.episodes; index += 1) {
      const share = index / Math.max(1, settings.episodes - 1);
      const skill = Math.min(0.98, share * 1.2);
      let run;
      if (room.id === 'room3') {
        run = reactorRun(next, epsilon, settings.maxSteps);
      } else if (room.id === 'room4') {
        run = flight(next, skill, epsilon, settings.maxSteps);
      } else if (room.id === 'room5') {
        run = storeRun(next, skill, epsilon, settings.lookAhead,
                       settings.maxSteps, settings.crates);
      } else {
        run = walk(next, epsilon, settings.bridgeCollapse, field,
                   settings.maxSteps, settings.slip);
      }

      episodes.push({
        index: index,
        steps: run.steps,
        totalReward: run.totalReward,
        outcome: run.outcome,
        epsilon: epsilon,
      });

      metrics.push({
        episode: index,
        reward: run.totalReward,
        steps: run.steps.length - 1,
        epsilon: epsilon,
        // Falls towards zero without quite reaching it, which is what a
        // convergence measure looks like.
        convergence: (1 - share) * (0.4 + next() * 0.35) + 0.004,
      });

      epsilon = Math.max(settings.epsilonMin, epsilon * settings.epsilonDecay);
    }

    forceLateSuccess(episodes, metrics, room, settings, field);
    return { episodes: episodes, metrics: metrics };
  }

  /** Guarantee one unmistakable success near the end. */
  function forceLateSuccess(episodes, metrics, room, settings, field) {
    if (!episodes.length) return;
    const at = Math.max(0, episodes.length - 3);
    if (episodes[at].outcome === 'success') return;

    const next = random(settings.seed + 991);
    // No exploration and no collapse, so it walks the route it settled on
    // and gets there. This is the episode "Show final route" will find.
    let run;
    if (room.id === 'room3') run = reactorRun(next, 0, settings.maxSteps);
    else if (room.id === 'room4') run = flight(next, 1, 0, settings.maxSteps);
    else if (room.id === 'room5') {
      run = storeRun(next, 1, 0, settings.lookAhead, settings.maxSteps,
                     settings.crates);
    } else run = walk(next, 0, 0, field, settings.maxSteps, 0);
    if (run.outcome !== 'success') return;

    episodes[at] = Object.assign({}, episodes[at], {
      steps: run.steps, totalReward: run.totalReward, outcome: 'success',
    });
    metrics[at] = Object.assign({}, metrics[at], {
      reward: run.totalReward, steps: run.steps.length - 1,
    });
  }

  /* ---------------------------------------------------------------------
     What the shell asks for
     ------------------------------------------------------------------- */

  const ROOMS = {
    room2: gridRoom,
    room3: gridRoom3,
    room4: tunnelRoom,
    room5: storeRoom,
  };

  return {
    available: Object.keys(ROOMS),

    /** The room. Parameters are passed in because some of them shape it. */
    room: function (id, parameters) {
      const build = ROOMS[id] || ROOMS.room2;
      return window.Contract.checkRoom(build(parameters));
    },

    /** What each route is worth right now, for the sidebar to show. */
    routeValues: routeValues,

    /** A fresh batch. `parameters` comes straight from the sidebar. */
    batch: function (room, parameters) {
      const settings = {};
      if (parameters) {
        if (parameters.epsilon !== undefined) settings.epsilonStart = parameters.epsilon;
        if (parameters.epsilonMin !== undefined) settings.epsilonMin = parameters.epsilonMin;
        if (parameters.epsilonDecay !== undefined) settings.epsilonDecay = parameters.epsilonDecay;
        if (parameters.bridgeCollapse !== undefined) {
          settings.bridgeCollapse = parameters.bridgeCollapse;
        }
        if (parameters.lookAhead !== undefined) {
          settings.lookAhead = parameters.lookAhead;
        }
        if (parameters.slip !== undefined) settings.slip = parameters.slip;
        if (parameters.crates !== undefined) settings.crates = parameters.crates;
        // Deliberately not the real episode count: a few hundred is all
        // the charts need to look honest, and this is a mock.
        if (parameters.episodes !== undefined) {
          settings.episodes = Math.max(40, Math.min(400,
            Math.round(parameters.episodes / 10)));
        }
      }
      return window.Contract.checkBatch(batch(room, settings));
    },
  };
})();
