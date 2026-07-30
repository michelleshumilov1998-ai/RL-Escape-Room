'use strict';

/* =====================================================================
   Comparing the methods a room offers, on one set of axes.

   A single curve says what one method did. It does not say whether that
   was any good, and every interesting claim in this project is
   comparative — Value Iteration and Policy Iteration reach the same plan
   in wildly different amounts of work; Q-Learning takes room 2's span and
   SARSA goes round. Neither shows up in one graph.

   This mounts on both screens, because both have something to compare:
   room 1 has two planners, rooms 2 and 3 have four learners. It owns its
   own session on the far end and never touches the room's own run, so
   comparing costs you nothing you were watching.

   ---------------------------------------------------------------------
   WHY THE CURVES CARRY THEIR OWN X VALUES
   ---------------------------------------------------------------------

   Because a point is not a unit of work. One call of Value Iteration is
   one sweep; one call of Policy Iteration is a whole policy evaluation —
   a couple of hundred sweeps — and then an improvement. Plotting both
   against point index would put 15 sweeps and 389 sweeps on the same tick
   and imply they were the same amount of effort. So every series is drawn
   against `curve.x`, which is what the method had actually done by then.
   ===================================================================== */

window.Compare = (function () {

  const C = window.ROOM_CONFIG;

  /* One CSS custom property per variant, in order. Four is enough for every
     room: the most any of them offers is four methods. The colours come from
     the stylesheet like everything else — nothing here writes a hex value. */
  const SERIES_COLORS = ['--accent', '--goal', '--hazard', '--muted'];

  function palette(count) {
    const computed = window.getComputedStyle(document.documentElement);
    const colours = [];
    for (let index = 0; index < count; index += 1) {
      colours.push(computed.getPropertyValue(
        SERIES_COLORS[index % SERIES_COLORS.length]).trim());
    }
    return {
      series: colours,
      hairline: computed.getPropertyValue('--hairline-faint').trim(),
      muted: computed.getPropertyValue('--muted').trim(),
    };
  }

  function element(tag, className, text) {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) node.textContent = text;
    return node;
  }

  /* ---- drawing ------------------------------------------------------- */

  /** Log-safe: the delta curves span orders of magnitude and reach zero. */
  function project(value, scale) {
    if (scale !== 'log') return value;
    return Math.log10(Math.max(value, 1e-12));
  }

  function extent(values) {
    let low = Infinity;
    let high = -Infinity;
    values.forEach(value => {
      if (!isFinite(value)) return;
      if (value < low) low = value;
      if (value > high) high = value;
    });
    if (low === Infinity) return { low: 0, high: 1 };
    if (low === high) return { low: low - 1, high: high + 1 };
    return { low: low, high: high };
  }

  /**
   * Every variant's curve, on shared axes.
   *
   * The x range is the union, so a method that needed ten times the work of
   * another is visibly ten times as wide rather than stretched to match.
   */
  function draw(canvas, variants, colours) {
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 260;
    const height = C.charts.height * 2;
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);

    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, width, height);

    const drawable = variants.filter(
      variant => variant.curve && variant.curve.points.length);
    if (!drawable.length) return;

    const scale = drawable[0].curve.scale;
    const pad = C.charts.padding;
    const left = pad.left;
    const right = width - pad.right;
    const top = pad.top;
    const bottom = height - pad.bottom;

    let xs = [];
    let ys = [];
    drawable.forEach(variant => {
      const curve = variant.curve;
      curve.points.forEach((value, index) => {
        const at = curve.x ? curve.x[index] : index;
        if (at !== undefined) xs.push(at);
        ys.push(project(value, scale));
      });
    });

    const xRange = extent(xs);
    const yRange = extent(ys);
    const threshold = drawable[0].curve.threshold;
    if (threshold) ys.push(project(threshold, scale));

    const screenX = value => left
      + (right - left) * (value - xRange.low) / (xRange.high - xRange.low);
    const screenY = value => bottom
      - (bottom - top) * (value - yRange.low) / (yRange.high - yRange.low);

    // The stopping threshold, where the room has one: it is what "finished"
    // means, so it belongs on the picture rather than in a caption.
    if (threshold) {
      ctx.save();
      ctx.strokeStyle = colours.hairline;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      const at = Math.round(screenY(project(threshold, scale))) + 0.5;
      ctx.moveTo(left, at);
      ctx.lineTo(right, at);
      ctx.stroke();
      ctx.restore();
    }

    drawable.forEach((variant, index) => {
      const curve = variant.curve;
      ctx.strokeStyle = colours.series[variants.indexOf(variant)];
      ctx.lineWidth = 1.5;
      ctx.lineJoin = 'round';
      ctx.beginPath();
      curve.points.forEach((value, at) => {
        const x = screenX(curve.x ? curve.x[at] : at);
        const y = screenY(project(value, scale));
        if (at === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
  }

  /* ---- the panel ----------------------------------------------------- */

  function mount(host, roomNumber) {
    let identifier = null;
    let describe = null;
    let snapshot = null;
    let running = false;
    /* The room's parameters as they stand on screen, so a comparison runs the
       settings you are looking at rather than the room's defaults. */
    let parameters = null;

    host.innerHTML = '';

    const button = element('button', 'btn wide', 'Compare methods');
    button.type = 'button';
    const hint = element('p', 'hint', 'Runs every method this chamber offers '
      + 'against the same layout, the same parameters and the same seed, so a '
      + 'difference between two curves is a difference between two methods.');
    const label = element('p', 'group-label');
    const canvas = element('canvas', 'chart-canvas');
    const legend = element('ul', 'legend compare-legend');
    const table = element('dl', 'readout');

    host.appendChild(button);
    host.appendChild(hint);
    host.appendChild(label);
    host.appendChild(canvas);
    host.appendChild(legend);
    host.appendChild(table);

    function nameOf(key) {
      const found = (describe.algorithms || []).filter(
        entry => entry.key === key);
      return found.length ? found[0].label : key;
    }

    function paint() {
      if (!snapshot) return;
      const variants = snapshot.variants;
      const colours = palette(variants.length);

      const curve = variants[0].curve;
      label.textContent = curve.label + ' · by ' + (curve.xLabel || 'step');
      draw(canvas, variants, colours);

      legend.innerHTML = '';
      variants.forEach((variant, index) => {
        const item = element('li');
        const swatch = element('span', 'compare-swatch');
        swatch.style.background = colours.series[index];
        item.appendChild(swatch);
        item.appendChild(element('span', null, nameOf(variant.algorithm)));
        legend.appendChild(item);
      });

      // One row per variant: how much work it took, and where it ended up.
      table.innerHTML = '';
      variants.forEach(variant => {
        const term = element('dt', null, nameOf(variant.algorithm));
        const value = variant.metric.value;
        const shown = (value === null || value === undefined)
          ? '—' : Number(value).toFixed(1);
        table.appendChild(term);
        table.appendChild(element('dd', null,
          variant.progress.count + ' ' + snapshot.describeUnit
          + '  ·  ' + variant.metric.label + ' ' + shown));
      });
    }

    async function post(path, body) {
      const response = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body || {}),
      });
      const text = await response.text();
      let payload = null;
      try {
        payload = text ? JSON.parse(text) : {};
      } catch (problem) {
        throw new Error('POST ' + path + ': the response is not JSON');
      }
      if (!response.ok) {
        throw new Error(payload.error || ('POST ' + path + ' failed'));
      }
      return payload;
    }

    async function run(parameters) {
      if (running) return;
      running = true;
      button.disabled = true;
      button.textContent = 'Comparing…';

      try {
        const created = await post('/api/comparison', {
          room: roomNumber,
          parameters: parameters || null,
        });
        identifier = created.comparison;
        describe = created.describe;
        snapshot = created.snapshot;
        snapshot.describeUnit = describe.unit;
        paint();

        // Guarded the same way the room's own loop is: a backstop against a
        // far end that never reports finished, not a policy.
        let slices = 0;
        while (!snapshot.finished && slices < 40000) {
          snapshot = await post('/api/comparison/' + identifier + '/advance',
                                { budgetMs: C.turboBudgetMs });
          snapshot.describeUnit = describe.unit;
          slices += 1;
          // Every so often rather than every slice: the curves move a long
          // way between redraws and the work is on the far end anyway.
          if (slices % 8 === 0) paint();
        }
        paint();
        button.textContent = 'Compare again';
      } catch (problem) {
        label.textContent = 'Comparison failed: '
                          + ((problem && problem.message) || problem);
        button.textContent = 'Compare methods';
      } finally {
        running = false;
        button.disabled = false;
      }
    }

    button.addEventListener('click', () => run(parameters));

    return {
      setParameters(values) { parameters = values; },
      release() {
        if (!identifier) return;
        const path = '/api/comparison/' + identifier;
        identifier = null;
        fetch(path, { method: 'DELETE', keepalive: true }).catch(() => {});
      },
    };
  }

  return { mount: mount };
})();
