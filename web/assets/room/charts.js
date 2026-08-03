'use strict';

/* =====================================================================
   The charts.

   Four small line charts of the metrics array, hand-rolled on canvases.
   No chart library, and no per-chart special casing: every one of them is
   the same function given a different accessor and a different label.

   Three cases have to work and are the reason this file is careful:

     no episodes yet     an empty array draws the frame and says so,
                         rather than dividing by a zero range
     one episode         a single point is a dot, not a line of length
                         zero, and its autoscale cannot be a zero span
     thousands           the series is bucketed down to a few hundred
                         columns before it is drawn, so a long run costs
                         the same as a short one

   Each chart shows the raw series faintly and a rolling average over it.
   The average is what the eye should follow; the raw series is there so
   the noise it is smoothing is visible rather than implied.
   ===================================================================== */

window.Charts = (function () {

  const C = window.ROOM_CONFIG;

  /* The learner's four series. `key` reads one number off an entry in the
     metrics array. Adding a fifth is one entry here and nothing else.

     A series may also carry:
       scale       'log' for something that falls over orders of magnitude
       threshold   a level to draw as a dashed line, so "finished" has a
                   visible meaning

     A screen may pass its own list to `build` instead; room 1 is a
     planner and measures itself per sweep rather than per episode, so it
     does exactly that. */
  const LEARNER_SERIES = [
    { key: 'reward', label: 'Reward per episode' },
    { key: 'steps', label: 'Steps per episode' },
    { key: 'epsilon', label: 'Exploration rate' },
    { key: 'convergence', label: 'Convergence measure' },
  ];

  /* A series may instead name several `keys` and a `source`, which draws them
     on one pair of axes in different colours with a small key underneath.
     Room 5 uses it for the one measurement that is not a property of a
     training episode: how the frozen policy does on the layouts it trains on
     against the layouts it has never seen. Those points come from
     `checkpoints` rather than from the per-episode history, which is why a
     series carries where to read itself from. */
  const MULTI_COLOURS = ['accent', 'warn', 'goal'];

  function palette() {
    const computed = window.getComputedStyle(document.documentElement);
    return {
      accent: computed.getPropertyValue(C.colors.accent).trim(),
      muted: computed.getPropertyValue(C.colors.muted).trim(),
      hairline: computed.getPropertyValue(C.colors.hairlineFaint).trim(),
      warn: computed.getPropertyValue(C.colors.warn).trim(),
      goal: computed.getPropertyValue(C.colors.goal).trim(),
    };
  }

  /**
   * Reduce a long series to at most `limit` points.
   *
   * Averaged within each bucket rather than sampled, so a spike is not
   * silently dropped just because it fell between two sample points.
   */
  function bucket(values, limit) {
    if (values.length <= limit) return values;
    const size = values.length / limit;
    const out = [];
    for (let index = 0; index < limit; index += 1) {
      const from = Math.floor(index * size);
      const to = Math.min(values.length, Math.floor((index + 1) * size));
      let total = 0;
      for (let at = from; at < to; at += 1) total += values[at];
      out.push(total / Math.max(1, to - from));
    }
    return out;
  }

  /**
   * Which index of the full series each bucket ended at.
   *
   * Bucketing throws away *where* a point was, and the x axis needs it back:
   * without this the axis can only count buckets, and a run of 4000 episodes
   * bucketed to 900 would be labelled 0..900. Same arithmetic as `bucket`, so
   * the two cannot drift apart.
   */
  function bucketIndices(length, limit) {
    if (length <= limit) {
      const all = new Array(length);
      for (let index = 0; index < length; index += 1) all[index] = index;
      return all;
    }
    const size = length / limit;
    const out = [];
    for (let index = 0; index < limit; index += 1) {
      const to = Math.min(length, Math.floor((index + 1) * size));
      out.push(Math.max(0, to - 1));
    }
    return out;
  }

  /** A trailing rolling average, same length as the input. */
  function rollingAverage(values, window_) {
    if (values.length === 0) return [];
    const out = new Array(values.length);
    let total = 0;
    for (let index = 0; index < values.length; index += 1) {
      total += values[index];
      if (index >= window_) total -= values[index - window_];
      const count = Math.min(index + 1, window_);
      out[index] = total / count;
    }
    return out;
  }

  function extent(values) {
    let low = Infinity;
    let high = -Infinity;
    values.forEach(value => {
      if (value < low) low = value;
      if (value > high) high = value;
    });
    if (low === Infinity) return { low: 0, high: 1 };
    if (high - low < 1e-9) {
      // A flat series still needs a band to be drawn in, or it would be a
      // line on the frame's edge. Widen around the value itself.
      const pad = Math.max(1e-6, Math.abs(high) * 0.1 || 1);
      return { low: low - pad, high: high + pad };
    }
    return { low: low, high: high };
  }

  /**
   * Draw one chart.
   *
   * Pure with respect to `values` — it is read, bucketed into a new array,
   * and never written to.
   */
  function drawSeries(canvas, rawValues, colours, options) {
    const settings = options || {};
    // A log series is transformed once, here, and everything downstream —
    // the extent, the smoothing, the threshold line — works in that
    // space. Smoothing a decaying quantity in log space is what keeps the
    // average from being dragged around by its own first few points.
    const values = settings.scale === 'log'
      ? rawValues.map(value => Math.log10(Math.max(Math.abs(value), 1e-12)))
      : rawValues;
    const ratio = window.devicePixelRatio || 1;
    const cssWidth = canvas.clientWidth || 1;
    const cssHeight = canvas.clientHeight || C.charts.height;

    canvas.width = Math.round(cssWidth * ratio);
    canvas.height = Math.round(cssHeight * ratio);

    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    const pad = C.charts.padding;
    const plot = {
      left: pad.left,
      top: pad.top,
      width: Math.max(1, cssWidth - pad.left - pad.right),
      height: Math.max(1, cssHeight - pad.top - pad.bottom),
    };

    if (!values.length) {
      ctx.fillStyle = colours.hairline;
      ctx.font = '10px ' + window.getComputedStyle(document.body).fontFamily;
      ctx.fillText('no episodes yet', plot.left + 2, plot.top + plot.height / 2);
      return;
    }

    /* THE ROLLING AVERAGE IS TAKEN BEFORE THE BUCKETING, NOT AFTER.
       It used to be the other way round: the series was bucketed down to 900
       points and *then* averaged over a window scaled to the bucket count. At
       4000 episodes that is a mean of roughly four buckets of four episodes —
       an average of averages, and not the 20-episode moving average the label
       claimed. Now the window is exactly `smoothingWindow` real episodes,
       computed on the full series, and only then reduced for drawing.

           smooth[i] = mean(reward[max(0, i-19) .. i])                       */
    const smoothedFull = rollingAverage(values, C.charts.smoothingWindow);
    const drawn = bucket(values, C.charts.maxPointsDrawn);
    const smoothed = bucket(smoothedFull, C.charts.maxPointsDrawn);
    // Where each drawn point sits in the full series, so the x axis can be
    // labelled with real episode numbers rather than bucket indices.
    const at = bucketIndices(values.length, C.charts.maxPointsDrawn);
    const labelFor = index => (settings.xValues
      ? settings.xValues[at[index]]
      : at[index] + 1);
    // The threshold has to be inside the band or the line marking it
    // would sit on the frame's edge and say nothing.
    const level = settings.threshold === undefined ? null
      : (settings.scale === 'log' ? Math.log10(Math.max(settings.threshold, 1e-12))
                                  : settings.threshold);
    const span = extent(level === null ? drawn : drawn.concat([level]));

    function plotX(index) {
      if (drawn.length === 1) return plot.left + plot.width / 2;
      return plot.left + (index / (drawn.length - 1)) * plot.width;
    }
    function plotY(value) {
      const share = (value - span.low) / (span.high - span.low);
      return plot.top + plot.height - share * plot.height;
    }

    if (level !== null) {
      ctx.strokeStyle = colours.hairline;
      ctx.setLineDash([3, 4]);
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(plot.left, plotY(level));
      ctx.lineTo(plot.left + plot.width, plotY(level));
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // A single episode is a point. Drawing it as a path would put nothing
    // on the screen at all, which reads as a broken chart.
    if (drawn.length === 1) {
      ctx.fillStyle = colours.accent;
      ctx.beginPath();
      ctx.arc(plotX(0), plotY(drawn[0]), 2, 0, Math.PI * 2);
      ctx.fill();
      return;
    }

    // A chart may ask for the moving average alone. The dashboard shows the
    // raw return and the smoothed return as two separate charts, the way the
    // course material does, and drawing the noise again underneath the second
    // one would only make it harder to read.
    ctx.strokeStyle = colours.muted;
    ctx.globalAlpha = settings.smoothOnly ? 0 : C.charts.rawOpacity;
    ctx.lineWidth = 1;
    ctx.beginPath();
    drawn.forEach((value, index) => {
      const x = plotX(index);
      const y = plotY(value);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.globalAlpha = 1;
    ctx.strokeStyle = colours.accent;
    ctx.lineWidth = 1.4;
    ctx.beginPath();
    smoothed.forEach((value, index) => {
      const x = plotX(index);
      const y = plotY(value);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    drawAxes(ctx, plot, colours, {
      spanLow: span.low,
      spanHigh: span.high,
      log: settings.scale === 'log',
      firstLabel: labelFor(0),
      lastLabel: labelFor(drawn.length - 1),
      xLabel: settings.xLabel || 'episode',
    });
  }

  /**
   * The frame, the tick labels and the axis names.
   *
   * Charts here are small — 74 css px tall — so this is deliberately sparse:
   * a baseline, a left rule, three y ticks and the first and last x value.
   * Enough to read a magnitude and a range off, without turning a sparkline
   * into a full plot.
   *
   * A log chart's tick labels are un-logged before they are written, so the
   * numbers on the axis are the real quantity and never its logarithm.
   */
  function drawAxes(ctx, plot, colours, options) {
    const font = C.charts.axisFont + ' '
               + window.getComputedStyle(document.body).fontFamily;
    ctx.save();
    ctx.font = font;
    ctx.strokeStyle = colours.hairline;
    ctx.fillStyle = colours.hairline;
    ctx.lineWidth = 1;
    ctx.globalAlpha = 0.75;

    // The two rules.
    ctx.beginPath();
    ctx.moveTo(plot.left, plot.top);
    ctx.lineTo(plot.left, plot.top + plot.height);
    ctx.lineTo(plot.left + plot.width, plot.top + plot.height);
    ctx.stroke();

    const shown = value => {
      const real = options.log ? Math.pow(10, value) : value;
      const size = Math.abs(real);
      if (size !== 0 && (size < 0.01 || size >= 100000)) {
        return real.toExponential(0);
      }
      if (size >= 100) return real.toFixed(0);
      if (size >= 1) return real.toFixed(1);
      return real.toFixed(3);
    };

    // Y ticks, from the top of the band down.
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const ticks = Math.max(2, C.charts.ticksY);
    for (let index = 0; index < ticks; index += 1) {
      const share = index / (ticks - 1);
      const value = options.spanHigh - share * (options.spanHigh
                                                - options.spanLow);
      const y = plot.top + share * plot.height;
      ctx.fillText(shown(value), plot.left - 3, y);
    }

    // X: the first and last real value, and what the axis counts.
    ctx.textBaseline = 'top';
    ctx.textAlign = 'left';
    ctx.fillText(String(options.firstLabel), plot.left,
                 plot.top + plot.height + 3);
    ctx.textAlign = 'right';
    ctx.fillText(String(options.lastLabel), plot.left + plot.width,
                 plot.top + plot.height + 3);
    ctx.textAlign = 'center';
    ctx.globalAlpha = 0.55;
    ctx.fillText(options.xLabel, plot.left + plot.width / 2,
                 plot.top + plot.height + 3);
    ctx.restore();
  }

  /**
   * Several series on one pair of axes, each its own colour.
   *
   * Unlike `drawSeries` this draws the values as they are: no rolling
   * average, because these are already summary measurements taken every few
   * hundred episodes rather than a noisy per-episode series, and smoothing a
   * sixteen-point line would only blur the thing it is meant to show. The
   * band is shared across all the lines so they can be compared by height,
   * which is the entire point of putting them together.
   */
  function drawLines(canvas, series, colours) {
    const ratio = window.devicePixelRatio || 1;
    const cssWidth = canvas.clientWidth || 1;
    const cssHeight = canvas.clientHeight || C.charts.height;

    canvas.width = Math.round(cssWidth * ratio);
    canvas.height = Math.round(cssHeight * ratio);

    const ctx = canvas.getContext('2d');
    ctx.setTransform(ratio, 0, 0, ratio, 0, 0);
    ctx.clearRect(0, 0, cssWidth, cssHeight);

    const pad = C.charts.padding;
    const plot = {
      left: pad.left,
      top: pad.top,
      width: Math.max(1, cssWidth - pad.left - pad.right),
      height: Math.max(1, cssHeight - pad.top - pad.bottom),
    };

    const every = series.reduce(
      (all, line) => all.concat(line.values), []);
    if (!every.length) {
      ctx.fillStyle = colours.hairline;
      ctx.font = '10px ' + window.getComputedStyle(document.body).fontFamily;
      ctx.fillText('no measurements yet',
                   plot.left + 2, plot.top + plot.height / 2);
      return;
    }

    // Rates share a band of 0 to 1 so that "a fifth of the way up" means the
    // same on every line and between one repaint and the next.
    const span = extent(every.concat([0]));
    const count = Math.max(...series.map(line => line.values.length));

    function plotX(index) {
      if (count === 1) return plot.left + plot.width / 2;
      return plot.left + (index / (count - 1)) * plot.width;
    }
    function plotY(value) {
      const share = (value - span.low) / (span.high - span.low);
      return plot.top + plot.height - share * plot.height;
    }

    series.forEach(line => {
      if (!line.values.length) return;
      ctx.strokeStyle = line.colour;
      ctx.fillStyle = line.colour;
      ctx.lineWidth = 1.4;
      if (line.values.length === 1) {
        ctx.beginPath();
        ctx.arc(plotX(0), plotY(line.values[0]), 2, 0, Math.PI * 2);
        ctx.fill();
        return;
      }
      ctx.beginPath();
      line.values.forEach((value, index) => {
        const x = plotX(index);
        const y = plotY(value);
        if (index === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.stroke();
    });
  }

  /* ---------------------------------------------------------------------
     Building and refreshing the set
     ------------------------------------------------------------------- */

  /**
   * Build the four charts into `host` and return a repaint function.
   *
   * The returned function is what everything else calls; the DOM is built
   * once and only the canvases are redrawn afterwards.
   */
  function build(host, series) {
    host.innerHTML = '';
    const built = (series || LEARNER_SERIES).map(series => {
      const wrapper = document.createElement('div');
      wrapper.className = 'chart';

      const head = document.createElement('div');
      head.className = 'chart-head';
      const label = document.createElement('span');
      label.textContent = series.label;
      const latest = document.createElement('span');
      latest.className = 'chart-latest';
      head.appendChild(label);
      head.appendChild(latest);

      const canvas = document.createElement('canvas');
      canvas.className = 'chart-canvas';
      canvas.style.height = C.charts.height + 'px';

      wrapper.appendChild(head);
      wrapper.appendChild(canvas);

      // A multi-line chart needs saying which line is which, and the colours
      // are the only thing distinguishing them.
      let key = null;
      if (series.keys) {
        key = document.createElement('ul');
        key.className = 'chart-key';
        series.keys.forEach((name, index) => {
          const item = document.createElement('li');
          const swatch = document.createElement('span');
          swatch.className = 'chart-key-swatch';
          swatch.dataset.tone = MULTI_COLOURS[index % MULTI_COLOURS.length];
          item.appendChild(swatch);
          item.appendChild(document.createTextNode(name));
          key.appendChild(item);
        });
        wrapper.appendChild(key);
      }

      host.appendChild(wrapper);
      return { series: series, canvas: canvas, latest: latest, key: key };
    });

    /**
     * Redraw them all.
     *
     * `data` is either the metrics array — which is what rooms 1 to 4 pass and
     * what this always took — or an object carrying several named series.
     * Room 5 needs the second because not everything it graphs is a property
     * of a training episode: the train / validation / unseen-test comparison
     * is measured periodically with the weights frozen and lives in
     * `checkpoints`, on its own x-axis of episode numbers.
     *
     * `context` is passed to any series whose threshold moves — room 1's
     * stopping threshold is a parameter, so its line has to follow it.
     */
    return function repaint(data, context) {
      const colours = palette();
      const bundle = Array.isArray(data) ? { history: data } : (data || {});
      // `history` is every episode; `metrics` is the sampled few kept frame by
      // frame. A room that sends only one of them gets it used for both.
      const history = bundle.history || bundle.metrics || [];

      built.forEach(chart => {
        /* Where this chart's rows come from.

           `checkpoints` is room 5's frozen-policy evaluation, on its own
           x-axis. `sweeps` is room 1's planner curve: Value Iteration has no
           episodes, so its progress is one point per sweep of the state
           space and it cannot be read from the episode history at all. */
        const source = chart.series.source;
        const rows = source === 'checkpoints' ? (bundle.checkpoints || [])
                   : source === 'sweeps' ? (bundle.sweeps || [])
                   : history;

        if (chart.series.keys) {
          const lines = chart.series.keys.map((name, index) => ({
            colour: colours[MULTI_COLOURS[index % MULTI_COLOURS.length]],
            values: rows.map(entry => entry[name])
              .filter(value => typeof value === 'number' && isFinite(value)),
          }));
          drawLines(chart.canvas, lines, colours);
          const last = lines.map(line => line.values[line.values.length - 1]);
          chart.latest.textContent = last.every(
            value => value === undefined)
            ? ''
            : last.map(value => value === undefined ? '—'
                                                    : formatLatest(value))
                  .join(' / ');
          return;
        }

        const values = rows.map(entry => entry[chart.series.key])
          .filter(value => typeof value === 'number' && isFinite(value));
        /* The stopping line. A room declared in JSON cannot hand over a
           function, so it names a parameter instead and it is resolved here
           — which is how room 1's dashed theta line follows its slider. */
        let threshold = chart.series.threshold;
        if (typeof threshold === 'function') threshold = threshold(context);
        if (chart.series.thresholdKey && context && context.parameters) {
          const named = context.parameters[chart.series.thresholdKey];
          if (typeof named === 'number' && isFinite(named)) threshold = named;
        }

        drawSeries(chart.canvas, values, colours, {
          scale: chart.series.scale,
          threshold: threshold,
          smoothOnly: chart.series.smoothOnly,
          xLabel: chart.series.xLabel,
          // Real x values, so a bucketed axis still reads in episode numbers.
          xValues: rows.map(entry => entry[chart.series.x || 'episode']),
        });
        // The label always shows the real number, never the logarithm.
        chart.latest.textContent = values.length
          ? formatLatest(values[values.length - 1])
          : '';
      });
    };
  }

  function formatLatest(value) {
    const size = Math.abs(value);
    if (size !== 0 && size < 0.01) return value.toExponential(1);
    if (Number.isInteger(value)) return String(value);
    return value.toFixed(size < 1 ? 3 : 1);
  }

  return {
    build: build,
    LEARNER_SERIES: LEARNER_SERIES,
    // Exported because they are the parts worth checking directly.
    rollingAverage: rollingAverage,
    bucket: bucket,
  };
})();
