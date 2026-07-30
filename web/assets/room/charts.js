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

  function palette() {
    const computed = window.getComputedStyle(document.documentElement);
    return {
      accent: computed.getPropertyValue(C.colors.accent).trim(),
      muted: computed.getPropertyValue(C.colors.muted).trim(),
      hairline: computed.getPropertyValue(C.colors.hairlineFaint).trim(),
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

    const drawn = bucket(values, C.charts.maxPointsDrawn);
    const smoothed = rollingAverage(
      drawn, Math.max(2, Math.round(C.charts.smoothingWindow
                                    * (drawn.length / values.length) || 1)));
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

    ctx.strokeStyle = colours.muted;
    ctx.globalAlpha = C.charts.rawOpacity;
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
      host.appendChild(wrapper);

      return { series: series, canvas: canvas, latest: latest };
    });

    /**
     * Redraw them all.
     *
     * `context` is passed to any series whose threshold moves — room 1's
     * stopping threshold is a parameter, so its line has to follow it.
     */
    return function repaint(metrics, context) {
      const colours = palette();
      built.forEach(chart => {
        const values = (metrics || []).map(entry => entry[chart.series.key])
          .filter(value => typeof value === 'number' && isFinite(value));
        const threshold = typeof chart.series.threshold === 'function'
          ? chart.series.threshold(context)
          : chart.series.threshold;

        drawSeries(chart.canvas, values, colours, {
          scale: chart.series.scale,
          threshold: threshold,
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
