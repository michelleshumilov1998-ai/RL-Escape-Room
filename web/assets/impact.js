'use strict';

/* =====================================================================
   The one sound in the game: a bridge section failing.

   SYNTHESISED, NOT LOADED
   Everything here is made from oscillators and a noise buffer at runtime, so
   the project still ships no binary assets and no request is made for one. A
   heavy steel structure failing is mostly filtered noise with a couple of
   ringing metal partials over it and a long low tail, which is a thing
   WebAudio does well.

   Three layers, as briefed:

     break     the metal parting — a hard noise transient through a
               band-pass, plus two detuned ringing partials
     debris    plates hitting things on the way down — a handful of short
               clicks scattered over half a second, quieter as they fall
     echo      the shaft itself — the break again, delayed, low-passed and
               fed back on itself so it answers from a long way down

   WHY IT IS SAFE TO CALL WHENEVER
   Browsers refuse to start audio before the player has interacted with the
   page, and a refused start throws. So the context is created lazily on the
   first play, every call is wrapped, and a failure is silent — a missing
   sound effect must never break a replay. Nothing here blocks or waits.
   ===================================================================== */

window.Impact = (function () {

  /* Kept well below the top: this fires while the player is reading graphs,
     and a collapse should be a thump rather than an event. */
  const LEVEL = 0.22;

  let context = null;
  let noise = null;
  /* The last time a collapse was played, so the several places that can
     notice one cannot stack three copies of it on the same frame. */
  let lastAt = 0;

  function ready() {
    if (context) return context;
    const Ctor = window.AudioContext || window.webkitAudioContext;
    if (!Ctor) return null;
    context = new Ctor();
    // Two seconds of white noise, made once and reused by every layer.
    const frames = Math.floor(context.sampleRate * 2);
    noise = context.createBuffer(1, frames, context.sampleRate);
    const data = noise.getChannelData(0);
    for (let index = 0; index < frames; index += 1) {
      data[index] = Math.random() * 2 - 1;
    }
    return context;
  }

  /** One burst of the shared noise buffer, shaped and filtered. */
  function burst(at, duration, options) {
    const settings = options || {};
    const source = context.createBufferSource();
    source.buffer = noise;
    source.loop = true;

    const filter = context.createBiquadFilter();
    filter.type = settings.type || 'bandpass';
    filter.frequency.setValueAtTime(settings.from || 900, at);
    if (settings.to) {
      filter.frequency.exponentialRampToValueAtTime(
        settings.to, at + duration);
    }
    filter.Q.value = settings.q === undefined ? 1.1 : settings.q;

    const gain = context.createGain();
    const peak = (settings.level === undefined ? 1 : settings.level) * LEVEL;
    gain.gain.setValueAtTime(0.0001, at);
    gain.gain.exponentialRampToValueAtTime(peak, at + (settings.attack || 0.004));
    gain.gain.exponentialRampToValueAtTime(0.0001, at + duration);

    source.connect(filter);
    filter.connect(gain);
    gain.connect(settings.destination || context.destination);
    source.start(at);
    source.stop(at + duration + 0.02);
  }

  /** A ringing metal partial: a struck bar, decaying. */
  function ring(at, frequency, duration, level) {
    const osc = context.createOscillator();
    osc.type = 'triangle';
    osc.frequency.setValueAtTime(frequency, at);
    // Metal drops slightly in pitch as it stops ringing.
    osc.frequency.exponentialRampToValueAtTime(frequency * 0.86, at + duration);

    const gain = context.createGain();
    gain.gain.setValueAtTime(0.0001, at);
    gain.gain.exponentialRampToValueAtTime(level * LEVEL, at + 0.006);
    gain.gain.exponentialRampToValueAtTime(0.0001, at + duration);

    osc.connect(gain);
    gain.connect(context.destination);
    osc.start(at);
    osc.stop(at + duration + 0.02);
  }

  return {
    /**
     * A bridge section giving way.
     *
     * Idempotent within a tenth of a second, because the shake, the frame
     * loop and the live view can all notice the same collapse.
     */
    collapse: function () {
      try {
        const audio = ready();
        if (!audio) return;
        // Suspended until the page has been interacted with; resuming is a
        // promise that may reject, and that is fine.
        if (audio.state === 'suspended' && audio.resume) {
          audio.resume().catch(function () {});
        }
        const now = audio.currentTime;
        if (now - lastAt < 0.1) return;
        lastAt = now;

        // 1. The break: a hard, bright transient sweeping downwards.
        burst(now, 0.28, { from: 2600, to: 420, q: 0.8, level: 1.0 });
        // Two ringing partials, detuned, so it is steel and not a drum.
        ring(now + 0.005, 430, 0.5, 0.5);
        ring(now + 0.012, 611, 0.34, 0.32);

        // 2. Debris: five short clicks over half a second, getting quieter
        //    and duller as the pieces fall away from the listener.
        for (let index = 0; index < 5; index += 1) {
          const when = now + 0.09 + index * 0.075 + Math.random() * 0.04;
          const fade = 1 - index / 6;
          burst(when, 0.05, { from: 1500 * fade + 300, q: 2.2,
                              level: 0.32 * fade });
        }

        // 3. The shaft answering: the break again, late, dark, and fed back
        //    on itself a couple of times so the room sounds deep.
        const delay = audio.createDelay(1.0);
        delay.delayTime.value = 0.19;
        const feedback = audio.createGain();
        feedback.gain.value = 0.42;
        const tone = audio.createBiquadFilter();
        tone.type = 'lowpass';
        tone.frequency.value = 700;
        const level = audio.createGain();
        level.gain.value = 0.5;

        delay.connect(feedback);
        feedback.connect(delay);          // the repeats
        delay.connect(tone);
        tone.connect(level);
        level.connect(audio.destination);
        burst(now + 0.02, 0.3, { from: 1400, to: 260, q: 0.7, level: 0.85,
                                 destination: delay });

        // The echo network is torn down once it has rung out, so a long run
        // does not accumulate a node per collapse.
        window.setTimeout(function () {
          try {
            level.disconnect();
            tone.disconnect();
            feedback.disconnect();
            delay.disconnect();
          } catch (problem) { /* already gone */ }
        }, 2500);
      } catch (problem) {
        // No sound. Never a broken replay.
      }
    },

    /** Whether audio is available at all, for anything that wants to know. */
    get available() {
      return Boolean(window.AudioContext || window.webkitAudioContext);
    },
  };
})();
