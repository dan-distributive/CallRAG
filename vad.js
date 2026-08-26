// Lightweight pure-JS energy-based voice activity detection. No model, no
// wasm -- just frame-energy thresholding with hangover padding. Used to
// trim silence before transcription: feeding a Whisper model long silent
// stretches causes it to hallucinate repeated phrases (a well-known failure
// mode), so trimming first both fixes that and reduces compute.
const FRAME_MS = 30;
const HANGOVER_MS = 250; // pad before/after each speech region
const MIN_SILENCE_MS = 300; // bridge gaps shorter than this
const MIN_SPEECH_MS = 200; // drop blips shorter than this

/**
 * @param {Float32Array} samples
 * @param {number} sr
 * @returns {Array<{start: number, end: number}>} speech regions, in seconds
 */
function detectSpeechRegions(samples, sr) {
  const frameLen = Math.floor((FRAME_MS / 1000) * sr);
  const nFrames = Math.floor(samples.length / frameLen);
  const energies = new Float64Array(nFrames);
  for (let i = 0; i < nFrames; i++) {
    let sum = 0;
    const off = i * frameLen;
    for (let j = 0; j < frameLen; j++) sum += samples[off + j] ** 2;
    energies[i] = Math.sqrt(sum / frameLen);
  }
  const sorted = Float64Array.from(energies).sort();
  const noiseFloor = sorted[Math.floor(sorted.length * 0.15)]; // robust low-percentile estimate
  const peak = sorted[sorted.length - 1];
  const threshold = noiseFloor + (peak - noiseFloor) * 0.08;

  const voiced = Array.from(energies, (e) => e > threshold);

  const regions = [];
  let start = null;
  const minSilenceFrames = MIN_SILENCE_MS / FRAME_MS;
  let silenceRun = 0;
  for (let i = 0; i < nFrames; i++) {
    if (voiced[i]) {
      if (start === null) start = i;
      silenceRun = 0;
    } else if (start !== null) {
      silenceRun++;
      if (silenceRun >= minSilenceFrames) {
        regions.push([start, i - silenceRun]);
        start = null;
        silenceRun = 0;
      }
    }
  }
  if (start !== null) regions.push([start, nFrames - silenceRun]);

  const hangoverFrames = HANGOVER_MS / FRAME_MS;
  const minSpeechFrames = MIN_SPEECH_MS / FRAME_MS;
  return regions
    .filter(([s, e]) => e - s >= minSpeechFrames)
    .map(([s, e]) => ({
      start: Math.max(0, (s - hangoverFrames) * FRAME_MS) / 1000,
      end: (Math.min(nFrames, e + hangoverFrames) * FRAME_MS) / 1000,
    }));
}

/**
 * Concatenates only the speech regions of `samples` into one Float32Array,
 * and returns a mapping to translate timestamps in the trimmed audio back
 * to timestamps in the original.
 * @param {Float32Array} samples
 * @param {number} sr
 * @param {Array<{start: number, end: number}>} regions
 */
function trimToSpeech(samples, sr, regions) {
  const totalLen = regions.reduce((s, r) => s + Math.round((r.end - r.start) * sr), 0);
  const trimmed = new Float32Array(totalLen);
  const mapping = []; // {trimmedStart, trimmedEnd, originalStart}
  let cursor = 0;
  for (const r of regions) {
    const s0 = Math.round(r.start * sr);
    const s1 = Math.round(r.end * sr);
    const len = s1 - s0;
    trimmed.set(samples.subarray(s0, s1), cursor);
    mapping.push({ trimmedStart: cursor / sr, trimmedEnd: (cursor + len) / sr, originalStart: r.start });
    cursor += len;
  }
  return { trimmed, mapping };
}

/** Maps a timestamp in trimmed-audio time back to original-audio time. */
function remapTime(t, mapping) {
  for (const m of mapping) {
    if (t >= m.trimmedStart && t <= m.trimmedEnd) return m.originalStart + (t - m.trimmedStart);
  }
  const last = mapping[mapping.length - 1];
  return last ? last.originalStart + (t - last.trimmedStart) : t;
}

module.exports = { detectSpeechRegions, trimToSpeech, remapTime };
