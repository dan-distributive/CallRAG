// 16kHz mono Float32Array -> speaker turns, via onnx-community/pyannote-segmentation-3.0
// (PyAnnoteForAudioFrameClassification). This is the segmentation stage only
// -- it labels "speaker slot" changes within a single continuous forward
// pass over the given audio, but there is no separate speaker-embedding +
// clustering stage here (that's the rest of pyannote's full pipeline, which
// isn't a transformers.js-native model). That means: turn boundaries and
// same-speaker-vs-different-speaker WITHIN one call are real, but a given
// speaker id is only reliably the same physical person within the single
// continuous audio buffer given to one _call -- not verified to stay
// consistent from one end of a very long recording to the other, and never
// consistent across different calls/files (no enrollment/identity matching
// exists). Good enough for "who's turn is this" within a call; not (yet)
// "this is the same Alice across all 50 calls".
require('./polyfills');
const installModelFetchPatch = require('./modelFetchPatch');

let modelPromise = null;
/**
 * @param {Record<string, string>} modelFiles { filename: base64 } map for onnx-community/pyannote-segmentation-3.0
 */
function getModel(modelFiles) {
  if (!modelPromise) {
    installModelFetchPatch(modelFiles);
    const { AutoModelForAudioFrameClassification, AutoProcessor, env } = require('@huggingface/transformers');
    require('./setupOrt')(env);
    env.useBrowserCache = false;
    env.allowLocalModels = false;
    modelPromise = Promise.all([
      AutoModelForAudioFrameClassification.from_pretrained('onnx-community/pyannote-segmentation-3.0', {
        device: 'wasm',
        dtype: 'fp32', // tiny model (~6MB) -- no need to fight the QDQ/quantization crash at all
        session_options: { graphOptimizationLevel: 'disabled' }, // same fix as whisper/bge-small, cheap insurance
      }),
      AutoProcessor.from_pretrained('onnx-community/pyannote-segmentation-3.0'),
    ]);
  }
  return modelPromise;
}

const SAMPLE_RATE = 16000;
const WINDOW_S = 30; // matches Whisper's own chunk_length_s -- no particular need to match, just consistent with the rest of the pipeline

/**
 * @param {Float32Array} samples 16kHz mono PCM
 * @param {Record<string, string>} modelFiles { filename: base64 } map for onnx-community/pyannote-segmentation-3.0
 * @param {() => void} [onChunk] called after each processed window -- see
 *   transcribeAudio.js's onToken for why this matters (DCP's ENOPROGRESS
 *   timeout). Feeding an entire long recording through this model in one
 *   forward pass, with no progress calls during it, both looks "frozen" to
 *   an observer and risks the scheduler actually killing the job.
 * @returns {Promise<Array<{ id: number, start: number, end: number, confidence: number }>>} speaker turns, in seconds, on the original timeline
 */
async function diarize(samples, modelFiles, onChunk) {
  const [model, processor] = await getModel(modelFiles);
  const windowSamples = SAMPLE_RATE * WINDOW_S;
  const MIN_CHUNK_SAMPLES = SAMPLE_RATE; // 1s -- pyannote's SincNet conv stack can't handle a very short trailing remainder (confirmed: a 478-sample leftover chunk crashed with "Invalid input shape: {1}")

  const allTurns = [];
  let offset = 0;
  while (offset < samples.length) {
    let end = Math.min(offset + windowSamples, samples.length);
    // absorb a too-short trailing remainder into this (final) chunk instead
    // of processing it separately
    if (samples.length - end < MIN_CHUNK_SAMPLES) end = samples.length;
    const chunk = samples.subarray(offset, end);
    const inputs = await processor(chunk);
    const { logits } = await model(inputs);
    const [turns] = processor.post_process_speaker_diarization(logits, chunk.length);
    const offsetSeconds = offset / SAMPLE_RATE;
    for (const t of turns) allTurns.push({ ...t, start: t.start + offsetSeconds, end: t.end + offsetSeconds });
    if (onChunk) onChunk();
    offset = end;
  }
  return allTurns;
}

/**
 * Labels each transcript segment with the diarization turn whose time range
 * covers its midpoint. `turns` and `segments` must be on the same timeline
 * (both original-recording time, not VAD-trimmed time) -- diarize() should
 * always be run on the full untrimmed audio for exactly this reason.
 *
 * If `turns` came from ingest.js's voice-fingerprint step (each turn
 * carrying `voiceMatch`/`voiceEmbedding` from speakerEmbed.js), those are
 * copied through too -- `speaker` is the local per-call slot id (see the
 * module doc above), `voiceMatch` is a cross-call name match if one was
 * found, and `voiceEmbedding` lets the viewer enroll/update a voice
 * profile later even for segments that didn't match anything yet.
 * @param {Array<{start: number, end: number}>} segments
 * @param {Array<{ id: number, start: number, end: number, voiceMatch?: {name: string, score: number}|null, voiceEmbedding?: number[]|null }>} turns
 * @returns {Array} segments with `speaker`/`voiceMatch`/`voiceEmbedding` fields added (null if no turn covers it)
 */
function assignSpeakers(segments, turns) {
  return segments.map((seg) => {
    const mid = (seg.start + seg.end) / 2;
    const turn = turns.find((t) => mid >= t.start && mid < t.end);
    return {
      ...seg,
      speaker: turn ? turn.id : null,
      voiceMatch: turn?.voiceMatch ?? null,
      voiceEmbedding: turn?.voiceEmbedding ?? null,
    };
  });
}

module.exports = diarize;
module.exports.assignSpeakers = assignSpeakers;
