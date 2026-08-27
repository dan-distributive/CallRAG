// 16kHz mono Float32Array -> 512-dim speaker voice-fingerprint embedding,
// via Xenova/wavlm-base-plus-sv (WavLMForXVector). This is the piece
// diarize.js explicitly doesn't do: diarize.js's pyannote-segmentation
// model answers "when did the speaker change" with local per-call slot
// IDs; this model answers "whose voice is this", producing a vector that
// can be compared (cosine similarity) against OTHER clips -- including
// ones from a completely different call -- to recognize the same person.
//
// Sanity-checked locally: two different 3s windows of the same real
// speaker produced a 0.73 cosine similarity -- a solid same-speaker
// signal at the same normalization this module uses.
require('./polyfills');
const installModelFetchPatch = require('./modelFetchPatch');
const { loadWithGpuFallback } = require('./gpuFallback');

let modelPromise = null;
/**
 * @param {Record<string, string>} modelFiles { filename: base64 } map for Xenova/wavlm-base-plus-sv
 */
function getModel(modelFiles) {
  if (!modelPromise) {
    installModelFetchPatch(modelFiles);
    const { AutoModel, AutoProcessor, env } = require('@huggingface/transformers');
    require('./setupOrt')(env);
    env.useBrowserCache = false;
    env.allowLocalModels = false;
    modelPromise = loadWithGpuFallback(
      (device) =>
        Promise.all([
          AutoModel.from_pretrained('Xenova/wavlm-base-plus-sv', {
            device,
            dtype: 'q8',
            ...(device === 'wasm' ? { session_options: { graphOptimizationLevel: 'disabled' } } : {}), // same qdq_actions.cc fix as every other quantized model in this stack, wasm-only
          }),
          AutoProcessor.from_pretrained('Xenova/wavlm-base-plus-sv'),
        ]),
      'wavlm',
    );
  }
  return modelPromise;
}

function normalize(vec) {
  const norm = Math.sqrt(vec.reduce((s, x) => s + x * x, 0));
  return norm > 0 ? vec.map((x) => x / norm) : vec;
}

const MIN_SAMPLES = 16000; // 1s -- same class of bug as diarize.js's tiny-trailing-chunk crash; untested below this, don't risk it

/**
 * @param {Float32Array} samples 16kHz mono PCM -- a single speaker turn's audio, not a whole recording
 * @param {Record<string, string>} modelFiles { filename: base64 } map for Xenova/wavlm-base-plus-sv
 * @returns {Promise<number[]|null>} 512-dim normalized embedding, or null if the clip is too short to embed reliably
 */
async function speakerEmbed(samples, modelFiles) {
  if (samples.length < MIN_SAMPLES) return null;
  const [model, processor] = await getModel(modelFiles);
  const inputs = await processor(samples);
  const { embeddings } = await model(inputs);
  return normalize(Array.from(embeddings.data));
}

function cosineSim(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

/**
 * @param {number[]} embedding
 * @param {Record<string, {embedding: number[], count: number}>} voiceProfiles name -> known profile
 * @param {number} [threshold=0.6]
 * @returns {{name: string, score: number}|null}
 */
function matchVoice(embedding, voiceProfiles, threshold = 0.6) {
  let best = null;
  for (const [name, profile] of Object.entries(voiceProfiles)) {
    const score = cosineSim(embedding, profile.embedding);
    if (score >= threshold && (!best || score > best.score)) best = { name, score };
  }
  return best;
}

module.exports = speakerEmbed;
module.exports.cosineSim = cosineSim;
module.exports.matchVoice = matchVoice;
