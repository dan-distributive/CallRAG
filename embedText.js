// Text -> normalized embedding vector, via Xenova/bge-small-en-v1.5.
// Self-contained except for the model files themselves, which are passed in
// rather than imported -- same pattern as transcribeAudio.js, so either
// module can ship via job.requires(), a job argument, or a published
// package without the other knowing.
require('./polyfills'); // AbortController -- see polyfills.js
const installModelFetchPatch = require('./modelFetchPatch');
const { loadWithGpuFallback } = require('./gpuFallback');

let pipelinePromise = null;
/**
 * @param {Record<string, string>} modelFiles { filename: base64 } map
 */
function getEmbedder(modelFiles) {
  if (!pipelinePromise) {
    installModelFetchPatch(modelFiles);
    const { pipeline, env } = require('@huggingface/transformers');
    require('./setupOrt')(env); // must come after importing transformers.js -- see setupOrt.js
    env.useBrowserCache = false;
    env.allowLocalModels = false;
    pipelinePromise = loadWithGpuFallback(
      (device) =>
        pipeline('feature-extraction', 'Xenova/bge-small-en-v1.5', {
          device, // 'wasm' on the real (browser-hosted) DCP worker, not 'cpu' -- see dcp_transformers_restart memory
          dtype: 'q8',
          // Same qdq_actions.cc/TransposeDQWeightsForMatMulNBits crash as
          // whisper hits, wasm-backend-specific -- see transcribeAudio.js.
          ...(device === 'wasm' ? { session_options: { graphOptimizationLevel: 'disabled' } } : {}),
        }),
      'bge-small',
    );
  }
  return pipelinePromise;
}

// bge-small-en-v1.5's tokenizer_config.json has no model_max_length, so the
// feature-extraction pipeline's internal `truncation: true` is a silent
// no-op (transformers.js's tokenizer only actually truncates when an
// explicit max_length is known -- see tokenization_utils.js). Without this,
// an overlong segment overflows the model's 512-token position embeddings
// and crashes ONNX Runtime ("Attempting to broadcast an axis by a dimension
// other than 1"). Confirmed in production: a Whisper repetition-loop
// hallucination on a real 53-minute call recording produced a 555-token
// segment that took down the entire embedding job for that file -- not just
// that one segment, ALL of the file's embeddings were lost since the job
// only had that one un-caught exception to report. Truncating defensively
// by character count (not exact, but segments are normally short spoken
// utterances -- this only ever fires on degenerate/hallucinated text).
const MAX_CHARS = 1000; // conservatively under 512 tokens even for dense repeated short tokens

/**
 * @param {string} text
 * @param {Record<string, string>} modelFiles { filename: base64 } map for Xenova/bge-small-en-v1.5 (q8)
 * @returns {Promise<number[]>} 384-dim normalized embedding
 */
async function embedText(text, modelFiles) {
  const embedder = await getEmbedder(modelFiles);
  const output = await embedder(text.slice(0, MAX_CHARS), { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

module.exports = embedText;
