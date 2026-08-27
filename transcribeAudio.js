// PCM samples (16kHz mono Float32Array) -> Whisper transcript segments.
// Self-contained except for the model files themselves, which are passed in
// rather than imported -- keeps this module decoupled from any one model's
// embedded data, and lets the caller choose how the bytes get to the worker
// (job.requires()-bundled local file, job argument, published package,
// whatever fits).
require('./polyfills'); // AbortController -- see polyfills.js
const installModelFetchPatch = require('./modelFetchPatch');
const { loadWithGpuFallback } = require('./gpuFallback');

let pipelinePromise = null;
/**
 * @param {Record<string, string>} modelFiles { filename: base64 } map
 * @param {string} dtype passed straight through to transformers.js's pipeline() dtype option
 * @param {string} modelName HF repo id, e.g. 'Xenova/whisper-tiny.en' or 'Xenova/whisper-base.en'
 */
function getTranscriber(modelFiles, dtype, modelName) {
  if (!pipelinePromise) {
    installModelFetchPatch(modelFiles);
    const { pipeline, env } = require('@huggingface/transformers');
    require('./setupOrt')(env); // must come after importing transformers.js -- see setupOrt.js
    env.useBrowserCache = false;
    env.allowLocalModels = false;
    pipelinePromise = loadWithGpuFallback(
      (device) =>
        pipeline('automatic-speech-recognition', modelName, {
          device, // 'wasm' on the real (browser-hosted) DCP worker, not 'cpu' (that's the local-Node-only value) -- see dcp_transformers_restart memory
          dtype,
          // qdq_actions.cc:137 TransposeDQWeightsForMatMulNBits crashes
          // session creation for ANY QDQ-format graph on this sandbox's
          // onnxruntime-web wasm backend specifically (hit identically at
          // q8, uint8, and fp32 -- whisper's merged decoder always has at
          // least one DequantizeLinear node, used for weight-sharing
          // between its with/without-past branches even at fp32). A
          // graph-optimization fusion-pass bug, not a hard runtime
          // limitation -- disabling extended optimizations skips it.
          // Confirmed NOT needed on webgpu (a real dispatch succeeded with
          // no override at all), so only apply it for the wasm fallback.
          ...(device === 'wasm' ? { session_options: { graphOptimizationLevel: 'disabled' } } : {}),
        }),
      'whisper',
    );
  }
  return pipelinePromise;
}

/**
 * @param {Float32Array} samples 16kHz mono PCM
 * @param {Record<string, string>} modelFiles { filename: base64 } map matching modelName
 * @param {string} [dtype='fp32']
 * @param {string} [modelName='Xenova/whisper-tiny.en']
 * @param {() => void} [onToken] called on every generated token, across every
 *   chunk -- DCP's own scheduler kills a job with ENOPROGRESS if progress()
 *   isn't called at least every ~30s (see dcp-client's own remote-data
 *   example comment), and a long recording's Whisper decode can easily run
 *   that long between our own coarse per-stage progress() calls otherwise.
 * @returns {Promise<Array<{start: number, end: number, text: string}>>}
 */
async function transcribeAudio(samples, modelFiles, dtype = 'fp32', modelName = 'Xenova/whisper-tiny.en', onToken) {
  const transcriber = await getTranscriber(modelFiles, dtype, modelName);
  const { BaseStreamer } = require('@huggingface/transformers');
  class ProgressStreamer extends BaseStreamer {
    put() {
      if (onToken) onToken();
    }
    end() {}
  }
  const output = await transcriber(samples, {
    return_timestamps: true,
    chunk_length_s: 30,
    stride_length_s: 5,
    streamer: onToken ? new ProgressStreamer() : undefined,
  });
  const chunks = Array.isArray(output) ? output[0].chunks : output.chunks;
  return chunks.map((c) => ({
    start: c.timestamp[0],
    end: c.timestamp[1] ?? samples.length / 16000,
    text: c.text.trim(),
  }));
}

module.exports = transcribeAudio;
