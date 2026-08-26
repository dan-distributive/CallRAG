// Stage 3 (WebGPU): investigated and shelved for now. device: 'webgpu'
// reaches real GPU adapter/device setup (navigator.gpu.requestAdapter()
// succeeds) but crashes inside onnxruntime-web 1.26.0-dev's JSEP session
// creation with "TypeError: Cannot convert undefined to a BigInt" -- and
// critically, this escapes normal promise rejection entirely (uncaught at
// the job level, not reachable via try/catch around the pipeline() call),
// so it's not a safe "attempt and fall back" operation as-is. Possibly
// specific to this onnxruntime-web dev build's WebGPU path interacting with
// this sandbox's Dawn implementation -- onnx-inference-dcp's working example
// uses the DCP-team's own dcp-ort.js/dcp-wasm.js build (a different pinned
// version, 1.23.0) via the package manager, not raw npm onnxruntime-web;
// that may have already avoided this. Revisit once dcp-wasm.js/dcp-ort.js
// content is available as local files, or a different onnxruntime-web
// version is tried.
//
// device: 'wasm' (CPU) is what actually ships for now -- proven reliable
// (Stage 2).
require('./polyfills');
require('./modelFetchPatch')();
const { pipeline, env } = require('@huggingface/transformers');

// Must come AFTER importing transformers.js -- configures wasmBinary on ITS
// OWN exported env.backends.onnx, not a separately-required onnxruntime-web
// instance (see setupOrt.js for why that silently didn't work).
require('./setupOrt')(env);

// The Cache Storage API (`caches`) isn't available in DCP's worker sandbox --
// transformers.js otherwise tries to use it by default for model caching.
env.useBrowserCache = false;
env.allowLocalModels = false;

async function testPipeline() {
  const extractor = await pipeline('feature-extraction', 'Xenova/bge-small-en-v1.5', {
    device: 'wasm',
    dtype: 'q8', // quantized, much smaller than the default fp32 (129MB -> ~34MB)
  });
  const output = await extractor('hello world', { pooling: 'mean', normalize: true });
  return {
    device: 'wasm',
    dims: output.dims,
    dataLength: output.data.length,
    first5: Array.from(output.data.slice(0, 5)),
  };
}

module.exports = testPipeline;
