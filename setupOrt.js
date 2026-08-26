// Configures onnxruntime-web's wasm backend to load entirely from an
// embedded binary -- no fetch() anywhere (DCP's sandbox blocks fetch()
// outright for disallowed origins -- see ~/DCP/wasm-package-howto.md).
//
// IMPORTANT: takes the already-imported `env` from @huggingface/transformers
// itself (env.backends.onnx), NOT a separate `require('onnxruntime-web/webgpu')`
// -- a separate import risks resolving to a DIFFERENT module instance (DCP's
// bundler may resolve transformers.js's internal ESM `import` and our own
// CommonJS `require()` of the same subpath to different physical files,
// .mjs vs .js, each with its own separate `env` object), silently making our
// wasmBinary override invisible to the actual code path transformers.js
// uses. Confirmed empirically: setting wasmBinary via a separately-required
// onnxruntime-web/webgpu had no effect on a real DCP worker (it kept trying
// to fetch from jsdelivr.net and failed). Going through transformers.js's
// own exported `env` guarantees we're touching the same object.
// Reverted from the JSEP variant back to asyncify: using ort-wasm-simd-
// threaded.jsep.wasm for the plain 'wasm' (CPU) device -- not just
// 'webgpu' -- caused the SAME "Cannot convert undefined to a BigInt" crash
// we saw with webgpu, reproduced twice in a row. The jsep.wasm binary
// apparently does some GPU-capability check internally regardless of which
// execution provider is actually requested at the ort.InferenceSession
// level -- it's not the drop-in CPU+GPU superset it looked like on paper.
// asyncify is what Stage 2's real, repeated successes actually used.
const wasmBase64 = require('./ortWasmAsyncifyBase64');
const { base64ToBytes } = require('./base64');

let wasmBytes = null;
function getWasmBytes() {
  if (!wasmBytes) wasmBytes = base64ToBytes(wasmBase64);
  return wasmBytes;
}

/** @param {import('@huggingface/transformers').env} env */
function setupOrt(env) {
  const onnx = env.backends.onnx;
  onnx.wasm.wasmBinary = getWasmBytes();
  onnx.wasm.numThreads = 1;
  onnx.wasm.proxy = false;
}

module.exports = setupOrt;
