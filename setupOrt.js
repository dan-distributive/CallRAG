// Configures onnxruntime-web's wasm backend to load entirely from embedded
// binaries -- no fetch() anywhere (DCP's sandbox blocks fetch() outright
// for disallowed origins) and no dynamic import() of a relative specifier
// either (DCP evaluates job code with `about:blank` as its base URL, so a
// relative import() can never resolve -- see docs.dcp.dev's "Getting
// WebGPU-accelerated libraries working in DCP work functions"). The second
// problem only bites the webgpu backend (it does a real dynamic import()
// of a companion .mjs during init; the plain wasm backend never does),
// but setting wasmPaths.mjs is harmless when device:'wasm' ends up being
// used instead, so this same setup covers both without needing to know in
// advance which one will actually run.
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
// asyncify is what Stage 2's real, repeated successes actually used, and
// (once the wasmPaths.mjs fix below is applied) what the webgpu path
// actually needs internally too.
const wasmBase64 = require('./ortWasmAsyncifyBase64');
const mjsBase64 = require('./ortWasmAsyncifyMjsBase64');
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
  onnx.wasm.wasmPaths = { mjs: 'data:text/javascript;base64,' + mjsBase64 };
  onnx.wasm.numThreads = 1;
  onnx.wasm.proxy = false;
}

module.exports = setupOrt;
