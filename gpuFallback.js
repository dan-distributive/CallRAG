// Try device:'webgpu' first, fall back to device:'wasm' on any failure --
// used by every model-loading module in this project (transcribeAudio.js,
// embedText.js, diarize.js, speakerEmbed.js) so the fallback logic exists
// in exactly one place. `loadFn` is caller-defined and just takes a device
// string, so this works whether the caller uses transformers.js's
// pipeline() (one call) or AutoModel.from_pretrained() + AutoProcessor
// separately (two calls) -- this module doesn't need to know which.
//
// Always attempts webgpu first, unconditionally -- no navigator.gpu
// precheck. That check used to gate the attempt, but DCP's sandbox only
// ever exposes navigator.gpu when the job itself set
// job.requirements.environment.webgpu = true at dispatch time (see
// access-lists.js in dcp-client) -- so the precheck wasn't detecting
// per-worker GPU capability, it was just mirroring whatever ingest.js
// requested. Trying unconditionally and catching the failure gets the same
// result on a CPU-only/no-webgpu-requirement worker (an immediate throw,
// caught, falls to wasm) without needing that flag to agree with reality,
// which is what lets ingest.js toggle
// job.requirements.environment.webgpu on/off per-dispatch (GPU compute
// group vs CPU compute group) and have this module do the right thing
// either way.
async function loadWithGpuFallback(loadFn, label) {
  try {
    const result = await loadFn('webgpu');
    console.log(`[gpuFallback] ${label}: using webgpu`);
    return result;
  } catch (err) {
    console.warn(`[gpuFallback] ${label}: webgpu failed (${err.message}), falling back to wasm`);
  }
  const result = await loadFn('wasm');
  console.log(`[gpuFallback] ${label}: using wasm`);
  return result;
}

module.exports = { loadWithGpuFallback };
