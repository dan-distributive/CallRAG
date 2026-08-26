// mp3 bytes -> PCM, pure JS/wasm, no ffmpeg, no fetch() anywhere.
//
// Uses minimp3-wasm's raw `Decoder` class directly instead of its
// `createDecoder()` convenience wrapper -- that wrapper does
// `fetch(wasmUrl)` internally, and DCP's worker sandbox blocks fetch()
// outright, even for data: URIs (see ~/DCP/wasm-package-howto.md). Decoder
// itself takes raw `WebAssembly.instantiate(bytes, {}).instance.exports`
// directly, no abstraction layer, no imports object needed -- this is the
// simplest possible case of the docs' proven "supply the module yourself,
// skip any default loader" pattern.
//
// Validated (2026-08-25, transformers.js-dcp prototype): 1.0000 cosine
// similarity vs ffmpeg's own decode, after trimming minimp3's fixed
// ~529-sample decoder-delay offset (a normal characteristic of mp3
// decoding -- different decoders have different priming latency, not a bug).
const { Decoder } = require('minimp3-wasm');
const wasmBase64 = require('./decoderWasmBase64');
const { base64ToBytes } = require('./base64');

const DECODER_DELAY_SAMPLES = 529;

let wasmInstancePromise = null;
function ensureWasmInstance() {
  if (!wasmInstancePromise) {
    wasmInstancePromise = WebAssembly.instantiate(base64ToBytes(wasmBase64), {});
  }
  return wasmInstancePromise;
}

/**
 * @param {Uint8Array} mp3Bytes
 * @returns {Promise<{ mono: Float32Array, sampleRate: number }>}
 */
async function decodeMp3(mp3Bytes) {
  const { instance } = await ensureWasmInstance();
  const decoder = new Decoder(instance.exports, mp3Bytes);
  const result = decoder.decode(decoder.duration);

  const samplesPerChannel = result.numSamples / result.numChannels;
  const mono = new Float32Array(Math.max(0, samplesPerChannel - DECODER_DELAY_SAMPLES));
  for (let i = 0; i < mono.length; i++) {
    const srcIdx = i + DECODER_DELAY_SAMPLES;
    let sum = 0;
    for (let c = 0; c < result.numChannels; c++) sum += result.pcm[srcIdx * result.numChannels + c];
    mono[i] = sum / result.numChannels / 32768;
  }

  return { mono, sampleRate: result.samplingRate };
}

module.exports = decodeMp3;
