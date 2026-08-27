// Generates the large embedded-base64 model files this project ships to
// DCP workers as job arguments/job.requires() modules. Not checked into
// git (see .gitignore) -- these are large (7.6MB-100MB), fully
// re-derivable from public model repos, and re-committing them on every
// change would bloat the repo for no reason.
//
// Run once after `npm install`: `node scripts/prepare-models.js`
const fs = require('fs');
const path = require('path');
const ROOT = path.join(__dirname, '..');

async function writeBundle(outFile, files) {
  const bundle = {};
  for (const [name, filePath] of Object.entries(files)) {
    bundle[name] = fs.readFileSync(filePath).toString('base64');
  }
  fs.writeFileSync(path.join(ROOT, outFile), `module.exports = ${JSON.stringify(bundle)};\n`);
  const size = fs.statSync(path.join(ROOT, outFile)).size;
  console.log(`wrote ${outFile} (${(size / 1024 / 1024).toFixed(1)}MB)`);
}

async function main() {
  const { pipeline, AutoModel, AutoModelForAudioFrameClassification, AutoProcessor } = require('@huggingface/transformers');

  console.log('downloading whisper-base.en (uint8)...');
  await pipeline('automatic-speech-recognition', 'Xenova/whisper-base.en', { dtype: 'uint8' });
  const whisperBase = path.join(ROOT, 'node_modules/@huggingface/transformers/.cache/Xenova/whisper-base.en');
  await writeBundle('whisperBaseEnModelFilesBase64.js', {
    'config.json': `${whisperBase}/config.json`,
    'tokenizer_config.json': `${whisperBase}/tokenizer_config.json`,
    'tokenizer.json': `${whisperBase}/tokenizer.json`,
    'generation_config.json': `${whisperBase}/generation_config.json`,
    'preprocessor_config.json': `${whisperBase}/preprocessor_config.json`,
    'onnx/encoder_model_uint8.onnx': `${whisperBase}/onnx/encoder_model_uint8.onnx`,
    'onnx/decoder_model_merged_uint8.onnx': `${whisperBase}/onnx/decoder_model_merged_uint8.onnx`,
  });

  console.log('downloading bge-small-en-v1.5 (q8)...');
  await pipeline('feature-extraction', 'Xenova/bge-small-en-v1.5', { dtype: 'q8' });
  const bgeSmall = path.join(ROOT, 'node_modules/@huggingface/transformers/.cache/Xenova/bge-small-en-v1.5');
  await writeBundle('bgeSmallModelFilesBase64.js', {
    'config.json': `${bgeSmall}/config.json`,
    'tokenizer_config.json': `${bgeSmall}/tokenizer_config.json`,
    'tokenizer.json': `${bgeSmall}/tokenizer.json`,
    'onnx/model_quantized.onnx': `${bgeSmall}/onnx/model_quantized.onnx`,
  });

  console.log('downloading pyannote-segmentation-3.0 (fp32)...');
  await AutoModelForAudioFrameClassification.from_pretrained('onnx-community/pyannote-segmentation-3.0', { dtype: 'fp32' });
  await AutoProcessor.from_pretrained('onnx-community/pyannote-segmentation-3.0');
  const pyannote = path.join(ROOT, 'node_modules/@huggingface/transformers/.cache/onnx-community/pyannote-segmentation-3.0');
  await writeBundle('pyannoteModelFilesBase64.js', {
    'config.json': `${pyannote}/config.json`,
    'preprocessor_config.json': `${pyannote}/preprocessor_config.json`,
    'onnx/model.onnx': `${pyannote}/onnx/model.onnx`,
  });

  console.log('downloading wavlm-base-plus-sv (q8)...');
  await AutoModel.from_pretrained('Xenova/wavlm-base-plus-sv', { dtype: 'q8' });
  await AutoProcessor.from_pretrained('Xenova/wavlm-base-plus-sv');
  const wavlm = path.join(ROOT, 'node_modules/@huggingface/transformers/.cache/Xenova/wavlm-base-plus-sv');
  await writeBundle('wavlmModelFilesBase64.js', {
    'config.json': `${wavlm}/config.json`,
    'preprocessor_config.json': `${wavlm}/preprocessor_config.json`,
    'onnx/model_quantized.onnx': `${wavlm}/onnx/model_quantized.onnx`,
  });

  console.log('embedding onnxruntime-web wasm binary + mjs glue file...');
  const ortWasm = path.join(ROOT, 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.wasm');
  const wasmBase64 = fs.readFileSync(ortWasm).toString('base64');
  fs.writeFileSync(path.join(ROOT, 'ortWasmAsyncifyBase64.js'), `module.exports = ${JSON.stringify(wasmBase64)};\n`);
  console.log(`wrote ortWasmAsyncifyBase64.js (${(wasmBase64.length / 1024 / 1024).toFixed(1)}MB)`);

  // Needed by setupOrt.js's wasmPaths.mjs fix -- see docs.dcp.dev's "Getting
  // WebGPU-accelerated libraries working in DCP work functions" for why the
  // webgpu backend needs this (a dynamic import() the plain wasm backend
  // never does) and why a data: URL is the fix for DCP's sandboxed eval.
  const ortMjs = path.join(ROOT, 'node_modules/onnxruntime-web/dist/ort-wasm-simd-threaded.asyncify.mjs');
  const mjsBase64 = fs.readFileSync(ortMjs).toString('base64');
  fs.writeFileSync(path.join(ROOT, 'ortWasmAsyncifyMjsBase64.js'), `module.exports = ${JSON.stringify(mjsBase64)};\n`);
  console.log(`wrote ortWasmAsyncifyMjsBase64.js (${(mjsBase64.length / 1024).toFixed(0)}KB)`);

  console.log('\nAll model bundles generated. Run `node ingest.js <dir>` to ingest mp3s.');
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
