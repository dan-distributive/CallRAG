// Stage 4: full pipeline, composed from independently-testable local
// modules -- decodeMp3 (Stage 1), vad (new, pure JS), transcribeAudio (new,
// wraps Stage 2/3's proven transformers.js setup). Each module has no
// knowledge of the others; the work function just wires them together.
//
// Model: Xenova/whisper-tiny.en, uint8 quantized, shipped via plain
// job.requires() (small enough to bundle, ~57MB base64). Getting a
// session-creation crash to go away for QDQ-format quantized whisper
// decoders on this DCP sandbox's onnxruntime-web took a real detour --
// see dcp_transformers_restart memory -- the fix was
// session_options.graphOptimizationLevel:'disabled' in transcribeAudio.js,
// NOT a bigger fp32 model or a model-file patch.
async function main() {
  const compute = require('dcp/compute');
  const identity = require('dcp/identity');
  const wallet = require('dcp/wallet');
  const fs = require('fs');

  await identity.set('0xf1512793d2dcb94a0102d53e6ab55ac8b145982342eae999be826aed54533ec7');
  const payKey = await wallet.get('default');
  await wallet.add(payKey);

  // short local test clip first (fast iteration) -- swap for the full
  // recording once this proves out. Lives in the project dir (testdata/),
  // not the session scratchpad -- that's ephemeral and doesn't survive a
  // restart (confirmed the hard way: 2026-08-25 crash wiped it mid-dispatch).
  const mp3Bytes = fs.readFileSync(`${__dirname}/testdata/test_slice.mp3`);
  const inputSet = [{ mp3Base64: mp3Bytes.toString('base64'), sourceFile: 'test_slice.mp3' }];

  async function workFunction(unit) {
    progress(0);
    const base64 = require('./base64');
    const decodeMp3 = require('./decodeMp3');
    const { detectSpeechRegions, trimToSpeech, remapTime } = require('./vad');
    const transcribeAudio = require('./transcribeAudio');
    const modelFiles = require('./whisperTinyEnModelFilesBase64');

    const mp3Bytes = base64.base64ToBytes(unit.mp3Base64);
    const { mono, sampleRate } = await decodeMp3(mp3Bytes);
    console.log(`decoded ${(mono.length / sampleRate).toFixed(1)}s @ ${sampleRate}Hz`);
    progress(0.2);

    const regions = detectSpeechRegions(mono, sampleRate);
    const speechSeconds = regions.reduce((s, r) => s + (r.end - r.start), 0);
    console.log(`VAD: ${regions.length} regions, ${speechSeconds.toFixed(1)}s of ${(mono.length / sampleRate).toFixed(1)}s kept`);
    const { trimmed, mapping } = trimToSpeech(mono, sampleRate, regions);
    progress(0.3);

    const rawSegments = await transcribeAudio(trimmed, modelFiles, 'uint8');
    const segments = rawSegments.map((s) => ({
      start: remapTime(s.start, mapping),
      end: remapTime(s.end, mapping),
      text: s.text,
    }));
    progress(1);

    return { sourceFile: unit.sourceFile, segments };
  }

  const job = compute.for(inputSet, workFunction);
  job.requires(['./base64', './decodeMp3', './vad', './transcribeAudio', './whisperTinyEnModelFilesBase64']);
  job.computeGroups = [{ joinKey: 'ibm', joinSecret: 'dcp' }];
  // NOT { webgpu: true } -- this pipeline runs on device:'wasm' (CPU)
  // everywhere; requiring webgpu-capable workers for a capability nothing
  // here uses restricts the eligible worker pool for no reason. See
  // README Gotcha 6 (webgpu shelved) and ingest.js for the fuller note --
  // confirmed as the real cause of a dispatch that looked stuck for 50+ minutes.
  job.public = {
    name: 'stage4-transcribe',
    description: 'Stage 4: mp3 decode + VAD + Whisper, composed from separate local modules',
  };

  job.on('readystatechange', (ev) => console.log(`Ready state: ${ev}`));
  job.on('accepted', () => console.log(`  Job id: ${job.id}\n  Awaiting results...`));
  job.on('result', (ev) => console.log(JSON.stringify(ev, null, 2)));
  job.on('error', (error) => console.error('  Job error:', error));
  job.on('console', (con) => console.dir(con, { depth: Infinity }));

  const results = await job.exec();
  console.log('FINAL:', JSON.stringify(results, null, 2));

  // Persist each slice's transcript locally so a later stage (embedding,
  // retrieval, etc.) can pick it up without re-running transcription.
  fs.mkdirSync(`${__dirname}/results`, { recursive: true });
  for (const r of results) {
    if (!r.sourceFile) continue; // skip error results
    const outPath = `${__dirname}/results/${r.sourceFile}.transcript.json`;
    fs.writeFileSync(outPath, JSON.stringify(r, null, 2));
    console.log(`wrote ${outPath}`);
  }
}
require('dcp-client').init('https://scheduler.distributed.computer').then(main);
