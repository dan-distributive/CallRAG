// Batch ingest: point at a directory of mp3 call recordings, transcribe +
// embed every one that hasn't been processed yet. One DCP job, one slice
// per file, each slice running the full decode->VAD->transcribe->embed
// pipeline -- DCP parallelize whole-file pipelines across workers
//
// Usage: node ingest.js <directory-of-mp3s>
async function main() {
  const compute = require('dcp/compute');
  const identity = require('dcp/identity');
  const wallet = require('dcp/wallet');
  const fs = require('fs');
  const path = require('path');
  const { parseCallDate } = require('./callDate');

  const dir = process.argv[2];
  if (!dir) {
    console.error('Usage: node ingest.js <directory-of-mp3s>');
    process.exit(1);
  }

  await identity.set('0xf1512793d2dcb94a0102d53e6ab55ac8b145982342eae999be826aed54533ec7');
  const payKey = await wallet.get('default');
  await wallet.add(payKey);

  const resultsDir = `${__dirname}/results`;
  fs.mkdirSync(resultsDir, { recursive: true });

  function readJSON(p, fallback) {
    try {
      return JSON.parse(fs.readFileSync(p, 'utf8'));
    } catch {
      return fallback;
    }
  }

  function persist(r) {
    if (!r.sourceFile) return; // an error result with no sourceFile -- nothing to write
    const callDate = parseCallDate(r.sourceFile);
    fs.writeFileSync(
      `${resultsDir}/${r.sourceFile}.transcript.json`,
      JSON.stringify({ sourceFile: r.sourceFile, callDate, segments: r.segments }, null, 2),
    );
    if (r.embedded) {
      fs.writeFileSync(
        `${resultsDir}/${r.sourceFile}.embedded.json`,
        JSON.stringify({ sourceFile: r.sourceFile, callDate, segments: r.embedded }, null, 2),
      );
      console.log(
        `${r.sourceFile} (${callDate ?? 'no date'}): transcribed (${r.segments.length} segments) + embedded (${r.embedded.length}) in ${((r.timings?.totalMs ?? 0) / 1000).toFixed(1)}s`,
      );
    } else {
      console.warn(`${r.sourceFile}: transcribed (${r.segments.length} segments) but embedding failed -- ${r.embedError}`);
    }
    if (r.timings) console.log(`  stage breakdown (s): ${JSON.stringify(Object.fromEntries(Object.entries(r.timings).map(([k, v]) => [k, (v / 1000).toFixed(1)])))}`);

    if (r.timings) {
      const durationSec = r.segments.length ? r.segments[r.segments.length - 1].end : 0;
      fs.writeFileSync(
        `${resultsDir}/${r.sourceFile}.timings.json`,
        JSON.stringify({ sourceFile: r.sourceFile, callDate, durationSec, segmentCount: r.segments.length, timings: r.timings }, null, 2),
      );
    }
  }

  const allMp3s = fs.readdirSync(dir).filter((f) => f.toLowerCase().endsWith('.mp3'));
  const toProcess = allMp3s.filter((f) => !fs.existsSync(`${resultsDir}/${f}.embedded.json`));
  console.log(`${allMp3s.length} mp3 files found, ${toProcess.length} need processing`);
  if (toProcess.length === 0) return;

  const inputSet = toProcess.map((f) => ({
    mp3Base64: fs.readFileSync(path.join(dir, f)).toString('base64'),
    sourceFile: f,
  }));

  async function pipelineWork(unit, whisperModelFiles, bgeModelFiles, pyannoteModelFiles, wavlmModelFiles, voiceProfiles) {
    progress(0);
    const base64 = require('./base64');
    const decodeMp3 = require('./decodeMp3');
    const { resample } = require('./resample');
    const { detectSpeechRegions, trimToSpeech, remapTime } = require('./vad');
    const transcribeAudio = require('./transcribeAudio');
    const embedText = require('./embedText');
    const diarize = require('./diarize');
    const { assignSpeakers } = diarize;
    const speakerEmbed = require('./speakerEmbed');
    const { matchVoice } = speakerEmbed;

    const TARGET_SR = 16000; // what both Whisper and pyannote-segmentation expect

    const t0 = Date.now();
    const timings = {};

    // diarize()'s/speakerEmbed()'s/transcribeAudio()'s per-chunk/per-token
    // callbacks are only there to satisfy DCP's ENOPROGRESS timeout (jobs
    // killed if progress() isn't called for ~30s) -- calling the real
    // progress() far more often than that (per token, easily several times
    // a second) is just noise in anything watching the job's console.
    // Gate it to at most once a second; the exact percentage reported on a
    // gated-out call is lost, but the next call within the loop reports
    // whatever's current, so nothing meaningful is missed.
    let lastProgressAt = 0;
    function throttledProgress(value) {
      const now = Date.now();
      if (now - lastProgressAt >= 1000) {
        lastProgressAt = now;
        progress(value);
      }
    }

    const mp3Bytes = base64.base64ToBytes(unit.mp3Base64);
    const { mono, sampleRate } = await decodeMp3(mp3Bytes);
    // transformers.js's audio pipelines only resample when given a
    // URL/string -- a raw Float32Array is passed straight to the model as-is
    // (see resample.js). Our mp3s decode at whatever rate they're actually
    // encoded at (8kHz for these telephony recordings), so this resample is
    // required, not optional -- without it Whisper silently treats 8kHz
    // audio as 16kHz, halving every timestamp and doubling apparent pitch/speed.
    const resampled = resample(mono, sampleRate, TARGET_SR);
    timings.decodeAndResample = Date.now() - t0;
    progress(0.1);

    // Diarize the FULL untrimmed audio, not the VAD-trimmed speech-only
    // audio -- turn boundaries need real silence gaps to be meaningful, and
    // this keeps diarization on the same original-recording timeline as the
    // (already remapped) transcript segments, so assignSpeakers can match
    // them directly without its own separate time-remapping step.
    // diarize() and transcribeAudio() both take a progress callback that
    // fires many times during their (potentially very long, for a real
    // multi-minute recording) internal loops -- without this, DCP's own
    // ENOPROGRESS timeout (jobs are killed if progress() isn't called at
    // least every ~30s) can trip, and even short of that, a job sitting at
    // one fixed percentage for a long time is indistinguishable from a
    // frozen worker to anyone watching. Confirmed as a real (not just
    // theoretical) risk -- see dcp_transformers_restart memory.
    const totalDiarizeChunks = Math.ceil(resampled.length / (16000 * 30));
    let diarizeChunksDone = 0;
    const turns = await diarize(resampled, pyannoteModelFiles, () => {
      diarizeChunksDone++;
      throttledProgress(0.1 + 0.1 * (diarizeChunksDone / totalDiarizeChunks));
    });
    timings.diarize = Date.now() - t0 - timings.decodeAndResample;
    progress(0.2);

    // Voice-fingerprint each diarization turn and match it against known,
    // named voice profiles (built up via the viewer -- see server.js).
    // diarize.js's turn IDs are only consistent within THIS call; matching
    // against a voice embedding is what makes a name carry across calls.
    // Turns too short to embed reliably (see speakerEmbed.js) just don't
    // get a voice match -- they keep their local speaker id only.
    const turnsWithVoice = [];
    for (let i = 0; i < turns.length; i++) {
      const turn = turns[i];
      const startSample = Math.round(turn.start * TARGET_SR);
      const endSample = Math.round(turn.end * TARGET_SR);
      const voiceEmbedding = await speakerEmbed(resampled.subarray(startSample, endSample), wavlmModelFiles);
      const voiceMatch = voiceEmbedding ? matchVoice(voiceEmbedding, voiceProfiles) : null;
      turnsWithVoice.push({ ...turn, voiceEmbedding, voiceMatch });
      throttledProgress(0.2 + 0.05 * ((i + 1) / turns.length)); // a real recording can have hundreds of turns -- same ENOPROGRESS risk as diarize()/transcribeAudio()
    }
    timings.speakerEmbed = Date.now() - t0 - timings.decodeAndResample - timings.diarize;

    const regions = detectSpeechRegions(resampled, TARGET_SR);
    const { trimmed, mapping } = trimToSpeech(resampled, TARGET_SR, regions);
    timings.vad = Date.now() - t0 - timings.decodeAndResample - timings.diarize - timings.speakerEmbed;
    progress(0.3);

    // Whisper's token-level callback has no way to know how many tokens are
    // left (unlike diarize's/speakerEmbed's chunk loops above, which know
    // their total up front) -- a fixed progress(0.4) here just repeats the
    // same percentage for the whole transcription stage, which is alive
    // enough to dodge ENOPROGRESS but LOOKS identical to a frozen worker to
    // anyone watching. Climb asymptotically toward (not reaching) 0.5
    // instead, so it visibly moves without ever overshooting into the next
    // stage's range before transcription actually finishes.
    let transcribeTokens = 0;
    const rawSegments = await transcribeAudio(trimmed, whisperModelFiles, 'uint8', 'Xenova/whisper-base.en', () => {
      transcribeTokens++;
      throttledProgress(0.4 + 0.1 * (1 - 1 / (1 + transcribeTokens / 200)));
    });
    timings.transcribe = Date.now() - t0 - timings.decodeAndResample - timings.diarize - timings.speakerEmbed - timings.vad;
    const withTime = rawSegments.map((s) => ({
      start: remapTime(s.start, mapping),
      end: remapTime(s.end, mapping),
      text: s.text,
    }));
    const segments = assignSpeakers(withTime, turnsWithVoice);
    progress(0.5);

    // Transcription is the expensive, slow-to-redo half -- guarantee it's
    // always returned even if embedding blows up in some way the per-segment
    // try/catch below doesn't catch (e.g. model loading itself failing),
    // rather than losing it along with the embedding failure.
    let embedded;
    let embedError;
    try {
      embedded = [];
      for (let i = 0; i < segments.length; i++) {
        const seg = segments[i];
        try {
          const embedding = await embedText(seg.text, bgeModelFiles);
          embedded.push({ ...seg, embedding });
        } catch (err) {
          console.warn(`embed failed for segment ${i} (${seg.text.length} chars), skipping:`, err.message);
        }
        throttledProgress(0.5 + 0.5 * ((i + 1) / segments.length));
      }
    } catch (err) {
      embedded = undefined;
      embedError = err.message;
    }
    timings.embed = Date.now() - t0 - timings.decodeAndResample - timings.diarize - timings.speakerEmbed - timings.vad - timings.transcribe;
    timings.totalMs = Date.now() - t0;
    console.log(`timings (ms) for ${unit.sourceFile}:`, JSON.stringify(timings));

    return { sourceFile: unit.sourceFile, segments, embedded, embedError, timings };
  }

  // Both model bundles ship as job ARGUMENTS, not job.requires() -- both
  // whisper-base.en (~100MB base64) and even bge-small (~46MB) risk the
  // local webpack bundler's ERR_WORKER_OUT_OF_MEMORY ceiling once combined
  // into one job requiring both, so keep both on the transport already
  // proven safe at this size.
  const whisperModelFiles = require('./whisperBaseEnModelFilesBase64');
  const bgeModelFiles = require('./bgeSmallModelFilesBase64');
  const pyannoteModelFiles = require('./pyannoteModelFilesBase64'); // tiny (~7.6MB); shipped as a job argument for consistency with the other two models, though job.requires() would work fine at this size too
  const wavlmModelFiles = require('./wavlmModelFilesBase64'); // ~129MB -- job argument, well past the job.requires() ceiling
  // Known named voices, built up via the viewer's speaker-naming endpoint
  // (see server.js) -- passed in fresh on every dispatch so newly-named
  // speakers get recognized on the NEXT file processed, not retroactively.
  const voiceProfiles = readJSON(`${resultsDir}/voiceProfiles.json`, {});
  
  // DCP JOB
  const job = compute.for(inputSet, pipelineWork, [whisperModelFiles, bgeModelFiles, pyannoteModelFiles, wavlmModelFiles, voiceProfiles]);
  
  // Require local modules
  job.requires([
    './base64',
    './decodeMp3',
    './resample',
    './vad',
    './transcribeAudio',
    './embedText',
    './diarize',
    './speakerEmbed'
  ]);

  // COMPUTE GROUP(S)
  job.computeGroups = [
    { joinKey: 'ssc-icelab', joinSecret: 'r2whez1w' }
  ];

  // JOB REQUIREMENTS
  // NOT `{ webgpu: true }` -- this whole pipeline runs on device:'wasm'
  // (CPU) everywhere; WebGPU was investigated and shelved (uncatchable
  // crash in onnxruntime-web's JSEP session creation -- see README Gotcha
  // 6). Requiring webgpu:true here restricted the eligible worker pool to
  // only GPU-capable devices for a capability nothing in the job actually
  // uses -- confirmed as the real cause of a dispatch that looked "stuck"
  // for 50+ minutes (an empty compute group was the first suspicion, but
  // this was the actual blocker: plenty of workers existed, almost none of
  // them GPU-capable).

  // JOB PUBLIC INFO
  job.public = { 
    name: '📞 CallRAG ingest-pipeline', 
    description: `Batch transcribe+embed ${toProcess.length} file(s)`,
    link: 'https://distributive.network',
  };

  // EVENTS
  job.on('readystatechange', (ev) => console.log(`[${new Date().toISOString()}] ready state: ${ev}`));
  job.on('accepted', () => console.log(`  Job id: ${job.id}`));
  job.on('console', (con) => console.dir(con, { depth: Infinity }));
  job.on('error', (error) => console.error('  Job error:', error));
  // Persist each file's result as soon as its slice completes, rather than
  // waiting on job.exec() for every slice -- so a later file's failure
  // can't take an earlier file's already-finished output down with it.
  job.on('result', (ev) => {
    if (ev.result) persist(ev.result);
    else console.error('  Slice error:', ev);
  });

  // EXEC
  await job.exec();
}
require('dcp-client').init('https://scheduler.distributed.computer').then(main);
