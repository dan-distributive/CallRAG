// Local viewer/query server for the ingest pipeline's output. Plain Node
// `http`, no framework -- consistent with this project's habit of avoiding
// dependencies where the built-in is enough. Serves:
//   GET  /                        -> viewer.html
//   GET  /api/recordings          -> list of ingested recordings
//   GET  /api/recordings/:file    -> one recording's segments, with
//                                    corrections/speaker-names already merged in
//   GET  /audio/:file             -> the raw mp3 (Range-request aware, for scrubbing)
//   POST /api/speakers/:file      -> body: {speakerId: name} map, persisted,
//                                    and enrolls/updates a GLOBAL voice
//                                    profile per name from that speaker's
//                                    voice-fingerprint embeddings (see
//                                    speakerEmbed.js / diarize.js) so future
//                                    recordings can recognize the same voice
//   POST /api/corrections/:file   -> body: {segmentIndex: newSpeakerId} map, persisted
//   GET  /api/voices              -> known global voice profiles (name + clip count)
//   GET  /api/review              -> every not-yet-confirmed segment across
//                                    every recording, most uncertain first
//                                    (no threshold param -- the viewer
//                                    filters client-side)
//   POST /api/review/answer       -> body: {sourceFile, segmentIndex, name},
//                                    assigns a fresh local speaker id to
//                                    that one clip and names it (reuses the
//                                    corrections + speakers endpoints' own
//                                    logic, including voice-profile enrollment)
//   POST /api/query               -> body: {question, topK, history}, runs
//                                    the same retrieval+synthesis as
//                                    query.js -- history is prior turns'
//                                    {role, content} for follow-up questions
//   GET  /api/timings              -> per-recording stage-by-stage compute
//                                    time (ms) from ingest.js -- populated
//                                    for recordings ingested after this
//                                    endpoint was added; older ones just won't appear
//   GET  /api/voice-space          -> every segment's 512-dim voice
//                                    embedding (see speakerEmbed.js)
//                                    PCA-projected to 2D for visualization
const http = require('http');
const fs = require('fs');
const path = require('path');
const queryEngine = require('./queryEngine');

const PORT = 8177;
const RESULTS_DIR = path.join(__dirname, 'results');
// Every directory that might hold a source mp3 -- searched in order when
// serving /audio/:file. Add more here if you ingest from elsewhere.
const AUDIO_DIRS = [
  path.join(__dirname, 'testdata'),
  path.join(__dirname, 'testdata_5min'),
  path.join(__dirname, 'recordings'),
];

function readJSON(p, fallback) {
  try {
    return JSON.parse(fs.readFileSync(p, 'utf8'));
  } catch {
    return fallback;
  }
}

function listRecordings() {
  if (!fs.existsSync(RESULTS_DIR)) return [];
  const files = fs.readdirSync(RESULTS_DIR).filter((f) => f.endsWith('.transcript.json'));
  return files.map((f) => {
    const sourceFile = f.replace(/\.transcript\.json$/, '');
    const transcript = readJSON(path.join(RESULTS_DIR, f), { segments: [] });
    const hasEmbeddings = fs.existsSync(path.join(RESULTS_DIR, `${sourceFile}.embedded.json`));
    const audioPath = findAudio(sourceFile);
    return {
      sourceFile,
      callDate: transcript.callDate,
      segmentCount: transcript.segments.length,
      duration: transcript.segments.length ? transcript.segments[transcript.segments.length - 1].end : 0,
      hasEmbeddings,
      hasAudio: !!audioPath,
    };
  });
}

function listTimings() {
  if (!fs.existsSync(RESULTS_DIR)) return [];
  const files = fs.readdirSync(RESULTS_DIR).filter((f) => f.endsWith('.timings.json'));
  return files.map((f) => readJSON(path.join(RESULTS_DIR, f), null)).filter(Boolean);
}

// Plain power-iteration PCA -- projects the 512-dim wavlm voice embeddings
// (see speakerEmbed.js) down to 2D so they're actually visualizable. No
// dependency pulled in for this; at these dimensions (a few hundred points,
// 512 dims) a from-scratch implementation is well under a second.
function powerIteration(cov, dim, iterations = 100) {
  let vec = Array.from({ length: dim }, () => Math.random() - 0.5);
  for (let it = 0; it < iterations; it++) {
    const next = new Array(dim).fill(0);
    for (let i = 0; i < dim; i++) {
      let s = 0;
      const row = cov[i];
      for (let j = 0; j < dim; j++) s += row[j] * vec[j];
      next[i] = s;
    }
    const norm = Math.sqrt(next.reduce((s, x) => s + x * x, 0)) || 1;
    for (let i = 0; i < dim; i++) next[i] = next[i] / norm;
    vec = next;
  }
  return vec;
}

function pca2D(vectors) {
  const n = vectors.length;
  const dim = vectors[0].length;

  const mean = new Array(dim).fill(0);
  for (const v of vectors) for (let i = 0; i < dim; i++) mean[i] += v[i] / n;
  const centered = vectors.map((v) => v.map((x, i) => x - mean[i]));

  // Covariance matrix (dim x dim) -- the expensive part, O(n * dim^2).
  const cov = Array.from({ length: dim }, () => new Array(dim).fill(0));
  for (const v of centered) {
    for (let i = 0; i < dim; i++) {
      const vi = v[i];
      if (vi === 0) continue;
      const row = cov[i];
      for (let j = 0; j < dim; j++) row[j] += vi * v[j];
    }
  }

  const pc1 = powerIteration(cov, dim);
  // Deflate: remove PC1's contribution before finding PC2, or power
  // iteration just converges to the same dominant direction twice.
  const covPc1 = pc1.map((_, i) => cov[i].reduce((s, c, j) => s + c * pc1[j], 0));
  const lambda1 = pc1.reduce((s, x, i) => s + x * covPc1[i], 0);
  const cov2 = cov.map((row, i) => row.map((c, j) => c - lambda1 * pc1[i] * pc1[j]));
  const pc2 = powerIteration(cov2, dim);

  return centered.map((v) => [v.reduce((s, x, i) => s + x * pc1[i], 0), v.reduce((s, x, i) => s + x * pc2[i], 0)]);
}

// Every segment with a voice embedding, projected to 2D. Segments from the
// same diarization turn share the same embedding (see diarize.js's
// assignSpeakers) and so land on the same point -- left as-is rather than
// deduped, since duplicate points just overlap harmlessly and dedup would
// need to reconstruct turn boundaries that aren't tracked past this point.
function voiceSpace() {
  if (!fs.existsSync(RESULTS_DIR)) return [];
  const files = fs.readdirSync(RESULTS_DIR).filter((f) => f.endsWith('.transcript.json'));
  const points = [];
  for (const f of files) {
    const sourceFile = f.replace(/\.transcript\.json$/, '');
    const transcript = readJSON(path.join(RESULTS_DIR, f), { segments: [] });
    const speakerNames = readJSON(path.join(RESULTS_DIR, `${sourceFile}.speakers.json`), {});
    const corrections = readJSON(path.join(RESULTS_DIR, `${sourceFile}.corrections.json`), {});
    transcript.segments.forEach((seg, i) => {
      if (!seg.voiceEmbedding) return;
      const speaker = corrections[i] !== undefined ? corrections[i] : seg.speaker;
      const label = speakerNames[speaker] ?? seg.voiceMatch?.name ?? null;
      points.push({ sourceFile, start: seg.start, text: seg.text, speaker, label, embedding: seg.voiceEmbedding });
    });
  }
  if (points.length < 2) return [];

  const coords = pca2D(points.map((p) => p.embedding));
  return points.map(({ embedding, ...rest }, i) => ({ ...rest, x: coords[i][0], y: coords[i][1] }));
}

function findAudio(sourceFile) {
  for (const dir of AUDIO_DIRS) {
    const p = path.join(dir, sourceFile);
    if (fs.existsSync(p)) return p;
  }
  return null;
}

function getRecording(sourceFile) {
  const transcript = readJSON(path.join(RESULTS_DIR, `${sourceFile}.transcript.json`), null);
  if (!transcript) return null;
  const speakerNames = readJSON(path.join(RESULTS_DIR, `${sourceFile}.speakers.json`), {});
  const corrections = readJSON(path.join(RESULTS_DIR, `${sourceFile}.corrections.json`), {});
  const segments = transcript.segments.map((seg, i) => {
    const speaker = corrections[i] !== undefined ? corrections[i] : seg.speaker;
    // A manual name for this call's local speaker id always wins; otherwise
    // fall back to the cross-call voice-fingerprint match (see
    // speakerEmbed.js) if one was found, flagged as auto so the UI can show
    // it's a guess rather than something you confirmed.
    const manualName = speakerNames[speaker] ?? null;
    const speakerName = manualName ?? seg.voiceMatch?.name ?? null;
    return {
      ...seg,
      index: i,
      speaker,
      speakerName,
      autoMatched: !manualName && !!seg.voiceMatch,
      voiceMatchScore: seg.voiceMatch?.score ?? null,
      corrected: corrections[i] !== undefined,
    };
  });
  return { sourceFile, callDate: transcript.callDate, segments, speakerNames };
}

// Named speakers are only ever local slot-ids per recording (see
// diarize.js) unless a voice-fingerprint match already resolved one to a
// cross-call name. Naming a speaker in the viewer both labels this
// recording AND enrolls/updates a GLOBAL voice profile (results/voiceProfiles.json,
// separate from any one recording's files) from that speaker's own
// voiceEmbeddings, so the NEXT recording ingested can recognize the same
// voice automatically. Merged as a running average, not overwritten, so
// one bad clip doesn't wreck an otherwise-good profile.
function updateVoiceProfiles(sourceFile, names) {
  const transcript = readJSON(path.join(RESULTS_DIR, `${sourceFile}.transcript.json`), null);
  if (!transcript) return;
  const corrections = readJSON(path.join(RESULTS_DIR, `${sourceFile}.corrections.json`), {});
  const profilesPath = path.join(RESULTS_DIR, 'voiceProfiles.json');
  const profiles = readJSON(profilesPath, {});

  const embeddingsBySpeaker = {};
  transcript.segments.forEach((seg, i) => {
    const speaker = corrections[i] !== undefined ? corrections[i] : seg.speaker;
    if (speaker == null || !seg.voiceEmbedding) return;
    (embeddingsBySpeaker[speaker] ??= []).push(seg.voiceEmbedding);
  });

  for (const [speakerId, name] of Object.entries(names)) {
    const embeddings = embeddingsBySpeaker[speakerId];
    if (!embeddings || embeddings.length === 0) continue;
    const dims = embeddings[0].length;
    const sum = new Array(dims).fill(0);
    for (const e of embeddings) for (let d = 0; d < dims; d++) sum[d] += e[d];

    const existing = profiles[name];
    const newCount = embeddings.length;
    if (existing) {
      const total = existing.count + newCount;
      profiles[name] = {
        embedding: existing.embedding.map((v, d) => (v * existing.count + sum[d]) / total),
        count: total,
      };
    } else {
      profiles[name] = { embedding: sum.map((v) => v / newCount), count: newCount };
    }
  }

  fs.writeFileSync(profilesPath, JSON.stringify(profiles, null, 2));
}

// Every segment, across every recording, that doesn't yet have a confirmed
// name and isn't long enough into a confident voice-fingerprint match --
// i.e. worth showing a human. The viewer filters this list by confidence
// client-side (no threshold param here) so adjusting the slider doesn't
// need a round trip. Segments already manually named (via corrections ->
// speakerNames) are excluded entirely -- once you've answered a clip, it's
// answered, low-confidence match or not.
function listReviewCandidates() {
  if (!fs.existsSync(RESULTS_DIR)) return [];
  const files = fs.readdirSync(RESULTS_DIR).filter((f) => f.endsWith('.transcript.json'));
  const out = [];
  for (const f of files) {
    const sourceFile = f.replace(/\.transcript\.json$/, '');
    const transcript = readJSON(path.join(RESULTS_DIR, f), { segments: [] });
    const speakerNames = readJSON(path.join(RESULTS_DIR, `${sourceFile}.speakers.json`), {});
    const corrections = readJSON(path.join(RESULTS_DIR, `${sourceFile}.corrections.json`), {});
    transcript.segments.forEach((seg, i) => {
      const speaker = corrections[i] !== undefined ? corrections[i] : seg.speaker;
      const manualName = speakerNames[speaker] ?? null;
      if (manualName) return; // already confirmed, not a review candidate
      out.push({
        sourceFile,
        callDate: transcript.callDate,
        index: i,
        start: seg.start,
        end: seg.end,
        text: seg.text,
        speaker,
        voiceMatchName: seg.voiceMatch?.name ?? null,
        voiceMatchScore: seg.voiceMatch?.score ?? null,
      });
    });
  }
  // Most uncertain first: no match at all, then lowest-confidence matches
  return out.sort((a, b) => (a.voiceMatchScore ?? -1) - (b.voiceMatchScore ?? -1));
}

// Answering one review clip: assign it a fresh local speaker id in its own
// recording (so it doesn't drag along whatever else that recording's
// diarization happened to lump into the same slot) via the existing
// corrections mechanism, then name that id via the existing speakers
// mechanism -- which also enrolls/updates the global voice profile. Reuses
// both existing endpoints' logic rather than inventing a third data path.
function answerReviewCandidate(sourceFile, segmentIndex, name) {
  const corrections = readJSON(path.join(RESULTS_DIR, `${sourceFile}.corrections.json`), {});
  const speakerNames = readJSON(path.join(RESULTS_DIR, `${sourceFile}.speakers.json`), {});
  const transcript = readJSON(path.join(RESULTS_DIR, `${sourceFile}.transcript.json`), null);
  if (!transcript) throw new Error(`no such recording: ${sourceFile}`);

  const allIds = transcript.segments.map((s, i) => (corrections[i] !== undefined ? corrections[i] : s.speaker));
  const newId = Math.max(-1, ...allIds.map(Number).filter((n) => !Number.isNaN(n))) + 1;

  corrections[segmentIndex] = newId;
  speakerNames[newId] = name;

  fs.mkdirSync(RESULTS_DIR, { recursive: true });
  fs.writeFileSync(path.join(RESULTS_DIR, `${sourceFile}.corrections.json`), JSON.stringify(corrections, null, 2));
  fs.writeFileSync(path.join(RESULTS_DIR, `${sourceFile}.speakers.json`), JSON.stringify(speakerNames, null, 2));
  updateVoiceProfiles(sourceFile, { [newId]: name });
}


function sendJSON(res, status, obj) {
  const body = JSON.stringify(obj);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(body) });
  res.end(body);
}

function readBody(req) {
  return new Promise((resolve, reject) => {
    let data = '';
    req.on('data', (c) => (data += c));
    req.on('end', () => {
      try {
        resolve(data ? JSON.parse(data) : {});
      } catch (err) {
        reject(err);
      }
    });
    req.on('error', reject);
  });
}

function serveAudio(req, res, sourceFile) {
  const audioPath = findAudio(sourceFile);
  if (!audioPath) return sendJSON(res, 404, { error: 'audio not found' });
  const stat = fs.statSync(audioPath);
  const range = req.headers.range;
  if (range) {
    const [startStr, endStr] = range.replace('bytes=', '').split('-');
    const start = parseInt(startStr, 10);
    const end = endStr ? parseInt(endStr, 10) : stat.size - 1;
    res.writeHead(206, {
      'content-range': `bytes ${start}-${end}/${stat.size}`,
      'accept-ranges': 'bytes',
      'content-length': end - start + 1,
      'content-type': 'audio/mpeg',
    });
    fs.createReadStream(audioPath, { start, end }).pipe(res);
  } else {
    res.writeHead(200, { 'content-type': 'audio/mpeg', 'content-length': stat.size, 'accept-ranges': 'bytes' });
    fs.createReadStream(audioPath).pipe(res);
  }
}

const server = http.createServer(async (req, res) => {
  try {
    const url = new URL(req.url, `http://localhost:${PORT}`);

    if (req.method === 'GET' && url.pathname === '/') {
      const html = fs.readFileSync(path.join(__dirname, 'viewer.html'));
      res.writeHead(200, { 'content-type': 'text/html' });
      return res.end(html);
    }

    if (req.method === 'GET' && url.pathname === '/api/recordings') {
      return sendJSON(res, 200, listRecordings());
    }

    if (req.method === 'GET' && url.pathname === '/api/voices') {
      const profiles = readJSON(path.join(RESULTS_DIR, 'voiceProfiles.json'), {});
      return sendJSON(
        res,
        200,
        Object.entries(profiles).map(([name, p]) => ({ name, clipsEnrolled: p.count })),
      );
    }

    if (req.method === 'GET' && url.pathname === '/api/review') {
      return sendJSON(res, 200, listReviewCandidates());
    }

    if (req.method === 'GET' && url.pathname === '/api/timings') {
      return sendJSON(res, 200, listTimings());
    }

    if (req.method === 'GET' && url.pathname === '/api/voice-space') {
      return sendJSON(res, 200, voiceSpace());
    }

    if (req.method === 'POST' && url.pathname === '/api/review/answer') {
      const { sourceFile, segmentIndex, name } = await readBody(req);
      if (!sourceFile || segmentIndex == null || !name) {
        return sendJSON(res, 400, { error: 'sourceFile, segmentIndex, and name are required' });
      }
      try {
        answerReviewCandidate(sourceFile, segmentIndex, name);
      } catch (err) {
        return sendJSON(res, 404, { error: err.message });
      }
      return sendJSON(res, 200, { ok: true });
    }

    let m;
    if (req.method === 'GET' && (m = url.pathname.match(/^\/api\/recordings\/(.+)$/))) {
      const rec = getRecording(decodeURIComponent(m[1]));
      return rec ? sendJSON(res, 200, rec) : sendJSON(res, 404, { error: 'not found' });
    }

    if (req.method === 'GET' && (m = url.pathname.match(/^\/audio\/(.+)$/))) {
      return serveAudio(req, res, decodeURIComponent(m[1]));
    }

    if (req.method === 'POST' && (m = url.pathname.match(/^\/api\/speakers\/(.+)$/))) {
      const sourceFile = decodeURIComponent(m[1]);
      const body = await readBody(req);
      fs.mkdirSync(RESULTS_DIR, { recursive: true });
      fs.writeFileSync(path.join(RESULTS_DIR, `${sourceFile}.speakers.json`), JSON.stringify(body, null, 2));
      updateVoiceProfiles(sourceFile, body);
      return sendJSON(res, 200, { ok: true });
    }

    if (req.method === 'POST' && (m = url.pathname.match(/^\/api\/corrections\/(.+)$/))) {
      const sourceFile = decodeURIComponent(m[1]);
      const body = await readBody(req);
      fs.mkdirSync(RESULTS_DIR, { recursive: true });
      fs.writeFileSync(path.join(RESULTS_DIR, `${sourceFile}.corrections.json`), JSON.stringify(body, null, 2));
      return sendJSON(res, 200, { ok: true });
    }

    if (req.method === 'POST' && url.pathname === '/api/query') {
      const { question, topK, history } = await readBody(req);
      if (!question) return sendJSON(res, 400, { error: 'question is required' });
      const result = await queryEngine.runQuery(question, Number(topK) || 8, Array.isArray(history) ? history : []);
      return sendJSON(res, 200, result);
    }

    sendJSON(res, 404, { error: 'not found' });
  } catch (err) {
    console.error(err);
    sendJSON(res, 500, { error: err.message });
  }
});

server.listen(PORT, () => console.log(`viewer running at http://localhost:${PORT}`));
