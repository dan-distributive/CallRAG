// Local viewer/query server for the ingest pipeline's output. Plain Node
// `http`, no framework -- consistent with this project's habit of avoiding
// dependencies where the built-in is enough. Serves:
//   GET  /                        -> viewer.html
//   GET  /api/recordings          -> list of ingested recordings
//   GET  /api/recordings/:file    -> one recording's segments, with
//                                    corrections/speaker-names already merged in
//   GET  /audio/:file             -> the raw mp3 (Range-request aware, for scrubbing)
//   POST /api/speakers/:file      -> body: {speakerId: name} map, persisted
//   POST /api/corrections/:file   -> body: {segmentIndex: newSpeakerId} map, persisted
//   POST /api/query               -> body: {question, topK}, runs the same
//                                    retrieval+synthesis as query.js
const http = require('http');
const fs = require('fs');
const path = require('path');

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
    return { ...seg, index: i, speaker, speakerName: speakerNames[speaker] ?? null, corrected: corrections[i] !== undefined };
  });
  return { sourceFile, callDate: transcript.callDate, segments, speakerNames };
}

function cosineSim(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot;
}

let embedQueryFn = null;
async function embedQuery(text) {
  if (!embedQueryFn) {
    const installModelFetchPatch = require('./modelFetchPatch');
    const modelFiles = require('./bgeSmallModelFilesBase64');
    installModelFetchPatch(modelFiles);
    const { pipeline, env } = require('@huggingface/transformers');
    env.allowLocalModels = false;
    const embedder = await pipeline('feature-extraction', 'Xenova/bge-small-en-v1.5', { dtype: 'q8' });
    embedQueryFn = async (t) => Array.from((await embedder(t, { pooling: 'mean', normalize: true })).data);
  }
  return embedQueryFn(text);
}

function loadAllSegmentsForQuery() {
  if (!fs.existsSync(RESULTS_DIR)) return [];
  const files = fs.readdirSync(RESULTS_DIR).filter((f) => f.endsWith('.embedded.json'));
  const out = [];
  for (const f of files) {
    const sourceFile = f.replace(/\.embedded\.json$/, '');
    const data = readJSON(path.join(RESULTS_DIR, f), { segments: [] });
    const speakerNames = readJSON(path.join(RESULTS_DIR, `${sourceFile}.speakers.json`), {});
    const corrections = readJSON(path.join(RESULTS_DIR, `${sourceFile}.corrections.json`), {});
    data.segments.forEach((seg, i) => {
      const speaker = corrections[i] !== undefined ? corrections[i] : seg.speaker;
      out.push({
        sourceFile,
        callDate: data.callDate,
        start: seg.start,
        end: seg.end,
        text: seg.text,
        speaker,
        speakerName: speakerNames[speaker] ?? null,
        embedding: seg.embedding,
      });
    });
  }
  return out;
}

async function runQuery(question, topK) {
  const t0 = Date.now();
  const allSegments = loadAllSegmentsForQuery();
  const qEmbedding = await embedQuery(question);
  const ranked = allSegments
    .map((s) => ({ ...s, score: cosineSim(qEmbedding, s.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);
  const retrievalMs = Date.now() - t0;

  const apiKey = process.env.ANTHROPIC_API_KEY;
  const chronological = [...ranked].sort((a, b) => (a.callDate ?? '').localeCompare(b.callDate ?? ''));
  const context = chronological
    .map((m, i) => {
      const when = m.callDate ? new Date(m.callDate).toISOString().slice(0, 16).replace('T', ' ') : 'unknown date';
      const speaker = m.speakerName ?? (m.speaker != null ? `speaker ${m.speaker}` : 'unknown speaker');
      return `[${i + 1}] (${when}, ${m.sourceFile} @ ${m.start.toFixed(1)}s-${m.end.toFixed(1)}s, ${speaker}) ${m.text}`;
    })
    .join('\n');

  let answer = null;
  let llmMs = null;
  if (apiKey) {
    const t1 = Date.now();
    try {
      const res = await fetch('https://api.anthropic.com/v1/messages', {
        method: 'POST',
        headers: { 'content-type': 'application/json', 'x-api-key': apiKey, 'anthropic-version': '2023-06-01' },
        body: JSON.stringify({
          model: 'claude-sonnet-5',
          max_tokens: 1024,
          messages: [
            {
              role: 'user',
              content: `The excerpts below are from phone/conference calls, given in chronological order with each one's date, source recording, and speaker label. Speaker labels are only consistent WITHIN a single recording unless a real name is shown. Answer the question using only these excerpts, citing excerpt numbers. If the question asks how something evolved or was decided over time, trace it chronologically. If the excerpts don't answer the question, say so.\n\nExcerpts:\n${context}\n\nQuestion: ${question}`,
            },
          ],
        }),
      });
      const data = await res.json();
      answer = res.ok ? data.content.map((c) => c.text).join('') : `LLM error: ${JSON.stringify(data)}`;
    } catch (err) {
      answer = `LLM request failed: ${err.message}`;
    }
    llmMs = Date.now() - t1;
  }

  return {
    matches: ranked.map(({ embedding, ...rest }) => rest),
    context,
    answer,
    stats: {
      segmentsSearched: allSegments.length,
      recordingsSearched: new Set(allSegments.map((s) => s.sourceFile)).size,
      topScore: ranked[0]?.score ?? null,
      retrievalMs,
      llmMs,
      usedLLM: !!apiKey,
      totalMs: Date.now() - t0,
    },
  };
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
      const { question, topK } = await readBody(req);
      if (!question) return sendJSON(res, 400, { error: 'question is required' });
      const result = await runQuery(question, Number(topK) || 8);
      return sendJSON(res, 200, result);
    }

    sendJSON(res, 404, { error: 'not found' });
  } catch (err) {
    console.error(err);
    sendJSON(res, 500, { error: err.message });
  }
});

server.listen(PORT, () => console.log(`viewer running at http://localhost:${PORT}`));
