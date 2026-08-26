// Query the ingested call recordings: embed the question locally (no DCP
// dispatch needed for one short string -- plain `node` resolves
// @huggingface/transformers to onnxruntime-node, proven fine for local use),
// cosine-search every persisted results/*.embedded.json segment, and hand
// the top-k matches + question to an LLM for synthesis if ANTHROPIC_API_KEY
// is set; otherwise just print the retrieved context so retrieval itself is
// verifiable without an API key.
//
// Usage: node query.js "<question>" [topK]
const fs = require('fs');
const path = require('path');

function cosineSim(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // both vectors are already normalized (pooling: 'mean', normalize: true)
}

function loadStore() {
  const resultsDir = `${__dirname}/results`;
  const files = fs.existsSync(resultsDir) ? fs.readdirSync(resultsDir).filter((f) => f.endsWith('.embedded.json')) : [];
  const segments = [];
  for (const f of files) {
    const data = JSON.parse(fs.readFileSync(path.join(resultsDir, f), 'utf8'));
    for (const seg of data.segments) {
      segments.push({
        sourceFile: data.sourceFile,
        callDate: data.callDate,
        start: seg.start,
        end: seg.end,
        text: seg.text,
        speaker: seg.speaker, // local speaker-slot id within this call -- see diarize.js; not consistent across different calls
        embedding: seg.embedding,
      });
    }
  }
  return segments;
}

async function embedQuery(text) {
  const installModelFetchPatch = require('./modelFetchPatch');
  const modelFiles = require('./bgeSmallModelFilesBase64');
  installModelFetchPatch(modelFiles);
  const { pipeline, env } = require('@huggingface/transformers');
  env.allowLocalModels = false;
  const embedder = await pipeline('feature-extraction', 'Xenova/bge-small-en-v1.5', { dtype: 'q8' });
  const output = await embedder(text, { pooling: 'mean', normalize: true });
  return Array.from(output.data);
}

async function synthesize(question, matches) {
  const apiKey = process.env.ANTHROPIC_API_KEY;
  // Selection (which segments made the cut) is by relevance score, but the
  // excerpts are presented to the LLM in chronological order -- it can't
  // reconstruct "how did we end up here" from a relevance-ranked jumble, it
  // needs the actual sequence of calls to reason about how a topic evolved.
  const chronological = [...matches].sort((a, b) => (a.callDate ?? '').localeCompare(b.callDate ?? ''));
  const context = chronological
    .map((m, i) => {
      const when = m.callDate ? new Date(m.callDate).toISOString().slice(0, 16).replace('T', ' ') : 'unknown date';
      const speaker = m.speaker != null ? `speaker ${m.speaker}` : 'unknown speaker';
      return `[${i + 1}] (${when}, ${m.sourceFile} @ ${m.start.toFixed(1)}s-${m.end.toFixed(1)}s, ${speaker}) ${m.text}`;
    })
    .join('\n');

  if (!apiKey) {
    console.log('\n(no ANTHROPIC_API_KEY set -- skipping LLM synthesis, showing retrieved context only)\n');
    console.log(context);
    return;
  }

  const res = await fetch('https://api.anthropic.com/v1/messages', {
    method: 'POST',
    headers: {
      'content-type': 'application/json',
      'x-api-key': apiKey,
      'anthropic-version': '2023-06-01',
    },
    body: JSON.stringify({
      model: 'claude-sonnet-5',
      max_tokens: 1024,
      messages: [
        {
          role: 'user',
          content: `The excerpts below are from phone/conference calls, given in chronological order with each one's date, source recording, and speaker label. Speaker labels (e.g. "speaker 2") are only consistent WITHIN a single recording -- "speaker 2" in one call is not necessarily the same person as "speaker 2" in a different call, so never claim two excerpts from different recordings are the same speaker. Answer the question using only these excerpts, citing excerpt numbers. If the question asks how something evolved or was decided over time, trace it chronologically across the excerpts -- note when a topic first came up, how it changed across later calls, and what (if anything) was ultimately decided and when. If the excerpts don't answer the question, say so.\n\nExcerpts:\n${context}\n\nQuestion: ${question}`,
        },
      ],
    }),
  });
  const data = await res.json();
  if (!res.ok) {
    console.error('LLM request failed:', data);
    return;
  }
  console.log('\n' + data.content.map((c) => c.text).join(''));
}

async function main() {
  const question = process.argv[2];
  const topK = Number(process.argv[3]) || 5;
  if (!question) {
    console.error('Usage: node query.js "<question>" [topK]');
    process.exit(1);
  }

  const segments = loadStore();
  if (segments.length === 0) {
    console.error('No embedded results found in results/ -- run ingest.js first.');
    process.exit(1);
  }
  console.log(`searching ${segments.length} segments across ${new Set(segments.map((s) => s.sourceFile)).size} recording(s)`);

  const qEmbedding = await embedQuery(question);
  const ranked = segments
    .map((s) => ({ ...s, score: cosineSim(qEmbedding, s.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK);

  console.log('\nTop matches:');
  for (const m of ranked) {
    const when = m.callDate ? m.callDate.slice(0, 10) : 'no date';
    console.log(`  ${m.score.toFixed(3)}  ${when}  ${m.sourceFile} @ ${m.start.toFixed(1)}s  "${m.text}"`);
  }

  await synthesize(question, ranked);
}
main();
