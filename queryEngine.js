// Shared retrieval + synthesis logic used by both server.js (the viewer's
// query box) and query.js (the CLI). Embeds locally (no DCP dispatch --
// see query.js's original header comment for why), then hands the LLM
// (llm.js -- Claude by default, or a local OpenAI-compatible server) a
// search tool it can call again on its own, rather than a single fixed
// top-K pull. Was the single highest-value upgrade identified when
// planning this: a fixed top-K often can't surface a topic thinly
// scattered across many calls, which is exactly the "how did we end up
// here" style query this project was built toward.
const fs = require('fs');
const path = require('path');
const llm = require('./llm');

const RESULTS_DIR = path.join(__dirname, 'results');
const MAX_TOOL_ROUNDS = 4; // caps runaway cost/latency if the model gets search-happy

function readJSON(p, fallback) {
  try {
    return JSON.parse(fs.readFileSync(p, 'utf8'));
  } catch {
    return fallback;
  }
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

function loadAllSegments() {
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
        speakerName: speakerNames[speaker] ?? seg.voiceMatch?.name ?? null,
        embedding: seg.embedding,
      });
    });
  }
  return out;
}

async function search(allSegments, query, topK) {
  const qEmbedding = await embedQuery(query);
  return allSegments
    .map((s) => ({ ...s, score: cosineSim(qEmbedding, s.embedding) }))
    .sort((a, b) => b.score - a.score)
    .slice(0, topK)
    .map(({ embedding, ...rest }) => rest);
}

function formatExcerpts(matches) {
  const chronological = [...matches].sort((a, b) => (a.callDate ?? '').localeCompare(b.callDate ?? ''));
  return chronological
    .map((m, i) => {
      const when = m.callDate ? new Date(m.callDate).toISOString().slice(0, 16).replace('T', ' ') : 'unknown date';
      const speaker = m.speakerName ?? (m.speaker != null ? `speaker ${m.speaker}` : 'unknown speaker');
      return `[${i + 1}] (${when}, ${m.sourceFile} @ ${m.start.toFixed(1)}s-${m.end.toFixed(1)}s, ${speaker}) ${m.text}`;
    })
    .join('\n');
}

const SYSTEM_PROMPT = `You answer questions about phone/conference calls using a search_calls tool that searches a transcript archive by semantic similarity. You've been given an initial set of search results already. If they're enough to answer, answer directly, citing excerpt numbers. If the question asks how something evolved, was argued about, or was decided over time, or if the initial results seem incomplete or only tangentially related, call search_calls again with different/more specific queries to dig further -- a single search often only surfaces one mention of a topic that's actually discussed across several calls. Speaker labels are only consistent within a single recording unless a real name is shown. If, after searching, the archive genuinely doesn't answer the question, say so plainly. This may be a follow-up in an ongoing conversation -- prior turns are included for context; a short or ambiguous question ("what about pricing?") likely refers back to something discussed earlier.`;

const TOOLS = [
  {
    name: 'search_calls',
    description: 'Search across all ingested call recordings for segments semantically similar to a query. Returns matching excerpts with source recording, date, timestamp, and speaker.',
    input_schema: {
      type: 'object',
      properties: {
        query: { type: 'string', description: 'What to search for' },
        topK: { type: 'number', description: 'How many results to return (default 8)' },
      },
      required: ['query'],
    },
  },
];

/**
 * @param {string} question
 * @param {number} [topK=8]
 * @param {Array<{role: 'user'|'assistant', content: string}>} [history=[]]
 *   prior turns' visible question/answer text, for follow-ups -- NOT the
 *   internal tool-use exchange from earlier turns (that's re-derived fresh
 *   each call via a new search rooted in the current question, since a
 *   follow-up usually needs different excerpts than the original question did).
 * @returns {Promise<{matches: Array, searchLog: Array, answer: string|null, stats: Object}>}
 */
async function runQuery(question, topK = 8, history = []) {
  const t0 = Date.now();
  const allSegments = loadAllSegments();
  if (allSegments.length === 0) {
    return { matches: [], searchLog: [], answer: null, stats: { segmentsSearched: 0, recordingsSearched: 0, error: 'No embedded results found -- run ingest.js first.' } };
  }

  const initialMatches = await search(allSegments, question, topK);
  const retrievalMs = Date.now() - t0;
  const searchLog = [{ query: question, topK, resultCount: initialMatches.length }];

  const noLlmConfigured = (process.env.LLM_PROVIDER || 'anthropic') === 'anthropic' && !process.env.ANTHROPIC_API_KEY;
  if (noLlmConfigured) {
    return {
      matches: initialMatches,
      searchLog,
      answer: null,
      stats: {
        segmentsSearched: allSegments.length,
        recordingsSearched: new Set(allSegments.map((s) => s.sourceFile)).size,
        topScore: initialMatches[0]?.score ?? null,
        retrievalMs,
        totalMs: Date.now() - t0,
        usedLLM: false,
        searchRounds: 1,
      },
    };
  }

  let messages = [
    ...history.map((h) => ({ role: h.role, content: h.content })),
    { role: 'user', content: `Question: ${question}\n\nInitial search results:\n${formatExcerpts(initialMatches)}` },
  ];
  let finalAnswer = null;
  let error = null;
  const usage = { input_tokens: 0, output_tokens: 0 };
  let rounds = 0;

  for (; rounds < MAX_TOOL_ROUNDS; rounds++) {
    const result = await llm.chat({ system: SYSTEM_PROMPT, messages, tools: TOOLS });
    if (result.error) {
      error = result.error;
      break;
    }
    usage.input_tokens += result.usage.input_tokens;
    usage.output_tokens += result.usage.output_tokens;
    messages.push({ role: 'assistant', content: result.content });

    const toolUses = result.content.filter((c) => c.type === 'tool_use');
    if (result.stopReason !== 'tool_use' || toolUses.length === 0) {
      finalAnswer = result.content
        .filter((c) => c.type === 'text')
        .map((c) => c.text)
        .join('');
      break;
    }

    const toolResults = [];
    for (const tu of toolUses) {
      const q = tu.input.query;
      const k = tu.input.topK || 8;
      const matches = await search(allSegments, q, k);
      searchLog.push({ query: q, topK: k, resultCount: matches.length });
      toolResults.push({ type: 'tool_result', tool_use_id: tu.id, content: formatExcerpts(matches) || '(no matches)' });
    }
    messages.push({ role: 'user', content: toolResults });
  }

  return {
    matches: initialMatches,
    searchLog,
    answer: finalAnswer ?? (error ? `LLM error: ${error}` : 'No answer produced (hit the search-round limit).'),
    stats: {
      segmentsSearched: allSegments.length,
      recordingsSearched: new Set(allSegments.map((s) => s.sourceFile)).size,
      topScore: initialMatches[0]?.score ?? null,
      retrievalMs,
      totalMs: Date.now() - t0,
      usedLLM: true,
      searchRounds: searchLog.length,
      usage,
    },
  };
}

module.exports = { runQuery };
