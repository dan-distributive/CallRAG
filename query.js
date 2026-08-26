// CLI for the same retrieval + agentic synthesis logic the viewer's query
// box uses -- see queryEngine.js. Set LLM_PROVIDER=local (+ LLM_BASE_URL)
// to point at an on-prem OpenAI-compatible server instead of Claude; see
// llm.js.
//
// Usage: node query.js "<question>" [topK]
const queryEngine = require('./queryEngine');

async function main() {
  const question = process.argv[2];
  const topK = Number(process.argv[3]) || 8;
  if (!question) {
    console.error('Usage: node query.js "<question>" [topK]');
    process.exit(1);
  }

  const { matches, searchLog, answer, stats } = await queryEngine.runQuery(question, topK);

  if (stats.error) {
    console.error(stats.error);
    process.exit(1);
  }

  console.log(`searching ${stats.segmentsSearched} segments across ${stats.recordingsSearched} recording(s)`);
  console.log('\nInitial matches:');
  for (const m of matches) {
    const when = m.callDate ? m.callDate.slice(0, 10) : 'no date';
    console.log(`  ${m.score.toFixed(3)}  ${when}  ${m.sourceFile} @ ${m.start.toFixed(1)}s  "${m.text}"`);
  }

  if (searchLog.length > 1) {
    console.log(`\n(model searched ${searchLog.length} times total, digging beyond the initial results)`);
    for (const s of searchLog.slice(1)) console.log(`  -> "${s.query}" (${s.resultCount} results)`);
  }

  if (!stats.usedLLM) {
    console.log('\n(no LLM configured -- see llm.js for LLM_PROVIDER/ANTHROPIC_API_KEY -- showing retrieved matches only)');
    return;
  }

  console.log('\n' + answer);
  console.log(`\n(${stats.searchRounds} search round(s), ${stats.usage.input_tokens}+${stats.usage.output_tokens} tokens, ${(stats.totalMs / 1000).toFixed(1)}s total)`);
}
main();
