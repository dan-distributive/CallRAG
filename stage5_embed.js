// Stage 5: transcript segments (from Stage 4's persisted results/*.json)
// -> embedded segments, stored locally for later retrieval. Deliberately a
// separate dispatch from Stage 4 -- no audio decoding needed here, just
// text -- so it can be re-run independently (e.g. after a model/dtype
// change) without re-transcribing anything.
async function main() {
  const compute = require('dcp/compute');
  const identity = require('dcp/identity');
  const wallet = require('dcp/wallet');
  const fs = require('fs');

  await identity.set('0xf1512793d2dcb94a0102d53e6ab55ac8b145982342eae999be826aed54533ec7');
  const payKey = await wallet.get('default');
  await wallet.add(payKey);

  const sourceFile = process.argv[2] || 'test_slice.mp3';
  const transcriptPath = `${__dirname}/results/${sourceFile}.transcript.json`;
  const transcript = JSON.parse(fs.readFileSync(transcriptPath, 'utf8'));

  const modelFiles = require('./bgeSmallModelFilesBase64');
  const inputSet = [{ sourceFile: transcript.sourceFile, segments: transcript.segments }];

  async function workFunction(unit) {
    progress(0);
    const embedText = require('./embedText');
    const modelFiles = require('./bgeSmallModelFilesBase64');

    const embedded = [];
    for (let i = 0; i < unit.segments.length; i++) {
      const seg = unit.segments[i];
      try {
        const embedding = await embedText(seg.text, modelFiles);
        embedded.push({ ...seg, embedding });
      } catch (err) {
        console.warn(`embed failed for segment ${i} (${seg.text.length} chars), skipping:`, err.message);
      }
      progress((i + 1) / unit.segments.length);
    }

    return { sourceFile: unit.sourceFile, segments: embedded };
  }

  const job = compute.for(inputSet, workFunction);
  job.requires(['./embedText', './bgeSmallModelFilesBase64']);
  job.computeGroups = [{ joinKey: 'ibm', joinSecret: 'dcp' }];
  job.requirements = job.requirements || {};
  job.requirements.environment = { webgpu: true };
  job.public = {
    name: 'stage5-embed',
    description: 'Stage 5: embed transcript segments (bge-small-en-v1.5) for later retrieval',
  };

  job.on('readystatechange', (ev) => console.log(`Ready state: ${ev}`));
  job.on('accepted', () => console.log(`  Job id: ${job.id}\n  Awaiting results...`));
  job.on('error', (error) => console.error('  Job error:', error));
  job.on('console', (con) => console.dir(con, { depth: Infinity }));

  const results = await job.exec();

  fs.mkdirSync(`${__dirname}/results`, { recursive: true });
  for (const r of results) {
    if (!r.sourceFile) continue;
    const outPath = `${__dirname}/results/${r.sourceFile}.embedded.json`;
    fs.writeFileSync(outPath, JSON.stringify(r, null, 2));
    console.log(`wrote ${outPath} (${r.segments.length} segments, ${r.segments[0].embedding.length}-dim)`);
  }
}
require('dcp-client').init('https://scheduler.distributed.computer').then(main);
