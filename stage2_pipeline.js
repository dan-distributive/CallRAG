// Stage 2: prove @huggingface/transformers + onnxruntime-web (wasm backend,
// device: 'cpu') works as local job.requires() modules on a real DCP worker.
// Model weights are loaded via transformers.js's default remote fetch for
// now (not yet bundled locally) -- this also tests whether DCP's sandbox
// fetch() origin allowlist permits huggingface.co at all.
async function main() {
  const compute = require('dcp/compute');
  const identity = require('dcp/identity');
  const wallet = require('dcp/wallet');

  await identity.set('0xf1512793d2dcb94a0102d53e6ab55ac8b145982342eae999be826aed54533ec7');
  const payKey = await wallet.get('default');
  await wallet.add(payKey);

  const inputSet = [1];

  async function workFunction() {
    progress();
    const testPipeline = require('./testPipeline');
    return await testPipeline();
  }

  const job = compute.for(inputSet, workFunction);
  job.requires(['./testPipeline']);
  job.computeGroups = [{ joinKey: 'ibm', joinSecret: 'dcp' }];
  job.requirements = job.requirements || {};
  job.requirements.environment = { webgpu: true };
  job.public = {
    name: 'stage2-transformers-pipeline',
    description: 'Stage 2: transformers.js + onnxruntime-web wasm backend via local job.requires()',
  };

  job.on('readystatechange', (ev) => console.log(`Ready state: ${ev}`));
  job.on('accepted', () => console.log(`  Job id: ${job.id}\n  Awaiting results...`));
  job.on('result', (ev) => console.log(JSON.stringify(ev, null, 2)));
  job.on('error', (error) => console.error('  Job error:', error));
  job.on('console', (con) => console.dir(con, { depth: Infinity }));

  const results = await job.exec();
  console.log('FINAL:', JSON.stringify(results, null, 2));
}
require('dcp-client').init('https://scheduler.distributed.computer').then(main);
