// Stage 0: prove the basic local-modules job dispatch pattern works at all,
// before building anything audio/ML-related on top of it. Exact shape from
// https://docs.dcp.dev/tutorials/node/to-upper-case.html, using the 'ibm'
// compute group (known to have active workers this session) instead of
// demo/public.
async function main() {
  const compute = require('dcp/compute');
  const identity = require('dcp/identity');
  const wallet = require('dcp/wallet');

  await identity.set('0xf1512793d2dcb94a0102d53e6ab55ac8b145982342eae999be826aed54533ec7');

  const payKey = await wallet.get('default');
  await wallet.add(payKey);

  /* INPUT SET */
  const inputSet = Array.from('yelling!');

  /* WORK FUNCTION */
  async function workFunction(letter) {
    progress();
    const charInfo = require('./charInfo');
    return { upper: letter.toUpperCase(), kind: charInfo(letter) };
  }

  /* COMPUTE FOR */
  const job = compute.for(inputSet, workFunction);

  /* LOCAL MODULES */
  job.requires(['./charInfo']);

  /* COMPUTE GROUPS */
  job.computeGroups = [{ joinKey: 'ibm', joinSecret: 'dcp' }];

  /* PUBLIC INFO */
  job.public = {
    name: 'stage0-toUpperCase',
    description: 'Stage 0: proving local job.requires() bundling works',
  };

  /* EVENTS */
  job.on('readystatechange', (ev) => console.log(`Ready state: ${ev}`));
  job.on('accepted', () => console.log(`  Job id: ${job.id}\n  Awaiting results...`));
  job.on('result', (ev) => console.log(ev));
  job.on('error', (error) => console.error('  Job error:', error));
  job.on('nofunds', (ev) => console.log(ev));
  job.on('console', (con) => console.dir(con, { depth: Infinity }));

  /* EXECUTION */
  let results = await job.exec();

  /* RESULT POST-PROCESSING */
  let RESULTS = results.map((r) => r.upper).join('');
  console.log(RESULTS);
  console.log(results.map((r) => `${r.upper}:${r.kind}`).join('  '));
}
require('dcp-client').init('https://scheduler.distributed.computer').then(main);
