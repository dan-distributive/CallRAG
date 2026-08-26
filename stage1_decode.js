// Stage 1: prove minimp3-wasm mp3 decode works as a local job.requires()
// module on a real DCP worker (not just locally in Node).
async function main() {
  const compute = require('dcp/compute');
  const fs = require('fs');

  const mp3Bytes = fs.readFileSync(
    '/Users/dandesjardins/Downloads/11167358-260821153015.mp3'
  );
  const mp3Base64 = mp3Bytes.toString('base64');

  const inputSet = [{ mp3Base64 }];

  async function workFunction(unit) {
    progress();
    // dependency-free base64 decode, matching decodeMp3.js's own -- avoids
    // assuming Buffer/atob are available in the actual worker runtime.
    const B64_CHARS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/';
    function base64ToBytes(b64) {
      const clean = b64.replace(/=+$/, '');
      const out = new Uint8Array(Math.floor((clean.length * 3) / 4));
      let outIdx = 0;
      for (let i = 0; i < clean.length; i += 4) {
        const c0 = B64_CHARS.indexOf(clean[i]);
        const c1 = B64_CHARS.indexOf(clean[i + 1]);
        const c2 = clean[i + 2] !== undefined ? B64_CHARS.indexOf(clean[i + 2]) : -1;
        const c3 = clean[i + 3] !== undefined ? B64_CHARS.indexOf(clean[i + 3]) : -1;
        out[outIdx++] = (c0 << 2) | (c1 >> 4);
        if (c2 >= 0) out[outIdx++] = ((c1 & 0xf) << 4) | (c2 >> 2);
        if (c3 >= 0) out[outIdx++] = ((c2 & 0x3) << 6) | c3;
      }
      return out;
    }

    const decodeMp3 = require('./decodeMp3');
    const mp3Bytes = base64ToBytes(unit.mp3Base64);
    const { mono, sampleRate } = await decodeMp3(mp3Bytes);
    return {
      numSamples: mono.length,
      sampleRate,
      durationSec: mono.length / sampleRate,
      firstSamples: Array.from(mono.slice(1000, 1010)),
    };
  }

  const job = compute.for(inputSet, workFunction);
  job.requires(['./decodeMp3']);
  job.computeGroups = [{ joinKey: 'ibm', joinSecret: 'dcp' }];
  job.public = {
    name: 'stage1-mp3-decode',
    description: 'Stage 1: minimp3-wasm decode via local job.requires()',
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
