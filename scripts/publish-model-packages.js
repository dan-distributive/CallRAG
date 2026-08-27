// Publishes the packages built by build-model-packages.js to the DCP
// package manager, smallest first -- pyannote (~8MB) acts as a canary
// before committing the two large ones (whisper ~105MB, wavlm ~136MB),
// since no package this large has been published before (largest known
// precedent is ffmpeg-wasm-test at 13.9MB -- see docs.dcp.dev's
// "Publishing a DCP package").
//
// Usage: node scripts/publish-model-packages.js [pkgName]
// With no argument, publishes all four in size order. With an argument,
// publishes just that one package (for retrying a single failure).
const path = require('path');
const ROOT = path.join(__dirname, '..');
const PKG_DIR = path.join(ROOT, 'packages');

const PACKAGES_IN_SIZE_ORDER = [
  'callrag-pyannote-seg3-fp32',
  'callrag-bge-small-en-q8',
  'callrag-whisper-base-en-uint8',
  'callrag-wavlm-base-sv-q8',
];

async function main() {
  const only = process.argv[2];
  const toPublish = only ? [only] : PACKAGES_IN_SIZE_ORDER;

  for (const pkgName of toPublish) {
    const manifestPath = path.join(PKG_DIR, pkgName, 'package.dcp');
    console.log(`\n=== publishing ${pkgName} ===`);
    const t0 = Date.now();
    try {
      const result = await require('dcp/publish').publish(manifestPath);
      console.log(`published: ${result.name}@${Object.keys(result.versions).join(',')} (${((Date.now() - t0) / 1000).toFixed(1)}s)`);
    } catch (err) {
      console.error(`FAILED to publish ${pkgName}:`, err.code || '', err.message);
      throw err;
    }
  }
  console.log('\nAll requested packages published.');
  process.exit(0);
}

require('dcp-client').init('https://scheduler.distributed.computer').then(main).catch((err) => {
  console.error('publish run failed:', err.message);
  process.exit(1);
});
