// Serves model files from an embedded local copy instead of fetching them
// from huggingface.co, which DCP's sandbox blocks outright (EPERM_ORIGIN --
// see ~/DCP/wasm-package-howto.md). Matches URLs by filename suffix. Falls
// through to the real (DCP-sandboxed) fetch for anything not registered.
//
// Generic by design: takes a { filename: base64 } map as an argument rather
// than importing one hardcoded model's files -- callable once per model
// (bge-small, whisper, etc.), registrations accumulate.
const { base64ToBytes } = require('./base64');

let patchInstalled = false;
const registeredFiles = {};

function installModelFetchPatch(modelFiles) {
  Object.assign(registeredFiles, modelFiles);
  if (patchInstalled) return;
  patchInstalled = true;

  const oldFetch = globalThis.fetch;
  globalThis.fetch = async function (...args) {
    const url = args[0].toString();
    // Longest name first: several real filenames are substrings of each
    // other (e.g. "generation_config.json", "tokenizer_config.json", and
    // "preprocessor_config.json" all contain "config.json"), so a plain
    // first-match would silently serve the WRONG file's bytes for
    // whichever names happen to iterate first -- confirmed as a real bug
    // (preprocessor_config.json requests were getting config.json's
    // content, causing "No image_processor_type or feature_extractor_type
        // found in the config").
    const names = Object.keys(registeredFiles).sort((a, b) => b.length - a.length);
    for (const name of names) {
      if (url.includes(name)) {
        return new Response(base64ToBytes(registeredFiles[name]));
      }
    }
    return oldFetch.apply(this, args);
  };
}

module.exports = installModelFetchPatch;
