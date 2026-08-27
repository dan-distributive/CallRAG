// Fixes a self-location crash at onnxruntime-web's WEBGPU BACKEND'S OWN
// TOP-LEVEL ENTRY POINT (`onnxruntime-web/webgpu`'s package.json `exports`
// map resolves straight to these three files) -- this fires at module
// load/init time, in DCP's sandboxed `about:blank`-base-URL `eval` context,
// before any of our own setupOrt.js configuration code even runs. This is
// a SEPARATE bug from the one setupOrt.js's `wasmPaths.mjs` data: URL
// fixes (a deeper dynamic import() during the actual WASM glue-file load,
// well after this module has already finished initializing) -- see
// docs.dcp.dev's "Getting WebGPU-accelerated libraries working in DCP work
// functions" for that one. Both are required together for `device:
// 'webgpu'` to work at all: this patch lets the module load in the first
// place; wasmPaths.mjs is what makes the load succeed once it's past this.
//
// Wired into `npm install` via package.json's `postinstall` script --
// without that, a fresh install silently loses this and webgpu breaks
// again with no obvious link back to this fix.
const fs = require('fs');
const path = require('path');

function patch(file, replacements) {
  const p = path.join(__dirname, '..', file);
  let code = fs.readFileSync(p, 'utf-8');
  for (const [before, after, label] of replacements) {
    const count = code.split(before).length - 1;
    if (count === 0) {
      console.warn(`  SKIP (pattern not found, already patched or file changed): ${label}`);
      continue;
    }
    code = code.split(before).join(after);
    console.log(`  patched: ${label} (${count} occurrence(s))`);
  }
  fs.writeFileSync(p, code);
}

console.log('patching onnxruntime-web/dist/ort.webgpu.min.js...');
patch('node_modules/onnxruntime-web/dist/ort.webgpu.min.js', [
  [
    'en=()=>{if(!!1)return typeof document<"u"?document.currentScript?.src:typeof self<"u"?self.location?.href:void 0}',
    'en=()=>void 0',
    'self-location detection (document.currentScript/self.location) -- crashed as "Failed to construct URL: Invalid URL" in DCP\'s worker sandbox; unneeded since we supply wasmBinary directly',
  ],
]);

// The .js patch above didn't clear the crash -- DCP's own local job.requires()
// bundler apparently resolves the "import" condition for 'onnxruntime-web/webgpu'
// (an .mjs file), not the "require"/.js one, regardless of us using require()
// syntax ourselves. Both .mjs variants have the SAME self-location pattern,
// just via `import.meta.url` instead of document.currentScript/self.location --
// webpack can't preserve import.meta.url semantics when compiling ESM source
// down to bravojs's non-ESM module.declare() wrapper, so it likely resolves to
// something non-URL-like, and `new URL(...)` throws before any of our own code
// runs. Patch both, since which one actually gets picked isn't directly
// observable from here.
console.log('patching onnxruntime-web/dist/ort.webgpu.min.mjs...');
patch('node_modules/onnxruntime-web/dist/ort.webgpu.min.mjs', [
  [
    'if(en){let t=URL;return new URL(new t("ort.webgpu.min.mjs",import.meta.url).href,cr).href}return import.meta.url',
    'return import.meta.url',
    'self-location via import.meta.url (min.mjs)',
  ],
]);

console.log('patching onnxruntime-web/dist/ort.webgpu.bundle.min.mjs...');
patch('node_modules/onnxruntime-web/dist/ort.webgpu.bundle.min.mjs', [
  [
    'if(tn){let a=URL;return new URL(new a("ort.webgpu.bundle.min.mjs",import.meta.url).href,as).href}return import.meta.url',
    'return import.meta.url',
    'self-location via import.meta.url (bundle.min.mjs)',
  ],
]);

console.log('done');
