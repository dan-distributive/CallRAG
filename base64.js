// Dependency-free base64 decode -- avoids assuming atob() or Buffer are
// available in the actual worker runtime (uncertain: workers may be
// browser-hosted regardless of the job author's own Node platform).
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

module.exports = { base64ToBytes };
