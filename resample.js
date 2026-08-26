// Linear-interpolation resampler. transformers.js's ASR/audio pipelines only
// resample when given a URL/string (via read_audio) -- a raw Float32Array is
// passed straight to the model, silently assumed to already be at the
// model's expected rate (see prepareAudios in
// node_modules/@huggingface/transformers/src/pipelines/_base.js). Our mp3s
// decode at whatever rate they're actually encoded at (8kHz for these
// telephony recordings), so every downstream model (Whisper: 16kHz,
// pyannote-segmentation: 16kHz) needs an explicit resample first, or it
// silently treats 8kHz audio as if it were 16kHz -- doubling apparent
// playback speed/pitch and halving every reported timestamp. Confirmed as a
// real bug (not hypothetical): a 30s/8kHz test clip's Whisper transcript
// only ever reached ~15.7s, exactly half -- see dcp_transformers_restart memory.
function resample(samples, fromRate, toRate) {
  if (fromRate === toRate) return samples;
  const ratio = fromRate / toRate;
  const outLength = Math.round(samples.length / ratio);
  const out = new Float32Array(outLength);
  for (let i = 0; i < outLength; i++) {
    const srcPos = i * ratio;
    const i0 = Math.floor(srcPos);
    const i1 = Math.min(i0 + 1, samples.length - 1);
    const frac = srcPos - i0;
    out[i] = samples[i0] * (1 - frac) + samples[i1] * frac;
  }
  return out;
}

module.exports = { resample };
