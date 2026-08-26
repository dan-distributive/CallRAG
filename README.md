# CallRAG: mp3 call recordings → searchable, LLM-queryable knowledge, on DCP

A pipeline that ingests mp3 call recordings, transcribes and diarizes and
embeds them via DCP-dispatched jobs (local models, no cloud STT/embedding
APIs), and lets you query across the whole backlog from a local viewer.

Built from scratch in one session, restarting an earlier, abandoned attempt
at the same idea. This document exists because almost every step of getting
transformers.js to actually run inside a DCP worker's browser sandbox hit a
real, non-obvious wall — recorded here so the next person (or the next
session) doesn't have to rediscover them.

## Setup

```
npm install
node scripts/prepare-models.js   # downloads + embeds the models below (~314MB of local cache; one-time)
```

You'll also need a DCP identity/wallet — see `dcp-client`'s own docs if you
don't already have one; `ingest.js` currently has a placeholder private key
and `computeGroups: [{ joinKey: 'ibm', joinSecret: 'dcp' }]` hardcoded near
the top, which you'll want to change to your own.

```
node ingest.js path/to/your/mp3s     # dispatches one DCP job, transcribes+diarizes+embeds every new file
node server.js                        # viewer + query UI at http://localhost:8177
```

`node query.js "<question>"` also works standalone from the terminal if you
don't want the viewer. Set `ANTHROPIC_API_KEY` in the environment first if
you want LLM-synthesized answers rather than just raw retrieved excerpts
(both `query.js` and the viewer's query box work fine without it — they
just skip straight to showing retrieval results).

## Pipeline stages

```
mp3 file
  -> decode (minimp3-wasm)              8kHz-or-whatever mono PCM
  -> resample to 16kHz                  (required -- see Gotcha 9)
  -> diarize (pyannote-segmentation-3.0) speaker turns, chunked 30s windows
  -> voice-fingerprint each turn (wavlm) 512-dim embeddings, matched against known named voices
  -> VAD (pure JS energy-based)          trims silence before transcription
  -> transcribe (Whisper base.en)        segments with text + timestamps
  -> assign speakers                     match each segment to a diarization turn (+ voice match, if any)
  -> embed each segment (bge-small)      384-dim vectors for retrieval
  -> persist to results/*.json           one file per recording, per stage
```

All of the above runs as **one DCP job, one slice per recording file** —
see `ingest.js`. Query-time retrieval (`query.js` / `server.js`) runs
**locally**, not as a DCP job — embedding a single question is cheap enough
that dispatch overhead isn't worth it.

## The model stack, and what else fits

| Stage | Current model | Size (as shipped) | Bigger alternative | Size | Smaller alternative | Size |
|---|---|---|---|---|---|---|
| Transcription | `Xenova/whisper-base.en` (uint8) | ~100MB | whisper-small.en | ~237MB | whisper-tiny.en | ~57MB |
| | | | whisper-medium.en | ~740MB | | |
| Diarization | `onnx-community/pyannote-segmentation-3.0` (fp32) | 7.6MB | *(no bigger segmentation variant — better diarization means adding a speaker-embedding model, not a bigger segmentation model)* | | int8/quantized | ~1.5MB |
| Text embedding | `Xenova/bge-small-en-v1.5` (q8) | ~46MB | bge-base-en-v1.5 | ~105MB | all-MiniLM-L6-v2 | ~22MB |
| Speaker identity | `Xenova/wavlm-base-plus-sv` (q8) | ~129MB | — (fp32, ~384MB) | ~384MB | — | — |

Swapping any of these is mechanically cheap — `dtype` and `modelName` are
already parameters on `transcribeAudio()`/`embedText()`, and every model
ships as a `{filename: base64}` map. The expensive part is re-validating
the model against the gotchas below (QDQ crash, size ceiling, `device`
value), not the code change itself.

## Current per-worker payload

Every `ingest.js` job dispatch ships this much data to whichever worker
picks up a slice, split across two transports (see Gotcha 3):

| Transport | Contents | Size |
|---|---|---|
| `job.requires()` (webpack-bundled locally, then shipped) | `base64.js`, `decodeMp3.js` + `decoderWasmBase64.js`, `resample.js`, `vad.js`, `transcribeAudio.js`, `embedText.js`, `diarize.js`, `speakerEmbed.js`, `setupOrt.js` + **`ortWasmAsyncifyBase64.js`**, `modelFetchPatch.js`, `polyfills.js` | **~30.0MB** (almost entirely the onnxruntime-web wasm binary) |
| Job arguments (kvin-serialized, not bundled) | whisper-base.en, bge-small, pyannote-segmentation, wavlm-base-plus-sv model files + known voice profiles | **~281.6MB** (~100.4MB + ~44.2MB + ~7.6MB + ~129.4MB) |
| **Total per slice** | | **~311.6MB** |

That's a lot of data re-sent per file. It's not currently cached
worker-side across slices of the same job (each slice is an independent
sandbox) — a real optimization opportunity if ingesting a large backlog
gets slow on the dispatch/upload side rather than the compute side.

## Gotchas, in the order they actually bit

1. **`job.requires(['./local-file'])` triggers a real webpack build at
   dispatch time** — this is the whole point of the local-modules approach
   (vs. the abandoned prior attempt's manual bundle-then-string-inject),
   but it means local dependency resolution has to actually work, and it
   has a real memory ceiling (see #3).

2. **DCP's worker sandbox fetch() is origin-allowlisted (`EPERM_ORIGIN`)** —
   blocks huggingface.co, jsdelivr.net, and even the library's own default
   wasm-loading fetches. Fix: bundle every model file locally as base64 and
   monkey-patch `globalThis.fetch` (`modelFetchPatch.js`) to serve them by
   filename match, falling through to real fetch otherwise.
   - **Sub-gotcha**: naive substring matching on filenames is wrong —
     `"config.json"` is a substring of `"generation_config.json"`,
     `"tokenizer_config.json"`, and `"preprocessor_config.json"`, so a
     first-match approach silently served the wrong file's bytes. Fixed by
     matching longest-filename-first.

3. **`job.requires()`'s local bundler has a real, lower-than-expected
   memory ceiling** — a ~195MB embedded base64 string crashed it
   (`ERR_WORKER_OUT_OF_MEMORY` / an opaque `webpack: child process
   returned exit code 1`), but ~100MB was also too much once combined with
   other job.requires() modules in the same job. Fix: ship large model
   files as **job arguments** instead (`compute.for(inputSet, workFn,
   [modelFiles])` — JSON/kvin transport, not webpack-parsed) — a
   completely different size ceiling.

4. **`onnxruntime-web`'s self-location logic crashes in the sandbox**
   (`"Failed to construct 'URL': Invalid URL"`, from `document.currentScript`
   / `self.location` / `import.meta.url` lookups that don't resolve the
   way the library expects). Fixed by supplying the wasm binary directly
   (`env.backends.onnx.wasm.wasmBinary`) so the library's own fetch-based
   loader never runs at all.
   - **Sub-gotcha**: this must be set on transformers.js's *own* exported
     `env`, not a separately-`require()`'d `onnxruntime-web` instance — a
     separate import can resolve to a different physical module instance
     (CJS vs ESM), silently making the override invisible.

5. **Local (`node`) testing resolves a different physical build than the
   real (browser-hosted) DCP worker** — local Node → `onnxruntime-node`
   (`device: 'cpu'`/`'coreml'`/`'webgpu'`), real worker → `onnxruntime-web`
   (`device: 'wasm'`/`'webgpu'`). These are not interchangeable; using the
   wrong one throws immediately. **Local testing cannot validate which
   value is correct for the real worker** — only real dispatch can.

6. **WebGPU crashes uncatchably** — `device: 'webgpu'` reaches real GPU
   setup but crashes inside onnxruntime-web's JSEP session creation
   (`"Cannot convert undefined to a BigInt"`), escaping normal
   `try/catch` entirely (uncaught at the job level). Shelved; CPU/`wasm`
   only ships. The `ort-wasm-simd-threaded.jsep.wasm` binary also breaks
   the *CPU* path if used there by mistake — it's not a safe CPU+GPU
   superset despite its naming; use `.asyncify.wasm` for CPU specifically.

7. **QDQ (quantized) ONNX graphs crash session creation on this sandbox's
   onnxruntime-web** — `qdq_actions.cc:137 TransposeDQWeightsForMatMulNBits
   ... Missing required scale`. Hit identically at every quantized dtype
   tried (q8, uint8), and even at fp32 for Whisper specifically (its
   "merged" decoder graph uses a `DequantizeLinear` node for weight-sharing
   between its with/without-past branches, regardless of overall dtype).
   **Root cause: an ONNX Runtime graph-optimization fusion pass bug**, not
   a hard QDQ incompatibility. Fix: `session_options: {
   graphOptimizationLevel: 'disabled' }` in the `pipeline()` call — one
   line, no correctness downside (slightly slower inference), resolved
   what looked like a deep architectural wall.

8. **A tokenizer with no configured `model_max_length` makes
   `truncation: true` silently a no-op** — `bge-small-en-v1.5` has no
   `model_max_length` in its `tokenizer_config.json`, so transformers.js's
   internal truncation never actually caps input length. A 555-token
   Whisper hallucination overflowed the model's 512-token position
   embeddings and crashed **the entire file's** embedding job (all
   segments lost, not just the bad one). Fixed with defensive
   character-length truncation before the model ever sees the text, plus
   per-segment `try/catch` so one bad segment can never again cost a whole
   file.

9. **transformers.js's audio pipelines don't resample a raw array input**
   — `prepareAudios()` only calls `read_audio()` (which resamples) when
   given a URL/string; a plain `Float32Array` is passed straight to the
   model, silently assumed to already be at the model's expected rate.
   Our mp3s decode at whatever rate they're actually encoded at (8kHz for
   these telephony recordings) — feeding that to a 16kHz-expecting model
   doubles apparent pitch/speed and halves every reported timestamp.
   **This was silently corrupting every transcription in the project until
   found** — confirmed by exact math: a 30s/8kHz clip's Whisper transcript
   only ever reached ~15s (240,000 samples / 16000 assumed-Hz = 15s).
   Fixed with an explicit linear-interpolation resampler (`resample.js`)
   applied right after decode.

10. **A model with no built-in chunking will crash (or silently misbehave)
    on unexpectedly-short input** — `pyannote-segmentation-3.0`'s SincNet
    conv stack can't handle a very short trailing remainder
    (`Invalid input shape: {1}` from a 478-sample leftover chunk after
    naive fixed-window chunking). Fixed by absorbing any
    too-short trailing remainder into the previous chunk instead of
    processing it alone.

11. **DCP's scheduler expects `progress()` calls at least every ~30s, or
    it can kill the job (`ENOPROGRESS`)** — and even short of that, a
    long-running stage with no progress calls looks indistinguishable from
    a frozen worker to anyone watching the portal. A single Whisper
    `generate()` call over a multi-minute recording, or one un-chunked
    diarization forward pass, can easily run that long. Fixed by hooking
    a custom `BaseStreamer` into Whisper's `generate()` (fires per
    generated token) and chunking diarization with a progress callback
    between windows.

12. **`compute.RemoteDataSet([url])` fetches remote data as UTF-8 text, not
    raw bytes** — confirmed by fetching a real mp3 from an
    origin-allowlisted S3 bucket and finding the result was byte-for-byte
    identical to Python's `bytes.decode('utf-8', errors='replace')` on the
    same source bytes: lossy, irreversible, ~290KB of a 6.39MB file
    silently gone. No binary/arraybuffer option exists in `RemoteDataSet`
    as of this dcp-client version. **Origin-allowlisting itself works
    fine** (confirmed separately) — the failure is purely in how the
    fetched bytes get deserialized.

13. **Passing a raw Node `Buffer` as a job argument does not preserve it**
    — tested directly (all 256 byte values + a real mp3 chunk): it arrived
    on the worker as an **empty plain object**, not a `Buffer` or
    `Uint8Array`. kvin does not transparently round-trip `Buffer` objects.
    **Conclusion**: the base64-string approach used throughout this
    project for binary data isn't unnecessary overhead — it's currently
    the *only* working transport for binary data through DCP's job
    dispatch or remote-data-fetch paths.

14. **A "keep-alive" progress callback with a fixed value defeats its own
    purpose.** Whisper's per-token `streamer` callback (added for Gotcha
    11's `ENOPROGRESS` fix) initially called `progress(0.4)` on every
    token — technically satisfies the scheduler's liveness requirement, but
    to anyone watching the portal, a percentage that repeats exactly for
    minutes looks identical to a genuinely frozen worker (indistinguishable
    without checking timestamps). Fixed by climbing asymptotically toward
    the next real milestone as tokens accumulate, since the total token
    count isn't knowable in advance the way diarization's/speaker-embedding's
    chunk counts are.

15. **`job.requirements.environment = { webgpu: true }` silently shrinks the
    eligible worker pool to almost nothing, even though nothing in the job
    uses WebGPU.** This pipeline runs on `device: 'wasm'` (CPU) everywhere —
    WebGPU was investigated and shelved (Gotcha 6) — but a leftover
    requirement from that investigation kept requiring WebGPU-*capable*
    workers to even be offered the job at all. The result looks exactly
    like a hung dispatch: `readystatechange` never advances, zero console
    output, near-zero local CPU usage, for 50+ minutes — genuinely
    indistinguishable from an empty/unreachable compute group without
    knowing to suspect this specific line, since the job isn't stuck, it's
    just waiting for a worker capability that almost nothing has and that
    the job never actually needed. Fixed by removing the requirement
    entirely (commented out, not deleted, in case a real future WebGPU
    attempt wants it back).

## Speaker identity (voice-fingerprint matching)

`pyannote-segmentation-3.0` alone only answers "when did the speaker
change" with local per-call slot IDs — "speaker 2" in one call is not the
same physical person as "speaker 2" in another. `speakerEmbed.js` adds the
other half: a `Xenova/wavlm-base-plus-sv` voice-fingerprint embedding per
diarization turn, matched (cosine similarity) against a store of
known, *named* voices (`results/voiceProfiles.json`, built up via the
viewer, keyed by name and shared across every recording).

Naming a speaker anywhere in the viewer (the main panel's legend, or the
dedicated **Needs Review** panel — a cross-recording queue of every
unmatched or low-confidence clip, with an adjustable confidence slider and
inline audio playback) enrolls or updates that name's profile as a running
average from that speaker's own embeddings. Every subsequent `ingest.js`
run loads the current profile store fresh and passes it into the worker,
so **later recordings recognize the same voice automatically** — validated
end to end, not just built: enrolling one clip as a name, then re-ingesting
a different recording of the same speaker from scratch, produced
0.89–1.00 confidence auto-matches on every segment with zero manual input.

This only applies going forward: recordings ingested before this feature
existed have no `voiceEmbedding` data to enroll from, so folding older
files into voice-based recognition means re-ingesting them.

## Chat, and swapping in a different LLM

Retrieval and synthesis run entirely locally except for one optional
network call: the actual LLM chat completion, made through `llm.js`, a
provider-agnostic interface. Default is Claude (`LLM_PROVIDER=anthropic`,
the default if unset), reading `ANTHROPIC_API_KEY` from the environment.
Set `LLM_PROVIDER=local` and `LLM_BASE_URL` to point at any OpenAI
chat-completions-compatible server instead — Ollama, vLLM, llama.cpp's
server mode, and LM Studio all implement that shape natively, so this
should work as a drop-in swap with no code changes. **Caveat:** the local
path is structurally complete but was built and tested without an actual
local LLM server available — the request/response shape matches the
documented OpenAI spec, but "matches the spec" isn't the same as "verified
against a real server."

`queryEngine.js` is the shared retrieval+synthesis logic behind both the
viewer's chat panel and the `query.js` CLI (previously duplicated between
them — now one module, one behavior). Rather than one fixed top-K
retrieval pass, the LLM gets a `search_calls` tool and can call it again
(up to 4 rounds) to dig further on its own — the single highest-value
change identified when planning this, since a fixed top-K often can't
surface a topic thinly scattered across many calls, which is exactly the
kind of query this project was built toward. The chat panel also carries
conversation history across turns for follow-up questions, sent fresh with
each request (client keeps history, server is stateless per-request).

Two things about the LLM call specifically that shape the architecture:
it cannot move into the browser the way the retrieval/embedding half
conceivably could (Anthropic's API doesn't send CORS headers permitting
direct browser calls, and putting an API key in page source would expose
it to anyone opening devtools), and none of the rest of the app — browsing,
playback, corrections, review queue, voice-space, performance — needs a
key at all; only the synthesized-answer step does.

## Known limitations

- **Voice matching is per-named-profile, not perfect identity resolution.**
  It's cosine-similarity matching against whatever's been enrolled — two
  similar-sounding voices can cross-match, and a voice with very little
  enrolled audio matches less reliably. The Needs Review panel's confidence
  threshold is there specifically to catch and correct this.
- **No true waveform visualization** — the viewer's timeline shows
  diarization turns as colored blocks, not actual audio amplitude.
- **The "vector store" is flat JSON files, one per recording per stage** —
  fine for dozens of files; `query.js`/`server.js` load and brute-force
  scan every embedded file on every query. Will need a real index
  (SQLite + a vector extension, or similar) before this scales to a large
  backlog.
- **Residual transcription hallucination** — much reduced by the
  whisper-base.en upgrade and the sample-rate fix, but Whisper's
  repetition-loop failure mode on quiet/low-confidence audio hasn't been
  eliminated, only reduced.
- **~312MB re-shipped per worker per file** (see payload table above) —
  no cross-slice model caching exists yet.

## File reference

- `ingest.js` — the main entry point: batch transcribe+diarize+embed a
  directory of mp3s via one DCP job.
- `query.js` — CLI for the same retrieval+synthesis as the viewer's chat
  panel (no DCP dispatch).
- `server.js` — backs `viewer.html`: recordings, playback, corrections,
  voice enrollment, Needs Review, voice-space visualization, performance
  timings, and the `/api/query` chat endpoint.
- `queryEngine.js` — shared retrieval + agentic (tool-use) synthesis logic
  used by both `query.js` and `server.js`.
- `llm.js` — provider-agnostic chat interface (Claude by default, or any
  OpenAI-compatible local server) that `queryEngine.js` calls through.
- `decodeMp3.js`, `resample.js`, `vad.js`, `transcribeAudio.js`,
  `embedText.js`, `diarize.js`, `speakerEmbed.js` — independently-reusable
  pipeline stage modules, each taking model files as a parameter rather
  than importing them, so any module can ship via `job.requires()`, a job
  argument, or a published package without the others knowing.
- `modelFetchPatch.js`, `setupOrt.js`, `polyfills.js`, `base64.js`,
  `callDate.js` — shared DCP-sandbox-compatibility shims and small utilities
  (see Gotchas 2, 4, 9 in spirit).
- `scripts/prepare-models.js` — downloads and embeds the model bundles
  (gitignored — see Setup above).
- `stage0`–`stage4` scripts — the original incremental proofs (basic
  dispatch → mp3 decode → transformers.js pipeline → full pipeline),
  kept as standalone regression checks for individual stages.
