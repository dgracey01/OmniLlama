# mtmd.cpp fork fixes (OmniLlama)

**File:** `X:\llamacpp\tools\mtmd\mtmd.cpp` (`\\GOLIATH\i\llamacpp\...` = `I:\llamacpp\...` on Goliath)

`X:\llamacpp` is a **non-git clone**, so an engine pull/overwrite can revert these.
Both edits are tagged in-source with `// OmniLlama:` — grep for that tag to confirm they survive.
Needed so a **gen-audio-only mmproj** (the Omni voice stack: MTP + Code2Wav, no input/speaker
encoder) loads for text→speech. Without them: Fix 1 → "Failed to load CLIP model"; Fix 2 → SEGV
0xC0000005 (null deref). Discovered/validated during 5b-4 (2026-10-28).

## Fix 1 — accept a gen-audio-only mmproj (~line 586)

Region: right after `ctx_gen_a = res.ctx_gen_a;`

```cpp
// OmniLlama: a gen-audio-only mmproj (ctx_gen_a, e.g. the Omni voice stack with no input/speaker
// encoder) is valid for text->speech; only reject when nothing at all loaded.
if (!ctx_v && !ctx_a && !ctx_gen_a) {
    throw std::runtime_error(string_format("Failed to load CLIP model from %s\n", mmproj_fname));
}
```

**Upstream (reverted) form:** `if (!ctx_v && !ctx_a) {` — add `&& !ctx_gen_a`.

## Fix 2 — guard the input-encoder n_embd check (~line 603)

The original dereferenced `ctx_v ? ctx_v : ctx_a` unconditionally, which is null for a gen-only
mmproj. Wrap it in `if (ctx_v || ctx_a)`. The gen-audio n_embd is validated separately in the
`if (ctx_gen_a)` block just below (that block is upstream, unchanged).

```cpp
// since we already validate n_embd of vision and audio mmproj,
// we can safely assume that they are the same
// OmniLlama: skip for a gen-audio-only mmproj (no input encoder) — the gen-audio n_embd is
// validated separately below; the original ctx_v?ctx_v:ctx_a would deref null here.
if (ctx_v || ctx_a) {
    int n_embd_clip = clip_n_mmproj_embd(ctx_v ? ctx_v : ctx_a);
    if (n_embd_text > 0 && n_embd_text != n_embd_clip) {
        throw std::runtime_error(string_format(
            "mismatch between text model (n_embd = %d) and mmproj (n_embd = %d)\n"
            "hint: you may be using wrong mmproj\n",
            n_embd_text, n_embd_clip));
    }
}
```

**Upstream (reverted) form:** the body above with no `if (ctx_v || ctx_a) { ... }` wrapper
(i.e. `int n_embd_clip = clip_n_mmproj_embd(ctx_v ? ctx_v : ctx_a);` at top level).

## Re-apply check (after any engine pull)

```
grep -c "OmniLlama:" X:\llamacpp\tools\mtmd\mtmd.cpp    # expect >= 2
```
If 0, both fixes were reverted — re-apply the two blocks above, then rebuild
`llama-talker-tts-test`.

## Also fork-local (same non-git revert risk)

New tool sources + CMake targets, all mirrored in `F:\VC Projects\OmniLlama\src\`:
- `tools/mtmd/talker-embed-test.cpp`  (5b-3 teacher-forced harness)
- `tools/mtmd/talker-tts-test.cpp`     (5b-4 end-to-end text→WAV orchestrator)
- `tools/mtmd/CMakeLists.txt` targets: `llama-talker-embed-test`, `llama-talker-tts-test`
