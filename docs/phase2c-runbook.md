# OmniLlama — Phase 2c/2d Runbook (run on Goliath the moment the 3090 is vacant)

Everything below is staged and drop-in-ready. Target: **convert → build → proper test** with no
authoring in the window. Current 3090 job ends ~48h from 2026-10-16. Run on **Goliath** (CUDA, model
local via E:, llama.cpp at `X:\llamacpp`).

> ⚠️ **SeedLora (the month-long run) was killed by a Windows auto-reboot. Do STEP 0 FIRST** so the same
> thing can't nuke this work (or the next training). Then consider restarting SeedLora from its last
> checkpoint once we're off the 3090.

## STEP 0 — Guard against Windows auto-reboot (admin PowerShell on Goliath)
```powershell
# Pause Windows Update ~5 weeks + block auto-reboot while logged in
$d=(Get-Date).AddDays(35).ToString('yyyy-MM-ddTHH:mm:ssZ')
reg add "HKLM\SOFTWARE\Microsoft\WindowsUpdate\UX\Settings" /v PauseUpdatesExpiryTime /t REG_SZ /d $d /f
reg add "HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU" /v NoAutoRebootWithLoggedOnUsers /t REG_DWORD /d 1 /f
reg add "HKLM\SOFTWARE\Policies\Microsoft\Windows\WindowsUpdate\AU" /v AUOptions /t REG_DWORD /d 2 /f  # notify, don't auto-install
# Disable the Update Orchestrator reboot task (re-enable later)
schtasks /Change /TN "\Microsoft\Windows\UpdateOrchestrator\Reboot" /DISABLE 2>$null
Get-Service wuauserv | Stop-Service -Force; Set-Service wuauserv -StartupType Manual
```
Re-enable updates after the test window if desired. (See [[reference_dotnet_update_failure]] for the
recurring Win-Update failure context.)

## STEP 1 — Snapshot X:\llamacpp (it is NOT a git repo → no undo otherwise)
```powershell
robocopy "X:\llamacpp" "X:\_backup\llamacpp_pre_omnillama" /MIR /XD build .git /NFL /NDL /NJH /NJS
# or: cd X:\llamacpp; git init; git add -A; git commit -m "pre-OmniLlama snapshot"
```

## STEP 2 — Apply the OmniLlama patches (from F:\VC Projects\OmniLlama\src)
1. `copy omnillama_voice_convert.py  X:\llamacpp\conversion\`
2. `gguf-py/gguf/constants.py`: add `A_GEN_WAV_CODE_EMBD` enum + name `"a.gen.wav.code_embd"` (see code2wav_omni_variant.md §1).
3. `conversion/__init__.py`: register the two Omni classes so the converter can select them for
   `Qwen3OmniMoeForConditionalGeneration` (Talker text model + voice-gen mmproj), mirroring how
   `qwen3vl` exposes the Thinker text + mmproj classes. Confirm the dispatch/CLI flag that picks each.
4. `tools/mtmd/models/qwen3tts-gen.cpp` + hparams + clip loader: apply the flat-embed decode variant
   (code2wav_omni_variant.md §2–4). Make `pre_conv`/`tfm_in_proj`/`tfm_out_proj` nullable.

## STEP 3 — Convert (produces the GGUFs)
```powershell
cd X:\llamacpp
# Thinker (already supported: qwen3vl path) -> text GGUF + mmproj (image/audio INPUT)
python convert_hf_to_gguf.py "Z:\Qwen3-Omni-30B-A3B-Instruct" --outfile "X:\models\omnillama\thinker-f16.gguf"
python convert_hf_to_gguf.py "Z:\Qwen3-Omni-30B-A3B-Instruct" --mmproj --outfile "X:\models\omnillama\mmproj-f16.gguf"
# Talker (QWEN3TTS text) + voice-gen (CodePredictor + Code2Wav) -> via the Omni classes
#   exact flags depend on the __init__.py dispatch wired in STEP 2.3; produce:
#   talker-f16.gguf  and  voicegen-f16.gguf
```
Expect: first run surfaces missing/extra-tensor or dtype errors → fix the converter map, re-run. Keep
voice tensors F16/F32 (parent `tensor_force_quant` rules; vocoders break low-bit).

## STEP 4 — Build llama.cpp (CUDA) on Goliath
```powershell
cmake -S X:\llamacpp -B X:\llamacpp\build -DGGML_CUDA=ON -DCMAKE_BUILD_TYPE=Release
cmake --build X:\llamacpp\build --target llama-tts llama-mtmd-cli llama-cli -j
```
Fix compile errors from the C++ variant (expect 1–3 rounds; the ggml int-offset in quant_decode_flat
is the likeliest fix point — see code2wav_omni_variant.md §3 note).

## STEP 5 — PROPER TEST (demonstrable = a WAV you can hear)
**5a. Code2Wav in isolation** (validates the one new C++): feed a known 16×T codec-token sequence to the
voice-gen model → a WAV. Compare to a transformers reference WAV for the same codes (bit-rough is fine;
listen for coherent audio, not noise). This proves the flat-embed vocoder.

**5b. End-to-end** (the real milestone): Thinker (text prompt) → Talker → Code Predictor → Code2Wav → WAV.
```powershell
X:\llamacpp\build\bin\llama-mtmd-cli -m X:\models\omnillama\talker-f16.gguf `
   --mmproj X:\models\omnillama\voicegen-f16.gguf --tts --prompt "Say: the beast is online." -o out.wav
# (exact flags per the llama-tts/mtmd gen interface; adapt from tools/tts/README.md + mtmd gen docs)
```
Success = `out.wav` is intelligible speech. That is "done" per [[user_nonprogrammer]] (demonstrable).

**5c.** Only after 5b: quantize the Thinker (imatrix IQ3) for the 580, keep voice F16/Q8; re-test.

## If it doesn't converge in the window
Fall back to the de-risk ladder: get **5a** (Code2Wav WAV) first — it isolates the only new C++ and is the
single most informative result. If 5a produces audio, the architecture is proven and the rest is wiring.

## After the test
Re-enable Windows Update (reverse STEP 0) if you want it back on, and restart SeedLora from its last
checkpoint when the 3090 is free again.
