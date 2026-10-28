// OmniLlama 5b-4: end-to-end Omni voice orchestrator (text-fixed) -> WAV.
//
// Ties the verified pieces together: the Talker backbone GGUF (llama, arch qwen3omnitalker) drives the
// per-frame loop, and the voice mmproj (clip/mtmd) supplies the MTP code predictor + Code2Wav vocoder via
// the existing GEN_CODE / GEN_WAV processes. Per frame: Talker decode -> greedy codebook-0 + last hidden
// -> GEN_CODE(h_state, cb0) -> 16 codes + feedback embd (cb0 embed + sum of 15 residual-codebook embeds)
// -> add the text stream overlay (trailing_text_hidden[step], then tts_pad) -> feed back to the Talker.
// Codes are windowed through GEN_WAV (code2wav, 24 kHz) into PCM.
//
// This uses the 5b-2 gold-reference prompt embeds (fixed sentence) so no thinker/text_projection front-end
// is needed yet; arbitrary-text input (thinker embed_tokens -> text_projection) is a later step (5b-5).
// Greedy cb0 matches the reference's deterministic Talker; the MTP samples (seeded here for reproducibility),
// so the WAV is coherent Omni speech from the same prompt, not a bit-identical copy of the reference.
//
// usage: llama-talker-tts-test <talker.gguf> <mmproj.gguf> <ref_dir> <out.wav>

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper-common.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

static std::vector<float> read_f32(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "FAIL: cannot open %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<float> v(n / sizeof(float));
    if (fread(v.data(), 1, n, f) != (size_t) n) { fprintf(stderr, "FAIL: short read %s\n", p.c_str()); exit(1); }
    fclose(f); return v;
}

static bool write_wav16(const std::string & path, const std::vector<float> & pcm, int rate) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t data_sz = (uint32_t) (pcm.size() * 2);
    const uint32_t riff_sz = 36 + data_sz, fmt_sz = 16, byte_rate = (uint32_t) rate * 2, rate32 = (uint32_t) rate;
    const uint16_t fmt = 1, ch = 1, align = 2, bits = 16;
    auto put = [&](const void * p, size_t n){ fwrite(p, 1, n, f); };
    put("RIFF",4); put(&riff_sz,4); put("WAVE",4); put("fmt ",4); put(&fmt_sz,4);
    put(&fmt,2); put(&ch,2); put(&rate32,4); put(&byte_rate,4); put(&align,2); put(&bits,2);
    put("data",4); put(&data_sz,4);
    for (float v : pcm) { int16_t s = (int16_t) (std::max(-1.0f, std::min(1.0f, v)) * 32767.0f); put(&s,2); }
    fclose(f); return true;
}

int main(int argc, char ** argv) {
    if (argc < 5) { fprintf(stderr, "usage: %s <talker.gguf> <mmproj.gguf> <ref_dir> <out.wav>\n", argv[0]); return 1; }
    const std::string gguf = argv[1], mmproj = argv[2], dir = argv[3], outwav = argv[4];
    const std::string sep = (dir.back() == '/' || dir.back() == '\\') ? "" : "/";
    auto RP = [&](const char * n){ return dir + sep + n; };

    const int   codec_eos   = 2150;        // stop token (verified 5b-3)
    const int   suppress_lo = 2048;        // reserved range masked (except eos) when picking cb0
    const int   max_frames  = 600;
    const size_t window_frames = 72;       // == hparams.wav_tfm_swa, mirrors the TTS helper's flush window
    const uint32_t seed     = 1234;

    std::vector<float> prompt   = read_f32(RP("talker_input_embed_f32.bin"));
    std::vector<float> trailing = read_f32(RP("trailing_text_hidden_f32.bin"));
    std::vector<float> pad      = read_f32(RP("tts_pad_embed_f32.bin"));

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(gguf.c_str(), mp);
    if (!model) { fprintf(stderr, "FAIL: talker load\n"); return 1; }
    const int n_embd = llama_model_n_embd(model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);

    const int Pn = (int) (prompt.size()   / n_embd);
    const int R  = (int) (trailing.size() / n_embd);
    fprintf(stderr, "IN: n_embd=%d n_vocab=%d prompt=%d trailing=%d\n", n_embd, n_vocab, Pn, R);

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t) (Pn + max_frames + 8);
    cp.n_batch = cp.n_ctx;
    cp.embeddings = true;                       // we need the last hidden state per step for GEN_CODE
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;  // per-token embeddings, logits still available
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "FAIL: ctx\n"); return 1; }

    mtmd_context_params mctxp = mtmd_context_params_default();
    mctxp.use_gpu = true; mctxp.print_timings = false; mctxp.warmup = false;
    mtmd_context * mctx = mtmd_init_from_file(mmproj.c_str(), model, mctxp);
    if (!mctx) { fprintf(stderr, "FAIL: mmproj init\n"); return 1; }
    mtmd_gen_audio_info info = mtmd_gen_audio_get_info(mctx);
    fprintf(stderr, "MMPROJ: gen_audio_type=%d sample_rate=%d\n", (int) info.type, info.sample_rate);

    // greedy cb0 over real-codec range (+eos), mirroring the reference suppress mask
    auto pick_cb0 = [&](const float * lg)->int {
        int am = 0; float best = -1e30f;
        for (int v = 0; v < n_vocab; ++v) {
            if (v >= suppress_lo && v != codec_eos) continue;
            if (lg[v] > best) { best = lg[v]; am = v; }
        }
        return am;
    };

    // --- prompt decode ---
    decode_embd_batch pb(prompt.data(), Pn, 1, n_embd);
    pb.set_position_normal(0, 0);
    pb.batch.logits[Pn - 1] = 1;
    if (llama_decode(ctx, pb.batch) != 0) { fprintf(stderr, "FAIL: prompt decode\n"); return 1; }
    int pos = Pn;

    std::vector<int32_t> codes_buf;   // flat [frame*16]
    std::vector<float>   pcm;
    std::vector<uint8_t> c2w_state;
    std::vector<float>   fb(n_embd);

    auto flush_wav = [&](bool final_flush)->bool {
        if (codes_buf.empty()) return true;
        const size_t frames = codes_buf.size() / 16;
        if (!final_flush && frames < window_frames) return true;
        mtmd_gen_inp gi = mtmd_gen_inp_default(mctx);
        gi.type = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
        gi.codes = codes_buf.data(); gi.n_codes = codes_buf.size();
        gi.seed = seed;
        gi.state_data = c2w_state.empty() ? nullptr : (const char *) c2w_state.data();
        gi.state_size = c2w_state.size();
        mtmd_gen_out go{};
        if (mtmd_gen_audio_process(mctx, &gi, &go) != 0) { fprintf(stderr, "FAIL: GEN_WAV\n"); return false; }
        pcm.insert(pcm.end(), go.audio, go.audio + go.n_samples);
        c2w_state.assign(go.state_data, go.state_data + go.state_size);
        codes_buf.clear();
        return true;
    };

    int frame = 0, stop_reason = 0;
    for (; frame < max_frames; ++frame) {
        const float * lg = llama_get_logits_ith(ctx, -1);
        if (!lg) { fprintf(stderr, "FAIL: null logits frame %d\n", frame); return 1; }
        const int cb0 = pick_cb0(lg);
        if (cb0 == codec_eos) { stop_reason = 1; break; }

        const float * he = llama_get_embeddings_ith(ctx, -1);
        if (!he) { fprintf(stderr, "FAIL: null embeddings frame %d\n", frame); return 1; }

        mtmd_gen_inp gi = mtmd_gen_inp_default(mctx);
        gi.type = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
        gi.code0 = cb0;
        gi.embd  = const_cast<float *>(he);
        gi.seed  = seed;
        mtmd_gen_out go{};
        if (mtmd_gen_audio_process(mctx, &gi, &go) != 0) { fprintf(stderr, "FAIL: GEN_CODE frame %d\n", frame); return 1; }

        codes_buf.insert(codes_buf.end(), go.codes, go.codes + go.n_codes);
        if (!flush_wav(false)) return 1;

        // next talker input = MTP feedback embd + text-stream overlay (trailing[frame], then tts_pad)
        const float * ov = (frame < R) ? &trailing[(size_t) frame * n_embd] : pad.data();
        for (int d = 0; d < n_embd; ++d) fb[d] = go.embd[d] + ov[d];

        decode_embd_batch sb(fb.data(), 1, 1, n_embd);
        sb.set_position_normal(pos, 0);
        sb.batch.logits[0] = 1;
        if (llama_decode(ctx, sb.batch) != 0) { fprintf(stderr, "FAIL: step decode frame %d\n", frame); return 1; }
        pos++;

        if (frame < 5 || frame % 20 == 0)
            fprintf(stderr, "frame %3d: cb0=%-5d codes[1..3]=%d,%d,%d pos=%d\n",
                    frame, cb0, go.codes[1], go.codes[2], go.codes[3], pos);
    }
    if (!flush_wav(true)) return 1;

    double rms = 0.0; for (float v : pcm) rms += (double) v * v;
    rms = pcm.empty() ? 0.0 : sqrt(rms / pcm.size());
    fprintf(stderr, "GEN DONE: frames=%d stop=%s samples=%zu (%.2fs) rms=%.4f\n",
            frame, stop_reason ? "codec_eos" : "max_frames", pcm.size(),
            pcm.size() / (double) info.sample_rate, rms);

    if (!write_wav16(outwav, pcm, info.sample_rate)) { fprintf(stderr, "FAIL: wav write\n"); return 1; }
    // raw PCM for numeric compare vs the reference
    FILE * pf = fopen((outwav + ".pcm_f32.bin").c_str(), "wb");
    if (pf) { fwrite(pcm.data(), sizeof(float), pcm.size(), pf); fclose(pf); }
    fprintf(stderr, "WROTE %s (%zu samples)\n", outwav.c_str(), pcm.size());

    if (frame > 0 && !pcm.empty() && rms > 0.005) fprintf(stderr, "TALKER_TTS_TEST_PASS\n");
    else                                          fprintf(stderr, "TALKER_TTS_TEST_FAIL\n");

    mtmd_free(mctx);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
