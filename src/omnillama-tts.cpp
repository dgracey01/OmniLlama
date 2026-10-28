// OmniLlama 5b-5 Tier-2: SELF-CONTAINED Omni text->speech in llama.cpp (no Python / no transformers).
//
//   omnillama-tts <frontend.gguf> <talker.gguf> <voice_mmproj.gguf> "<text>" <out.wav> [speaker]
//
// Front-end GGUF (arch qwen3vlmoe, built by conversion/omnillama_textfe.py) carries: the Qwen tokenizer,
// token_embd [152064,2048], the talker text_projection (ResizeMLP 2048->2048 SiLU ->1024), and
// codec_special_embd [9,1024] (the talker codec-embedding rows for the Omni special ids). This tool
// reproduces build_talker_prompt.py's assembly in C++ (tokenize chatml -> embed -> text_projection ->
// assistant_parts layout) then runs the verified per-frame Talker->MTP->Code2Wav loop (see
// talker-tts-test.cpp) to a WAV. Greedy cb0; MTP sampling seeded.
//
// speaker in {ethan(default), chelsie, aiden}.

#include "llama.h"
#include "mtmd.h"
#include "mtmd-helper-common.h"
#include "ggml.h"
#include "gguf.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

// ---- Omni constants (verified from config via probe_textproj.py / probe_chatml.py) ----
static const int TOK_IM_START = 151644, TOK_SYSTEM = 8948, TOK_USER = 872, TOK_ASSISTANT = 77091;
static const int TTS_BOS = 151672, TTS_EOS = 151673, TTS_PAD = 151671;
static const int CODEC_EOS = 2150, SUPPRESS_LO = 2048;
// codec_special_embd row order baked by the converter:
// 0 nothink 1 think_bos 2 think_eos 3 pad 4 bos 5 eos 6 chelsie 7 ethan 8 aiden
enum { CSP_NOTHINK=0, CSP_THINK_BOS=1, CSP_THINK_EOS=2, CSP_PAD=3, CSP_BOS=4, CSP_EOS=5,
       CSP_CHELSIE=6, CSP_ETHAN=7, CSP_AIDEN=8 };

static bool write_wav16(const std::string & path, const std::vector<float> & pcm, int rate) {
    FILE * f = fopen(path.c_str(), "wb");
    if (!f) return false;
    const uint32_t data_sz = (uint32_t)(pcm.size() * 2);
    const uint32_t riff_sz = 36 + data_sz, fmt_sz = 16, byte_rate = (uint32_t)rate * 2, rate32 = (uint32_t)rate;
    const uint16_t fmt = 1, ch = 1, align = 2, bits = 16;
    auto put = [&](const void * p, size_t n){ fwrite(p, 1, n, f); };
    put("RIFF",4); put(&riff_sz,4); put("WAVE",4); put("fmt ",4); put(&fmt_sz,4);
    put(&fmt,2); put(&ch,2); put(&rate32,4); put(&byte_rate,4); put(&align,2); put(&bits,2);
    put("data",4); put(&data_sz,4);
    for (float v : pcm) { int16_t s = (int16_t)(std::max(-1.0f, std::min(1.0f, v)) * 32767.0f); put(&s,2); }
    fclose(f); return true;
}

// a baked tensor read from the front-end GGUF, held as F32 with ggml ne order {ne0(fast), ne1}
struct FeTensor {
    std::vector<float> data;
    int64_t ne0 = 0, ne1 = 1;
    float   at(int64_t row, int64_t col) const { return data[row * ne0 + col]; } // row in [0,ne1), col in [0,ne0)
};

static FeTensor load_fe_tensor(struct ggml_context * c, const char * name) {
    FeTensor t;
    ggml_tensor * g = ggml_get_tensor(c, name);
    if (!g) { fprintf(stderr, "FAIL: front-end tensor '%s' missing\n", name); exit(1); }
    t.ne0 = g->ne[0]; t.ne1 = g->ne[1] == 0 ? 1 : g->ne[1];
    const int64_t n = t.ne0 * t.ne1;
    t.data.resize((size_t)n);
    if (g->type == GGML_TYPE_F16) {
        const ggml_fp16_t * s = (const ggml_fp16_t *) g->data;
        for (int64_t i = 0; i < n; ++i) t.data[i] = ggml_fp16_to_fp32(s[i]);
    } else if (g->type == GGML_TYPE_F32) {
        memcpy(t.data.data(), g->data, (size_t)n * sizeof(float));
    } else {
        fprintf(stderr, "FAIL: tensor '%s' unexpected type %d\n", name, (int)g->type); exit(1);
    }
    return t;
}

int main(int argc, char ** argv) {
    if (argc < 6) {
        fprintf(stderr, "usage: %s <frontend.gguf> <talker.gguf> <voice_mmproj.gguf> \"<text>\" <out.wav> [speaker]\n", argv[0]);
        return 1;
    }
    const std::string fe_path = argv[1], talker_path = argv[2], mmproj_path = argv[3];
    const std::string spoken  = argv[4], outwav = argv[5];
    const std::string speaker = (argc > 6) ? argv[6] : "ethan";
    const std::string prompt  = "Please read this aloud.";   // user turn (context only; not spoken)
    int sp_row = CSP_ETHAN;
    if      (speaker == "chelsie") sp_row = CSP_CHELSIE;
    else if (speaker == "aiden")   sp_row = CSP_AIDEN;
    else if (speaker != "ethan") { fprintf(stderr, "unknown speaker '%s' (ethan|chelsie|aiden)\n", speaker.c_str()); return 1; }

    const uint32_t seed = 1234;
    const int   max_frames = 2000;
    const size_t window_frames = 72;

    llama_backend_init();

    // ================= front-end: load baked tensors + vocab =================
    struct ggml_context * fe_ctx = nullptr;
    struct gguf_init_params gp { /*no_alloc*/ false, /*ctx*/ &fe_ctx };
    gguf_context * gc = gguf_init_from_file(fe_path.c_str(), gp);
    if (!gc || !fe_ctx) { fprintf(stderr, "FAIL: gguf_init_from_file front-end\n"); return 1; }
    FeTensor token_embd = load_fe_tensor(fe_ctx, "token_embd.weight");          // ne0=2048, ne1=152064
    FeTensor fc1_w = load_fe_tensor(fe_ctx, "text_projection.linear_fc1.weight"); // ne0=2048, ne1=2048
    FeTensor fc1_b = load_fe_tensor(fe_ctx, "text_projection.linear_fc1.bias");   // ne0=2048
    FeTensor fc2_w = load_fe_tensor(fe_ctx, "text_projection.linear_fc2.weight"); // ne0=2048, ne1=1024
    FeTensor fc2_b = load_fe_tensor(fe_ctx, "text_projection.linear_fc2.bias");   // ne0=1024
    FeTensor csp   = load_fe_tensor(fe_ctx, "codec_special_embd.weight");         // ne0=1024, ne1=9
    const int D_IN = (int) token_embd.ne0;   // 2048
    const int D    = (int) fc2_w.ne1;         // 1024 (talker hidden)
    fprintf(stderr, "FE: token_embd[%lld,%lld] D_in=%d D=%d csp_rows=%lld\n",
            (long long)token_embd.ne1, (long long)token_embd.ne0, D_IN, D, (long long)csp.ne1);

    // vocab-only load of the same file for tokenization
    llama_model_params femp = llama_model_default_params();
    femp.vocab_only = true;
    llama_model * fe_model = llama_model_load_from_file(fe_path.c_str(), femp);
    if (!fe_model) { fprintf(stderr, "FAIL: front-end vocab load\n"); return 1; }
    const llama_vocab * fe_vocab = llama_model_get_vocab(fe_model);

    // ---- render chatml (matches transformers apply_chat_template, no system turn) ----
    std::string chatml =
        "<|im_start|>user\n"      + prompt + "<|im_end|>\n" +
        "<|im_start|>assistant\n" + spoken + "<|im_end|>\n";

    std::vector<llama_token> ids(chatml.size() + 64);
    int n_ids = llama_tokenize(fe_vocab, chatml.c_str(), (int32_t) chatml.size(),
                               ids.data(), (int32_t) ids.size(), /*add_special*/ false, /*parse_special*/ true);
    if (n_ids < 0) { fprintf(stderr, "FAIL: tokenize (buffer %d)\n", -n_ids); return 1; }
    ids.resize(n_ids);
    fprintf(stderr, "TOK: %d ids; first= %d %d %d ... last= %d %d\n", n_ids,
            ids[0], ids[1], ids[2], ids[n_ids-2], ids[n_ids-1]);

    // ---- text_projection(embed(id)) on CPU for every token ----
    auto embed_row = [&](int id, float * x){ const float * r = &token_embd.data[(size_t)id * D_IN]; memcpy(x, r, D_IN*sizeof(float)); };
    auto text_projection = [&](const float * x, float * out){
        std::vector<float> h(D_IN);
        for (int j = 0; j < D_IN; ++j) {
            double acc = fc1_b.data[j];
            const float * w = &fc1_w.data[(size_t)j * D_IN];
            for (int i = 0; i < D_IN; ++i) acc += (double) w[i] * x[i];
            float v = (float) acc;
            h[j] = v / (1.0f + std::exp(-v));          // SiLU
        }
        for (int k = 0; k < D; ++k) {
            double acc = fc2_b.data[k];
            const float * w = &fc2_w.data[(size_t)k * D_IN];
            for (int j = 0; j < D_IN; ++j) acc += (double) w[j] * h[j];
            out[k] = (float) acc;
        }
    };
    // proj[i] = text_projection(embed(ids[i]))  [n_ids, D]
    std::vector<float> proj((size_t)n_ids * D);
    { std::vector<float> x(D_IN);
      for (int i = 0; i < n_ids; ++i) { embed_row(ids[i], x.data()); text_projection(x.data(), &proj[(size_t)i*D]); } }
    auto proj_row = [&](int i){ return &proj[(size_t)i * D]; };
    // special projected vectors
    std::vector<float> tts_bos(D), tts_eos(D), tts_pad(D);
    { std::vector<float> x(D_IN);
      embed_row(TTS_BOS, x.data()); text_projection(x.data(), tts_bos.data());
      embed_row(TTS_EOS, x.data()); text_projection(x.data(), tts_eos.data());
      embed_row(TTS_PAD, x.data()); text_projection(x.data(), tts_pad.data()); }

    // ---- assemble talker_input_embed + trailing (mirror generate()/assistant_parts) ----
    std::vector<float> talker_input;   // [P, D]
    std::vector<float> trailing;       // [R, D]
    auto push = [&](std::vector<float> & dst, const float * v){ dst.insert(dst.end(), v, v + D); };

    std::vector<int> im_starts;
    for (int i = 0; i < n_ids; ++i) if (ids[i] == TOK_IM_START) im_starts.push_back(i);
    im_starts.push_back(n_ids);
    const int n_seg = (int) im_starts.size() - 1;
    for (int si = 0; si < n_seg; ++si) {
        const int s = im_starts[si], e = im_starts[si+1];
        const int role = ids[s + 1];
        if (role == TOK_SYSTEM) continue;
        if (role == TOK_USER) {
            for (int i = s; i < e; ++i) push(talker_input, proj_row(i));
        } else if (role == TOK_ASSISTANT && si == n_seg - 1) {
            // assistant_text = [ah0,ah1,ah2, pad,pad,pad,pad, tts_bos, ah3]  (ah j = proj[s+j])
            // assistant_codec= [0,0,0, csp[nothink],csp[think_bos],csp[think_eos],csp[speaker],csp[pad],csp[bos]]
            // inp = assistant_text + assistant_codec (elementwise)
            std::vector<float> zero(D, 0.0f);
            const float * atext[9] = { proj_row(s+0), proj_row(s+1), proj_row(s+2),
                                       tts_pad.data(), tts_pad.data(), tts_pad.data(), tts_pad.data(),
                                       tts_bos.data(), proj_row(s+3) };
            const float * acodec[9] = { zero.data(), zero.data(), zero.data(),
                                        &csp.data[(size_t)CSP_NOTHINK*D], &csp.data[(size_t)CSP_THINK_BOS*D],
                                        &csp.data[(size_t)CSP_THINK_EOS*D], &csp.data[(size_t)sp_row*D],
                                        &csp.data[(size_t)CSP_PAD*D], &csp.data[(size_t)CSP_BOS*D] };
            for (int k = 0; k < 9; ++k) { std::vector<float> v(D); for (int d=0; d<D; ++d) v[d]=atext[k][d]+acodec[k][d]; push(talker_input, v.data()); }
            // trailing = [proj[s+4 .. e-1], tts_eos]
            for (int i = s + 4; i < e; ++i) push(trailing, proj_row(i));
            push(trailing, tts_eos.data());
        }
    }
    const int Pn = (int)(talker_input.size() / D);
    const int R  = (int)(trailing.size()     / D);
    fprintf(stderr, "ASM: talker_input=%d trailing=%d (expect prompt+9, trailing=assistant_text_rest+1)\n", Pn, R);
    llama_model_free(fe_model);
    gguf_free(gc);

    // ================= talker + voice mmproj: the verified per-frame loop =================
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(talker_path.c_str(), mp);
    if (!model) { fprintf(stderr, "FAIL: talker load\n"); return 1; }
    const int n_embd = llama_model_n_embd(model);
    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int n_vocab = llama_vocab_n_tokens(vocab);
    if (n_embd != D) { fprintf(stderr, "FAIL: talker n_embd %d != front-end D %d\n", n_embd, D); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = (uint32_t)(Pn + max_frames + 8);
    cp.n_batch = cp.n_ctx;
    cp.embeddings = true;
    cp.pooling_type = LLAMA_POOLING_TYPE_NONE;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "FAIL: ctx\n"); return 1; }

    mtmd_context_params mctxp = mtmd_context_params_default();
    mctxp.use_gpu = true; mctxp.print_timings = false; mctxp.warmup = false;
    mtmd_context * mctx = mtmd_init_from_file(mmproj_path.c_str(), model, mctxp);
    if (!mctx) { fprintf(stderr, "FAIL: mmproj init\n"); return 1; }
    mtmd_gen_audio_info info = mtmd_gen_audio_get_info(mctx);
    fprintf(stderr, "MMPROJ: gen_audio_type=%d sample_rate=%d\n", (int)info.type, info.sample_rate);

    auto pick_cb0 = [&](const float * lg)->int {
        int am = 0; float best = -1e30f;
        for (int v = 0; v < n_vocab; ++v) { if (v >= SUPPRESS_LO && v != CODEC_EOS) continue; if (lg[v] > best) { best = lg[v]; am = v; } }
        return am;
    };

    decode_embd_batch pb(talker_input.data(), Pn, 1, n_embd);
    pb.set_position_normal(0, 0);
    pb.batch.logits[Pn - 1] = 1;
    if (llama_decode(ctx, pb.batch) != 0) { fprintf(stderr, "FAIL: prompt decode\n"); return 1; }
    int pos = Pn;

    std::vector<int32_t> codes_buf;
    std::vector<float>   pcm;
    std::vector<uint8_t> c2w_state;
    std::vector<float>   fb(n_embd);

    auto flush_wav = [&](bool final_flush)->bool {
        if (codes_buf.empty()) return true;
        const size_t frames = codes_buf.size() / 16;
        if (!final_flush && frames < window_frames) return true;
        mtmd_gen_inp gi = mtmd_gen_inp_default(mctx);
        gi.type = MTMD_GEN_PROCESS_TYPE_GEN_WAV;
        gi.codes = codes_buf.data(); gi.n_codes = codes_buf.size(); gi.seed = seed;
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
        if (cb0 == CODEC_EOS) { stop_reason = 1; break; }
        const float * he = llama_get_embeddings_ith(ctx, -1);
        if (!he) { fprintf(stderr, "FAIL: null embeddings frame %d\n", frame); return 1; }

        mtmd_gen_inp gi = mtmd_gen_inp_default(mctx);
        gi.type = MTMD_GEN_PROCESS_TYPE_GEN_CODE;
        gi.code0 = cb0; gi.embd = const_cast<float *>(he); gi.seed = seed;
        mtmd_gen_out go{};
        if (mtmd_gen_audio_process(mctx, &gi, &go) != 0) { fprintf(stderr, "FAIL: GEN_CODE frame %d\n", frame); return 1; }
        codes_buf.insert(codes_buf.end(), go.codes, go.codes + go.n_codes);
        if (!flush_wav(false)) return 1;

        const float * ov = (frame < R) ? &trailing[(size_t)frame * n_embd] : tts_pad.data();
        for (int d = 0; d < n_embd; ++d) fb[d] = go.embd[d] + ov[d];
        decode_embd_batch sb(fb.data(), 1, 1, n_embd);
        sb.set_position_normal(pos, 0);
        sb.batch.logits[0] = 1;
        if (llama_decode(ctx, sb.batch) != 0) { fprintf(stderr, "FAIL: step decode frame %d\n", frame); return 1; }
        pos++;
        if (frame < 5 || frame % 20 == 0)
            fprintf(stderr, "frame %3d: cb0=%-5d codes[1..3]=%d,%d,%d pos=%d\n", frame, cb0, go.codes[1], go.codes[2], go.codes[3], pos);
    }
    if (!flush_wav(true)) return 1;

    double rms = 0.0; for (float v : pcm) rms += (double)v*v;
    rms = pcm.empty() ? 0.0 : sqrt(rms / pcm.size());
    fprintf(stderr, "GEN DONE: frames=%d stop=%s samples=%zu (%.2fs) rms=%.4f\n",
            frame, stop_reason ? "codec_eos" : "max_frames", pcm.size(), pcm.size()/(double)info.sample_rate, rms);

    if (!write_wav16(outwav, pcm, info.sample_rate)) { fprintf(stderr, "FAIL: wav write\n"); return 1; }
    fprintf(stderr, "WROTE %s (%zu samples)\n", outwav.c_str(), pcm.size());
    if (frame > 0 && !pcm.empty() && rms > 0.005) fprintf(stderr, "OMNILLAMA_TTS_PASS\n");
    else                                          fprintf(stderr, "OMNILLAMA_TTS_FAIL\n");

    mtmd_free(mctx);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
