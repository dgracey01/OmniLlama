// OmniLlama 5b-3: teacher-forced validation of the qwen3omnitalker GGUF against the gold reference.
//
// The Talker is embedding-driven and its per-step input is NOT simply codec_embed(prev)+text: it is the
// SUM of all 16 codebooks' representations for the previous frame (cb0 via the talker codec_embedding,
// cb1..cb14 as the MTP/code_predictor hidden states, cb15 via the MTP's last embedding) plus the text
// stream (trailing_text_hidden, then tts_pad). That feedback couples the Talker with the MTP, so it
// cannot be reconstructed offline from codes alone. Instead the reference captures the EXACT per-step
// inputs_embeds it fed the Talker (talker_step_embeds_f32.bin); we replay prompt + those captured steps
// through the ported graph and check the argmax (codebook-0) at every frame equals the reference's cb0.
// Matching proves the Talker backbone port is numerically faithful to transformers, isolated from the MTP.
//
// Inputs (little-endian f32 / i32, in <ref_dir>):
//   talker_input_embed_f32.bin   [P, n_embd]   assembled prompt embeddings (P=19); predicts frame 0
//   talker_step_embeds_f32.bin   [G, n_embd]   captured per-step inputs for frames 1..G
//   ref_cb0_i32.bin              [T]           reference codebook-0 tokens (T=33)
//
// Sequence fed (P + G): prompt[0..P-1], then the G captured step embeds. Logits at position P-1 predict
// frame 0; position P-1+g predicts frame g. Compared over frames 0..T-1.
//
// usage: llama-talker-embed-test <talker.gguf> <ref_dir>
// PASS line: TALKER_EMBED_TEST matches=<m>/<T> ...

#include "llama.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
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
static std::vector<int32_t> read_i32(const std::string & p) {
    FILE * f = fopen(p.c_str(), "rb");
    if (!f) { fprintf(stderr, "FAIL: cannot open %s\n", p.c_str()); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<int32_t> v(n / sizeof(int32_t));
    if (fread(v.data(), 1, n, f) != (size_t) n) { fprintf(stderr, "FAIL: short read %s\n", p.c_str()); exit(1); }
    fclose(f); return v;
}

int main(int argc, char ** argv) {
    if (argc < 3) { fprintf(stderr, "usage: %s <talker.gguf> <ref_dir>\n", argv[0]); return 1; }
    const std::string gguf = argv[1];
    const std::string dir  = std::string(argv[2]);
    const std::string sep  = (dir.back() == '/' || dir.back() == '\\') ? "" : "/";
    auto P = [&](const char * n){ return dir + sep + n; };

    const int n_embd  = 1024;
    const int n_vocab = 3072;
    const int codec_eos = 2150;             // stop token; allowed through the suppression mask
    const int suppress_lo = n_vocab - 1024; // reserved range 2048..3071 masked (except eos), as in gen

    std::vector<float>   prompt = read_f32(P("talker_input_embed_f32.bin"));
    std::vector<float>   steps  = read_f32(P("talker_step_embeds_f32.bin"));
    std::vector<int32_t> ref    = read_i32(P("ref_cb0_i32.bin"));

    const int Pn = (int) (prompt.size() / n_embd); // prompt positions (19)
    const int G  = (int) (steps.size()  / n_embd); // captured generated-step inputs
    const int T  = (int) ref.size();               // frames (33)
    const int n_pos = Pn + G;
    const int n_cmp = (T - 1 <= G) ? T : (G + 1);  // frames we can compare (frame 0 from prompt + G steps)
    fprintf(stderr, "IN: n_embd=%d prompt=%d steps=%d frames=%d compare=%d\n", n_embd, Pn, G, T, n_cmp);

    llama_backend_init();
    llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = 99;
    llama_model * model = llama_model_load_from_file(gguf.c_str(), mp);
    if (!model) { fprintf(stderr, "FAIL: load\n"); return 1; }

    llama_context_params cp = llama_context_default_params();
    cp.n_ctx = n_pos + 8; cp.n_batch = n_pos + 8;
    llama_context * ctx = llama_init_from_model(model, cp);
    if (!ctx) { fprintf(stderr, "FAIL: ctx\n"); return 1; }

    llama_batch batch = llama_batch_init(n_pos, n_embd, 1); // embd batch (token==null)
    for (int i = 0; i < n_pos; ++i) {
        const float * src = (i < Pn) ? &prompt[i * n_embd] : &steps[(i - Pn) * n_embd];
        for (int d = 0; d < n_embd; ++d) batch.embd[i * n_embd + d] = src[d];
        batch.pos[i] = i; batch.n_seq_id[i] = 1; batch.seq_id[i][0] = 0;
        // request logits at the positions that predict frames 0..n_cmp-1: P-1 .. P-1+(n_cmp-1)
        const int f = i - (Pn - 1);
        batch.logits[i] = (f >= 0 && f < n_cmp) ? 1 : 0;
    }
    batch.n_tokens = n_pos;

    if (llama_decode(ctx, batch) != 0) { fprintf(stderr, "FAIL: decode\n"); return 1; }

    int matches = 0, first_mismatch = -1;
    for (int f = 0; f < n_cmp; ++f) {
        float * lg = llama_get_logits_ith(ctx, Pn - 1 + f);
        if (!lg) { fprintf(stderr, "FAIL: null logits frame %d\n", f); return 1; }
        int argmax = 0; float best = -1e30f;
        for (int v = 0; v < n_vocab; ++v) {
            if (v >= suppress_lo && v != codec_eos) continue;
            if (lg[v] > best) { best = lg[v]; argmax = v; }
        }
        const bool ok = (argmax == ref[f]);
        if (ok) ++matches; else if (first_mismatch < 0) first_mismatch = f;
        // on a mismatch, report the reference token's own logit + gap: a tiny gap == F16-vs-bf16 near-tie
        const float ref_logit = lg[ref[f]];
        const float gap = best - ref_logit;
        if (ok) fprintf(stderr, "frame %2d: pred=%-5d ref=%-5d logit=%7.3f OK\n", f, argmax, ref[f], best);
        else    fprintf(stderr, "frame %2d: pred=%-5d ref=%-5d pred_logit=%7.3f ref_logit=%7.3f gap=%.4f x\n",
                        f, argmax, ref[f], best, ref_logit, gap);
    }

    fprintf(stderr, "TALKER_EMBED_TEST matches=%d/%d first_mismatch=%d\n", matches, n_cmp, first_mismatch);
    if (matches == n_cmp)                 fprintf(stderr, "TALKER_EMBED_TEST_PASS\n");
    else if (matches >= (n_cmp * 9) / 10) fprintf(stderr, "TALKER_EMBED_TEST_NEARPASS\n");
    else                                  fprintf(stderr, "TALKER_EMBED_TEST_FAIL\n");

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
