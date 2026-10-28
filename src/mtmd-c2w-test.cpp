// OmniLlama 5a: standalone Code2Wav test.
// Loads an Omni voice mmproj, feeds a fixed frame-major code sequence straight into the code2wav
// path (clip_encode GEN_WAV, no talker / no llama text model), and writes the decoded PCM to WAV
// (+ optional raw f32) so it can be diffed against the transformers gold reference.
//
// usage: llama-mtmd-c2w-test <mmproj.gguf> <codes_fm_i32.bin> <out.wav> [out_f32.bin]
//   codes_fm_i32.bin = raw little-endian int32, frame-major, length n_frames*16 (code for group g,
//   frame f at index f*16+g). n_frames must be <= the code2wav window (72), so it fits one call.

#include "clip.h"
#include "clip-impl.h"   // clip_image_f32 / clip_image_f32_batch

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

static std::vector<int32_t> read_i32(const char * path) {
    FILE * f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<int32_t> v(n / 4);
    if (fread(v.data(), 4, v.size(), f) != v.size()) { fprintf(stderr, "short read %s\n", path); exit(1); }
    fclose(f);
    return v;
}

static void write_wav16(const char * path, const std::vector<float> & pcm, int rate) {
    FILE * f = fopen(path, "wb");
    const uint32_t dsz = (uint32_t)(pcm.size() * 2), rsz = 36 + dsz, fsz = 16, br = (uint32_t) rate * 2;
    const uint16_t fmt = 1, ch = 1, al = 2, bits = 16;
    const uint32_t r32 = (uint32_t) rate;
    fwrite("RIFF", 1, 4, f); fwrite(&rsz, 4, 1, f); fwrite("WAVE", 1, 4, f);
    fwrite("fmt ", 1, 4, f); fwrite(&fsz, 4, 1, f); fwrite(&fmt, 2, 1, f); fwrite(&ch, 2, 1, f);
    fwrite(&r32, 4, 1, f); fwrite(&br, 4, 1, f); fwrite(&al, 2, 1, f); fwrite(&bits, 2, 1, f);
    fwrite("data", 1, 4, f); fwrite(&dsz, 4, 1, f);
    for (float v : pcm) { int16_t s = (int16_t)(std::max(-1.0f, std::min(1.0f, v)) * 32767.0f); fwrite(&s, 2, 1, f); }
    fclose(f);
}

int main(int argc, char ** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <mmproj.gguf> <codes_fm_i32.bin> <out.wav> [out_f32.bin]\n", argv[0]);
        return 1;
    }
    const char * mmproj     = argv[1];
    const char * codes_path = argv[2];
    const char * out_wav    = argv[3];
    const char * out_bin    = argc > 4 ? argv[4] : nullptr;

    clip_context_params cp{};
    cp.use_gpu          = true;
    cp.device           = nullptr;
    cp.flash_attn_type  = CLIP_FLASH_ATTN_TYPE_AUTO;
    cp.image_min_tokens = -1;
    cp.image_max_tokens = -1;
    cp.warmup           = false;
    cp.cb_eval          = nullptr;
    cp.cb_eval_user_data = nullptr;
    cp.no_alloc         = false;
    cp.progress_callback = nullptr;
    cp.progress_callback_user_data = nullptr;

    clip_init_result r = clip_init(mmproj, cp);
    clip_ctx * ctx = r.ctx_gen_a;
    if (!ctx) { fprintf(stderr, "no gen-audio (code2wav) context in this mmproj\n"); return 1; }

    std::vector<int32_t> codes = read_i32(codes_path); // frame-major [n_frames*16]
    const int n_codes  = 16;
    const int n_frames = (int) codes.size() / n_codes;
    fprintf(stderr, "loaded %d frames (%zu codes) from %s\n", n_frames, codes.size(), codes_path);
    if (n_frames <= 0 || (int) codes.size() % n_codes != 0) { fprintf(stderr, "bad codes length\n"); return 1; }

    // GEN_WAV has no hidden-state input; the batch entry is an unused placeholder (mirror mtmd.cpp)
    clip_image_f32 dummy;
    dummy.set_size({1, 1}, false, true);
    dummy.cpy_buf(std::vector<float>(1, 0.0f));
    clip_image_f32_batch batch;
    batch.is_audio = true;
    batch.entries.push_back(std::move(dummy));

    std::vector<float>   pcm;
    std::vector<uint8_t> st_out;
    clip_encode_params p;
    p.imgs        = &batch;
    p.n_threads   = 4;
    p.gen_process = CLIP_GEN_PROCESS_GEN_WAV;
    p.seed        = 1234;
    p.codes       = &codes;
    p.out_audio   = &pcm;
    p.state_in    = nullptr; // cold start, single window
    p.state_out   = &st_out;

    if (!clip_encode(ctx, &p)) { fprintf(stderr, "clip_encode (code2wav) failed\n"); return 1; }

    double rms = 0, amax = 0;
    for (float v : pcm) { rms += (double) v * v; amax = std::max(amax, (double) std::fabs(v)); }
    rms = pcm.empty() ? 0 : std::sqrt(rms / pcm.size());
    fprintf(stderr, "OUT: samples=%zu dur=%.3fs rms=%.4f absmax=%.4f\n",
            pcm.size(), pcm.size() / 24000.0, rms, amax);

    write_wav16(out_wav, pcm, 24000);
    if (out_bin) { FILE * f = fopen(out_bin, "wb"); fwrite(pcm.data(), 4, pcm.size(), f); fclose(f); }
    fprintf(stderr, "wrote %s%s%s\n", out_wav, out_bin ? " + " : "", out_bin ? out_bin : "");

    if (r.ctx_v)     clip_free(r.ctx_v);
    if (r.ctx_a)     clip_free(r.ctx_a);
    if (r.ctx_gen_a) clip_free(r.ctx_gen_a);
    return 0;
}
