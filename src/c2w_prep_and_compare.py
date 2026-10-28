#!/usr/bin/env python3
"""OmniLlama 5a helper (Goliath): mode 'prep' converts the gold reference into the harness's binary
inputs; mode 'compare' diffs the harness output against the reference.

  prep:    codes.npy [1,16,T] -> codes_fm_i32.bin (frame-major int32, index f*16+g)
           ref_wav.npy        -> ref_wav_f32.bin   (raw float32)
  compare: ref_wav_f32.bin vs <out_f32.bin> over the common prefix (len, RMS err, corr, maxdiff)
"""
import sys, os
sys.stdout.reconfigure(encoding="utf-8")
import numpy as np

REF = r"I:\_bridge\c2w_ref"

def prep():
    codes = np.load(os.path.join(REF, "codes.npy"))  # [1,16,T] int64
    assert codes.ndim == 3 and codes.shape[0] == 1 and codes.shape[1] == 16, codes.shape
    T = codes.shape[2]
    fm = np.ascontiguousarray(codes[0].T.astype(np.int32))  # [T,16] -> frame-major f*16+g
    fm.reshape(-1).tofile(os.path.join(REF, "codes_fm_i32.bin"))
    ref = np.load(os.path.join(REF, "ref_wav.npy")).reshape(-1).astype(np.float32)
    ref.tofile(os.path.join(REF, "ref_wav_f32.bin"))
    print(f"prep: T={T} frames -> codes_fm_i32.bin ({fm.size} int32); ref_wav_f32.bin ({ref.size} f32, {ref.size/24000:.3f}s)")

def compare(out_bin):
    ref = np.fromfile(os.path.join(REF, "ref_wav_f32.bin"), dtype=np.float32)
    out = np.fromfile(out_bin, dtype=np.float32)
    n = min(ref.size, out.size)
    print(f"compare: ref={ref.size} out={out.size} common={n} ({n/24000:.3f}s)")
    if n == 0:
        print("EMPTY output"); return
    a, b = ref[:n], out[:n]
    rms_err = float(np.sqrt(np.mean((a - b) ** 2)))
    ref_rms = float(np.sqrt(np.mean(a ** 2)))
    maxdiff = float(np.max(np.abs(a - b)))
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    corr = float(np.dot(a, b) / denom) if denom > 0 else 0.0
    print(f"  ref_rms={ref_rms:.4f}  rms_err={rms_err:.4f}  rms_err/ref={rms_err/ref_rms if ref_rms else 0:.3%}")
    print(f"  corr={corr:.5f}  maxdiff={maxdiff:.4f}")
    # fp16 vocoder vs fp32 reference + conv-edge differences -> expect corr very high, not bit-exact
    print("PASS (vocoder path matches reference)" if corr > 0.99 and rms_err < 0.05
          else "REVIEW (corr/err outside expected band — inspect)")

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "prep"
    if mode == "prep":
        prep()
    elif mode == "compare":
        compare(sys.argv[2] if len(sys.argv) > 2 else os.path.join(REF, "out_f32.bin"))
    else:
        print("unknown mode", mode)
