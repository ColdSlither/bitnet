# bitnet.cpp — I2_S CUDA Offload Debug

**Handoff for:** GLM 5.2
**Author:** Oya (Hermes Agent)
**Date:** 2026-06-24
**Method:** Zixuan Li prompt architecture (task description + five-round polish)

---

## 01 / TASK DESCRIPTION

### 1. Product Overview

Fix GPU offloading for I2_S (1.58-bit ternary) tensors in bitnet.cpp's ggml-cuda backend. Currently `--ngl>=1` crashes with `GGML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src))` at `ggml.c:4107` when any I2_S layer is offloaded to the NVIDIA RTX 3080. The objective is to make GPU offloading work for I2_S layers so text generation latency drops from 33 tok/s (CPU-only) toward potential 100+ tok/s.

### 2. Tech Stack & Code Locations

- **Repo:** `ColdSlither/bitnet` (fork of `microsoft/BitNet`), branch `main`
- **Build dir:** `/home/rell/bitnet/build/`
- **Venv:** `/home/rell/bitnet/.venv/`
- **GPU:** RTX 3080 10GB, CUDA 12.0, sm_86
- **Test model:** `models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf` (1.10 GiB, 332 tensors, 31 layers)

**Key files:**
- `3rdparty/llama.cpp/ggml/src/ggml-cuda.cu` — main CUDA backend dispatch
- `3rdparty/llama.cpp/ggml/src/ggml.c` — `ggml_nbytes` I2_S formula, `GGML_ASSERT` at line 4107
- `3rdparty/llama.cpp/src/llama.cpp` — tensor loading, `create_tensor`, `check_tensor_dims`

### 3. Architecture Detail

**I2_S tensor format:**
- `type_size = 1` (int8_t), `blck_size = 1`, `ncols = 4`
- 4 ternary values `{-1, 0, +1}` packed into 1 byte
- GGUF stores logical shapes `(2560, 2560)` for 2560-dim model
- `ggml_nbytes = ne[0]*ne[1]*1/4 + 32` (accounts for packing)
- Native BitNet models work at `ngl=0` (CUDA backend active, 259 tok/s prompt processing)
- `ngl>=1` crashes in `ggml_cuda_can_mul_mat` or tensor view creation

**Crash trace:**
1. `llama-cli` loads model, offloads `ngl` layers to GPU
2. `ggml_cuda_can_mul_mat` checks if tensor type is supported for GPU matmul
3. I2_S type (value 36) is not in the supported types switch
4. Falls back to CPU, but tensor view creation for CPU→GPU transfer hits `ggml.c:4107`

### 4. Module Specifications

**4.1 Identify all ggml-cuda dispatch points** that need I2_S entries:
- `ggml_cuda_can_mul_mat` — add `case GGML_TYPE_I2_S: return true;`
- `ggml_cuda_nop` — add I2_S support
- `ggml_cuda_compute_forward` — add dispatch for GGML_OP_MUL_MAT with I2_S

**4.2 Implement I2_S CUDA kernel:**
- The I2_S GEMV kernel exists in `3rdparty/llama.cpp/ggml/src/ggml-cuda.cu`? Search for `i2_s` or `I2_S` patterns
- Native CPU I2_S kernels: `ggml_gemv_i2_i8_s`, `ggml_gemm_i2_i8_s`
- Equivalent CUDA kernel path: use the bitnet-gpu kernel from `gpu/` directory as reference
- The GPU kernel in `gpu/bitnet_kernels/` is a standalone implementation — it uses W2A8 (2-bit weight × 8-bit activation) GEMV, not full I2_S matmul

**4.3 Fallback path:**
- If I2_S CUDA kernel is not feasible, ensure `ngl>=1` gracefully falls back to CPU for I2_S layers without crashing

**4.4 Test with the native BitNet-b1.58-2B-4T model:**
```bash
cd /home/rell/bitnet
./build/bin/llama-cli -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "The meaning of life is" -n 30 -t 20 --temp 0.8 -c 512 -ngl 1
```

### 5. Models & API

- **GLM-5V-Turbo** for vision analysis of the crash (not needed here — pure C++/CUDA)
- **GLM-5.2-0416** for code generation and debugging

### 6. Local Persistence

All changes go into the `ColdSlither/bitnet` git repo. Submodule at `3rdparty/llama.cpp/` points to `Eddie-Wang1120/llama.cpp` (upstream BitNet fork, commit `1f86f058`). Patches to the submodule should be committed separately or saved under `patches/`.

### 7. Acceptance Criteria

**AC1** `ngl=1` loads the BitNet 2B model without crashing
**AC2** `ngl=31` (all layers) produces correct output — `"The meaning of life is"` generates coherent text, not garbage like `GGGGGGGGGG`
**AC3** Text generation speed at `ngl=31` exceeds 100 tok/s (vs 33 tok/s CPU-only)
**AC4** Prompt processing at `ngl=31` exceeds 500 tok/s (vs 259 tok/s at ngl=0)
**AC5** No memory leaks — repeated 10 invocations stay within 10GB VRAM
**AC6** The Llama3-8B-1.58 model also loads and generates coherent text at `ngl>=1`

### 8. What Is Forbidden

**DO NOT** rewrite the existing ggml.c infrastructure. Only add I2_S entries to existing dispatch points.
**DO NOT** change the I2_S type definition (`blck_size`, `ncols`, `type_size`).
**DO NOT** change the I2_S data layout — the native model works correctly at ngl=0.
**DO NOT** expand tensor dimensions or modify tensor shapes — use the packed shapes as stored in the GGUF.
**DO NOT** spend time on non-I2_S quant types (TL1, TL2, TQ1_0, TQ2_0).
**DO NOT** use placeholder implementations — every function must be real.
**DO NOT** ask for clarification — verify your own work by running the acceptance tests.

---

## 02 / ROUND 1 — INVENTORY ALL GPU BACKEND SOURCES

Audit every file in `3rdparty/llama.cpp/ggml/src/ggml-cuda/` that contains matmul, can_mul_mat, or tensor-type dispatch. List all files that need I2_S entries. Verify each file is real (not a placeholder). Report the full file list and count.

---

## 03 / ROUND 2 — ADD I2_S TO DISPATCH POINTS

For every file inventoried in Round 1, add the `case GGML_TYPE_I2_S:` entry. The dispatch pattern is `switch(src0->type)`. Each case should call the equivalent float implementation or a I2_S-specific implementation when available. After each file, recompile and verify the build succeeds.

**Quantifiable constraints:**
- Each dispatch addition: at most 3 lines of code
- Compile time per iteration: under 30 seconds (CUDA only changed files via make)
- Test passes: model loads without crash

---

## 04 / ROUND 3 — HANDLE ggml.c:4107 CRASH

The assertion at line 4107 in ggml.c is:
```c
GGML_ASSERT(view_src == NULL || data_size == 0 || data_size + view_offs <= ggml_nbytes(view_src));
```

This fires during `ggml_view_3d` or similar view creation. I2_S tensors have `nbytes` computed via the `/4 + 32` formula. The view system may not handle this correctly. Fix by:
1. Tracing which function calls ggml_view_3d for I2_S tensors
2. Ensuring the view size accounts for the packed data layout
3. Or routing I2_S tensor views through a custom path that skips the check

---

## 05 / ROUND 4 — BENCHMARK AND VERIFY TEXT QUALITY

Run the native 2B model through all `ngl` levels from 0 to 31:

```bash
cd /home/rell/bitnet
for ngl in 0 1 5 10 20 31; do
  ./build/bin/llama-bench -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
    -p 512 -n 128 -t 20 -ngl $ngl 2>&1
done
```

Verify output quality by running interactive generation:
```bash
./build/bin/llama-cli -m models/BitNet-b1.58-2B-4T/ggml-model-i2_s.gguf \
  -p "Write a haiku about CUDA programming" -n 100 -t 20 -ngl 31 --temp 0.7
```

The output must be readable English, not repeating characters or garbage tokens.

---

## 06 / ROUND 5 — STRIP THE TUTORIAL FEEL

Remove any debug printfs, test placeholders, or work-in-progress scaffolding. Ensure only the dispatch additions and necessary I2_S handling code remain. Clean up commented-out code. Verify the final diff is minimal and focused:

```bash
cd /home/rell/bitnet
git diff --stat 3rdparty/llama.cpp/ggml/src/ggml-cuda*.cu 3rdparty/llama.cpp/ggml/src/ggml-cuda*.cuh
```

The diff should be under 100 lines total across all files. Run the full acceptance criteria set one final time and record all results.
