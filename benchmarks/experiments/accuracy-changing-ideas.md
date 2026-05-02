# Accuracy-Changing Ideas (Approximation Algorithms)

These ideas trade mathematical exactness for potential speed gains.
They are saved here for reference but are **not being pursued** under the
current directive (exact arithmetic, no approximations).

---

## FP16 Column Storage

**Proposed:** 2026-04-30  
**Status:** Rejected — changes the math (approximation algorithm)

### Mechanism

Store column data as `uint16_t` (FP16 bit-patterns, 2 bytes/element) instead of
`float` (FP32, 4 bytes/element). This halves DRAM bandwidth for column reads.

Cache-line utilization improvement:
- FP32: average stride ~52 bytes vs 64-byte cache line → ~1.2 floats per cache-line load
- FP16: average stride ~26 bytes → ~2.5 floats per cache-line load → ~2× more useful data per DRAM access

Target machine: Intel Xeon Platinum 8488C (Sapphire Rapids) with F16C support.
`_mm256_cvtph_ps(__m128i)` converts 8 FP16 → 8 FP32 in one instruction.

### Kernel sketch (M=8 FP16 path)

```cpp
// Column data stored as uint16_t* (FP16 bit-patterns).
// 8 scalar uint16_t loads per feature, then _mm256_cvtph_ps, then FMA.
#if defined(__F16C__) && defined(__AVX2__) && defined(__FMA__)
for (; i + 8 <= rows_n; i += 8) {
    __m256 acc = _mm256_setzero_ps();
    for (const auto& feat : proj) {
        const uint16_t* col = reinterpret_cast<const uint16_t*>(
            evaluator.AttributeValues(feat.attribute_idx).data());
        __m256 vw = _mm256_set1_ps(feat.weight);
        uint16_t buf[8];
        for (int j = 0; j < 8; ++j) buf[j] = col[sel_ptr[i+j]];
        __m128i h = _mm_loadu_si128(reinterpret_cast<const __m128i*>(buf));
        __m256 vv = _mm256_cvtph_ps(h);
        acc = _mm256_fmadd_ps(vw, vv, acc);
    }
    _mm256_storeu_ps(out + i, acc);
}
#endif
```

### Implementation plan (if ever revisited)

1. Add to `.bazelrc`: `build:fp16_columns --cxxopt="-DFP16_COLUMNS=1" --cxxopt="-mf16c"`
2. In `oblique.cc` `EvaluateObliqueSplits`, convert `AttributeValues` float data to FP16 before the
   projection loop. Store as a per-node temporary `std::vector<uint16_t>` per column.
3. In `oblique_cpu_depthwise_1pass.cc`, add `#ifdef FP16_COLUMNS` branch that accepts
   `const uint16_t* const*` column pointers and uses the FP16 kernel above.
4. Microbench first in `gather_vs_scalar.cc` (add `kernel_fp16_scalar_m8` kernel,
   pre-convert column data to FP16 before timing).

### Expected gain
- Theoretical: up to 2× AP speedup (halved DRAM traffic)
- Microbench risk: 8 scalar uint16_t loads still required per feature per row-block;
  if scalar-load overhead dominates over bandwidth savings, gain may be < 20%
- Precision: FP16 has 3.3 decimal digits vs FP32's 7.2. For oblique projections
  (sum of ~2 weighted features), error is tiny but non-zero.
- Accuracy gate: ≤5% per dataset drop is the hard stop.

### Why rejected

FP16 introduces quantization error into every column value read during training.
The projection values fed to split-finding are approximate, not exact. This violates
the correctness constraint for the current experimental phase.

---

## INT8 Column Storage

**Proposed:** (future idea, not yet designed)  
**Status:** Not pursued

Store columns as `int8_t` (1 byte/element) with per-column scale+offset.
Would give 4× bandwidth reduction over FP32.
Requires fixed-point dot product with de-quantization at accumulation step.
Even larger precision impact than FP16.
