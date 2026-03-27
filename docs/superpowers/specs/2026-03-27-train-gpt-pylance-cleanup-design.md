# Pylance Cleanup Design for `train_gpt.py`

## Summary

Clean up the real code-level Pylance errors in [`records/track_10min_16mb/2026-03-20_RecurrentSharedDepth_SP1024_4x3_rank4/train_gpt.py`](/Users/kai/Desktop/parameter-golf/records/track_10min_16mb/2026-03-20_RecurrentSharedDepth_SP1024_4x3_rank4/train_gpt.py) without changing runtime behavior, model defaults, logging, serialization format, or architecture.

Out of scope:
- missing-import diagnostics caused by the local environment not having `numpy`, `sentencepiece`, or `torch`
- refactoring the file into modules
- training or submission changes

## Issues To Fix

1. Quantization payload typing is too loose.
   `dequantize_state_dict_int8` accepts `dict[str, object]`, so Pylance reports invalid `.get`, `.items`, and index access on `object`.

2. Rotary cache return values are typed as optional.
   `_cos_cached` and `_sin_cached` are `Tensor | None`, and Pylance does not prove they are non-`None` at the return site.

3. `lm_head` narrowing is incomplete in `forward_logits`.
   The untied-embedding path calls `self.lm_head(x)` without an explicit guard.

4. `zip(..., strict=True)` is not accepted by the configured checker target.
   The code needs an equivalent shape check that Pylance accepts.

## Proposed Changes

### 1. Quantization payload typing

- Introduce small local typed structures for the serialized quantization payload.
- Use typed aliases or `TypedDict`-style definitions for:
  - quantized tensors
  - scales
  - dtype names
  - passthrough tensors
  - quantization metadata
  - passthrough original dtypes
- Update `quantize_state_dict_int8` and `dequantize_state_dict_int8` to consume and produce those typed structures instead of raw `dict[str, object]`.
- Keep the serialized wire shape unchanged so saved artifacts are compatible with the current runtime logic.

### 2. Rotary cache narrowing

- Add explicit narrowing before returning cached tensors from `Rotary.forward`.
- Use a local assertion or local bound variables after the cache fill branch so Pylance can see the tensors are non-`None`.
- Do not change cache behavior or recomputation logic.

### 3. `lm_head` narrowing

- Mirror the main `forward` path in `forward_logits`.
- If `tie_embeddings` is false, explicitly check `self.lm_head is not None` before calling it.
- Keep the current runtime error behavior for invalid untied-head states.

### 4. `zip(strict=True)` replacement

- Replace the typed-checker-hostile `zip(..., strict=True)` loop with an explicit length assertion followed by plain `zip(...)`.
- Preserve the invariant that optimizer states and optimizers must stay aligned.

## Verification

Run these checks after the edit:

```bash
npx pyright records/track_10min_16mb/2026-03-20_RecurrentSharedDepth_SP1024_4x3_rank4/train_gpt.py
python3 -m py_compile records/track_10min_16mb/2026-03-20_RecurrentSharedDepth_SP1024_4x3_rank4/train_gpt.py
```

Success means:
- the remaining `pyright` diagnostics are only missing-import/environment issues
- the real code-level issues above are gone
- the script still parses cleanly

## Assumptions

- The user wants only the real code-level Pylance issues fixed.
- Behavioral changes are not acceptable unless required for type narrowing with equivalent runtime semantics.
- The cleanup should remain local to this single file.
