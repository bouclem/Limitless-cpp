# Changelog

## [0.0.1] - 2026-08-28

### Added
- **LMLC_Tensor** module — initial tensor data structure
  - `lmlc_tensor_t` struct with shape, strides, count, dtype, data pointer
  - `LMLC_DTYPE_F32` data type (others stubbed as TODO)
  - `lmlc_tensor_create` / `lmlc_tensor_free` — lifecycle management
  - `lmlc_tensor_zero` / `lmlc_tensor_fill` — initialization helpers
  - `lmlc_tensor_copy` — tensor-to-tensor copy
  - `lmlc_tensor_get` / `lmlc_tensor_set` — element access by indices
  - `lmlc_tensor_print` — debug print (flat, multi-dim TODO)
  - C-compatible API (`extern "C"`) callable from both C and C++
  - Row-major contiguous memory layout with computed strides

### TODO (future versions)
- Additional dtypes: F16, BF16, I8, I16, I32, I64, BOOL
- Shape ops: reshape, transpose, permute, squeeze, unsqueeze, view
- Math ops: add, mul, matmul, scale, dot, norm
- Tensor views (non-owning, with offset)
- Aligned allocation for SIMD
- Bounds checking on element access
- Multi-dimensional pretty print
- Dynamic shape array (remove fixed size 8 limit)
