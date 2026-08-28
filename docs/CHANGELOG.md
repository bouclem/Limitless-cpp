# Changelog

## [0.0.4] - 2026-08-29

### Added
- **CMake build system** — root `CMakeLists.txt` with library target, test integration
  - `lmlc_tensor` static library target
  - `enable_testing()` + `add_test()` for CTest
  - Warning flags (`-Wall -Wextra -Wpedantic` / `/W4`)
  - Debug build by default
- **BOOL dtype** — `LMLC_DTYPE_BOOL` with `lmlc_bool_t` (uint8_t)
  - Tensor create, fill, get, set, print all support BOOL
- **Cross-dtype support** — all ops convert through f32 internally
  - `lmlc_tensor_copy` — cross-dtype copy via element-wise conversion
  - `lmlc_tensor_add` / `lmlc_tensor_mul` — mixed dtype inputs
  - `lmlc_tensor_dot` / `lmlc_tensor_scale` — mixed dtype inputs
  - `lmlc_tensor_matmul` — mixed dtype inputs
- **Batched matmul** — `lmlc_tensor_matmul` now supports ndim >= 2
  - Batch dimension broadcasting (NumPy-style)
  - `[batch..., M, K] x [batch..., K, N] -> [batch..., M, N]`
- **Error reporting system**:
  - `lmlc_error_t` enum with 11 error codes
  - `lmlc_error_info_t` struct with code, message, function name, line number
  - `lmlc_last_error()` / `lmlc_last_error_msg()` / `lmlc_last_error_info()`
  - `lmlc_clear_error()` / `lmlc_error_string()`
  - `LMLC_SET_ERROR` macro captures `__func__` and `__LINE__`
- **Validation everywhere**:
  - `lmlc_tensor_create` — validates ndim (1..8), shape values (>0), dtype
  - `lmlc_tensor_get` / `lmlc_tensor_set` — bounds checking on all indices
  - `lmlc_tensor_reshape` — validates ndim, shape values, count match
  - `lmlc_tensor_permute` — validates permutation is valid (no duplicates, in range)
  - `lmlc_tensor_squeeze` — validates dim is size 1
  - `lmlc_tensor_unsqueeze` — validates ndim won't exceed 8
  - `lmlc_tensor_view` — validates ndim, shape values, count match
  - `lmlc_tensor_transpose` — validates dim bounds
- **Tests** — `test/test_LMLC_Tensor.cpp` with 22 test cases
  - create/free, invalid create, fill/zero, get/set bounds
  - BF16/F16/BOOL dtype conversions
  - add/mul broadcasting, 2D matmul, batched matmul
  - cross-dtype copy, cross-dtype add
  - scale, dot, norm, reshape, transpose, squeeze/unsqueeze, view, permute
  - error reporting verification

### Changed
- Removed dtype-match checks from all math ops (cross-dtype via f32 conversion)
- `lmlc_tensor_matmul` rewritten: batched + broadcasting + cross-dtype (was 2D-only)
- All functions now set error state on failure instead of silent return

## [0.0.3] - 2026-08-28

### Added
- **F16 dtype** — `LMLC_DTYPE_F16` with IEEE 754 half-precision conversion
  - `lmlc_f32_to_f16` / `lmlc_f16_to_f32` (round-to-nearest-even, subnormal support)
  - Tensor create, fill, get, set, print all support F16
- **Broadcasting** — NumPy-style broadcasting for element-wise ops
  - Right-aligned dimension matching with size-1 expansion
  - Used by `lmlc_tensor_add` and `lmlc_tensor_mul`
- **Math ops**:
  - `lmlc_tensor_mul` — element-wise multiply with broadcasting
  - `lmlc_tensor_scale` — scalar multiplication
  - `lmlc_tensor_matmul` — 2D matrix multiply (M×K · K×N → M×N)
  - `lmlc_tensor_dot` — dot product (1D, same count)
  - `lmlc_tensor_norm` — L2 norm (sqrt of sum of squares)

### Changed
- Refactored `lmlc_tensor_add` to use broadcasting (was same-shape only)
- Refactored `lmlc_tensor_get` / `lmlc_tensor_set` to use internal `tensor_get_flat` / `tensor_set_flat` helpers
- Added `<math.h>` include for `sqrtf`

## [0.0.2] - 2026-08-28

### Added
- **BF16 dtype** — `LMLC_DTYPE_BF16` with f32<->bf16 conversion functions
  - `lmlc_f32_to_bf16` / `lmlc_bf16_to_f32`
  - Tensor create, fill, get, set, copy, print all support BF16
- **lmlc_tensor_add** — element-wise addition (F32 + BF16)
- **Shape ops**:
  - `lmlc_tensor_reshape` — in-place reshape (count must match)
  - `lmlc_tensor_transpose` — swap two dimensions
  - `lmlc_tensor_permute` — reorder dimensions by permutation
  - `lmlc_tensor_squeeze` — remove a size-1 dimension
  - `lmlc_tensor_unsqueeze` — insert a size-1 dimension
  - `lmlc_tensor_view` — non-owning view with new shape

### Changed
- `float *data` → `void *data` in `lmlc_tensor_t` (multi-dtype support)
- Added `owns_data` field to `lmlc_tensor_t` (views don't free data)
- `lmlc_tensor_free` now checks `owns_data` before freeing

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
