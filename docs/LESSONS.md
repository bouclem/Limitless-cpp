# Lessons

Cumulative mistakes, patterns, and insights from building LMLC.

## C/C++ Comment Style

- **`//` for single-line comments** in C/C++. `/* */` is for multi-line only.
- Using `/* TODO: ... */` for single-line TODOs is wrong — use `// TODO: ...`.
- This was corrected in v0.0.1 after user feedback.

## Macro Parameter Naming

- **Never name macro parameters the same as struct field names.**
- `LMLC_SET_ERROR(code, msg)` expanded `g_error.code = (code)` to `g_error.LMLC_ERR_NULL = (LMLC_ERR_NULL)` — MSVC interpreted the enum as a field access, causing 100+ errors.
- Fix: renamed to `err_code` / `err_msg`.
- Rule: prefix macro params to avoid collisions with any struct member names.

## Test Macro Design (MSVC)

- `do { ... if (1) ... } while (0)` with a trailing `;` triggers MSVC C4390 ("empty controlled statement").
- The `if (1)` was unnecessary — just `do {` is enough to create a block scope.
- ASSERT macros should wrap their `if` in `do { ... } while (0)` to avoid dangling-else problems.

## Architecture Patterns

### `void *data` + `owns_data` Flag

- Changed from `float *data` to `void *data` early (v0.0.2) to support multiple dtypes.
- `owns_data` flag (0 for views, 1 for owners) prevents double-free.
- Views share the data pointer but don't own it. `lmlc_tensor_free` checks `owns_data` before calling `free(t->data)`.
- This was the right call — retrofitting this later would have been painful.

### Dtype-Agnostic Helpers

- `tensor_get_flat` / `tensor_set_flat` abstract away dtype dispatch.
- All math ops work in `float` internally — conversion happens at the helper boundary.
- This makes cross-dtype support trivial: just remove the dtype-match checks (done in v0.0.4).
- Trade-off: f32 intermediate loses precision for BF16/F16 chains. Acceptable for now.

### Broadcasting

- NumPy-style: right-aligned dimension matching, size-1 dims expand.
- `broadcast_shape` computes the output shape. `broadcast_offset` maps output indices to input offsets.
- Both helpers are static — kept in one place, reused by `add` and `mul`.
- Batched matmul (v0.0.4) reuses the same right-aligned logic for batch dims.

### Error Reporting

- Global `lmlc_error_info_t g_error` with `LMLC_SET_ERROR(code, msg)` macro.
- Macro captures `__func__` and `__LINE__` — gives function name + line number for debugging.
- `lmlc_last_error()` returns the code, `lmlc_last_error_info()` returns the full struct.
- Better than silent returns (which was the v0.0.1–0.0.3 approach).
- Thread-safety note: global error state is not thread-safe. Fine for now, TODO for later.

## Build System

- CMake with `enable_testing()` + `add_test()` for CTest integration.
- Static library target (`lmlc_tensor`) linked to test executable.
- Warning flags: `-Wall -Wextra -Wpedantic` (GCC/Clang), `/W4` (MSVC).
- Debug build by default — switch to Release for performance testing.
- Build: `cmake -B build -S . && cmake --build build --config Debug`
- Test: `.\build\test\Debug\test_lmlc_tensor.exe` (moved to root `test/` in v0.1.1)

## Dynamic Arrays: realloc Failure Handling (v0.1.0)

- `realloc` returns NULL on failure but **the original pointer is still valid**.
- Naive `t->shape = realloc(t->shape, ...)` leaks the old buffer on failure.
- Fix: save return to temp, only assign if non-NULL.

## In-Place Shape Ops Break Graph DAGs (v0.1.2)

- `reshape`/`transpose`/`squeeze`/`unsqueeze` modify the source tensor in-place.
- In a linear graph chain this works — each node sees the result of the previous.
- In a DAG, an earlier shape op changes the tensor for **all** downstream nodes, not just the intended one.
- Pattern: shape ops in a compute graph should produce a new tensor, not mutate in-place. (TODO for later.)

## Tracing Test Assertions (v0.1.2)

- Wrote `ASSERT(t->ndim == 3)` without tracing the actual op sequence.
- The ops were: reshape (2D→1D), transpose (fail, 1D), permute (fail, 1D), squeeze (fail, 1D), unsqueeze (1D→2D).
- Correct answer was 2, not 3. Test failed.
- Lesson: trace through the op chain before writing assertions.

## What Not To Do

- Don't rewrite entire files when adding features — use targeted edits.
- Don't add features outside the current version scope.
- Don't skip validation to save lines — bounds checking and error reporting matter.
- Don't use `/* */` for single-line comments in C++.
