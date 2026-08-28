#ifndef LMLC_TENSOR_H
#define LMLC_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// TODO: add I8, I16, I32, I64
typedef uint16_t lmlc_bf16_t;
typedef uint16_t lmlc_f16_t;
typedef uint8_t  lmlc_bool_t;

typedef enum {
    LMLC_DTYPE_F32 = 0,
    LMLC_DTYPE_BF16,
    LMLC_DTYPE_F16,
    LMLC_DTYPE_BOOL,
    LMLC_DTYPE_COUNT
} lmlc_dtype_t;

typedef struct {
    int   ndim;          // number of dimensions
    int64_t shape[8];    // TODO: dynamic shape array instead of fixed 8
    int64_t strides[8];  // element strides (not bytes)
    size_t  count;       // total number of elements
    lmlc_dtype_t dtype;
    void   *data;
    int    owns_data;    // 1 if this tensor owns the data buffer
} lmlc_tensor_t;

// ---- error reporting ----

typedef enum {
    LMLC_OK = 0,
    LMLC_ERR_NULL,
    LMLC_ERR_SHAPE_MISMATCH,
    LMLC_ERR_DTYPE_MISMATCH,
    LMLC_ERR_OUT_OF_BOUNDS,
    LMLC_ERR_INVALID_NDIM,
    LMLC_ERR_INVALID_SHAPE,
    LMLC_ERR_ALLOC_FAILED,
    LMLC_ERR_BROADCAST_FAILED,
    LMLC_ERR_INVALID_PERM,
    LMLC_ERR_MATMUL_DIM,
} lmlc_error_t;

typedef struct {
    lmlc_error_t code;
    const char*  msg;
    const char*  func;
    int          line;
} lmlc_error_info_t;

lmlc_error_t      lmlc_last_error(void);
const char*       lmlc_last_error_msg(void);
lmlc_error_info_t lmlc_last_error_info(void);
void              lmlc_clear_error(void);
const char*       lmlc_error_string(lmlc_error_t code);

// bf16 conversion
lmlc_bf16_t lmlc_f32_to_bf16(float f);
float       lmlc_bf16_to_f32(lmlc_bf16_t b);

// f16 conversion
lmlc_f16_t lmlc_f32_to_f16(float f);
float      lmlc_f16_to_f32(lmlc_f16_t h);

// create / free
lmlc_tensor_t* lmlc_tensor_create(int ndim, const int64_t* shape, lmlc_dtype_t dtype);
void           lmlc_tensor_free(lmlc_tensor_t* t);

// basic ops
void lmlc_tensor_zero(lmlc_tensor_t* t);
void lmlc_tensor_fill(lmlc_tensor_t* t, float value);
void lmlc_tensor_copy(const lmlc_tensor_t* src, lmlc_tensor_t* dst);

// shape ops
void lmlc_tensor_reshape(lmlc_tensor_t* t, int ndim, const int64_t* shape);
void lmlc_tensor_transpose(lmlc_tensor_t* t, int dim0, int dim1);
void lmlc_tensor_permute(lmlc_tensor_t* t, const int* perm);
void lmlc_tensor_squeeze(lmlc_tensor_t* t, int dim);
void lmlc_tensor_unsqueeze(lmlc_tensor_t* t, int dim);
lmlc_tensor_t* lmlc_tensor_view(lmlc_tensor_t* t, int ndim, const int64_t* shape);

// element access
float lmlc_tensor_get(const lmlc_tensor_t* t, const int64_t* indices);
void  lmlc_tensor_set(lmlc_tensor_t* t, const int64_t* indices, float value);

// math ops
void  lmlc_tensor_add(const lmlc_tensor_t* a, const lmlc_tensor_t* b, lmlc_tensor_t* out);
void  lmlc_tensor_mul(const lmlc_tensor_t* a, const lmlc_tensor_t* b, lmlc_tensor_t* out);
void  lmlc_tensor_scale(const lmlc_tensor_t* a, float scalar, lmlc_tensor_t* out);
void  lmlc_tensor_matmul(const lmlc_tensor_t* a, const lmlc_tensor_t* b, lmlc_tensor_t* out);
float lmlc_tensor_dot(const lmlc_tensor_t* a, const lmlc_tensor_t* b);
float lmlc_tensor_norm(const lmlc_tensor_t* t);

// debug
void lmlc_tensor_print(const lmlc_tensor_t* t);

#ifdef __cplusplus
}
#endif

#endif // LMLC_TENSOR_H
