#ifndef LMLC_TENSOR_H
#define LMLC_TENSOR_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* TODO: add F16, BF16, I8, I16, I32, I64, BOOL */
typedef enum {
    LMLC_DTYPE_F32 = 0,
    LMLC_DTYPE_COUNT
} lmlc_dtype_t;

typedef struct {
    int   ndim;          /* number of dimensions */
    int64_t shape[8];    /* TODO: dynamic shape array instead of fixed 8 */
    int64_t strides[8];  /* element strides (not bytes) */
    size_t  count;       /* total number of elements */
    lmlc_dtype_t dtype;
    float  *data;        /* TODO: void* for multi-dtype support */
} lmlc_tensor_t;

/* TODO: tensor view (no ownership), tensor with offset */

/* create / free */
lmlc_tensor_t* lmlc_tensor_create(int ndim, const int64_t* shape, lmlc_dtype_t dtype);
void           lmlc_tensor_free(lmlc_tensor_t* t);

/* basic ops */
void lmlc_tensor_zero(lmlc_tensor_t* t);
void lmlc_tensor_fill(lmlc_tensor_t* t, float value);
void lmlc_tensor_copy(const lmlc_tensor_t* src, lmlc_tensor_t* dst);

/* shape ops */
/* TODO: lmlc_tensor_reshape */
/* TODO: lmlc_tensor_transpose */
/* TODO: lmlc_tensor_permute */
/* TODO: lmlc_tensor_squeeze */
/* TODO: lmlc_tensor_unsqueeze */
/* TODO: lmlc_tensor_view */

/* element access */
float lmlc_tensor_get(const lmlc_tensor_t* t, const int64_t* indices);
void  lmlc_tensor_set(lmlc_tensor_t* t, const int64_t* indices, float value);

/* math ops */
/* TODO: lmlc_tensor_add */
/* TODO: lmlc_tensor_mul */
/* TODO: lmlc_tensor_matmul */
/* TODO: lmlc_tensor_scale */
/* TODO: lmlc_tensor_dot */
/* TODO: lmlc_tensor_norm */

/* debug */
void lmlc_tensor_print(const lmlc_tensor_t* t);

#ifdef __cplusplus
}
#endif

#endif /* LMLC_TENSOR_H */
