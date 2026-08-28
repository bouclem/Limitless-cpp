#include "LMLC_Tensor.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ---- bf16 conversion ----

lmlc_bf16_t lmlc_f32_to_bf16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    // round to nearest even, take upper 16 bits
    return (lmlc_bf16_t)((u + 0x8000) >> 16);
}

float lmlc_bf16_to_f32(lmlc_bf16_t b) {
    uint32_t u = (uint32_t)b << 16;
    float f;
    memcpy(&f, &u, sizeof(f));
    return f;
}

// ---- helpers ----

static void compute_strides(lmlc_tensor_t* t) {
    if (t->ndim <= 0) return;
    t->strides[t->ndim - 1] = 1;
    for (int i = t->ndim - 2; i >= 0; i--) {
        t->strides[i] = t->strides[i + 1] * t->shape[i + 1];
    }
}

static size_t dtype_size(lmlc_dtype_t dtype) {
    // TODO: handle other dtypes
    switch (dtype) {
        case LMLC_DTYPE_F32:  return sizeof(float);
        case LMLC_DTYPE_BF16: return sizeof(lmlc_bf16_t);
        default:              return sizeof(float);
    }
}

// ---- create / free ----

lmlc_tensor_t* lmlc_tensor_create(int ndim, const int64_t* shape, lmlc_dtype_t dtype) {
    // TODO: validate ndim > 0 and <= 8
    // TODO: validate shape values > 0

    lmlc_tensor_t* t = (lmlc_tensor_t*)calloc(1, sizeof(lmlc_tensor_t));
    if (!t) return NULL;

    t->ndim  = ndim;
    t->dtype = dtype;

    t->count = 1;
    for (int i = 0; i < ndim; i++) {
        t->shape[i] = shape[i];
        t->count *= (size_t)shape[i];
    }

    compute_strides(t);

    // TODO: aligned allocation (SIMD-friendly)
    t->data = calloc(t->count, dtype_size(dtype));
    if (!t->data) {
        free(t);
        return NULL;
    }
    t->owns_data = 1;

    return t;
}

void lmlc_tensor_free(lmlc_tensor_t* t) {
    if (!t) return;
    if (t->owns_data) free(t->data);
    free(t);
}

// ---- basic ops ----

void lmlc_tensor_zero(lmlc_tensor_t* t) {
    if (!t || !t->data) return;
    memset(t->data, 0, t->count * dtype_size(t->dtype));
}

void lmlc_tensor_fill(lmlc_tensor_t* t, float value) {
    if (!t || !t->data) return;
    if (t->dtype == LMLC_DTYPE_BF16) {
        lmlc_bf16_t* data = (lmlc_bf16_t*)t->data;
        lmlc_bf16_t bv = lmlc_f32_to_bf16(value);
        for (size_t i = 0; i < t->count; i++) data[i] = bv;
    } else {
        float* data = (float*)t->data;
        for (size_t i = 0; i < t->count; i++) data[i] = value;
    }
}

void lmlc_tensor_copy(const lmlc_tensor_t* src, lmlc_tensor_t* dst) {
    // TODO: check shape match
    // TODO: cross-dtype conversion
    if (!src || !dst || !src->data || !dst->data) return;
    if (src->count != dst->count) return;
    if (src->dtype != dst->dtype) return; // TODO: cross-dtype copy
    memcpy(dst->data, src->data, src->count * dtype_size(src->dtype));
}

// ---- element access ----

float lmlc_tensor_get(const lmlc_tensor_t* t, const int64_t* indices) {
    // TODO: bounds checking
    if (!t || !t->data || !indices) return 0.0f;
    size_t offset = 0;
    for (int i = 0; i < t->ndim; i++) {
        offset += (size_t)indices[i] * (size_t)t->strides[i];
    }
    if (t->dtype == LMLC_DTYPE_BF16) {
        return lmlc_bf16_to_f32(((lmlc_bf16_t*)t->data)[offset]);
    }
    return ((float*)t->data)[offset];
}

void lmlc_tensor_set(lmlc_tensor_t* t, const int64_t* indices, float value) {
    // TODO: bounds checking
    if (!t || !t->data || !indices) return;
    size_t offset = 0;
    for (int i = 0; i < t->ndim; i++) {
        offset += (size_t)indices[i] * (size_t)t->strides[i];
    }
    if (t->dtype == LMLC_DTYPE_BF16) {
        ((lmlc_bf16_t*)t->data)[offset] = lmlc_f32_to_bf16(value);
    } else {
        ((float*)t->data)[offset] = value;
    }
}

// ---- math ops ----

void lmlc_tensor_add(const lmlc_tensor_t* a, const lmlc_tensor_t* b, lmlc_tensor_t* out) {
    // TODO: check shapes match
    // TODO: check dtypes match
    // TODO: broadcasting
    if (!a || !b || !out || !a->data || !b->data || !out->data) return;
    if (a->count != b->count || a->count != out->count) return;
    if (a->dtype != b->dtype || a->dtype != out->dtype) return;

    if (a->dtype == LMLC_DTYPE_BF16) {
        lmlc_bf16_t* ad = (lmlc_bf16_t*)a->data;
        lmlc_bf16_t* bd = (lmlc_bf16_t*)b->data;
        lmlc_bf16_t* od = (lmlc_bf16_t*)out->data;
        for (size_t i = 0; i < a->count; i++) {
            od[i] = lmlc_f32_to_bf16(lmlc_bf16_to_f32(ad[i]) + lmlc_bf16_to_f32(bd[i]));
        }
    } else {
        float* ad = (float*)a->data;
        float* bd = (float*)b->data;
        float* od = (float*)out->data;
        for (size_t i = 0; i < a->count; i++) {
            od[i] = ad[i] + bd[i];
        }
    }
}

// ---- shape ops ----

void lmlc_tensor_reshape(lmlc_tensor_t* t, int ndim, const int64_t* shape) {
    // TODO: error reporting instead of silent return
    if (!t) return;
    size_t new_count = 1;
    for (int i = 0; i < ndim; i++) new_count *= (size_t)shape[i];
    if (new_count != t->count) return;
    t->ndim = ndim;
    for (int i = 0; i < ndim; i++) t->shape[i] = shape[i];
    compute_strides(t);
}

void lmlc_tensor_transpose(lmlc_tensor_t* t, int dim0, int dim1) {
    // TODO: bounds check dim0, dim1
    if (!t || dim0 < 0 || dim1 < 0 || dim0 >= t->ndim || dim1 >= t->ndim) return;
    int64_t tmp = t->shape[dim0];
    t->shape[dim0] = t->shape[dim1];
    t->shape[dim1] = tmp;
    tmp = t->strides[dim0];
    t->strides[dim0] = t->strides[dim1];
    t->strides[dim1] = tmp;
}

void lmlc_tensor_permute(lmlc_tensor_t* t, const int* perm) {
    // TODO: validate perm is a valid permutation
    if (!t || !perm) return;
    int64_t new_shape[8], new_strides[8];
    for (int i = 0; i < t->ndim; i++) {
        new_shape[i] = t->shape[perm[i]];
        new_strides[i] = t->strides[perm[i]];
    }
    for (int i = 0; i < t->ndim; i++) {
        t->shape[i] = new_shape[i];
        t->strides[i] = new_strides[i];
    }
}

void lmlc_tensor_squeeze(lmlc_tensor_t* t, int dim) {
    // TODO: check shape[dim] == 1
    if (!t || dim < 0 || dim >= t->ndim || t->shape[dim] != 1) return;
    for (int i = dim; i < t->ndim - 1; i++) {
        t->shape[i] = t->shape[i + 1];
        t->strides[i] = t->strides[i + 1];
    }
    t->ndim--;
}

void lmlc_tensor_unsqueeze(lmlc_tensor_t* t, int dim) {
    // TODO: check ndim < 8
    if (!t || dim < 0 || dim > t->ndim || t->ndim >= 8) return;
    for (int i = t->ndim; i > dim; i--) {
        t->shape[i] = t->shape[i - 1];
        t->strides[i] = t->strides[i - 1];
    }
    t->shape[dim] = 1;
    t->strides[dim] = 0; // size-1 dim, stride unused
    t->ndim++;
}

lmlc_tensor_t* lmlc_tensor_view(lmlc_tensor_t* t, int ndim, const int64_t* shape) {
    // TODO: check new count == old count
    if (!t) return NULL;
    lmlc_tensor_t* v = (lmlc_tensor_t*)calloc(1, sizeof(lmlc_tensor_t));
    if (!v) return NULL;
    v->ndim = ndim;
    v->dtype = t->dtype;
    v->count = 1;
    for (int i = 0; i < ndim; i++) {
        v->shape[i] = shape[i];
        v->count *= (size_t)shape[i];
    }
    compute_strides(v);
    v->data = t->data;
    v->owns_data = 0;
    return v;
}

// ---- debug ----

void lmlc_tensor_print(const lmlc_tensor_t* t) {
    // TODO: multi-dimensional pretty print
    // TODO: handle ndim > 2
    if (!t) {
        printf("(null tensor)\n");
        return;
    }

    printf("tensor[");
    for (int i = 0; i < t->ndim; i++) {
        printf("%lld", (long long)t->shape[i]);
        if (i < t->ndim - 1) printf(", ");
    }
    printf("] dtype=%d count=%zu\n", t->dtype, t->count);

    // flat print for now
    printf("  data: [");
    size_t show = t->count < 10 ? t->count : 10;
    for (size_t i = 0; i < show; i++) {
        float val;
        if (t->dtype == LMLC_DTYPE_BF16) {
            val = lmlc_bf16_to_f32(((lmlc_bf16_t*)t->data)[i]);
        } else {
            val = ((float*)t->data)[i];
        }
        printf("%.4f", val);
        if (i < show - 1) printf(", ");
    }
    if (t->count > 10) printf(", ...");
    printf("]\n");
}
