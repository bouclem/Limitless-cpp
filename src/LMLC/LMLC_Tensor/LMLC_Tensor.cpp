#include "LMLC_Tensor.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

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
        case LMLC_DTYPE_F32: return sizeof(float);
        default:             return sizeof(float);
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
    t->data = (float*)calloc(t->count, dtype_size(dtype));
    if (!t->data) {
        free(t);
        return NULL;
    }

    return t;
}

void lmlc_tensor_free(lmlc_tensor_t* t) {
    if (!t) return;
    free(t->data);
    free(t);
}

// ---- basic ops ----

void lmlc_tensor_zero(lmlc_tensor_t* t) {
    if (!t || !t->data) return;
    memset(t->data, 0, t->count * sizeof(float));
}

void lmlc_tensor_fill(lmlc_tensor_t* t, float value) {
    if (!t || !t->data) return;
    for (size_t i = 0; i < t->count; i++) {
        t->data[i] = value;
    }
}

void lmlc_tensor_copy(const lmlc_tensor_t* src, lmlc_tensor_t* dst) {
    // TODO: check shape match
    // TODO: check dtype match
    if (!src || !dst || !src->data || !dst->data) return;
    if (src->count != dst->count) return;
    memcpy(dst->data, src->data, src->count * sizeof(float));
}

// ---- element access ----

float lmlc_tensor_get(const lmlc_tensor_t* t, const int64_t* indices) {
    // TODO: bounds checking
    if (!t || !t->data || !indices) return 0.0f;
    size_t offset = 0;
    for (int i = 0; i < t->ndim; i++) {
        offset += (size_t)indices[i] * (size_t)t->strides[i];
    }
    return t->data[offset];
}

void lmlc_tensor_set(lmlc_tensor_t* t, const int64_t* indices, float value) {
    // TODO: bounds checking
    if (!t || !t->data || !indices) return;
    size_t offset = 0;
    for (int i = 0; i < t->ndim; i++) {
        offset += (size_t)indices[i] * (size_t)t->strides[i];
    }
    t->data[offset] = value;
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
        printf("%.4f", t->data[i]);
        if (i < show - 1) printf(", ");
    }
    if (t->count > 10) printf(", ...");
    printf("]\n");
}
