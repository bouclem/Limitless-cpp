#include "LMLC_Tensor.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

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

// ---- f16 conversion ----

lmlc_f16_t lmlc_f32_to_f16(float f) {
    uint32_t u;
    memcpy(&u, &f, sizeof(u));
    uint32_t sign = (u >> 16) & 0x8000;
    int32_t  exp  = (u >> 23) & 0xFF;
    uint32_t mant = u & 0x7FFFFF;

    if (exp == 0xFF) {
        // inf or nan
        return (lmlc_f16_t)(sign | 0x7C00 | (mant ? 0x200 : 0));
    }

    exp = exp - 127 + 15;

    if (exp >= 0x1F) {
        // overflow -> inf
        return (lmlc_f16_t)(sign | 0x7C00);
    }

    if (exp <= 0) {
        if (exp < -10) return (lmlc_f16_t)sign;
        // subnormal
        mant |= 0x800000;
        uint32_t shift = (uint32_t)(14 - exp);
        uint32_t half  = 1u << (shift - 1);
        uint32_t mask  = (1u << shift) - 1;
        uint32_t rounded = mant >> shift;
        if ((mant & mask) > half || ((mant & mask) == half && (rounded & 1)))
            rounded++;
        return (lmlc_f16_t)(sign | rounded);
    }

    // normal: round 23-bit mantissa to 10 bits
    uint32_t shift = 13;
    uint32_t half  = 1u << (shift - 1);
    uint32_t mask  = (1u << shift) - 1;
    uint32_t rounded = mant >> shift;
    if ((mant & mask) > half || ((mant & mask) == half && (rounded & 1))) {
        rounded++;
        if (rounded == 0x400) { rounded = 0; exp++; }
    }
    if (exp >= 0x1F) return (lmlc_f16_t)(sign | 0x7C00);
    return (lmlc_f16_t)(sign | ((uint32_t)exp << 10) | rounded);
}

float lmlc_f16_to_f32(lmlc_f16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000) << 16;
    int32_t  exp  = ((uint32_t)h >> 10) & 0x1F;
    uint32_t mant = (uint32_t)h & 0x3FF;

    if (exp == 0) {
        if (mant == 0) {
            uint32_t u = sign;
            float f; memcpy(&f, &u, sizeof(f));
            return f;
        }
        // subnormal -> normalize
        while (!(mant & 0x400)) { mant <<= 1; exp--; }
        exp++;
        mant &= ~0x400u;
    } else if (exp == 0x1F) {
        // inf or nan
        uint32_t u = sign | 0x7F800000 | (mant << 13);
        float f; memcpy(&f, &u, sizeof(f));
        return f;
    }

    uint32_t u = sign | ((uint32_t)(exp + 112) << 23) | (mant << 13);
    float f; memcpy(&f, &u, sizeof(f));
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

static float tensor_get_flat(const lmlc_tensor_t* t, size_t offset) {
    if (t->dtype == LMLC_DTYPE_BF16) return lmlc_bf16_to_f32(((lmlc_bf16_t*)t->data)[offset]);
    if (t->dtype == LMLC_DTYPE_F16)  return lmlc_f16_to_f32(((lmlc_f16_t*)t->data)[offset]);
    return ((float*)t->data)[offset];
}

static void tensor_set_flat(lmlc_tensor_t* t, size_t offset, float val) {
    if (t->dtype == LMLC_DTYPE_BF16) { ((lmlc_bf16_t*)t->data)[offset] = lmlc_f32_to_bf16(val); return; }
    if (t->dtype == LMLC_DTYPE_F16)  { ((lmlc_f16_t*)t->data)[offset] = lmlc_f32_to_f16(val); return; }
    ((float*)t->data)[offset] = val;
}

static int broadcast_shape(const lmlc_tensor_t* a, const lmlc_tensor_t* b,
                           int* out_ndim, int64_t* out_shape) {
    int max_ndim = a->ndim > b->ndim ? a->ndim : b->ndim;
    for (int i = 0; i < max_ndim; i++) {
        int64_t da = (i < max_ndim - a->ndim) ? 1 : a->shape[i - (max_ndim - a->ndim)];
        int64_t db = (i < max_ndim - b->ndim) ? 1 : b->shape[i - (max_ndim - b->ndim)];
        if (da != db && da != 1 && db != 1) return 0;
        out_shape[i] = da > db ? da : db;
    }
    *out_ndim = max_ndim;
    return 1;
}

static size_t broadcast_offset(const lmlc_tensor_t* t, const int64_t* out_idx, int out_ndim) {
    size_t offset = 0;
    int dim_offset = out_ndim - t->ndim;
    for (int i = 0; i < t->ndim; i++) {
        int64_t idx = out_idx[i + dim_offset];
        if (t->shape[i] == 1) idx = 0;
        offset += (size_t)idx * (size_t)t->strides[i];
    }
    return offset;
}

static size_t dtype_size(lmlc_dtype_t dtype) {
    // TODO: handle other dtypes
    switch (dtype) {
        case LMLC_DTYPE_F32:  return sizeof(float);
        case LMLC_DTYPE_BF16: return sizeof(lmlc_bf16_t);
        case LMLC_DTYPE_F16:  return sizeof(lmlc_f16_t);
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
    } else if (t->dtype == LMLC_DTYPE_F16) {
        lmlc_f16_t* data = (lmlc_f16_t*)t->data;
        lmlc_f16_t hv = lmlc_f32_to_f16(value);
        for (size_t i = 0; i < t->count; i++) data[i] = hv;
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
    return tensor_get_flat(t, offset);
}

void lmlc_tensor_set(lmlc_tensor_t* t, const int64_t* indices, float value) {
    // TODO: bounds checking
    if (!t || !t->data || !indices) return;
    size_t offset = 0;
    for (int i = 0; i < t->ndim; i++) {
        offset += (size_t)indices[i] * (size_t)t->strides[i];
    }
    tensor_set_flat(t, offset, value);
}

// ---- math ops ----

void lmlc_tensor_add(const lmlc_tensor_t* a, const lmlc_tensor_t* b, lmlc_tensor_t* out) {
    // TODO: cross-dtype support
    if (!a || !b || !out || !a->data || !b->data || !out->data) return;
    if (a->dtype != b->dtype || a->dtype != out->dtype) return;

    int out_ndim;
    int64_t bshape[8];
    if (!broadcast_shape(a, b, &out_ndim, bshape)) return;

    int64_t idx[8] = {0};
    for (size_t i = 0; i < out->count; i++) {
        size_t a_off = broadcast_offset(a, idx, out_ndim);
        size_t b_off = broadcast_offset(b, idx, out_ndim);
        tensor_set_flat(out, i, tensor_get_flat(a, a_off) + tensor_get_flat(b, b_off));
        for (int d = out_ndim - 1; d >= 0; d--) {
            if (++idx[d] < bshape[d]) break;
            idx[d] = 0;
        }
    }
}

void lmlc_tensor_mul(const lmlc_tensor_t* a, const lmlc_tensor_t* b, lmlc_tensor_t* out) {
    // TODO: cross-dtype support
    if (!a || !b || !out || !a->data || !b->data || !out->data) return;
    if (a->dtype != b->dtype || a->dtype != out->dtype) return;

    int out_ndim;
    int64_t bshape[8];
    if (!broadcast_shape(a, b, &out_ndim, bshape)) return;

    int64_t idx[8] = {0};
    for (size_t i = 0; i < out->count; i++) {
        size_t a_off = broadcast_offset(a, idx, out_ndim);
        size_t b_off = broadcast_offset(b, idx, out_ndim);
        tensor_set_flat(out, i, tensor_get_flat(a, a_off) * tensor_get_flat(b, b_off));
        for (int d = out_ndim - 1; d >= 0; d--) {
            if (++idx[d] < bshape[d]) break;
            idx[d] = 0;
        }
    }
}

void lmlc_tensor_scale(const lmlc_tensor_t* a, float scalar, lmlc_tensor_t* out) {
    // TODO: cross-dtype support
    if (!a || !out || !a->data || !out->data) return;
    if (a->dtype != out->dtype) return;
    if (a->count != out->count) return;

    for (size_t i = 0; i < a->count; i++) {
        tensor_set_flat(out, i, tensor_get_flat(a, i) * scalar);
    }
}

void lmlc_tensor_matmul(const lmlc_tensor_t* a, const lmlc_tensor_t* b, lmlc_tensor_t* out) {
    // TODO: batched matmul (ndim > 2)
    // TODO: broadcasting for batch dims
    // TODO: cross-dtype support
    if (!a || !b || !out || !a->data || !b->data || !out->data) return;
    if (a->ndim != 2 || b->ndim != 2 || out->ndim != 2) return;
    if (a->shape[1] != b->shape[0]) return;
    if (out->shape[0] != a->shape[0] || out->shape[1] != b->shape[1]) return;
    if (a->dtype != b->dtype || a->dtype != out->dtype) return;

    int64_t M = a->shape[0];
    int64_t K = a->shape[1];
    int64_t N = b->shape[1];

    for (int64_t i = 0; i < M; i++) {
        for (int64_t j = 0; j < N; j++) {
            float sum = 0.0f;
            for (int64_t k = 0; k < K; k++) {
                size_t a_off = (size_t)i * (size_t)a->strides[0] + (size_t)k * (size_t)a->strides[1];
                size_t b_off = (size_t)k * (size_t)b->strides[0] + (size_t)j * (size_t)b->strides[1];
                sum += tensor_get_flat(a, a_off) * tensor_get_flat(b, b_off);
            }
            size_t o_off = (size_t)i * (size_t)out->strides[0] + (size_t)j * (size_t)out->strides[1];
            tensor_set_flat(out, o_off, sum);
        }
    }
}

float lmlc_tensor_dot(const lmlc_tensor_t* a, const lmlc_tensor_t* b) {
    // TODO: cross-dtype support
    if (!a || !b || !a->data || !b->data) return 0.0f;
    if (a->count != b->count) return 0.0f;
    if (a->dtype != b->dtype) return 0.0f;

    float sum = 0.0f;
    for (size_t i = 0; i < a->count; i++) {
        sum += tensor_get_flat(a, i) * tensor_get_flat(b, i);
    }
    return sum;
}

float lmlc_tensor_norm(const lmlc_tensor_t* t) {
    if (!t || !t->data) return 0.0f;
    float sum = 0.0f;
    for (size_t i = 0; i < t->count; i++) {
        float v = tensor_get_flat(t, i);
        sum += v * v;
    }
    return sqrtf(sum);
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
        val = tensor_get_flat(t, i);
        printf("%.4f", val);
        if (i < show - 1) printf(", ");
    }
    if (t->count > 10) printf(", ...");
    printf("]\n");
}
