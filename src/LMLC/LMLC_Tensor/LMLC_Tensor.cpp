#include "LMLC_Tensor.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>
#include <math.h>

// ---- error reporting ----

static lmlc_error_info_t g_error = { LMLC_OK, "no error", NULL, 0 };

#define LMLC_SET_ERROR(err_code, err_msg) do { \
    g_error.code = (err_code); \
    g_error.msg = (err_msg); \
    g_error.func = __func__; \
    g_error.line = __LINE__; \
} while (0)

lmlc_error_t lmlc_last_error(void) { return g_error.code; }

const char* lmlc_last_error_msg(void) { return g_error.msg; }

lmlc_error_info_t lmlc_last_error_info(void) { return g_error; }

void lmlc_clear_error(void) {
    g_error.code = LMLC_OK;
    g_error.msg = "no error";
    g_error.func = NULL;
    g_error.line = 0;
}

const char* lmlc_error_string(lmlc_error_t code) {
    switch (code) {
        case LMLC_OK:                   return "ok";
        case LMLC_ERR_NULL:             return "null pointer";
        case LMLC_ERR_SHAPE_MISMATCH:   return "shape mismatch";
        case LMLC_ERR_DTYPE_MISMATCH:   return "dtype mismatch";
        case LMLC_ERR_OUT_OF_BOUNDS:    return "index out of bounds";
        case LMLC_ERR_INVALID_NDIM:     return "invalid number of dimensions";
        case LMLC_ERR_INVALID_SHAPE:    return "invalid shape value";
        case LMLC_ERR_ALLOC_FAILED:     return "memory allocation failed";
        case LMLC_ERR_BROADCAST_FAILED: return "broadcast failed";
        case LMLC_ERR_INVALID_PERM:     return "invalid permutation";
        case LMLC_ERR_MATMUL_DIM:       return "matmul dimension mismatch";
        default:                        return "unknown error";
    }
}

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
    if (t->dtype == LMLC_DTYPE_BOOL) return (float)((lmlc_bool_t*)t->data)[offset];
    return ((float*)t->data)[offset];
}

static void tensor_set_flat(lmlc_tensor_t* t, size_t offset, float val) {
    if (t->dtype == LMLC_DTYPE_BF16) { ((lmlc_bf16_t*)t->data)[offset] = lmlc_f32_to_bf16(val); return; }
    if (t->dtype == LMLC_DTYPE_F16)  { ((lmlc_f16_t*)t->data)[offset] = lmlc_f32_to_f16(val); return; }
    if (t->dtype == LMLC_DTYPE_BOOL) { ((lmlc_bool_t*)t->data)[offset] = (val != 0.0f) ? 1 : 0; return; }
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
    switch (dtype) {
        case LMLC_DTYPE_F32:  return sizeof(float);
        case LMLC_DTYPE_BF16: return sizeof(lmlc_bf16_t);
        case LMLC_DTYPE_F16:  return sizeof(lmlc_f16_t);
        case LMLC_DTYPE_BOOL: return sizeof(lmlc_bool_t);
        default:              return sizeof(float);
    }
}

// ---- create / free ----

lmlc_tensor_t* lmlc_tensor_create(int ndim, const int64_t* shape, lmlc_dtype_t dtype) {
    if (ndim <= 0 || ndim > 8) {
        LMLC_SET_ERROR(LMLC_ERR_INVALID_NDIM, "ndim must be 1..8");
        return NULL;
    }
    if (!shape) {
        LMLC_SET_ERROR(LMLC_ERR_NULL, "shape is NULL");
        return NULL;
    }
    if (dtype < 0 || dtype >= LMLC_DTYPE_COUNT) {
        LMLC_SET_ERROR(LMLC_ERR_DTYPE_MISMATCH, "invalid dtype");
        return NULL;
    }
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) {
            LMLC_SET_ERROR(LMLC_ERR_INVALID_SHAPE, "shape dim must be > 0");
            return NULL;
        }
    }

    lmlc_tensor_t* t = (lmlc_tensor_t*)calloc(1, sizeof(lmlc_tensor_t));
    if (!t) {
        LMLC_SET_ERROR(LMLC_ERR_ALLOC_FAILED, "calloc tensor struct");
        return NULL;
    }

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
        LMLC_SET_ERROR(LMLC_ERR_ALLOC_FAILED, "calloc tensor data");
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
    if (!t || !t->data) { LMLC_SET_ERROR(LMLC_ERR_NULL, "null tensor or data"); return; }
    if (t->dtype == LMLC_DTYPE_BF16) {
        lmlc_bf16_t* data = (lmlc_bf16_t*)t->data;
        lmlc_bf16_t bv = lmlc_f32_to_bf16(value);
        for (size_t i = 0; i < t->count; i++) data[i] = bv;
    } else if (t->dtype == LMLC_DTYPE_F16) {
        lmlc_f16_t* data = (lmlc_f16_t*)t->data;
        lmlc_f16_t hv = lmlc_f32_to_f16(value);
        for (size_t i = 0; i < t->count; i++) data[i] = hv;
    } else if (t->dtype == LMLC_DTYPE_BOOL) {
        lmlc_bool_t* data = (lmlc_bool_t*)t->data;
        lmlc_bool_t bv = (value != 0.0f) ? 1 : 0;
        memset(data, bv, t->count * sizeof(lmlc_bool_t));
    } else {
        float* data = (float*)t->data;
        for (size_t i = 0; i < t->count; i++) data[i] = value;
    }
}

void lmlc_tensor_copy(const lmlc_tensor_t* src, lmlc_tensor_t* dst) {
    if (!src || !dst || !src->data || !dst->data) {
        LMLC_SET_ERROR(LMLC_ERR_NULL, "null tensor or data");
        return;
    }
    if (src->count != dst->count) {
        LMLC_SET_ERROR(LMLC_ERR_SHAPE_MISMATCH, "src and dst count mismatch");
        return;
    }
    if (src->dtype == dst->dtype) {
        memcpy(dst->data, src->data, src->count * dtype_size(src->dtype));
    } else {
        // cross-dtype: convert through f32
        for (size_t i = 0; i < src->count; i++) {
            tensor_set_flat(dst, i, tensor_get_flat(src, i));
        }
    }
}

// ---- element access ----

float lmlc_tensor_get(const lmlc_tensor_t* t, const int64_t* indices) {
    if (!t || !t->data || !indices) {
        LMLC_SET_ERROR(LMLC_ERR_NULL, "null tensor, data, or indices");
        return 0.0f;
    }
    size_t offset = 0;
    for (int i = 0; i < t->ndim; i++) {
        if (indices[i] < 0 || indices[i] >= t->shape[i]) {
            LMLC_SET_ERROR(LMLC_ERR_OUT_OF_BOUNDS, "index out of bounds");
            return 0.0f;
        }
        offset += (size_t)indices[i] * (size_t)t->strides[i];
    }
    return tensor_get_flat(t, offset);
}

void lmlc_tensor_set(lmlc_tensor_t* t, const int64_t* indices, float value) {
    if (!t || !t->data || !indices) {
        LMLC_SET_ERROR(LMLC_ERR_NULL, "null tensor, data, or indices");
        return;
    }
    size_t offset = 0;
    for (int i = 0; i < t->ndim; i++) {
        if (indices[i] < 0 || indices[i] >= t->shape[i]) {
            LMLC_SET_ERROR(LMLC_ERR_OUT_OF_BOUNDS, "index out of bounds");
            return;
        }
        offset += (size_t)indices[i] * (size_t)t->strides[i];
    }
    tensor_set_flat(t, offset, value);
}

// ---- math ops ----

void lmlc_tensor_add(const lmlc_tensor_t* a, const lmlc_tensor_t* b, lmlc_tensor_t* out) {
    if (!a || !b || !out || !a->data || !b->data || !out->data) {
        LMLC_SET_ERROR(LMLC_ERR_NULL, "null tensor or data");
        return;
    }

    int out_ndim;
    int64_t bshape[8];
    if (!broadcast_shape(a, b, &out_ndim, bshape)) {
        LMLC_SET_ERROR(LMLC_ERR_BROADCAST_FAILED, "shapes cannot broadcast");
        return;
    }

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
    if (!a || !b || !out || !a->data || !b->data || !out->data) {
        LMLC_SET_ERROR(LMLC_ERR_NULL, "null tensor or data");
        return;
    }

    int out_ndim;
    int64_t bshape[8];
    if (!broadcast_shape(a, b, &out_ndim, bshape)) {
        LMLC_SET_ERROR(LMLC_ERR_BROADCAST_FAILED, "shapes cannot broadcast");
        return;
    }

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
    if (!a || !out || !a->data || !out->data) {
        LMLC_SET_ERROR(LMLC_ERR_NULL, "null tensor or data");
        return;
    }
    if (a->count != out->count) {
        LMLC_SET_ERROR(LMLC_ERR_SHAPE_MISMATCH, "a and out count mismatch");
        return;
    }

    for (size_t i = 0; i < a->count; i++) {
        tensor_set_flat(out, i, tensor_get_flat(a, i) * scalar);
    }
}

void lmlc_tensor_matmul(const lmlc_tensor_t* a, const lmlc_tensor_t* b, lmlc_tensor_t* out) {
    if (!a || !b || !out || !a->data || !b->data || !out->data) {
        LMLC_SET_ERROR(LMLC_ERR_NULL, "null tensor or data");
        return;
    }
    if (a->ndim < 2 || b->ndim < 2 || out->ndim < 2) {
        LMLC_SET_ERROR(LMLC_ERR_INVALID_NDIM, "matmul requires ndim >= 2");
        return;
    }

    int a_batch = a->ndim - 2;
    int b_batch = b->ndim - 2;
    int max_batch = a_batch > b_batch ? a_batch : b_batch;
    int out_ndim = max_batch + 2;

    if (out->ndim != out_ndim) {
        LMLC_SET_ERROR(LMLC_ERR_SHAPE_MISMATCH, "out ndim does not match broadcasted batch");
        return;
    }

    // check batch dims are broadcastable
    int64_t batch_shape[8];
    for (int i = 0; i < max_batch; i++) {
        int64_t da = (i < max_batch - a_batch) ? 1 : a->shape[i - (max_batch - a_batch)];
        int64_t db = (i < max_batch - b_batch) ? 1 : b->shape[i - (max_batch - b_batch)];
        if (da != db && da != 1 && db != 1) {
            LMLC_SET_ERROR(LMLC_ERR_BROADCAST_FAILED, "batch dims cannot broadcast");
            return;
        }
        batch_shape[i] = da > db ? da : db;
        if (out->shape[i] != batch_shape[i]) {
            LMLC_SET_ERROR(LMLC_ERR_SHAPE_MISMATCH, "out batch shape mismatch");
            return;
        }
    }

    int64_t M = a->shape[a->ndim - 2];
    int64_t K = a->shape[a->ndim - 1];
    int64_t K2 = b->shape[b->ndim - 2];
    int64_t N = b->shape[b->ndim - 1];

    if (K != K2) {
        LMLC_SET_ERROR(LMLC_ERR_MATMUL_DIM, "a cols != b rows");
        return;
    }
    if (out->shape[out_ndim - 2] != M || out->shape[out_ndim - 1] != N) {
        LMLC_SET_ERROR(LMLC_ERR_SHAPE_MISMATCH, "out shape must be [batch..., M, N]");
        return;
    }

    // compute total batch count
    size_t batch_count = 1;
    for (int i = 0; i < max_batch; i++) batch_count *= (size_t)batch_shape[i];

    // compute batch strides (in elements, not bytes)
    size_t a_batch_stride = (size_t)M * (size_t)K;
    size_t b_batch_stride = (size_t)K * (size_t)N;
    size_t o_batch_stride = (size_t)M * (size_t)N;

    int64_t bidx[8] = {0};
    for (size_t bi = 0; bi < batch_count; bi++) {
        // compute a and b batch offsets
        size_t a_off_base = 0, b_off_base = 0;
        int a_dim_off = max_batch - a_batch;
        int b_dim_off = max_batch - b_batch;
        for (int i = 0; i < a_batch; i++) {
            int64_t idx = bidx[i + a_dim_off];
            if (a->shape[i] == 1) idx = 0;
            a_off_base += (size_t)idx * a_batch_stride;
        }
        for (int i = 0; i < b_batch; i++) {
            int64_t idx = bidx[i + b_dim_off];
            if (b->shape[i] == 1) idx = 0;
            b_off_base += (size_t)idx * b_batch_stride;
        }
        size_t o_off_base = bi * o_batch_stride;

        for (int64_t i = 0; i < M; i++) {
            for (int64_t j = 0; j < N; j++) {
                float sum = 0.0f;
                for (int64_t k = 0; k < K; k++) {
                    size_t a_off = a_off_base + (size_t)i * (size_t)K + (size_t)k;
                    size_t b_off = b_off_base + (size_t)k * (size_t)N + (size_t)j;
                    sum += tensor_get_flat(a, a_off) * tensor_get_flat(b, b_off);
                }
                tensor_set_flat(out, o_off_base + (size_t)i * (size_t)N + (size_t)j, sum);
            }
        }

        // increment batch index
        for (int d = max_batch - 1; d >= 0; d--) {
            if (++bidx[d] < batch_shape[d]) break;
            bidx[d] = 0;
        }
    }
}

float lmlc_tensor_dot(const lmlc_tensor_t* a, const lmlc_tensor_t* b) {
    if (!a || !b || !a->data || !b->data) {
        LMLC_SET_ERROR(LMLC_ERR_NULL, "null tensor or data");
        return 0.0f;
    }
    if (a->count != b->count) {
        LMLC_SET_ERROR(LMLC_ERR_SHAPE_MISMATCH, "a and b count mismatch");
        return 0.0f;
    }

    float sum = 0.0f;
    for (size_t i = 0; i < a->count; i++) {
        sum += tensor_get_flat(a, i) * tensor_get_flat(b, i);
    }
    return sum;
}

float lmlc_tensor_norm(const lmlc_tensor_t* t) {
    if (!t || !t->data) {
        LMLC_SET_ERROR(LMLC_ERR_NULL, "null tensor or data");
        return 0.0f;
    }
    float sum = 0.0f;
    for (size_t i = 0; i < t->count; i++) {
        float v = tensor_get_flat(t, i);
        sum += v * v;
    }
    return sqrtf(sum);
}

// ---- shape ops ----

void lmlc_tensor_reshape(lmlc_tensor_t* t, int ndim, const int64_t* shape) {
    if (!t) { LMLC_SET_ERROR(LMLC_ERR_NULL, "null tensor"); return; }
    if (!shape) { LMLC_SET_ERROR(LMLC_ERR_NULL, "null shape"); return; }
    if (ndim <= 0 || ndim > 8) { LMLC_SET_ERROR(LMLC_ERR_INVALID_NDIM, "ndim must be 1..8"); return; }
    size_t new_count = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) { LMLC_SET_ERROR(LMLC_ERR_INVALID_SHAPE, "shape dim must be > 0"); return; }
        new_count *= (size_t)shape[i];
    }
    if (new_count != t->count) { LMLC_SET_ERROR(LMLC_ERR_SHAPE_MISMATCH, "reshape count mismatch"); return; }
    t->ndim = ndim;
    for (int i = 0; i < ndim; i++) t->shape[i] = shape[i];
    compute_strides(t);
}

void lmlc_tensor_transpose(lmlc_tensor_t* t, int dim0, int dim1) {
    if (!t) { LMLC_SET_ERROR(LMLC_ERR_NULL, "null tensor"); return; }
    if (dim0 < 0 || dim1 < 0 || dim0 >= t->ndim || dim1 >= t->ndim) {
        LMLC_SET_ERROR(LMLC_ERR_OUT_OF_BOUNDS, "transpose dim out of bounds");
        return;
    }
    int64_t tmp = t->shape[dim0];
    t->shape[dim0] = t->shape[dim1];
    t->shape[dim1] = tmp;
    tmp = t->strides[dim0];
    t->strides[dim0] = t->strides[dim1];
    t->strides[dim1] = tmp;
}

void lmlc_tensor_permute(lmlc_tensor_t* t, const int* perm) {
    if (!t) { LMLC_SET_ERROR(LMLC_ERR_NULL, "null tensor"); return; }
    if (!perm) { LMLC_SET_ERROR(LMLC_ERR_NULL, "null perm"); return; }
    // validate permutation
    int seen[8] = {0};
    for (int i = 0; i < t->ndim; i++) {
        if (perm[i] < 0 || perm[i] >= t->ndim) {
            LMLC_SET_ERROR(LMLC_ERR_INVALID_PERM, "perm index out of bounds");
            return;
        }
        if (seen[perm[i]]) {
            LMLC_SET_ERROR(LMLC_ERR_INVALID_PERM, "perm is not a valid permutation");
            return;
        }
        seen[perm[i]] = 1;
    }
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
    if (!t) { LMLC_SET_ERROR(LMLC_ERR_NULL, "null tensor"); return; }
    if (dim < 0 || dim >= t->ndim) { LMLC_SET_ERROR(LMLC_ERR_OUT_OF_BOUNDS, "squeeze dim out of bounds"); return; }
    if (t->shape[dim] != 1) { LMLC_SET_ERROR(LMLC_ERR_INVALID_SHAPE, "squeeze dim must be size 1"); return; }
    for (int i = dim; i < t->ndim - 1; i++) {
        t->shape[i] = t->shape[i + 1];
        t->strides[i] = t->strides[i + 1];
    }
    t->ndim--;
}

void lmlc_tensor_unsqueeze(lmlc_tensor_t* t, int dim) {
    if (!t) { LMLC_SET_ERROR(LMLC_ERR_NULL, "null tensor"); return; }
    if (dim < 0 || dim > t->ndim) { LMLC_SET_ERROR(LMLC_ERR_OUT_OF_BOUNDS, "unsqueeze dim out of bounds"); return; }
    if (t->ndim >= 8) { LMLC_SET_ERROR(LMLC_ERR_INVALID_NDIM, "ndim would exceed 8"); return; }
    for (int i = t->ndim; i > dim; i--) {
        t->shape[i] = t->shape[i - 1];
        t->strides[i] = t->strides[i - 1];
    }
    t->shape[dim] = 1;
    t->strides[dim] = 0; // size-1 dim, stride unused
    t->ndim++;
}

lmlc_tensor_t* lmlc_tensor_view(lmlc_tensor_t* t, int ndim, const int64_t* shape) {
    if (!t) { LMLC_SET_ERROR(LMLC_ERR_NULL, "null tensor"); return NULL; }
    if (!shape) { LMLC_SET_ERROR(LMLC_ERR_NULL, "null shape"); return NULL; }
    if (ndim <= 0 || ndim > 8) { LMLC_SET_ERROR(LMLC_ERR_INVALID_NDIM, "ndim must be 1..8"); return NULL; }

    size_t new_count = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0) { LMLC_SET_ERROR(LMLC_ERR_INVALID_SHAPE, "shape dim must be > 0"); return NULL; }
        new_count *= (size_t)shape[i];
    }
    if (new_count != t->count) { LMLC_SET_ERROR(LMLC_ERR_SHAPE_MISMATCH, "view count mismatch"); return NULL; }

    lmlc_tensor_t* v = (lmlc_tensor_t*)calloc(1, sizeof(lmlc_tensor_t));
    if (!v) { LMLC_SET_ERROR(LMLC_ERR_ALLOC_FAILED, "calloc view struct"); return NULL; }
    v->ndim = ndim;
    v->dtype = t->dtype;
    v->count = new_count;
    for (int i = 0; i < ndim; i++) v->shape[i] = shape[i];
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
