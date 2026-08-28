#include "LMLC_Tensor.h"
#include "LMLC_Graph.h"

#include <stdio.h>
#include <math.h>
#include <string.h>

static int tests_run = 0;
static int tests_passed = 0;

#define TEST(name) \
    tests_run++; \
    printf("  [RUN] %s\n", name); \
    do {

#define END_TEST \
        tests_passed++; \
        printf("  [OK]  %s\n", __func__); \
    } while (0)

#define ASSERT(cond, msg) \
    do { \
        if (!(cond)) { \
            printf("  [FAIL] %s: %s (line %d)\n", __func__, msg, __LINE__); \
            return; \
        } \
    } while (0)

#define ASSERT_F32(a, b, tol) \
    do { \
        if (fabsf((a) - (b)) > (tol)) { \
            printf("  [FAIL] %s: %.6f != %.6f (line %d)\n", __func__, (a), (b), __LINE__); \
            return; \
        } \
    } while (0)

static void test_create_free(void) {
    TEST("create_free");
    int64_t shape[] = {2, 3};
    lmlc_tensor_t* t = lmlc_tensor_create(2, shape, LMLC_DTYPE_F32);
    ASSERT(t != NULL, "create returned NULL");
    ASSERT(t->ndim == 2, "ndim wrong");
    ASSERT(t->shape[0] == 2 && t->shape[1] == 3, "shape wrong");
    ASSERT(t->count == 6, "count wrong");
    ASSERT(t->owns_data == 1, "owns_data wrong");
    ASSERT(t->strides[0] == 3 && t->strides[1] == 1, "strides wrong");
    lmlc_tensor_free(t);
    END_TEST;
}

static void test_create_invalid(void) {
    TEST("create_invalid");
    int64_t shape[] = {2, 3};
    lmlc_tensor_t* t = lmlc_tensor_create(0, shape, LMLC_DTYPE_F32);
    ASSERT(t == NULL, "ndim=0 should fail");
    ASSERT(lmlc_last_error() == LMLC_ERR_INVALID_NDIM, "should set invalid ndim error");

    t = lmlc_tensor_create(2, shape, (lmlc_dtype_t)99);
    ASSERT(t == NULL, "invalid dtype should fail");
    ASSERT(lmlc_last_error() == LMLC_ERR_DTYPE_MISMATCH, "should set dtype error");
    lmlc_clear_error();
    END_TEST;
}

static void test_fill_zero(void) {
    TEST("fill_zero");
    int64_t shape[] = {4};
    lmlc_tensor_t* t = lmlc_tensor_create(1, shape, LMLC_DTYPE_F32);
    lmlc_tensor_fill(t, 3.14f);
    int64_t idx[] = {2};
    ASSERT_F32(lmlc_tensor_get(t, idx), 3.14f, 1e-5f);
    lmlc_tensor_zero(t);
    ASSERT_F32(lmlc_tensor_get(t, idx), 0.0f, 1e-5f);
    lmlc_tensor_free(t);
    END_TEST;
}

static void test_get_set_bounds(void) {
    TEST("get_set_bounds");
    int64_t shape[] = {3};
    lmlc_tensor_t* t = lmlc_tensor_create(1, shape, LMLC_DTYPE_F32);
    int64_t idx_ok[] = {1};
    lmlc_tensor_set(t, idx_ok, 42.0f);
    ASSERT_F32(lmlc_tensor_get(t, idx_ok), 42.0f, 1e-5f);

    int64_t idx_bad[] = {5};
    lmlc_tensor_get(t, idx_bad);
    ASSERT(lmlc_last_error() == LMLC_ERR_OUT_OF_BOUNDS, "should set OOB error");
    lmlc_clear_error();
    lmlc_tensor_free(t);
    END_TEST;
}

static void test_bf16_conversion(void) {
    TEST("bf16_conversion");
    float vals[] = {1.0f, -1.5f, 3.14159f, 0.0f, 100.0f};
    for (int i = 0; i < 5; i++) {
        lmlc_bf16_t b = lmlc_f32_to_bf16(vals[i]);
        float back = lmlc_bf16_to_f32(b);
        ASSERT_F32(back, vals[i], 1.0f);
    }
    END_TEST;
}

static void test_f16_conversion(void) {
    TEST("f16_conversion");
    float vals[] = {1.0f, -2.0f, 0.5f, 0.0f, 100.0f};
    for (int i = 0; i < 5; i++) {
        lmlc_f16_t h = lmlc_f32_to_f16(vals[i]);
        float back = lmlc_f16_to_f32(h);
        ASSERT_F32(back, vals[i], 0.5f);
    }
    END_TEST;
}

static void test_bool_dtype(void) {
    TEST("bool_dtype");
    int64_t shape[] = {3};
    lmlc_tensor_t* t = lmlc_tensor_create(1, shape, LMLC_DTYPE_BOOL);
    lmlc_tensor_fill(t, 1.0f);
    int64_t idx[] = {0};
    ASSERT_F32(lmlc_tensor_get(t, idx), 1.0f, 1e-5f);
    lmlc_tensor_fill(t, 0.0f);
    ASSERT_F32(lmlc_tensor_get(t, idx), 0.0f, 1e-5f);
    lmlc_tensor_free(t);
    END_TEST;
}

static void test_add_broadcasting(void) {
    TEST("add_broadcasting");
    int64_t shape_a[] = {2, 3};
    int64_t shape_b[] = {3};
    int64_t shape_out[] = {2, 3};
    lmlc_tensor_t* a = lmlc_tensor_create(2, shape_a, LMLC_DTYPE_F32);
    lmlc_tensor_t* b = lmlc_tensor_create(1, shape_b, LMLC_DTYPE_F32);
    lmlc_tensor_t* out = lmlc_tensor_create(2, shape_out, LMLC_DTYPE_F32);

    lmlc_tensor_fill(a, 1.0f);
    lmlc_tensor_fill(b, 10.0f);
    lmlc_tensor_add(a, b, out);

    int64_t idx[] = {1, 2};
    ASSERT_F32(lmlc_tensor_get(out, idx), 11.0f, 1e-5f);

    lmlc_tensor_free(a);
    lmlc_tensor_free(b);
    lmlc_tensor_free(out);
    END_TEST;
}

static void test_mul_broadcasting(void) {
    TEST("mul_broadcasting");
    int64_t shape_a[] = {2, 3};
    int64_t shape_b[] = {1};
    int64_t shape_out[] = {2, 3};
    lmlc_tensor_t* a = lmlc_tensor_create(2, shape_a, LMLC_DTYPE_F32);
    lmlc_tensor_t* b = lmlc_tensor_create(1, shape_b, LMLC_DTYPE_F32);
    lmlc_tensor_t* out = lmlc_tensor_create(2, shape_out, LMLC_DTYPE_F32);

    lmlc_tensor_fill(a, 2.0f);
    lmlc_tensor_fill(b, 5.0f);
    lmlc_tensor_mul(a, b, out);

    int64_t idx[] = {0, 0};
    ASSERT_F32(lmlc_tensor_get(out, idx), 10.0f, 1e-5f);

    lmlc_tensor_free(a);
    lmlc_tensor_free(b);
    lmlc_tensor_free(out);
    END_TEST;
}

static void test_matmul_2d(void) {
    TEST("matmul_2d");
    int64_t shape_a[] = {2, 3};
    int64_t shape_b[] = {3, 2};
    int64_t shape_out[] = {2, 2};
    lmlc_tensor_t* a = lmlc_tensor_create(2, shape_a, LMLC_DTYPE_F32);
    lmlc_tensor_t* b = lmlc_tensor_create(2, shape_b, LMLC_DTYPE_F32);
    lmlc_tensor_t* out = lmlc_tensor_create(2, shape_out, LMLC_DTYPE_F32);

    // a = [[1,2,3],[4,5,6]]
    int64_t idx[2];
    float val = 1.0f;
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++) {
            idx[0] = i; idx[1] = j;
            lmlc_tensor_set(a, idx, val++);
        }

    // b = [[1,0],[0,1],[1,1]]
    int64_t idx2[2];
    idx2[0]=0; idx2[1]=0; lmlc_tensor_set(b, idx2, 1.0f);
    idx2[0]=0; idx2[1]=1; lmlc_tensor_set(b, idx2, 0.0f);
    idx2[0]=1; idx2[1]=0; lmlc_tensor_set(b, idx2, 0.0f);
    idx2[0]=1; idx2[1]=1; lmlc_tensor_set(b, idx2, 1.0f);
    idx2[0]=2; idx2[1]=0; lmlc_tensor_set(b, idx2, 1.0f);
    idx2[0]=2; idx2[1]=1; lmlc_tensor_set(b, idx2, 1.0f);

    lmlc_tensor_matmul(a, b, out);
    // out = [[1+0+3, 0+2+3],[4+0+6, 0+5+6]] = [[4,5],[10,11]]
    int64_t oi[2];
    oi[0]=0; oi[1]=0; ASSERT_F32(lmlc_tensor_get(out, oi), 4.0f, 1e-4f);
    oi[0]=0; oi[1]=1; ASSERT_F32(lmlc_tensor_get(out, oi), 5.0f, 1e-4f);
    oi[0]=1; oi[1]=0; ASSERT_F32(lmlc_tensor_get(out, oi), 10.0f, 1e-4f);
    oi[0]=1; oi[1]=1; ASSERT_F32(lmlc_tensor_get(out, oi), 11.0f, 1e-4f);

    lmlc_tensor_free(a);
    lmlc_tensor_free(b);
    lmlc_tensor_free(out);
    END_TEST;
}

static void test_matmul_batched(void) {
    TEST("matmul_batched");
    int64_t shape_a[] = {2, 2, 3};
    int64_t shape_b[] = {2, 3, 2};
    int64_t shape_out[] = {2, 2, 2};
    lmlc_tensor_t* a = lmlc_tensor_create(3, shape_a, LMLC_DTYPE_F32);
    lmlc_tensor_t* b = lmlc_tensor_create(3, shape_b, LMLC_DTYPE_F32);
    lmlc_tensor_t* out = lmlc_tensor_create(3, shape_out, LMLC_DTYPE_F32);

    lmlc_tensor_fill(a, 1.0f);
    lmlc_tensor_fill(b, 1.0f);
    lmlc_tensor_matmul(a, b, out);

    // each batch: [[1,1,1],[1,1,1]] x [[1,1],[1,1],[1,1]] = [[3,3],[3,3]]
    int64_t idx[3] = {0, 0, 0};
    ASSERT_F32(lmlc_tensor_get(out, idx), 3.0f, 1e-4f);
    idx[0]=1; idx[1]=1; idx[2]=1;
    ASSERT_F32(lmlc_tensor_get(out, idx), 3.0f, 1e-4f);

    lmlc_tensor_free(a);
    lmlc_tensor_free(b);
    lmlc_tensor_free(out);
    END_TEST;
}

static void test_cross_dtype_copy(void) {
    TEST("cross_dtype_copy");
    int64_t shape[] = {3};
    lmlc_tensor_t* src = lmlc_tensor_create(1, shape, LMLC_DTYPE_F32);
    lmlc_tensor_t* dst = lmlc_tensor_create(1, shape, LMLC_DTYPE_BF16);

    lmlc_tensor_fill(src, 2.5f);
    lmlc_tensor_copy(src, dst);

    int64_t idx[] = {1};
    ASSERT_F32(lmlc_tensor_get(dst, idx), 2.5f, 1.0f);

    lmlc_tensor_free(src);
    lmlc_tensor_free(dst);
    END_TEST;
}

static void test_cross_dtype_add(void) {
    TEST("cross_dtype_add");
    int64_t shape[] = {3};
    lmlc_tensor_t* a = lmlc_tensor_create(1, shape, LMLC_DTYPE_F32);
    lmlc_tensor_t* b = lmlc_tensor_create(1, shape, LMLC_DTYPE_F16);
    lmlc_tensor_t* out = lmlc_tensor_create(1, shape, LMLC_DTYPE_F32);

    lmlc_tensor_fill(a, 2.0f);
    lmlc_tensor_fill(b, 3.0f);
    lmlc_tensor_add(a, b, out);

    int64_t idx[] = {0};
    ASSERT_F32(lmlc_tensor_get(out, idx), 5.0f, 0.5f);

    lmlc_tensor_free(a);
    lmlc_tensor_free(b);
    lmlc_tensor_free(out);
    END_TEST;
}

static void test_scale(void) {
    TEST("scale");
    int64_t shape[] = {3};
    lmlc_tensor_t* a = lmlc_tensor_create(1, shape, LMLC_DTYPE_F32);
    lmlc_tensor_t* out = lmlc_tensor_create(1, shape, LMLC_DTYPE_F32);

    lmlc_tensor_fill(a, 2.0f);
    lmlc_tensor_scale(a, 3.0f, out);

    int64_t idx[] = {1};
    ASSERT_F32(lmlc_tensor_get(out, idx), 6.0f, 1e-5f);

    lmlc_tensor_free(a);
    lmlc_tensor_free(out);
    END_TEST;
}

static void test_dot(void) {
    TEST("dot");
    int64_t shape[] = {3};
    lmlc_tensor_t* a = lmlc_tensor_create(1, shape, LMLC_DTYPE_F32);
    lmlc_tensor_t* b = lmlc_tensor_create(1, shape, LMLC_DTYPE_F32);

    lmlc_tensor_fill(a, 2.0f);
    lmlc_tensor_fill(b, 3.0f);
    float result = lmlc_tensor_dot(a, b);
    ASSERT_F32(result, 18.0f, 1e-5f); // 3 * (2*3)

    lmlc_tensor_free(a);
    lmlc_tensor_free(b);
    END_TEST;
}

static void test_norm(void) {
    TEST("norm");
    int64_t shape[] = {3};
    lmlc_tensor_t* t = lmlc_tensor_create(1, shape, LMLC_DTYPE_F32);

    lmlc_tensor_fill(t, 4.0f);
    float result = lmlc_tensor_norm(t);
    ASSERT_F32(result, sqrtf(3.0f * 16.0f), 1e-5f); // sqrt(48)

    lmlc_tensor_free(t);
    END_TEST;
}

static void test_reshape(void) {
    TEST("reshape");
    int64_t shape1[] = {6};
    int64_t shape2[] = {2, 3};
    lmlc_tensor_t* t = lmlc_tensor_create(1, shape1, LMLC_DTYPE_F32);
    lmlc_tensor_reshape(t, 2, shape2);
    ASSERT(t->ndim == 2, "ndim should be 2");
    ASSERT(t->shape[0] == 2 && t->shape[1] == 3, "shape should be [2,3]");

    int64_t bad[] = {2, 4};
    lmlc_tensor_reshape(t, 2, bad);
    ASSERT(lmlc_last_error() == LMLC_ERR_SHAPE_MISMATCH, "should set shape mismatch");
    lmlc_clear_error();
    lmlc_tensor_free(t);
    END_TEST;
}

static void test_transpose(void) {
    TEST("transpose");
    int64_t shape[] = {2, 3};
    lmlc_tensor_t* t = lmlc_tensor_create(2, shape, LMLC_DTYPE_F32);
    lmlc_tensor_transpose(t, 0, 1);
    ASSERT(t->shape[0] == 3 && t->shape[1] == 2, "shape should be swapped");
    lmlc_tensor_free(t);
    END_TEST;
}

static void test_squeeze_unsqueeze(void) {
    TEST("squeeze_unsqueeze");
    int64_t shape[] = {1, 3};
    lmlc_tensor_t* t = lmlc_tensor_create(2, shape, LMLC_DTYPE_F32);
    lmlc_tensor_squeeze(t, 0);
    ASSERT(t->ndim == 1, "ndim should be 1 after squeeze");
    ASSERT(t->shape[0] == 3, "shape[0] should be 3");

    lmlc_tensor_unsqueeze(t, 0);
    ASSERT(t->ndim == 2, "ndim should be 2 after unsqueeze");
    ASSERT(t->shape[0] == 1, "shape[0] should be 1");
    lmlc_tensor_free(t);
    END_TEST;
}

static void test_view(void) {
    TEST("view");
    int64_t shape[] = {6};
    lmlc_tensor_t* t = lmlc_tensor_create(1, shape, LMLC_DTYPE_F32);
    lmlc_tensor_fill(t, 7.0f);

    int64_t vshape[] = {2, 3};
    lmlc_tensor_t* v = lmlc_tensor_view(t, 2, vshape);
    ASSERT(v != NULL, "view should not be NULL");
    ASSERT(v->owns_data == 0, "view should not own data");
    ASSERT(v->data == t->data, "view should share data");

    int64_t idx[] = {1, 2};
    ASSERT_F32(lmlc_tensor_get(v, idx), 7.0f, 1e-5f);

    lmlc_tensor_free(v);
    lmlc_tensor_free(t);
    END_TEST;
}

static void test_permute(void) {
    TEST("permute");
    int64_t shape[] = {2, 3, 4};
    lmlc_tensor_t* t = lmlc_tensor_create(3, shape, LMLC_DTYPE_F32);
    int perm[] = {2, 0, 1};
    lmlc_tensor_permute(t, perm);
    ASSERT(t->shape[0] == 4 && t->shape[1] == 2 && t->shape[2] == 3, "permuted shape wrong");

    int bad_perm[] = {0, 0, 1};
    lmlc_tensor_permute(t, bad_perm);
    ASSERT(lmlc_last_error() == LMLC_ERR_INVALID_PERM, "should set invalid perm error");
    lmlc_clear_error();
    lmlc_tensor_free(t);
    END_TEST;
}

static void test_error_reporting(void) {
    TEST("error_reporting");
    lmlc_clear_error();
    ASSERT(lmlc_last_error() == LMLC_OK, "should be OK after clear");

    lmlc_tensor_get(NULL, NULL);
    ASSERT(lmlc_last_error() == LMLC_ERR_NULL, "should set NULL error");

    lmlc_error_info_t info = lmlc_last_error_info();
    ASSERT(info.func != NULL, "func should be set");
    ASSERT(info.line > 0, "line should be set");

    const char* str = lmlc_error_string(LMLC_ERR_NULL);
    ASSERT(str != NULL && strlen(str) > 0, "error string should not be empty");
    lmlc_clear_error();
    END_TEST;
}

static void test_multidim_print(void) {
    TEST("multidim_print");
    int64_t shape[] = {2, 3};
    lmlc_tensor_t* t = lmlc_tensor_create(2, shape, LMLC_DTYPE_F32);
    float val = 1.0f;
    int64_t idx[2];
    for (int i = 0; i < 2; i++)
        for (int j = 0; j < 3; j++) {
            idx[0] = i; idx[1] = j;
            lmlc_tensor_set(t, idx, val++);
        }
    // just verify it doesn't crash
    lmlc_tensor_print(t);
    lmlc_tensor_free(t);
    END_TEST;
}

static void test_ndim_gt_8(void) {
    TEST("ndim_gt_8");
    int64_t shape[10];
    for (int i = 0; i < 10; i++) shape[i] = 1;
    lmlc_tensor_t* t = lmlc_tensor_create(10, shape, LMLC_DTYPE_F32);
    ASSERT(t != NULL, "ndim=10 should succeed with dynamic arrays");
    ASSERT(t->ndim == 10, "ndim should be 10");

    int64_t idx[10] = {0,0,0,0,0,0,0,0,0,0};
    lmlc_tensor_set(t, idx, 42.0f);
    ASSERT_F32(lmlc_tensor_get(t, idx), 42.0f, 1e-5f);

    lmlc_tensor_free(t);

    // ndim > LMLC_MAX_NDIM should fail
    int64_t big_shape[33];
    for (int i = 0; i < 33; i++) big_shape[i] = 1;
    lmlc_tensor_t* bad = lmlc_tensor_create(33, big_shape, LMLC_DTYPE_F32);
    ASSERT(bad == NULL, "ndim=33 should fail");
    ASSERT(lmlc_last_error() == LMLC_ERR_INVALID_NDIM, "should set invalid ndim");
    lmlc_clear_error();
    END_TEST;
}

static void test_graph_forward(void) {
    TEST("graph_forward");
    int64_t shape[] = {3};
    lmlc_tensor_t* a = lmlc_tensor_create(1, shape, LMLC_DTYPE_F32);
    lmlc_tensor_t* b = lmlc_tensor_create(1, shape, LMLC_DTYPE_F32);
    lmlc_tensor_t* c = lmlc_tensor_create(1, shape, LMLC_DTYPE_F32);
    lmlc_tensor_t* d = lmlc_tensor_create(1, shape, LMLC_DTYPE_F32);

    lmlc_tensor_fill(a, 2.0f);
    lmlc_tensor_fill(b, 3.0f);

    // c = a + b, then d = c * 2.0
    lmlc_cgraph_t* g = lmlc_graph_create(8);
    ASSERT(g != NULL, "graph create should succeed");

    lmlc_graph_add_binary(g, LMLC_OP_ADD, a, b, c);
    lmlc_graph_add_scale(g, c, 2.0f, d);
    ASSERT(g->n_nodes == 2, "should have 2 nodes");

    lmlc_graph_forward(g);

    int64_t idx[] = {0};
    // c = [5, 5, 5], d = [10, 10, 10]
    ASSERT_F32(lmlc_tensor_get(c, idx), 5.0f, 1e-5f);
    ASSERT_F32(lmlc_tensor_get(d, idx), 10.0f, 1e-5f);

    lmlc_graph_free(g);
    lmlc_tensor_free(a);
    lmlc_tensor_free(b);
    lmlc_tensor_free(c);
    lmlc_tensor_free(d);
    END_TEST;
}

static void test_graph_matmul(void) {
    TEST("graph_matmul");
    int64_t shape_a[] = {2, 3};
    int64_t shape_b[] = {3, 2};
    int64_t shape_out[] = {2, 2};
    lmlc_tensor_t* a = lmlc_tensor_create(2, shape_a, LMLC_DTYPE_F32);
    lmlc_tensor_t* b = lmlc_tensor_create(2, shape_b, LMLC_DTYPE_F32);
    lmlc_tensor_t* out = lmlc_tensor_create(2, shape_out, LMLC_DTYPE_F32);

    lmlc_tensor_fill(a, 1.0f);
    lmlc_tensor_fill(b, 1.0f);

    lmlc_cgraph_t* g = lmlc_graph_create(4);
    lmlc_graph_add_matmul(g, a, b, out);
    lmlc_graph_forward(g);

    int64_t idx[] = {0, 0};
    ASSERT_F32(lmlc_tensor_get(out, idx), 3.0f, 1e-4f);

    lmlc_graph_free(g);
    lmlc_tensor_free(a);
    lmlc_tensor_free(b);
    lmlc_tensor_free(out);
    END_TEST;
}

static void test_integer_dtypes(void) {
    TEST("integer_dtypes");
    int64_t shape[] = {4};

    // I8
    lmlc_tensor_t* t8 = lmlc_tensor_create(1, shape, LMLC_DTYPE_I8);
    lmlc_tensor_fill(t8, -5.0f);
    int64_t idx[] = {0};
    ASSERT_F32(lmlc_tensor_get(t8, idx), -5.0f, 0.0f);
    idx[0] = 3;
    ASSERT_F32(lmlc_tensor_get(t8, idx), -5.0f, 0.0f);
    lmlc_tensor_free(t8);

    // I16
    lmlc_tensor_t* t16 = lmlc_tensor_create(1, shape, LMLC_DTYPE_I16);
    lmlc_tensor_fill(t16, 1000.0f);
    idx[0] = 1;
    ASSERT_F32(lmlc_tensor_get(t16, idx), 1000.0f, 0.0f);
    lmlc_tensor_free(t16);

    // I32
    lmlc_tensor_t* t32 = lmlc_tensor_create(1, shape, LMLC_DTYPE_I32);
    lmlc_tensor_fill(t32, 100000.0f);
    idx[0] = 2;
    ASSERT_F32(lmlc_tensor_get(t32, idx), 100000.0f, 0.0f);
    lmlc_tensor_free(t32);

    // I64
    lmlc_tensor_t* t64 = lmlc_tensor_create(1, shape, LMLC_DTYPE_I64);
    lmlc_tensor_fill(t64, 999999.0f);
    idx[0] = 0;
    ASSERT_F32(lmlc_tensor_get(t64, idx), 999999.0f, 0.0f);
    lmlc_tensor_free(t64);
    END_TEST;
}

static void test_cross_dtype_int_to_f32(void) {
    TEST("cross_dtype_int_to_f32");
    int64_t shape[] = {3};
    lmlc_tensor_t* src = lmlc_tensor_create(1, shape, LMLC_DTYPE_I32);
    lmlc_tensor_t* dst = lmlc_tensor_create(1, shape, LMLC_DTYPE_F32);

    lmlc_tensor_fill(src, 42.0f);
    lmlc_tensor_copy(src, dst);

    int64_t idx[] = {1};
    ASSERT_F32(lmlc_tensor_get(dst, idx), 42.0f, 0.0f);

    lmlc_tensor_free(src);
    lmlc_tensor_free(dst);
    END_TEST;
}

int main(void) {
    printf("=== LMLC_Tensor Tests ===\n\n");

    test_create_free();
    test_create_invalid();
    test_fill_zero();
    test_get_set_bounds();
    test_bf16_conversion();
    test_f16_conversion();
    test_bool_dtype();
    test_add_broadcasting();
    test_mul_broadcasting();
    test_matmul_2d();
    test_matmul_batched();
    test_cross_dtype_copy();
    test_cross_dtype_add();
    test_scale();
    test_dot();
    test_norm();
    test_reshape();
    test_transpose();
    test_squeeze_unsqueeze();
    test_view();
    test_permute();
    test_error_reporting();
    test_multidim_print();
    test_ndim_gt_8();
    test_graph_forward();
    test_graph_matmul();
    test_integer_dtypes();
    test_cross_dtype_int_to_f32();

    printf("\n=== Results: %d/%d passed ===\n", tests_passed, tests_run);
    return (tests_passed == tests_run) ? 0 : 1;
}
