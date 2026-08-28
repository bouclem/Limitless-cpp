#ifndef LMLC_GRAPH_H
#define LMLC_GRAPH_H

#include "LMLC_Tensor.h"

#ifdef __cplusplus
extern "C" {
#endif

// TODO: backward pass / gradients
// TODO: memory planning / arena allocator
// TODO: topological sort for arbitrary DAGs
// TODO: op fusion
// TODO: lazy evaluation
// TODO: JIT code generation

typedef enum {
    LMLC_OP_NONE = 0,
    LMLC_OP_ADD,
    LMLC_OP_MUL,
    LMLC_OP_SCALE,
    LMLC_OP_MATMUL,
    LMLC_OP_DOT,
    LMLC_OP_NORM,
    LMLC_OP_RESHAPE,
    LMLC_OP_TRANSPOSE,
    LMLC_OP_PERMUTE,
    LMLC_OP_SQUEEZE,
    LMLC_OP_UNSQUEEZE,
    LMLC_OP_VIEW,
    LMLC_OP_COUNT
} lmlc_op_t;

typedef struct {
    lmlc_op_t    op;
    lmlc_tensor_t* src[2];  // input tensors (up to 2)
    lmlc_tensor_t* dst;     // output tensor
    float        scalar;    // for scale op
    int          perm[LMLC_MAX_NDIM]; // for permute
    int          ndim;      // for reshape/view
    int64_t      shape[LMLC_MAX_NDIM]; // for reshape/view
    int          dim0;      // for transpose/squeeze/unsqueeze
    int          dim1;      // for transpose
} lmlc_cnode_t;

typedef struct {
    lmlc_cnode_t* nodes;
    int           n_nodes;
    int           capacity;
} lmlc_cgraph_t;

// graph create / free
lmlc_cgraph_t* lmlc_graph_create(int capacity);
void           lmlc_graph_free(lmlc_cgraph_t* g);

// build — add nodes to the graph
void lmlc_graph_add_binary(lmlc_cgraph_t* g, lmlc_op_t op,
                           lmlc_tensor_t* src0, lmlc_tensor_t* src1,
                           lmlc_tensor_t* dst);
void lmlc_graph_add_scale(lmlc_cgraph_t* g,
                          lmlc_tensor_t* src, float scalar,
                          lmlc_tensor_t* dst);
void lmlc_graph_add_matmul(lmlc_cgraph_t* g,
                           lmlc_tensor_t* src0, lmlc_tensor_t* src1,
                           lmlc_tensor_t* dst);
// TODO: add helpers for reshape, transpose, permute, squeeze, unsqueeze, view

// execute — run all nodes in order
void lmlc_graph_forward(lmlc_cgraph_t* g);

// debug
void lmlc_graph_print(const lmlc_cgraph_t* g);
const char* lmlc_op_string(lmlc_op_t op);

#ifdef __cplusplus
}
#endif

#endif // LMLC_GRAPH_H
