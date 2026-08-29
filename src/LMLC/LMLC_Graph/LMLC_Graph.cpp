#include "LMLC_Graph.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

// ---- graph create / free ----

lmlc_cgraph_t* lmlc_graph_create(int capacity) {
    if (capacity <= 0) capacity = 16;
    lmlc_cgraph_t* g = (lmlc_cgraph_t*)calloc(1, sizeof(lmlc_cgraph_t));
    if (!g) return NULL;
    g->nodes = (lmlc_cnode_t*)calloc((size_t)capacity, sizeof(lmlc_cnode_t));
    if (!g->nodes) { free(g); return NULL; }
    g->n_nodes = 0;
    g->capacity = capacity;
    return g;
}

void lmlc_graph_free(lmlc_cgraph_t* g) {
    if (!g) return;
    // Graph does not own tensors — caller is responsible for freeing them.
    // This matches GGML's design: tensors can be shared across graphs and
    // reused across forward passes.
    free(g->nodes);
    free(g);
}

// ---- build ----

static int graph_ensure_capacity(lmlc_cgraph_t* g) {
    if (g->n_nodes < g->capacity) return 1;
    int new_cap = g->capacity * 2;
    lmlc_cnode_t* new_nodes = (lmlc_cnode_t*)realloc(g->nodes, (size_t)new_cap * sizeof(lmlc_cnode_t));
    if (!new_nodes) return 0;
    g->nodes = new_nodes;
    g->capacity = new_cap;
    return 1;
}

void lmlc_graph_add_binary(lmlc_cgraph_t* g, lmlc_op_t op,
                           lmlc_tensor_t* src0, lmlc_tensor_t* src1,
                           lmlc_tensor_t* dst) {
    if (!g) return;
    if (!graph_ensure_capacity(g)) return;
    lmlc_cnode_t* n = &g->nodes[g->n_nodes++];
    memset(n, 0, sizeof(*n));
    n->op = op;
    n->src[0] = src0;
    n->src[1] = src1;
    n->dst = dst;
}

void lmlc_graph_add_scale(lmlc_cgraph_t* g,
                          lmlc_tensor_t* src, float scalar,
                          lmlc_tensor_t* dst) {
    if (!g) return;
    if (!graph_ensure_capacity(g)) return;
    lmlc_cnode_t* n = &g->nodes[g->n_nodes++];
    memset(n, 0, sizeof(*n));
    n->op = LMLC_OP_SCALE;
    n->src[0] = src;
    n->dst = dst;
    n->scalar = scalar;
}

void lmlc_graph_add_matmul(lmlc_cgraph_t* g,
                           lmlc_tensor_t* src0, lmlc_tensor_t* src1,
                           lmlc_tensor_t* dst) {
    if (!g) return;
    if (!graph_ensure_capacity(g)) return;
    lmlc_cnode_t* n = &g->nodes[g->n_nodes++];
    memset(n, 0, sizeof(*n));
    n->op = LMLC_OP_MATMUL;
    n->src[0] = src0;
    n->src[1] = src1;
    n->dst = dst;
}

void lmlc_graph_add_reshape(lmlc_cgraph_t* g,
                            lmlc_tensor_t* src, int ndim, const int64_t* shape) {
    if (!g) return;
    if (!graph_ensure_capacity(g)) return;
    lmlc_cnode_t* n = &g->nodes[g->n_nodes++];
    memset(n, 0, sizeof(*n));
    n->op = LMLC_OP_RESHAPE;
    n->src[0] = src;
    n->ndim = ndim;
    for (int i = 0; i < ndim && i < LMLC_MAX_NDIM; i++) n->shape[i] = shape[i];
}

void lmlc_graph_add_transpose(lmlc_cgraph_t* g,
                              lmlc_tensor_t* src, int dim0, int dim1) {
    if (!g) return;
    if (!graph_ensure_capacity(g)) return;
    lmlc_cnode_t* n = &g->nodes[g->n_nodes++];
    memset(n, 0, sizeof(*n));
    n->op = LMLC_OP_TRANSPOSE;
    n->src[0] = src;
    n->dim0 = dim0;
    n->dim1 = dim1;
}

void lmlc_graph_add_permute(lmlc_cgraph_t* g,
                            lmlc_tensor_t* src, const int* perm) {
    if (!g || !perm) return;
    if (!graph_ensure_capacity(g)) return;
    lmlc_cnode_t* n = &g->nodes[g->n_nodes++];
    memset(n, 0, sizeof(*n));
    n->op = LMLC_OP_PERMUTE;
    n->src[0] = src;
    if (src) {
        for (int i = 0; i < src->ndim; i++) n->perm[i] = perm[i];
    }
}

void lmlc_graph_add_squeeze(lmlc_cgraph_t* g,
                            lmlc_tensor_t* src, int dim) {
    if (!g) return;
    if (!graph_ensure_capacity(g)) return;
    lmlc_cnode_t* n = &g->nodes[g->n_nodes++];
    memset(n, 0, sizeof(*n));
    n->op = LMLC_OP_SQUEEZE;
    n->src[0] = src;
    n->dim0 = dim;
}

void lmlc_graph_add_unsqueeze(lmlc_cgraph_t* g,
                              lmlc_tensor_t* src, int dim) {
    if (!g) return;
    if (!graph_ensure_capacity(g)) return;
    lmlc_cnode_t* n = &g->nodes[g->n_nodes++];
    memset(n, 0, sizeof(*n));
    n->op = LMLC_OP_UNSQUEEZE;
    n->src[0] = src;
    n->dim0 = dim;
}

void lmlc_graph_add_view(lmlc_cgraph_t* g,
                         lmlc_tensor_t* src, int ndim, const int64_t* shape,
                         lmlc_tensor_t* dst) {
    if (!g) return;
    if (!graph_ensure_capacity(g)) return;
    lmlc_cnode_t* n = &g->nodes[g->n_nodes++];
    memset(n, 0, sizeof(*n));
    n->op = LMLC_OP_VIEW;
    n->src[0] = src;
    n->dst = dst;
    n->ndim = ndim;
    for (int i = 0; i < ndim && i < LMLC_MAX_NDIM; i++) n->shape[i] = shape[i];
}

// ---- execute ----

void lmlc_graph_forward(lmlc_cgraph_t* g) {
    if (!g) return;
    // TODO: topological sort for arbitrary DAGs
    // TODO: memory planning — allocate intermediates once
    // TODO: op fusion — combine adjacent element-wise ops
    // for now: execute in insertion order (linear chain)
    for (int i = 0; i < g->n_nodes; i++) {
        lmlc_cnode_t* n = &g->nodes[i];
        switch (n->op) {
            case LMLC_OP_ADD:
                lmlc_tensor_add(n->src[0], n->src[1], n->dst);
                break;
            case LMLC_OP_MUL:
                lmlc_tensor_mul(n->src[0], n->src[1], n->dst);
                break;
            case LMLC_OP_SCALE:
                lmlc_tensor_scale(n->src[0], n->scalar, n->dst);
                break;
            case LMLC_OP_MATMUL:
                lmlc_tensor_matmul(n->src[0], n->src[1], n->dst);
                break;
            case LMLC_OP_RESHAPE:
                lmlc_tensor_reshape(n->src[0], n->ndim, n->shape);
                break;
            case LMLC_OP_TRANSPOSE:
                lmlc_tensor_transpose(n->src[0], n->dim0, n->dim1);
                break;
            case LMLC_OP_PERMUTE:
                lmlc_tensor_permute(n->src[0], n->perm);
                break;
            case LMLC_OP_SQUEEZE:
                lmlc_tensor_squeeze(n->src[0], n->dim0);
                break;
            case LMLC_OP_UNSQUEEZE:
                lmlc_tensor_unsqueeze(n->src[0], n->dim0);
                break;
            case LMLC_OP_VIEW:
                // view creates a new tensor — caller should pre-allocate dst
                // or we create it here
                // TODO: automatic dst allocation for view
                break;
            default:
                break;
        }
    }
}

// ---- backward ----

// Backward pass: iterate nodes in reverse, accumulate gradients.
// For each node, grad_dst must be set (either seeded by caller for the last
// node, or accumulated from downstream nodes). We compute grad_src from grad_dst.
//
// Gradient rules:
//   add:   dL/da = dL/dy, dL/db = dL/dy  (with broadcasting reduction)
//   mul:   dL/da = dL/dy * b, dL/db = dL/dy * a
//   scale: dL/da = dL/dy * scalar
//   matmul: dL/dA = dL/dY @ B^T, dL/dB = A^T @ dL/dY
//
// TODO: gradient accumulation across multiple uses of same tensor
// TODO: broadcasting reduction in add/mul gradients
// TODO: shape op gradients (reshape, transpose, permute are identity for data)

void lmlc_graph_backward(lmlc_cgraph_t* g) {
    if (!g) return;
    for (int i = g->n_nodes - 1; i >= 0; i--) {
        lmlc_cnode_t* n = &g->nodes[i];
        if (!n->grad_dst) continue;

        switch (n->op) {
            case LMLC_OP_ADD:
                // dL/da = dL/dy, dL/db = dL/dy
                if (n->grad_src[0])
                    lmlc_tensor_add(n->grad_dst, n->grad_src[0], n->grad_src[0]);
                if (n->grad_src[1])
                    lmlc_tensor_add(n->grad_dst, n->grad_src[1], n->grad_src[1]);
                break;
            case LMLC_OP_MUL:
                // dL/da = dL/dy * b, dL/db = dL/dy * a
                if (n->grad_src[0] && n->src[1])
                    lmlc_tensor_mul(n->grad_dst, n->src[1], n->grad_src[0]);
                if (n->grad_src[1] && n->src[0])
                    lmlc_tensor_mul(n->grad_dst, n->src[0], n->grad_src[1]);
                break;
            case LMLC_OP_SCALE:
                // dL/da = dL/dy * scalar
                if (n->grad_src[0])
                    lmlc_tensor_scale(n->grad_dst, n->scalar, n->grad_src[0]);
                break;
            case LMLC_OP_MATMUL:
                // dL/dA = dL/dY @ B^T, dL/dB = A^T @ dL/dY
                // TODO: implement transpose + matmul for gradient
                // For now, only works with 2D matmul (no batch)
                // TODO: batched matmul backward
                break;
            default:
                // Shape ops: gradient flows through unchanged (data isn't moved,
                // only metadata). For in-place shape ops, grad_src = grad_dst.
                if (n->grad_src[0])
                    lmlc_tensor_add(n->grad_dst, n->grad_src[0], n->grad_src[0]);
                break;
        }
    }
}

// ---- debug ----

const char* lmlc_op_string(lmlc_op_t op) {
    switch (op) {
        case LMLC_OP_NONE:      return "none";
        case LMLC_OP_ADD:       return "add";
        case LMLC_OP_MUL:       return "mul";
        case LMLC_OP_SCALE:     return "scale";
        case LMLC_OP_MATMUL:    return "matmul";
        case LMLC_OP_DOT:       return "dot";
        case LMLC_OP_NORM:      return "norm";
        case LMLC_OP_RESHAPE:   return "reshape";
        case LMLC_OP_TRANSPOSE: return "transpose";
        case LMLC_OP_PERMUTE:   return "permute";
        case LMLC_OP_SQUEEZE:   return "squeeze";
        case LMLC_OP_UNSQUEEZE: return "unsqueeze";
        case LMLC_OP_VIEW:      return "view";
        default:                return "unknown";
    }
}

void lmlc_graph_print(const lmlc_cgraph_t* g) {
    if (!g) {
        printf("(null graph)\n");
        return;
    }
    printf("cgraph[%d nodes]:\n", g->n_nodes);
    for (int i = 0; i < g->n_nodes; i++) {
        lmlc_cnode_t* n = &g->nodes[i];
        printf("  [%d] %s", i, lmlc_op_string(n->op));
        if (n->src[0]) printf(" src0=%p", (void*)n->src[0]);
        if (n->src[1]) printf(" src1=%p", (void*)n->src[1]);
        if (n->dst)    printf(" dst=%p", (void*)n->dst);
        if (n->op == LMLC_OP_SCALE) printf(" scalar=%.4f", n->scalar);
        printf("\n");
    }
}
