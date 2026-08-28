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
    // TODO: should we free tensors? For now, caller owns tensors.
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
            // TODO: implement remaining ops in forward pass
            // case LMLC_OP_DOT:
            // case LMLC_OP_NORM:
            // case LMLC_OP_RESHAPE:
            // case LMLC_OP_TRANSPOSE:
            // case LMLC_OP_PERMUTE:
            // case LMLC_OP_SQUEEZE:
            // case LMLC_OP_UNSQUEEZE:
            // case LMLC_OP_VIEW:
            default:
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
