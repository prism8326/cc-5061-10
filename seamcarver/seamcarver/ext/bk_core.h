/*
 * bk_core.h - Pure-C Boykov-Kolmogorov max-flow core (no Python dependency).
 *
 * Boykov & Kolmogorov (PAMI 2004) max-flow on a node grid plus a source and a
 * sink.  Arc convention (identical to the canonical reference):
 *   - every edge is an XOR-paired arc pair (a, a^1).
 *   - parent[v] is an arc in v's own adjacency list whose head is v's parent.
 *     Source-tree residual child->parent is arcs[parent^1].cap; sink-tree
 *     residual child->parent is arcs[parent].cap.
 *   - orphans get parent=NONE as soon as they are pushed; nodes that fail
 *     adoption leave their tree and are removed from the active ring queue.
 * Undirected Potts pair edges are one XOR pair with capacity c in each
 * direction (a label disagreement is charged c).
 *
 * This header is included both by the numpy extension (bkflow.c) and by the
 * standalone C profiling/test harness, keeping the inspected algorithm and
 * the shipped solver literally the same code.
 */
#ifndef BK_CORE_H
#define BK_CORE_H

#include <stdlib.h>

#define BK_NONE (-1)
#define BK_EPS 1e-12f

typedef struct { int head, next; float cap; } BkArc;

typedef struct {
    int V, source, sink, narc, acap;
    int *first;
    BkArc *arcs;
    int *tree;          /* 0 free, 1 source, 2 sink */
    int *parent;        /* arc in this node's list; head == parent node */
    int *active;        /* 1 if currently in the ring queue */
    int *orphan_mark;
    int *queue; int qh, qt, qcap;
    int *orphans; int n_orphan;
    double flow;
} BkGraph;

static int bk_reserve(BkGraph *g) {
    if (g->narc + 2 <= g->acap) return 0;
    int ncap = g->acap ? g->acap * 2 : 256;
    BkArc *p = (BkArc*)realloc(g->arcs, (size_t)ncap * sizeof(BkArc));
    if (!p) return -1;
    g->arcs = p; g->acap = ncap;
    return 0;
}

/* Directed capacity arc u->v plus zero reverse residual arc. */
static int bk_add_arc(BkGraph *g, int u, int v, float c) {
    if (bk_reserve(g)) return -1;
    int a = g->narc;
    g->arcs[a].head = v;   g->arcs[a].next = g->first[u]; g->arcs[a].cap = c;
    g->arcs[a+1].head = u; g->arcs[a+1].next = g->first[v]; g->arcs[a+1].cap = 0.0f;
    g->first[u] = a; g->first[v] = a + 1;
    g->narc = a + 2;
    return 0;
}

/* Undirected Potts edge: one XOR pair, capacity c each direction. */
static int bk_add_pair(BkGraph *g, int u, int v, float c) {
    if (bk_reserve(g)) return -1;
    int a = g->narc;
    g->arcs[a].head = v;   g->arcs[a].next = g->first[u]; g->arcs[a].cap = c;
    g->arcs[a+1].head = u; g->arcs[a+1].next = g->first[v]; g->arcs[a+1].cap = c;
    g->first[u] = a; g->first[v] = a + 1;
    g->narc = a + 2;
    return 0;
}

static void bk_enqueue(BkGraph *g, int n) {
    if (g->active[n]) return;
    g->active[n] = 1;
    g->queue[g->qt] = n; g->qt = (g->qt + 1) % g->qcap;
}
static int bk_dequeue(BkGraph *g) {
    while (g->qh != g->qt) {
        int n = g->queue[g->qh]; g->qh = (g->qh + 1) % g->qcap;
        if (g->active[n] && g->tree[n] != 0) { g->active[n] = 0; return n; }
    }
    return BK_NONE;
}
static void bk_push_orphan(BkGraph *g, int n) {
    if (g->orphan_mark[n]) return;
    g->orphan_mark[n] = 1;
    g->orphans[g->n_orphan++] = n;
}

/* Residual capacity of arc a leaving a node that belongs to tree tn. */
static inline float bk_res_out(BkGraph *g, int tn, int a) {
    return (tn == 1) ? g->arcs[a].cap : g->arcs[a ^ 1].cap;
}

static int bk_grow(BkGraph *g) {
    int u;
    while ((u = bk_dequeue(g)) != BK_NONE) {
        int tn = g->tree[u];
        for (int a = g->first[u]; a != BK_NONE; a = g->arcs[a].next) {
            if (bk_res_out(g, tn, a) <= BK_EPS) continue;
            int v = g->arcs[a].head;
            if (g->tree[v] == 0) {
                g->tree[v] = tn;
                g->parent[v] = a ^ 1;       /* sister arc lives in v, head=u */
                bk_enqueue(g, v);
            } else if (g->tree[v] != tn) {
                bk_enqueue(g, u);
                return (tn == 1) ? a : (a ^ 1);  /* oriented source->sink */
            }
        }
    }
    return BK_NONE;
}

static void bk_augment(BkGraph *g, int mid) {
    float f = g->arcs[mid].cap;

    int s = g->arcs[mid ^ 1].head;           /* source-side endpoint */
    while (s != g->source) {
        int pa = g->parent[s];
        if (g->arcs[pa ^ 1].cap < f) f = g->arcs[pa ^ 1].cap;
        s = g->arcs[pa].head;
    }
    int t = g->arcs[mid].head;               /* sink-side endpoint */
    while (t != g->sink) {
        int pa = g->parent[t];
        if (g->arcs[pa].cap < f) f = g->arcs[pa].cap;
        t = g->arcs[pa].head;
    }

    g->arcs[mid].cap -= f; g->arcs[mid ^ 1].cap += f;

    s = g->arcs[mid ^ 1].head;
    while (s != g->source) {
        int pa = g->parent[s];
        g->arcs[pa ^ 1].cap -= f; g->arcs[pa].cap += f;
        if (g->arcs[pa ^ 1].cap <= BK_EPS) {
            g->active[s] = 0; g->parent[s] = BK_NONE; bk_push_orphan(g, s);
        }
        s = g->arcs[pa].head;
    }
    t = g->arcs[mid].head;
    while (t != g->sink) {
        int pa = g->parent[t];
        g->arcs[pa].cap -= f; g->arcs[pa ^ 1].cap += f;
        if (g->arcs[pa].cap <= BK_EPS) {
            g->active[t] = 0; g->parent[t] = BK_NONE; bk_push_orphan(g, t);
        }
        t = g->arcs[pa].head;
    }
    g->flow += f;
}

/* Verify the parent chain from n reaches root.  A NONE parent means the chain
 * enters an unresolved orphan, which always fails - this prevents adoption
 * through orphans (and thus parent cycles).  The walk is intentionally
 * uncached: a timestamp short-circuit can outlive a node becoming an orphan. */
static int bk_find_root(BkGraph *g, int n, int root) {
    int v = n;
    while (v != root) {
        int pa = g->parent[v];
        if (pa == BK_NONE) return 0;         /* unresolved orphan */
        v = g->arcs[pa].head;
    }
    return 1;
}

static int bk_find_parent(BkGraph *g, int n) {
    int tn = g->tree[n];
    int root = (tn == 1) ? g->source : g->sink;
    for (int a = g->first[n]; a != BK_NONE; a = g->arcs[a].next) {
        float c = (tn == 1) ? g->arcs[a ^ 1].cap : g->arcs[a].cap;
        if (c <= BK_EPS) continue;
        int v = g->arcs[a].head;             /* candidate parent */
        if (g->tree[v] != tn) continue;
        if (bk_find_root(g, v, root)) { g->parent[n] = a; return 1; }
    }
    return 0;
}

static void bk_adopt(BkGraph *g) {
    while (g->n_orphan > 0) {
        int n = g->orphans[--g->n_orphan];
        g->orphan_mark[n] = 0;
        if (bk_find_parent(g, n)) { bk_enqueue(g, n); continue; }

        int tn = g->tree[n];
        g->tree[n] = 0; g->parent[n] = BK_NONE; g->active[n] = 0;
        for (int a = g->first[n]; a != BK_NONE; a = g->arcs[a].next) {
            int v = g->arcs[a].head;
            if (g->tree[v] != tn) continue;
            float c = (tn == 1) ? g->arcs[a ^ 1].cap : g->arcs[a].cap;
            if (c > BK_EPS) bk_enqueue(g, v);
            int pa = g->parent[v];
            if (pa != BK_NONE && g->arcs[pa].head == n) bk_push_orphan(g, v);
        }
    }
}

static double bk_maxflow(BkGraph *g) {
    for (;;) {
        int mid = bk_grow(g);
        if (mid == BK_NONE) {
            if (g->n_orphan == 0) break;
            bk_adopt(g);
        } else {
            bk_augment(g, mid);
            bk_adopt(g);
        }
    }
    return g->flow;
}

#endif /* BK_CORE_H */
