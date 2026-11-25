/*
 * bkflow.c - numpy binding for the pure-C Boykov-Kolmogorov core in bk_core.h.
 *
 * Inputs (contiguous float32 numpy arrays, passed flattened):
 *   term_src/term_sink : (T,H,W) unary terminal capacities
 *   cap_h : (T,H,W-1), cap_v : (T,H-1,W), cap_t : (T-1,H,W) (empty if T==1)
 * Returns labels (uint8; 0=source side, 1=sink side), flow (double).
 *
 * All graph mechanics live in bk_core.h so the exact algorithm can be built
 * and inspected without Python.
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <string.h>

#include "bk_core.h"

static void oom(void) { PyErr_SetString(PyExc_MemoryError, "bkflow: out of memory"); }

static const float* f32(PyArrayObject *a, npy_intp need, const char *name) {
    if (!a || PyArray_NDIM(a) != 1 || PyArray_TYPE(a) != NPY_FLOAT32 ||
        !PyArray_ISCARRAY_RO(a)) {
        PyErr_Format(PyExc_ValueError, "bkflow: %s must be 1D C float32", name);
        return NULL;
    }
    if (PyArray_SIZE(a) != need) {
        PyErr_Format(PyExc_ValueError, "bkflow: %s size %ld != %ld",
                     name, (long)PyArray_SIZE(a), (long)need);
        return NULL;
    }
    return (const float*)PyArray_DATA(a);
}

static PyObject* py_solve(PyObject* self, PyObject* args) {
    int W, H, T;
    PyArrayObject *sa=NULL, *ka=NULL, *cha=NULL, *cva=NULL, *cta=NULL;
    if (!PyArg_ParseTuple(args, "iiiO!O!O!O!O!", &W, &H, &T,
            &PyArray_Type, &sa, &PyArray_Type, &ka,
            &PyArray_Type, &cha, &PyArray_Type, &cva, &PyArray_Type, &cta))
        return NULL;
    if (W < 2 || H < 2 || T < 1) {
        PyErr_SetString(PyExc_ValueError, "bkflow: need W>=2,H>=2,T>=1");
        return NULL;
    }
    int N = W * H * T;
    npy_intp nh = (npy_intp)T * H * (W - 1);
    npy_intp nv = (npy_intp)T * (H - 1) * W;
    npy_intp nt = (npy_intp)(T - 1) * H * W;
    const float *ts = f32(sa, N, "term_src"); if (!ts) return NULL;
    const float *tk = f32(ka, N, "term_sink"); if (!tk) return NULL;
    const float *ch = f32(cha, nh, "cap_h"); if (!ch) return NULL;
    const float *cv = f32(cva, nv, "cap_v"); if (!cv) return NULL;
    const float *ctp = NULL;
    if (nt) { ctp = f32(cta, nt, "cap_t"); if (!ctp) return NULL; }
    else if (PyArray_SIZE(cta) != 0) {
        PyErr_SetString(PyExc_ValueError, "bkflow: cap_t must be empty for T==1");
        return NULL;
    }

    BkGraph g;
    g.V = N + 2; g.source = N; g.sink = N + 1;
    g.narc = 0; g.acap = 0; g.arcs = NULL; g.flow = 0.0;
    g.qh = g.qt = 0; g.n_orphan = 0;
    size_t V = (size_t)g.V;
    g.first = malloc(V * sizeof(int));
    g.tree = calloc(V, sizeof(int));
    g.parent = malloc(V * sizeof(int));
    g.active = calloc(V, sizeof(int));
    g.orphan_mark = calloc(V, sizeof(int));
    g.qcap = (int)(2 * V);
    g.queue = malloc((size_t)g.qcap * sizeof(int));
    g.orphans = malloc(V * sizeof(int));
    if (!g.first||!g.tree||!g.parent||!g.active||!g.orphan_mark||
        !g.queue||!g.orphans) { oom(); goto fail; }
    for (int i = 0; i < g.V; ++i) { g.first[i] = BK_NONE; g.parent[i] = BK_NONE; }
    /* terminal arcs (2 each) + grid pair arcs (2 each); exact upper bound */
    g.acap = (int)(2 * (npy_intp)N + 2 * (nh + nv + nt) + 16);
    g.arcs = malloc((size_t)g.acap * sizeof(BkArc));
    if (!g.arcs) { oom(); goto fail; }

    for (int n = 0; n < N; ++n) {
        if (ts[n] > 0.0f && bk_add_arc(&g, g.source, n, ts[n])) { oom(); goto fail; }
        if (tk[n] > 0.0f && bk_add_arc(&g, n, g.sink, tk[n])) { oom(); goto fail; }
    }
    int WH = W * H;
    for (int t = 0; t < T; ++t)
        for (int y = 0; y < H; ++y)
            for (int x = 0; x < W; ++x) {
                int n = t * WH + y * W + x;
                if (x + 1 < W) {
                    int i = t * H * (W - 1) + y * (W - 1) + x;
                    if (ch[i] > 0 && bk_add_pair(&g, n, n + 1, ch[i])) { oom(); goto fail; }
                }
                if (y + 1 < H) {
                    int i = t * (H - 1) * W + y * W + x;
                    if (cv[i] > 0 && bk_add_pair(&g, n, n + W, cv[i])) { oom(); goto fail; }
                }
                if (t + 1 < T) {
                    int i = t * H * W + y * W + x;
                    if (ctp[i] > 0 && bk_add_pair(&g, n, n + WH, ctp[i])) { oom(); goto fail; }
                }
            }

    g.tree[g.source] = 1; g.tree[g.sink] = 2;
    bk_enqueue(&g, g.source); bk_enqueue(&g, g.sink);
    double flow = bk_maxflow(&g);

    npy_intp dims[1] = { N };
    PyArrayObject *labels = (PyArrayObject*)PyArray_ZEROS(1, dims, NPY_UINT8, 0);
    if (!labels) goto fail;
    unsigned char *lab = PyArray_DATA(labels);
    for (int n = 0; n < N; ++n) lab[n] = (g.tree[n] == 1) ? 0 : 1;

    free(g.first); free(g.tree); free(g.parent); free(g.active);
    free(g.orphan_mark); free(g.queue); free(g.orphans); free(g.arcs);
    return Py_BuildValue("(Nd)", labels, flow);

fail:
    free(g.first); free(g.tree); free(g.parent); free(g.active);
    free(g.orphan_mark); free(g.queue); free(g.orphans); free(g.arcs);
    return NULL;
}

static PyMethodDef methods[] = {
    {"grid_mincut", py_solve, METH_VARARGS,
     "grid_mincut(W,H,T,term_src,term_sink,cap_h,cap_v,cap_t)->(labels,flow)"},
    {NULL, NULL, 0, NULL}
};
static struct PyModuleDef mod = { PyModuleDef_HEAD_INIT, "bkflow", NULL, -1, methods };
PyMODINIT_FUNC PyInit_bkflow(void) { import_array(); return PyModule_Create(&mod); }
