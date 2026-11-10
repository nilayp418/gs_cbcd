/* End-to-end RHF in C, timed from the C call: diagonalization (SCF), CG and CBCD. Python builds h, the Fock-build integrals (dense Q as
 * oracles.make_rhf_oracle_energy, or a DF tensor B) and the MINAO guess C0, then calls fast_scf(_df):
 *   E = fock(C0);  repeat { inner(F, C); E' = fock(C), F <- (1-damp) F_new + damp F } until |dE| < eps
 * Inner solvers (lowest-energy orthonormal C, nb x no, for fixed F):
 *   0 SCF : lowest no eigenvectors of F (dsyevr, lowest no only, if no/nb < 0.08; else dsyevd).
 *   1 CG  : rhf/cg_scf.cg_inner_loop (Fletcher-Reeves, quadratic-fit line search along qf(u - lam p),
 *           Armijo fallback); trial energies from no x no matrices (see CG section).
 *   2 CBCD: rhf/cbcd_scf.circle_optimize column choice (c = N_o: Gauss-Southwell; c = 0: random order;
 *           c >= 1: pivoted QR, first floor(no/c) pivots) + exact geodesic step; implicit projected
 *           gradient (M = C^T G kept, P = G - C M never stored).
 * Matched stopping rule (tol = inner_tol): CG stops when one iteration lowers E = Tr(C^T F C) by <= tol;
 * CBCD c = 0, 1, 2 (and c = N_o with cno_window) when its last no steps together do; c = N_o when one
 * step does (Python passes tol / no). Fock build: dense Q (rows p <= q, parallel) or DF; timed separately.
 * Layouts: h, F, D nb x nb; Q nb^2 x nb^2 row-major; B naux x nb x nb; C nb x no column-major.
 * out[]: 0 E, 1 wall, 2 Fock time, 3 outer its, 4 inner its, 5 inner time. Returns 0 if converged.
 * Notation: X[i + j*nb] = entry (i, j); X + j*nb = column j. GEMM: C <- a op(A) op(B) + b C;
 * GEMV: y <- a op(A) x + b y; DOT: x^T y; TRSM: X <- X R^-1 or R^-T X (R upper); CPY: copy. */
#include <dispatch/dispatch.h>
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#define ACCELERATE_NEW_LAPACK
#include <Accelerate/Accelerate.h> /* Apple's BLAS + LAPACK */

typedef __LAPACK_int li; /* LAPACK's integer type */
#define N_ CblasNoTrans
#define T_ CblasTrans
#define GEMM(ta, tb, ...) cblas_dgemm(CblasColMajor, ta, tb, __VA_ARGS__) /* (m, n, k, a, A, lda, B, ldb, b, C, ldc) */
#define GEMV(ta, ...) cblas_dgemv(CblasColMajor, ta, __VA_ARGS__)         /* (m, n, a, A, lda, x, 1, b, y, 1) */
#define TRSM(side, ta, m, n, R, ldr, X, ldx) cblas_dtrsm(CblasColMajor, side, CblasUpper, ta, CblasNonUnit, m, n, 1.0, R, ldr, X, ldx)
#define DOT(n, x, y) cblas_ddot((int)(n), x, 1, y, 1)
#define CPY(dst, src, n) memcpy(dst, src, sizeof(double) * (n))

static inline double now_s(void) { return clock_gettime_nsec_np(CLOCK_UPTIME_RAW) * 1e-9; } /* wall clock, s */

/* ---------------------------------------------------------------- shared helpers */

typedef struct
{
    void *p[40];
    int n;
} Pool; /* every allocation of one run, freed at the end */
static double *dget(Pool *P, size_t n) { return P->p[P->n++] = malloc(sizeof(double) * (n ? n : 1)); }
static li *iget(Pool *P, size_t n) { return P->p[P->n++] = malloc(sizeof(li) * (n ? n : 1)); }

typedef struct
{
    li lw, liw, qlw, *iw;
    double *wk, *qw, *tau;
} Lap;                                                          /* LAPACK scratch: dsyevd (order neig), QR (m x nqr) */
static void lap_init(Pool *P, Lap *L, int m, int nqr, int neig) /* workspace queries (lwork = -1), then allocate */
{
    li mm = m, n = nqr, ne = neig, q = -1, iq = 1, info;
    double a = 1, b = 1, c = 1, d = 1;
    if (neig)
        dsyevd_("V", "U", &ne, NULL, &ne, NULL, &a, &q, &iq, &q, &info);
    if (nqr)
    {
        dgeqrf_(&mm, &n, NULL, &mm, NULL, &b, &q, &info);
        dorgqr_(&mm, &n, &n, NULL, &mm, NULL, &c, &q, &info);
        dgeqp3_(&mm, &n, NULL, &mm, NULL, NULL, &d, &q, &info);
    }
    L->lw = (li)fmax(a, 1);
    L->liw = iq > 1 ? iq : 1;
    L->qlw = (li)fmax(fmax(b, c), fmax(d, 1));
    L->wk = dget(P, L->lw);
    L->iw = iget(P, L->liw);
    L->qw = dget(P, L->qlw);
    L->tau = dget(P, nqr);
}

static double sum(const double *w, int n)
{
    double s = 0;
    for (int i = 0; i < n; i++)
        s += w[i];
    return s;
}

static void proj(int nb, int no, const double *C, const double *X, double *out, double *M) /* out = (I - C C^T) X */
{
    GEMM(T_, N_, no, no, nb, 1.0, C, nb, X, nb, 0.0, M, no); /* M = C^T X */
    if (out != X)
        CPY(out, X, (size_t)nb * no);
    GEMM(N_, N_, nb, no, no, -1.0, C, nb, M, no, 1.0, out, nb); /* out = X - C M */
}

static void cholqr(int nb, int no, double *C, double *W) /* re-orthonormalize: C <- C R^-1, R^T R = C^T C */
{
    li n = no, info;
    cblas_dsyrk(CblasColMajor, CblasUpper, T_, no, nb, 1.0, C, nb, 0.0, W, no); /* W = C^T C */
    dpotrf_("U", &n, W, &n, &info);                                             /* W <- R (Cholesky) */
    TRSM(CblasRight, N_, nb, no, W, no, C, nb);                                 /* C <- C R^-1 */
}

/* ---------------------------------------------------------------- Fock build */

typedef struct
{
    int nb, no, naux;
    const double *h, *Q, *B;
    double *F, *D, *y, t, *J, *K, *Z, *d, *W;
} Sys;

#define DF_BLK 64 /* aux functions per DF block */
static void sys_init(Pool *P, Sys *s, int nb, int no, int naux, const double *h, const double *Q, const double *B)
{
    size_t n2 = (size_t)nb * nb;
    double *b = dget(P, 3 * n2 + (size_t)no * no); /* one block (as in the benchmarked runs) */
    *s = (Sys){.nb = nb, .no = no, .naux = naux, .h = h, .Q = Q, .B = B, .F = b, .D = b + n2, .y = b + 2 * n2, .W = b + 3 * n2};
    if (B)
    {
        s->J = dget(P, n2);
        s->K = dget(P, n2);
        s->d = dget(P, DF_BLK);
        s->Z = dget(P, (size_t)nb * DF_BLK * no);
    }
}

/* DF: (pq|rs) ~ sum_P B_P[pq] B_P[rs]  =>  y = Q vec(D) = 2J - K, J = sum_P Tr(C^T B_P C) B_P, K = sum_P (B_P C)(B_P C)^T */
static void df_y(Sys *s, const double *C)
{
    int nb = s->nb, no = s->no;
    size_t n2 = (size_t)nb * nb;
    double *J = s->J, *K = s->K, *Z = s->Z, *d = s->d;
    memset(J, 0, sizeof(double) * n2); /* J = 0 */
    for (int P0 = 0; P0 < s->naux; P0 += DF_BLK)
    { /* block of aux functions P0 .. P0+nk-1 */
        int nk = s->naux - P0 < DF_BLK ? s->naux - P0 : DF_BLK;
        const double *Bb = s->B + (size_t)P0 * n2;
        GEMM(T_, N_, nb * nk, no, nb, 1.0, Bb, nb, C, nb, 0.0, Z, nb * nk); /* Z_P = B_P C, all P in block */
        memset(d, 0, sizeof(double) * nk);
        for (int i = 0; i < no; i++)
            GEMV(T_, nb, nk, 1.0, Z + (size_t)nb * nk * i, nb, C + (size_t)nb * i, 1, 1.0, d, 1);   /* d_P = Tr(C^T B_P C) */
        GEMV(N_, (int)n2, nk, 1.0, Bb, (int)n2, d, 1, 1.0, J, 1);                                   /* J += sum_P d_P B_P */
        cblas_dsyrk(CblasColMajor, CblasUpper, N_, nb, nk * no, 1.0, Z, nb, P0 ? 1.0 : 0.0, K, nb); /* K += sum_P Z_P Z_P^T */
    }
    for (int j = 0; j < nb; j++)
        for (int i = 0; i <= j; i++)
            s->y[i + j * nb] = s->y[j + i * nb] = 2 * J[i + j * nb] - K[i + j * nb]; /* y = 2J - K */
}

/* E = D.(h + F_new), F_new = h + Q vec(D), D = C C^T; F <- F_new, or (1-mix) F_new + mix F */
static double fock(Sys *s, const double *C, double mix)
{
    double t0 = now_s(), *D = s->D, *y = s->y, e = 0;
    int nb = s->nb;
    size_t n2 = (size_t)nb * nb;
    cblas_dsyrk(CblasColMajor, CblasUpper, N_, nb, s->no, 1.0, C, nb, 0.0, D, nb); /* D = C C^T (upper) */
    for (int j = 0; j < nb; j++)
        for (int i = j + 1; i < nb; i++)
            D[i + j * nb] = D[j + i * nb]; /* lower = upper */
    const double *Q = s->Q;
    if (s->B)
        df_y(s, C); /* y = 2J - K */
    else
    {
        dispatch_apply(nb, dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE, 0), ^(size_t p) { /* rows p in parallel */
                                                                                                   for (size_t q = p; q < (size_t)nb; q++)
                                                                                                       y[p * nb + q] = DOT(n2, Q + (p * nb + q) * n2, D); /* y_pq = sum_rs Q[pq,rs] D_rs, p <= q */
        });
        for (int p = 0; p < nb; p++)
            for (int q = p + 1; q < nb; q++)
                y[q * nb + p] = y[p * nb + q]; /* y symmetric */
    }
    for (size_t i = 0; i < n2; i++)
    {
        double fn = s->h[i] + y[i];                              /* F_new = h + y */
        e += D[i] * (s->h[i] + fn);                              /* E = sum D o (h + F_new) */
        s->F[i] = mix > 0 ? (1 - mix) * fn + mix * s->F[i] : fn; /* damping */
    }
    s->t += now_s() - t0;
    return e;
}

/* ---------------------------------------------------------------- outer SCF loop (all solvers) */

typedef long (*Inner)(void *state, const double *F, double *C, double tol, long max_it); /* < 0: failed */

/* Optional energy profiles, as the Python drivers' opt_profiling (filled only after prof_set turns them on):
 *   outer[i]  = E after outer iteration i (outer[0]: the guess)                          -> scf_profile
 *   inner[..] = Tr(C^T F C) at the start of each inner solve, then after every CG iteration
 *               or CBCD step, all inner solves concatenated
 *   len[i]    = number of inner entries from outer iteration i + 1                      -> sub_iter_energies */
typedef struct
{
    double *outer, *inner;
    long *len, cap, n;
} Prof;
static Prof prof_buf, *prof = NULL; /* NULL: not profiling (the timed runs) */

static void rec(double e) /* append one inner energy (n > cap: overflowed, extra entries dropped) */
{
    if (prof->n < prof->cap)
        prof->inner[prof->n] = e;
    prof->n++;
}

static int outer(Sys *s, double *C, Inner inner_fn, void *state, int reorth, double eps, int iter_max,
                 double damp, double tol, double t0, double *out)
{
    int it = 0, rc = 0;
    long inner = 0;
    double t_inner = 0, e = fock(s, C, 0), dE = e; /* E_0, F from the guess (no damping) */
    if (prof)
        prof->outer[0] = e; /* profile: guess energy */
    while (fabs(dE) > eps)
    {
        if (it > iter_max)
        {
            rc = -1;
            break;
        }
        long n0 = prof ? prof->n : 0; /* profile: inner entries before this solve */
        double ti = now_s();
        long k = inner_fn(state, s->F, C, tol, iter_max); /* C <- argmin Tr(C^T F C), warm start */
        if (k < 0)
        {
            rc = -1;
            break;
        }
        inner += k;
        if (reorth)
            cholqr(s->nb, s->no, C, s->W); /* as cbcd_scf's retract(C, C, 0) */
        t_inner += now_s() - ti;
        it++;
        double e_prev = e;
        e = fock(s, C, damp); /* new E, damped F */
        dE = e - e_prev;
        if (prof)
        {
            prof->outer[it] = e;              /* profile: E after outer iteration it */
            prof->len[it - 1] = prof->n - n0; /* profile: entries this inner solve added */
        }
    }
    out[0] = e;
    out[1] = now_s() - t0;
    out[2] = s->t;
    out[3] = it;
    out[4] = (double)inner;
    out[5] = t_inner;
    return rc;
}

/* ---------------------------------------------------------------- 0: SCF (diagonalization) */

typedef struct
{
    li n, no, lw, liw, *iw, *sup;
    double *wk, *A, *w;
    int subset;
} Eig;

static void eig_call(Eig *e, double *A, double *Z, double *wk, li *iw, li lw, li liw) /* F = V diag(w) V^T */
{
    li info, il = 1, iu = e->no, m;
    double z = 0;
    if (e->subset)
        dsyevr_("V", "I", "L", &e->n, A, &e->n, &z, &z, &il, &iu, &z, &m, e->w, Z, &e->n, e->sup, wk, &lw, iw, &liw, &info); /* pairs 1..no -> Z */
    else
        dsyevd_("V", "L", &e->n, A, &e->n, e->w, wk, &lw, iw, &liw, &info); /* all pairs: A <- V */
}

static void eig_init(Pool *P, Eig *e, int nb, int no)
{
    double wq;
    li iwq;
    *e = (Eig){.n = nb, .no = no, .subset = (double)no / nb < 0.08, .w = dget(P, nb), .sup = iget(P, 2 * nb)}; /* subset is found empirically--either works, but Claude suggests 0.08 as a switching point */
    eig_call(e, NULL, NULL, &wq, &iwq, -1, -1);                                                                /* workspace query */
    e->lw = (li)wq;
    e->liw = iwq;
    e->wk = dget(P, e->lw);
    e->iw = iget(P, e->liw);
    e->A = dget(P, (size_t)nb * nb);
}

static long scf_inner(void *st, const double *F, double *C, double tol, long max_it)
{
    Eig *e = st;
    (void)tol;
    (void)max_it;
    CPY(e->A, F, (size_t)e->n * e->n); /* A = F (overwritten by LAPACK) */
    eig_call(e, e->A, C, e->wk, e->iw, e->lw, e->liw);
    if (!e->subset)
        CPY(C, e->A, (size_t)e->n * e->no); /* C = V[:, :no] */
    return 1;
}

/* ---------------------------------------------------------------- 1: CG
 * Trial energies: X = U - lam P, Q = qf(X) = X R^-1 (R^T R = X^T X) => Tr(Q^T F Q) = Tr[(X^T X)^-1 X^T F X] with
 * X^T X = UU - lam (S1 + S1^T) + lam^2 S2, X^T F X = A - lam (B + B^T) + lam^2 E (A = U^T F U, B = (FU)^T P,
 * E = P^T F P, S1 = U^T P, S2 = P^T P, UU = U^T U). One F P product per direction; each trial O(no^3).
 * Only the symmetric part of X^T F X enters the trace (R^-T W R^-1 of an antisymmetric W is antisymmetric,
 * so traceless), so W = A - 2 lam B + lam^2 E is used in its place (B itself is not symmetric).
 * U, P are stored side by side (X = [U | P]), as are FU, FP (FX), so UU, S1, S2 come from one dsyrk and
 * B, E from one GEMM. The last trial of every line search is at the accepted lam, so its R is reused:
 * U <- (U - lam P) R^-1, FU <- (FU - lam FP) R^-1 (FU is computed exactly only at the start of each solve;
 * measured drift of the carried FU stays at ~1e-15 relative, no growth, over 81 its).
 * In the code, U is u (the CG copy of the input U) and P is gc; FU, FP are g->FU, g->FP. */

typedef struct
{
    int nb, no;
    Lap *L;
    double *X, *FX, *gc, *FU, *FP, *pp, *A, *S, *T, *R, *W;
} CG;

static void cg_init(Pool *P, CG *g, Lap *L, int nb, int no)
{
    size_t N = (size_t)nb * no, n2 = (size_t)no * no;
    *g = (CG){.nb = nb, .no = no, .L = L, .X = dget(P, 2 * N), .FX = dget(P, 2 * N), .pp = dget(P, N), .A = dget(P, n2), .S = dget(P, 4 * n2), .T = dget(P, 2 * n2), .R = dget(P, n2), .W = dget(P, n2)};
    g->gc = g->X + N;
    g->FU = g->FX;
    g->FP = g->FX + N; /* X = [u | p], FX = [F u | F p] (nb x 2no each) */
}

static void cg_line_setup(CG *g, const double *F) /* p = g->gc */
{
    int nb = g->nb, no = g->no;
    GEMM(N_, N_, nb, no, nb, 1.0, F, nb, g->gc, nb, 0.0, g->FP, nb);                          /* FP = F p (the nb^2 no product) */
    cblas_dsyrk(CblasColMajor, CblasUpper, T_, 2 * no, nb, 1.0, g->X, nb, 0.0, g->S, 2 * no); /* S = [u p]^T [u p] = [UU S1; . S2] (upper) */
    GEMM(T_, N_, 2 * no, no, nb, 1.0, g->FX, nb, g->gc, nb, 0.0, g->T, 2 * no);               /* T = [Fu Fp]^T p = [B; E^T] */
}

static double cg_line_energy(CG *g, double lam) /* E(qf(u - lam p)); leaves chol(X^T X) in R; NAN if not PD */
{
    int no = g->no, m = 2 * no; /* m: leading dimension of S and T */
    li n = no, info;
    const double *S = g->S, *T = g->T;
    if (!isfinite(lam))
        return NAN;
    for (int j = 0; j < no; j++)
        for (int i = 0; i < no; i++)
        {
            g->W[i + j * no] = g->A[i + j * no] - 2 * lam * T[i + j * m] + lam * lam * T[no + i + j * m]; /* W = A - 2 lam B + lam^2 E^T (same trace as X^T F X) */
            if (i <= j)
                g->R[i + j * no] = S[i + j * m] - lam * (S[i + (no + j) * m] + S[j + (no + i) * m]) /* X^T X = UU - lam (S1 + S1^T) */
                                   + lam * lam * S[no + i + (no + j) * m];                          /*   + lam^2 S2 (upper half) */
        }
    dpotrf_("U", &n, g->R, &n, &info); /* R^T R = X^T X */
    if (info)
        return NAN;
    TRSM(CblasLeft, T_, no, no, g->R, no, g->W, no);  /* W <- R^-T W */
    TRSM(CblasRight, N_, no, no, g->R, no, g->W, no); /* W <- R^-T (X^T F X) R^-1 = Q^T F Q */
    double t = 0;
    for (int k = 0; k < no; k++)
        t += g->W[k + k * no]; /* Tr(Q^T F Q) */
    return t;
}

static void cg_accept(CG *g, double lam) /* u <- qf(u - lam p), FU = F u; R from the last trial (at lam) */
{
    int nb = g->nb, no = g->no;
    size_t N = (size_t)nb * no;
    for (size_t i = 0; i < N; i++)
        g->X[i] -= lam * g->gc[i]; /* u <- u - lam p (in place) */
    for (size_t i = 0; i < N; i++)
        g->FU[i] -= lam * g->FP[i];                    /* FU = F (u - lam p) */
    TRSM(CblasRight, N_, nb, no, g->R, no, g->X, nb);  /* u <- (u - lam p) R^-1 */
    TRSM(CblasRight, N_, nb, no, g->R, no, g->FU, nb); /* FU = F u */
}

static long cg_inner(void *st, const double *F, double *U, double eps, long iter_max)
{
    CG *g = st;
    int nb = g->nb, no = g->no;
    size_t N = (size_t)nb * no;
    double *u = g->X, *gc = g->gc, *pp = g->pp;
    CPY(u, U, N);                                                                                    /* u = U (first half of X) */
    GEMM(N_, N_, nb, no, nb, 1.0, F, nb, u, nb, 0.0, g->FU, nb);                                     /* FU = F u */
    double f_prev = DOT(N, g->FU, u), res_prev = 100, lam_prev = 1, f_curr = f_prev, del_E = f_prev; /* f = Tr(u^T F u) */
    long it = 0;
    if (prof)
        rec(f_prev);                   /* profile: starting energy */
    memset(pp, 0, sizeof(double) * N); /* p_prev = 0 */
    while (fabs(del_E) > eps)
    {
        if (it > iter_max)
            return -1;
        GEMM(T_, N_, no, no, nb, 1.0, u, nb, g->FU, nb, 0.0, g->A, no);                                    /* A = u^T F u */
        CPY(gc, g->FU, N);                                                                                 /* gc = F u */
        GEMM(N_, N_, nb, no, no, -1.0, u, nb, g->A, no, 1.0, gc, nb);                                      /* gc = F u - u A (alpha = -1, beta = 1: Accelerate's fast path) */
        cblas_dscal((int)N, 2.0, gc, 1);                                                                   /* gc = 2 (F u - u A) = (I - u u^T) 2 F u */
        double res = DOT(N, gc, gc), beta = it % (long)N == 0 ? 0 : res / res_prev; /* FR beta = res / res_prev */
        if (beta != 0)
            cblas_daxpy((int)N, beta, pp, 1, gc, 1); /* p = gc + beta p_prev */
        cg_line_setup(g, F);
        double v0 = f_curr, v1 = cg_line_energy(g, lam_prev), v2 = cg_line_energy(g, 2 * lam_prev); /* E at 0, lam, 2 lam */
        double b = -(-1.5 * v0 + 2 * v1 - 0.5 * v2) / (2 * (0.5 * v0 - v1 + 0.5 * v2));             /* quadratic-fit minimizer */
        double f_tmp = b == 0 ? f_curr : cg_line_energy(g, b * lam_prev), lam_curr = b * lam_prev;
        if (!(lam_curr > 0 && f_tmp < f_prev))
        { /* Armijo fallback along gc */
            if (beta != 0)
            {
                cblas_daxpy((int)N, -beta, pp, 1, gc, 1);
                cg_line_setup(g, F);
            }
            double step = 2 * lam_prev;
            while (!((f_tmp = cg_line_energy(g, step)) - (f_curr - 0.5 * step * res) <= 1e-12))
            { /* until sufficient decrease (NaN: halve) */
                if (step < 1e-14)
                    return -1;
                step /= 2;
            }
            lam_curr = step;
        }
        cg_accept(g, lam_curr); /* last trial was at lam_curr (finite, PD) */
        del_E = f_tmp - f_prev;
        f_curr = f_prev = f_tmp;
        lam_prev = lam_curr;
        res_prev = res;
        if (prof)
            rec(f_prev); /* profile: energy after this iteration */
        CPY(pp, gc, N);  /* p_prev = p */
        it++;
    }
    CPY(U, u, N); /* U = u */
    return it;
}

/* ---------------------------------------------------------------- 2: CBCD
 * State kept on the fly: G = F C, M = C^T G (symmetric), a_j = |G_j|^2, b_j = |M_j|^2, so the projected
 * gradient norms are |p_j|^2 = a_j - b_j (P = G - C M never stored). */

typedef struct
{
    int nb, no, cv, window;
    long max_steps;
    Lap *L;
    double *G, *P, *M, *a, *b, *r, *g, *fs, *hist;
    li *jpvt;
    int *order, head, len;
    uint64_t rng;
} CB;

static void cb_init(Pool *P, CB *c, Lap *L, int nb, int no, int cv, int window, uint64_t seed)
{
    size_t N = (size_t)nb * no;
    *c = (CB){.nb = nb, .no = no, .cv = cv, .window = window || cv != no, .max_steps = cv == no ? 100000000L : 100000L * no, .L = L, .G = dget(P, N), .P = dget(P, N), .M = dget(P, (size_t)no * no), .a = dget(P, no), .b = dget(P, no), .r = dget(P, no), .g = dget(P, nb), .fs = dget(P, nb), .hist = dget(P, no), .jpvt = iget(P, no), .order = (int *)iget(P, no), .rng = seed ? seed : 0x9E3779B97F4A7C15ULL};
}

static void cb_rebuild(int nb, int no, const double *F, const double *C, CB *c) /* exact state at the start of an inner solve, noting the factor of 2 is irrelevant since the step size is solved exactly! */
{
    GEMM(N_, N_, nb, no, nb, 1.0, F, nb, C, nb, 0.0, c->G, nb); /* G = F C */
    proj(nb, no, C, c->G, c->P, c->M);                          /* M = C^T G, P = G - C M */
    if (c->cv != no) /* a, b are only read by the Gauss-Southwell argmax (c = N_o) */
        return;
    for (int j = 0; j < no; j++)
    {
        c->a[j] = DOT(nb, c->G + j * nb, c->G + j * nb);           /* a_j = |G_j|^2 */
        c->b[j] = c->a[j] - DOT(nb, c->P + j * nb, c->P + j * nb); /* b_j = a_j - |p_j|^2 */
    }
}

/* one step on column k: rotate c_k toward its projected gradient by the exact minimizer; returns dE (0 if stationary) */
static double cb_step(int nb, int no, const double *F, double *C, int k, CB *c)
{
    double *M = c->M, *a = c->a, *b = c->b, *r = c->r, *g = c->g, *fs = c->fs, *ck = C + k * nb, *Gk = c->G + k * nb;
    CPY(g, Gk, nb);
    GEMV(N_, nb, no, -1.0, C, nb, M + k * no, 1, 1.0, g, 1); /* g = G_k - C M[:, k] = p_k */
    double nrm = sqrt(DOT(nb, g, g));                        /* |p_k| */
    if (nrm < 1e-10)
        return 0;
    cblas_dscal(nb, 1.0 / nrm, g, 1);                                                       /* g = p_k / |p_k| */
    GEMV(N_, nb, nb, 1.0, F, nb, g, 1, 0.0, fs, 1);                                         /* fs = F g (the nb^2 product) */
    double t1 = DOT(nb, ck, Gk), t2 = DOT(nb, g, fs), t3 = DOT(nb, g, Gk);                  /* c^T F c, g^T F g, g^T F c */
    double half = (t1 - t2) / 2, R = hypot(half, t3), c2 = -half / R, s2 = -t3 / R, cs, sn; /* E(th) = (t1+t2)/2 + half cos 2th + t3 sin 2th; min at (c2, s2) */
    if (c2 >= 0)
    {
        cs = sqrt((1 + c2) / 2);
        sn = s2 / (2 * cs);
    } /* half-angle formulas */
    else
    {
        double sa = sqrt((1 - c2) / 2);
        sn = copysign(sa, s2);
        cs = fabs(s2) / (2 * sa);
    }
    int gs = c->cv == no; /* a, b (projected-gradient norms) are only needed for the Gauss-Southwell argmax */
    double ak = 0, bk = 0;
    for (int i = 0; i < nb; i++)
    {
        ck[i] = cs * ck[i] + sn * g[i];  /* c_k <- cos th c_k + sin th g */
        Gk[i] = cs * Gk[i] + sn * fs[i]; /* G_k = F c_k, no new product */
    }
    GEMV(T_, nb, no, 1.0, c->G, nb, ck, 1, 0.0, r, 1); /* r = G^T c_k = new row/col k of M */
    if (gs)
    {
        for (int i = 0; i < nb; i++)
            ak += Gk[i] * Gk[i]; /* a_k = |G_k|^2 */
        a[k] = ak;
        for (int j = 0; j < no; j++)
        {
            if (j != k)
                b[j] += r[j] * r[j] - M[k + j * no] * M[k + j * no]; /* b_j changes only via M_kj */
            bk += r[j] * r[j];                                       /* b_k = |M_k|^2 */
        }
        b[k] = bk;
    }
    for (int j = 0; j < no; j++)
        M[k + j * no] = M[j + k * no] = r[j];
    return -half - R; /* E(th) - E(0) <= 0 */
}

static long cbcd_inner(void *st, const double *F, double *C, double tol, long max_it)
{
    CB *c = st;
    int nb = c->nb, no = c->no, cv = c->cv, kq = cv > 0 ? no / cv : 0, gs = cv == no; /* kq: columns per pivoted QR */
    (void)max_it;
    cb_rebuild(nb, no, F, C, c);
    long s = 0;
    double e = 0; /* profile: running Tr(C^T F C) */
    if (prof)
    {
        for (int j = 0; j < no; j++)
            e += c->M[j + j * no]; /* starts at Tr(M) = Tr(C^T F C) */
        rec(e);
    }
    c->head = c->len = 0;
    while (s < c->max_steps)
    {
        int k = 0;
        double best = -1e300;
        if (gs)
        {
            for (int j = 0; j < no; j++)
                if (c->a[j] - c->b[j] > best)
                {
                    best = c->a[j] - c->b[j];
                    k = j;
                }
        } /* argmax |p_j|^2 */
        else if (c->head < c->len)
            k = c->order[c->head++]; /* next queued column */
        else if (cv == 0)
        { /* random permutation (Fisher-Yates, xorshift64) */
            for (int i = 0; i < no; i++)
                c->order[i] = i;
            for (int i = no - 1; i > 0; i--)
            {
                uint64_t *x = &c->rng;
                *x ^= *x << 13;
                *x ^= *x >> 7;
                *x ^= *x << 17;
                int j = (int)(*x % (uint64_t)(i + 1)), t = c->order[i];
                c->order[i] = c->order[j];
                c->order[j] = t;
            }
            k = c->order[0];
            c->head = 1;
            c->len = no;
        }
        else
        { /* pivoted QR of P = G - C M */
            CPY(c->P, c->G, (size_t)nb * no);
            GEMM(N_, N_, nb, no, no, -1.0, C, nb, c->M, no, 1.0, c->P, nb);
            c->head = c->len = 0;
            if (kq <= 1)
            {
                best = -1;
                for (int j = 0; j < no; j++)
                {
                    double v = DOT(nb, c->P + j * nb, c->P + j * nb);
                    if (v > best)
                    {
                        best = v;
                        k = j;
                    }
                }
            }
            else
            {
                li m = nb, n = no, info;
                memset(c->jpvt, 0, sizeof(li) * no);
                dgeqp3_(&m, &n, c->P, &m, c->jpvt, c->L->tau, c->L->qw, &c->L->qlw, &info); /* P Pi = Q R */
                k = (int)c->jpvt[0] - 1;                                                    /* first pivot now, next kq - 1 queued */
                for (int i = 1; i < kq; i++)
                    c->order[c->len++] = (int)c->jpvt[i] - 1;
            }
        }
        double dE = cb_step(nb, no, F, C, k, c);
        if (gs && dE == 0)
            break; /* stationary */
        s++;
        if (prof)
            rec(e += dE); /* profile: energy after this step */
        if (!c->window)
        {
            if (fabs(dE) <= tol)
                break;
            continue;
        } /* one-step stop */
        c->hist[(s - 1) % no] = fabs(dE); /* window: last no steps */
        if (s >= no && sum(c->hist, no) <= tol)
            break;
    }
    return s;
}

/* ---------------------------------------------------------------- entry points */

static int drive(int method, int c_value, int cno_window, uint64_t seed, int nb, int no, int naux, const double *h,
                 const double *Q, const double *B, double *C, double eps, int iter_max, double damp, double inner_tol, double *out)
{
    double t0 = now_s(); /* timing starts here */
    Pool P = {0};
    Sys s;
    Lap L;
    Eig eg;
    CG cg;
    CB cb;
    Inner fn[] = {scf_inner, cg_inner, cbcd_inner};
    void *st[] = {&eg, &cg, &cb};
    sys_init(&P, &s, nb, no, naux, h, Q, B);
    if (method == 0)
        eig_init(&P, &eg, nb, no);
    else
    {
        lap_init(&P, &L, nb, no, 0);
        if (method == 1)
            cg_init(&P, &cg, &L, nb, no);
        else
            cb_init(&P, &cb, &L, nb, no, c_value, cno_window, seed);
    }
    int rc = outer(&s, C, fn[method], st[method], method == 2, eps, iter_max, damp, inner_tol, t0, out);
    for (int i = 0; i < P.n; i++)
        free(P.p[i]); /* free everything allocated above */
    return rc;
}

int fast_scf(int method, int c_value, int cno_window, uint64_t seed, int nb, int no, const double *h, const double *Q,
             double *C, double eps, int iter_max, double damp, double inner_tol, double *out) /* dense-Q Fock build */
{
    return drive(method, c_value, cno_window, seed, nb, no, 0, h, Q, NULL, C, eps, iter_max, damp, inner_tol, out);
}

/* profiling on (buffers from Python; outer, len: iter_max + 2 entries, inner: cap entries) or off (outer = NULL) */
void prof_set(double *outer, double *inner, long *len, long cap)
{
    prof_buf = (Prof){.outer = outer, .inner = inner, .len = len, .cap = cap, .n = 0};
    prof = outer ? &prof_buf : NULL;
}
long prof_count(void) { return prof_buf.n; } /* inner entries recorded (> cap: overflowed) */

int fast_scf_df(int method, int c_value, int cno_window, uint64_t seed, int nb, int no, int naux, const double *h, const double *B,
                double *C, double eps, int iter_max, double damp, double inner_tol, double *out) /* DF: B naux x nb x nb */
{
    return drive(method, c_value, cno_window, seed, nb, no, naux, h, NULL, B, C, eps, iter_max, damp, inner_tol, out);
}
