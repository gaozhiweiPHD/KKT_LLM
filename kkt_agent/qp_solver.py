"""Checked Schur/Cholesky equality solver and independent OSQP wrapper."""
from dataclasses import dataclass
from time import perf_counter
import numpy as np
from scipy.linalg import cho_factor,cho_solve

class NumericalError(ValueError): pass
@dataclass
class Solution:
    x: np.ndarray
    lam: np.ndarray
    seconds: float
    condition: float

def solve_active(qp,active,condition_limit=1e12):
    start=perf_counter(); active=list(active)
    if len(active)!=len(set(active)) or any(type(i) not in (int,np.int64,np.int32) or not 0<=i<qp.m for i in active):
        raise NumericalError("invalid/duplicate active index")
    if len(active)>qp.n: raise NumericalError("more active rows than variables")
    try:
        qp.validate(); cond=np.linalg.cond(qp.Q)
        if cond>condition_limit: raise NumericalError("Q exceeds condition limit")
        factor=cho_factor(qp.Q,lower=True); z=cho_solve(factor,-qp.c); lam=np.zeros(qp.m)
        if active:
            AI=qp.A[active]; norms=np.linalg.norm(AI,axis=1); B=AI/norms[:,None]
            if np.linalg.matrix_rank(B,tol=1e-10)<len(active): raise NumericalError("dependent active rows")
            W=cho_solve(factor,B.T); S=B@W
            cond=max(cond,float(np.linalg.cond(S)))
            if cond>condition_limit: raise NumericalError("Schur complement exceeds condition limit")
            mu=cho_solve(cho_factor((S+S.T)/2,lower=True),B@z-qp.b[active]/norms)
            x=z-W@mu; lam[active]=mu/norms
            err=np.max(np.abs(B@x-qp.b[active]/norms))
            if err>1e-7*(1+np.linalg.norm(x,np.inf)+np.max(np.abs(qp.b[active]/norms))):
                raise NumericalError("equality solve inaccurate")
        else: x=z
        if not np.isfinite(x).all() or not np.isfinite(lam).all(): raise NumericalError("nonfinite solution")
        return Solution(x,lam,perf_counter()-start,float(cond))
    except (np.linalg.LinAlgError,ValueError) as exc: raise NumericalError(str(exc)) from exc

def solve_osqp(qp):
    import osqp
    from scipy.sparse import csc_matrix
    start=perf_counter(); solver=osqp.OSQP()
    solver.setup(P=csc_matrix(qp.Q),q=qp.c,A=csc_matrix(qp.A),l=np.full(qp.m,-np.inf),u=qp.b,
                 verbose=False,eps_abs=1e-9,eps_rel=1e-9,max_iter=100000,polishing=True)
    result=solver.solve(raise_error=False)
    if result.x is None or result.y is None or result.info.status_val not in (1,2):
        raise NumericalError(f"OSQP: {result.info.status}")
    return Solution(result.x,result.y,perf_counter()-start,float(np.linalg.cond(qp.Q)))
