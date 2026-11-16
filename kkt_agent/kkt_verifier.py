"""Independent scale-normalized KKT verification, without active-set comparison."""
from dataclasses import dataclass,asdict
import numpy as np

@dataclass(frozen=True)
class Certificate:
    stationarity: float
    primal: float
    dual: float
    complementarity: float
    raw_stationarity: float
    raw_primal: float
    raw_dual: float
    raw_complementarity: float
    success: bool
    @property
    def merit(self): return self.stationarity+self.primal+self.dual+self.complementarity
    def to_dict(self): return {**asdict(self),"merit":self.merit}

def verify(qp,x,lam,tolerance=1e-5):
    x,lam=np.asarray(x),np.asarray(lam)
    if x.shape!=(qp.n,) or lam.shape!=(qp.m,): raise ValueError("candidate shape")
    if tolerance<=0 or not np.isfinite(tolerance): raise ValueError("invalid tolerance")
    if not all(np.isfinite(a).all() for a in (qp.Q,qp.c,qp.A,qp.b,x,lam)):
        return Certificate(*([float("inf")]*8),False)
    slack=qp.A@x-qp.b; stationarity=qp.Q@x+qp.c+qp.A.T@lam
    xs=max(1.,float(np.linalg.norm(x,np.inf))); row=np.sum(np.abs(qp.A),axis=1)
    tiny=np.finfo(float).tiny
    ps=np.maximum(row*xs+np.abs(qp.b),tiny)
    gs=max(float(np.linalg.norm(qp.Q,np.inf)*xs+np.linalg.norm(qp.c,np.inf)),tiny)
    raw=[np.max(np.abs(stationarity)),np.max(np.maximum(slack,0)),np.max(np.maximum(-lam,0)),np.max(np.abs(lam*slack))]
    norm=[raw[0]/gs,np.max(np.maximum(slack,0)/ps),np.max(np.maximum(-lam,0)*row)/gs,raw[3]/(gs*xs)]
    return Certificate(*map(float,norm),*map(float,raw),bool(np.isfinite(norm).all() and max(norm)<=tolerance))
