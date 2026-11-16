"""Feasible strictly convex QPs with separately stored primal-dual certificates."""
from dataclasses import dataclass, asdict
import numpy as np

@dataclass
class QP:
    Q: np.ndarray
    c: np.ndarray
    A: np.ndarray
    b: np.ndarray
    problem_id: str = ""

    @property
    def n(self): return len(self.c)
    @property
    def m(self): return len(self.b)

    def validate(self):
        if self.Q.shape != (self.n, self.n) or self.A.shape != (self.m, self.n):
            raise ValueError("inconsistent QP shapes")
        if not all(np.isfinite(a).all() for a in (self.Q,self.c,self.A,self.b)):
            raise ValueError("nonfinite QP")
        if not np.allclose(self.Q,self.Q.T,rtol=1e-12,atol=1e-12):
            raise ValueError("Q must be symmetric")
        np.linalg.cholesky(self.Q)
        if np.any(np.linalg.norm(self.A,axis=1)==0): raise ValueError("zero row")

    def to_dict(self):
        return {k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in asdict(self).items()}

    @classmethod
    def from_dict(cls,d):
        return cls(**{k:np.asarray(d[k],dtype=np.float64) for k in ("Q","c","A","b")},
                   problem_id=d.get("problem_id",""))

def generate_qp(seed,n=12,m=36,condition=10.,active_size=None,family="standard",problem_id=None):
    if not (m>=1 and n>=1 and condition>=1): raise ValueError("invalid dimensions/condition")
    rng=np.random.default_rng(seed)
    k=int(active_size if active_size is not None else rng.integers(min(3,n,m),min(10,n,m)+1))
    if not 0<=k<=min(n,m): raise ValueError("active size exceeds rank")
    U,_=np.linalg.qr(rng.normal(size=(n,n)))
    Q=(U*np.geomspace(1.,condition,n))@U.T
    Q=(Q+Q.T)/2
    active=np.sort(rng.choice(m,k,replace=False))
    for _ in range(100):
        A=rng.normal(size=(m,n))
        if family=="correlated": A=.8*A+.6*rng.normal(size=(1,n))
        A/=np.linalg.norm(A,axis=1,keepdims=True)
        if k==0 or np.linalg.cond(A[active])<1e3: break
    else: raise RuntimeError("active rows ill-conditioned")
    x=rng.normal(scale=.5,size=n)
    slack=rng.uniform(.2,2.,m)
    inactive=np.setdiff1d(np.arange(m),active)
    if family=="small_slack": slack[inactive]=rng.uniform(1e-4,1e-2,len(inactive))
    elif family=="nearly_active":
        near=rng.choice(inactive,min(5,len(inactive)),replace=False)
        slack[near]=rng.uniform(1e-7,1e-5,len(near))
    elif family not in ("standard","correlated"): raise ValueError("unknown family")
    slack[active]=0
    lam=np.zeros(m); lam[active]=rng.uniform(.2,2.,k)
    b=A@x+slack; c=-Q@x-A.T@lam
    qp=QP(Q,c,A,b,problem_id or f"seed-{seed}"); qp.validate()
    oracle={"problem_id":qp.problem_id,"x_star":x.tolist(),"lambda_star":lam.tolist(),"active_set":active.tolist()}
    meta={"problem_id":qp.problem_id,"seed":int(seed),"n":n,"m":m,"condition":condition,
          "active_size":k,"family":family,"active_row_condition":float(np.linalg.cond(A[active])) if k else 1.}
    return qp,oracle,meta
