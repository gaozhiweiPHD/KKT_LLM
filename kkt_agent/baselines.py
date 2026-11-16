"""Policies and a separate feasible primal active-set algorithm."""
from time import perf_counter
import numpy as np
from scipy.optimize import linprog
from .qp_generator import QP
from .qp_solver import solve_active,NumericalError
from .kkt_verifier import verify

def greedy(env):
    if env.certificate.success: return "STOP"
    lam=env.solution.lam
    negative=[i for i in env.active if lam[i]<-1e-9]
    if negative: return f"DROP({min(negative,key=lambda i:lam[i])})"
    gap=env.qp.A@env.solution.x-env.qp.b
    for i in np.argsort(-gap):
        if int(i) in env.active or gap[i]<=0: continue
        start=perf_counter()
        try: solve_active(env.qp,env.active+[int(i)]); return f"ADD({int(i)})"
        except NumericalError: pass
        finally: env.solver_time+=perf_counter()-start
    return f"DROP({min(env.active,key=lambda i:lam[i])})" if env.active else "STOP"

def random_valid(env,rng):
    candidates=["STOP"]+[f"DROP({i})" for i in env.active]
    if len(env.active)<env.qp.n: candidates += [f"ADD({i})" for i in range(env.qp.m) if i not in env.active]
    rng.shuffle(candidates)
    for action in candidates:
        if not action.startswith("ADD"): return action
        i=int(action[4:-1])
        start=perf_counter()
        try: solve_active(env.qp,env.active+[i]); return action
        except NumericalError: pass
        finally: env.solver_time+=perf_counter()-start
    return "STOP"

def classical_active_set(qp,max_steps=20):
    """Feasible primal method: equality search directions and blocking line search.
    Phase I is a separate LP and is included in wall-clock cost; no oracle access.
    """
    start=perf_counter()
    lp=linprog(np.r_[np.zeros(qp.n),-1.],A_ub=np.c_[qp.A,np.ones(qp.m)],b_ub=qp.b,
               bounds=[(None,None)]*qp.n+[(0.,1.)],method="highs",options={"threads":1})
    if not lp.success: raise NumericalError("phase-I LP failed")
    x=lp.x[:-1]; active=[]; history=[]; visits=set(); cycle=False
    lam=np.zeros(qp.m); reason="max_steps"; cert=verify(qp,x,lam)
    for step in range(1,max_steps+1):
        direction_qp=QP(qp.Q,qp.Q@x+qp.c,qp.A,np.zeros(qp.m))
        sol=solve_active(direction_qp,active); p=sol.x; lam=sol.lam
        if np.linalg.norm(p,np.inf)<=1e-9*(1+np.linalg.norm(x,np.inf)):
            cert=verify(qp,x,lam)
            if cert.success:
                reason="kkt_success"; history.append({"iteration":step,"active":active.copy(),"residuals":cert.to_dict(),"action":"STOP"}); break
            neg=[i for i in active if lam[i]<0]
            if not neg: reason="numerical_stagnation"; break
            drop=min(neg,key=lambda i:lam[i]); active.remove(drop); action=f"DROP({drop})"
        else:
            ap=qp.A@p; gap=qp.b-qp.A@x
            candidates=[i for i in range(qp.m) if i not in active and ap[i]>1e-12]
            alpha=1.; blocker=None
            for i in candidates:
                ratio=max(0.,float(gap[i]/ap[i]))
                if ratio<alpha: alpha,blocker=ratio,i
            x=x+alpha*p; action=f"LINESEARCH({alpha:.6g})"
            if blocker is not None:
                active.append(blocker); active.sort(); action+=f"+ADD({blocker})"
        state=tuple(active)
        # repeated working set alone isn't a cycle if a positive direction step occurred
        key=(state,tuple(np.round(x,10)))
        cycle |= key in visits; visits.add(key)
        cert=verify(qp,x,lam)
        history.append({"iteration":step,"active":active.copy(),"residuals":cert.to_dict(),"action":action})
    cert=verify(qp,x,lam)
    elapsed=perf_counter()-start
    rec={"problem_id":qp.problem_id,"success":reason=="kkt_success","reason":reason,"actions":step,
         "invalid_actions":0,"invalid_action_rate":0.,"incorrect_stop":False,"cycle":cycle,
         "inference_seconds":0.,"solver_seconds":elapsed,"wall_seconds":elapsed,"input_tokens":0,"output_tokens":0,
         **{k:v for k,v in cert.to_dict().items() if k!="success"},"kkt_candidate":cert.success}
    if history: history[-1]["candidate"]={"x":x.tolist(),"lambda":lam.tolist()}
    return rec,history
