import json
import numpy as np
SYSTEM=("You repair a convex quadratic program active set. Choose exactly one action: ADD(i), DROP(i), or STOP. "
        "Indices start at 0. ADD requires an inactive index; DROP requires an active index. "
        "Do not explain or output multiple actions. Use numerical feedback to repair primal or dual infeasibility. "
        "STOP only when verified=true. At most 20 actions including STOP. /no_think")

def observation(env,mode="compact",initial=False):
    q,s=env.qp,env.solution; gap=q.A@s.x-q.b; row=np.sum(np.abs(q.A),axis=1)
    xs=max(1.,np.linalg.norm(s.x,np.inf)); gs=np.linalg.norm(q.Q,np.inf)*xs+np.linalg.norm(q.c,np.inf)
    normalized=gap/(row*xs+np.abs(q.b))
    violations=sorted([(i,float(v)) for i,v in enumerate(normalized) if v>1e-8],key=lambda z:-z[1])
    fmt=lambda pairs: " ".join(f"{i}:{v:.3g}" for i,v in pairs) or "none"
    c=env.certificate
    text=(f"n={q.n} m={q.m} step={env.steps}/{env.max_steps} active={env.active}\n"
          f"verified={str(c.success).lower()} residuals stationarity={c.stationarity:.3g} primal={c.primal:.3g} "
          f"dual={c.dual:.3g} complementarity={c.complementarity:.3g}\n"
          f"violations(index:value descending): {fmt(violations)}\n"
          f"active_multipliers(index:scaled_value): {fmt([(i,float(s.lam[i]*row[i]/gs)) for i in env.active])}\n"
          f"last_error={env.error or 'none'} terminated={str(env.done).lower()}\nAction:")
    if mode=="full" and initial:
        text="QP data: "+json.dumps({k:v for k,v in q.to_dict().items() if k!="problem_id"},separators=(",",":"))+"\n"+text
    elif mode not in ("compact","full"): raise ValueError("unknown observation mode")
    return text

def messages(env,mode="compact"):
    return [{"role":"system","content":SYSTEM},{"role":"user","content":observation(env,mode,True)}]
