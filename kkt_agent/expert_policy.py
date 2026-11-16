"""Oracle access is restricted to offline teacher/curriculum construction."""
import numpy as np
from .active_set_env import ActiveSetEnv
from .baselines import greedy
from .observation import messages,observation
from .qp_solver import solve_active,NumericalError

def initial_set(qp,oracle,mode,rng,difficulty=3):
    truth=list(oracle["active_set"])
    if mode=="cold": return []
    if mode=="warm":
        env=ActiveSetEnv(qp)
        for _ in range(3):
            a=greedy(env)
            if a=="STOP": break
            env.step(a)
        return env.active
    active=truth.copy()
    if mode in ("missing","mixed"):
        count=min(len(active),int(rng.integers(1,difficulty+1)))
        remove=rng.choice(active,count,replace=False); active=[i for i in active if i not in remove]
    if mode in ("extra","mixed"):
        wrong=[i for i in range(qp.m) if i not in truth]; rng.shuffle(wrong)
        for i in wrong[:int(rng.integers(1,difficulty+1))]:
            try: solve_active(qp,active+[i]); active.append(i)
            except NumericalError: continue
    return sorted(active)

def expert_action(env,oracle):
    if env.certificate.success: return "STOP"
    truth=set(oracle["active_set"]); extra=set(env.active)-truth
    if extra: return f"DROP({min(extra,key=lambda i:env.solution.lam[i])})"
    missing=truth-set(env.active)
    if missing:
        gap=env.qp.A@env.solution.x-env.qp.b
        return f"ADD({max(missing,key=lambda i:gap[i])})"
    raise RuntimeError("oracle set failed certificate")

def trajectory(qp,oracle,mode,rng,difficulty=5):
    env=ActiveSetEnv(qp,initial_set(qp,oracle,mode,rng,difficulty))
    conversation=messages(env); rows=[]
    while not env.done:
        prompt=[dict(conversation[0]),dict(conversation[-1])]
        action=expert_action(env,oracle); state=env.step(action)
        result=observation(env)
        rows.append({"prompt":prompt,"completion":[{"role":"assistant","content":action}],
                     "problem_id":qp.problem_id,"action":action,"tool_result":result,
                     "next_observation":result,"mode":mode})
        conversation += [{"role":"assistant","content":action},{"role":"user","content":"apply_action result:\n"+result}]
    return env,rows
