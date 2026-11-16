import math
def reward(env,version="B",beta=.5,bound=1.,alpha=.01,gamma=.05):
    if version not in ("A","B","C"): raise ValueError("reward must be A, B, or C")
    result=float(env.success)
    if version=="A": return result
    v0,vt=env.initial_merit,env.certificate.merit
    improvement=math.log1p(v0)-math.log1p(vt) if math.isfinite(vt) else -bound
    result+=beta*max(-bound,min(bound,improvement))
    if version=="C": result-=alpha*env.steps+gamma*env.invalid
    return result
