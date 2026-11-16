"""Inspect the solver/verifier loop without a model download."""
import numpy as np
from kkt_agent.qp_generator import QP
from kkt_agent.active_set_env import ActiveSetEnv
from kkt_agent.baselines import greedy

qp = QP(Q=np.eye(2), c=np.array([-2., -3.]),
        A=np.array([[1., 0.], [0., 1.], [-1., 0.], [0., -1.]]),
        b=np.array([1., 1., 0., 0.]), problem_id="two-dimensional-box")
env = ActiveSetEnv(qp)
print("Minimize 0.5*(x1^2+x2^2)-2*x1-3*x2 subject to 0 <= x1,x2 <= 1")
while not env.done:
    action = greedy(env)
    env.step(action)
    print(f"{action:8s} active={env.active} x={env.solution.x.tolist()} verified={env.certificate.success}")
print("success =", env.success)
print("certificate =", env.certificate.to_dict())
assert env.success
np.testing.assert_allclose(env.solution.x, [1., 1.], atol=1e-12)
