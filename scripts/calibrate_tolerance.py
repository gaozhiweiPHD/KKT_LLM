"""Independent numerical calibration and row/objective rescaling audit."""
import numpy as np
from kkt_agent.qp_generator import generate_qp,QP
from kkt_agent.qp_solver import solve_osqp
from kkt_agent.kkt_verifier import verify
from kkt_agent.io import dump_json
rows=[]
for seed in range(40):
    q,o,_=generate_qp(90000000+seed,condition=[10,100,1000,10000][seed%4],family=["standard","small_slack","nearly_active","correlated"][seed%4])
    s=solve_osqp(q); base=verify(q,s.x,s.lam)
    scales=np.geomspace(1e-6,1e6,q.m); scaled=QP(q.Q*1e4,q.c*1e4,q.A*scales[:,None],q.b*scales)
    rescaled=verify(scaled,s.x,s.lam*1e4/scales)
    for delta in [0.,1e-6,1e-4,.01]:
        c=verify(q,s.x+delta*np.ones(q.n),s.lam)
        rows.append({"seed":seed,"perturbation":delta,**c.to_dict(),"osqp_merit":base.merit,"rescaled_osqp_merit":rescaled.merit})
dump_json("results/tolerance_calibration.json",{"tolerance":1e-5,"rows":rows,
    "max_osqp_merit":max(r["osqp_merit"] for r in rows),
    "note":"Normalized approximate certificate, not exact arithmetic or an absolute x-error bound."})
print("Calibration max OSQP merit",max(r["osqp_merit"] for r in rows))
