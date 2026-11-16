import json
from kkt_agent.qp_generator import generate_qp
from kkt_agent.qp_solver import solve_active,solve_osqp
from kkt_agent.kkt_verifier import verify
from kkt_agent.active_set_env import ActiveSetEnv
from kkt_agent.baselines import greedy
from kkt_agent.io import dump_json

rows=[]
for n,m,k in [(2,6,1),(5,15,4),(12,40,8)]:
    q,o,_=generate_qp(123+n,n,m,1000,k)
    for method,s in [("known_active",solve_active(q,o["active_set"])),("osqp",solve_osqp(q))]:
        rows.append({"n":n,"m":m,"method":method,**verify(q,s.x,s.lam).to_dict()})
    e=ActiveSetEnv(q)
    while not e.done: e.step(greedy(e))
    rows.append({"n":n,"m":m,"method":"greedy",**e.record()})
dump_json("results/numerical_smoke.json",rows)
print(json.dumps(rows,indent=2))
