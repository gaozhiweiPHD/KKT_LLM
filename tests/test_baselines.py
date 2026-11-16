import pytest
from kkt_agent.qp_generator import generate_qp
from kkt_agent.baselines import classical_active_set
@pytest.mark.parametrize("family",["standard","small_slack","nearly_active","correlated"])
@pytest.mark.parametrize("n",[2,5,12])
def test_classical_independent(family,n):
    q,_,_=generate_qp(200+n,n=n,m=max(10,3*n),active_size=min(n,5),family=family,condition=1000)
    record,_=classical_active_set(q,max_steps=200)
    assert record["success"]
    assert max(record[k] for k in ["stationarity","primal","dual","complementarity"])<=1e-5
