import numpy as np
import pytest
from kkt_agent.qp_generator import generate_qp
from kkt_agent.qp_solver import solve_active,solve_osqp,NumericalError
from kkt_agent.kkt_verifier import verify

@pytest.mark.parametrize("n,m,k",[(2,6,1),(5,15,4),(12,40,10),(16,50,8)])
@pytest.mark.parametrize("condition",[10.,1000.,10000.])
def test_certificates(n,m,k,condition):
    q,o,md=generate_qp(100+n,n,m,condition,k)
    assert verify(q,o["x_star"],o["lambda_star"]).success
    assert np.linalg.matrix_rank(q.A[o["active_set"]])==k
    assert np.linalg.cond(q.Q)==pytest.approx(condition,rel=1e-8)
    for s in [solve_active(q,o["active_set"]),solve_osqp(q)]:
        assert verify(q,s.x,s.lam).success
        assert np.max(np.abs(s.x-o["x_star"]))<2e-5

@pytest.mark.parametrize("family",["standard","small_slack","nearly_active","correlated"])
def test_hard_families(family):
    q,o,_=generate_qp(33,family=family)
    assert verify(q,o["x_star"],o["lambda_star"]).success
    assert np.min(q.b-q.A@o["x_star"])>=-1e-12

def test_invalid_solver():
    q,_,_=generate_qp(3,n=2,m=5,active_size=1)
    for active in [[0,0],[0,1,2],[6]]:
        with pytest.raises(NumericalError): solve_active(q,active)
    q.A[1]=q.A[0]
    with pytest.raises(NumericalError): solve_active(q,[0,1])
    q.Q[0,0]=np.nan
    with pytest.raises(NumericalError): solve_active(q,[])
