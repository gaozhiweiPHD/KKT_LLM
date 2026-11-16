import numpy as np
import pytest
from kkt_agent.qp_generator import QP,generate_qp
from kkt_agent.kkt_verifier import verify

def simple(c=0,b=0):
    return QP(np.eye(2),np.array([c,0.]),np.array([[1.,0.]]),np.array([float(b)]))

def test_correct(): assert verify(simple(c=-1),np.zeros(2),np.ones(1)).success
def test_primal():
    c=verify(simple(),np.array([1.,0.]),np.zeros(1))
    assert c.primal>1e-5 and not c.success
def test_dual():
    c=verify(simple(c=1),np.zeros(2),np.array([-1.]))
    assert c.stationarity==0 and c.dual>1e-5 and not c.success
def test_complementarity():
    c=verify(simple(c=-1,b=1),np.zeros(2),np.ones(1))
    assert c.stationarity==0 and c.primal==0 and c.dual==0 and c.complementarity>1e-5 and not c.success
def test_nonfinite(): assert not verify(simple(),[np.nan,0],[0]).success

@pytest.mark.parametrize("objective_scale",[1e-8,1.,1e8])
def test_scaling_invariance(objective_scale):
    q,o,_=generate_qp(13); rng=np.random.default_rng(14)
    x=np.array(o["x_star"])+.1; lam=np.array(o["lambda_star"])-.1
    a=verify(q,x,lam); row_scale=np.geomspace(1e-6,1e6,q.m)
    scaled=QP(q.Q*objective_scale,q.c*objective_scale,q.A*row_scale[:,None],q.b*row_scale)
    b=verify(scaled,x,lam*objective_scale/row_scale)
    for field in ["stationarity","primal","dual","complementarity"]:
        assert getattr(a,field)==pytest.approx(getattr(b,field),rel=1e-9,abs=1e-13)
    assert a.success==b.success
