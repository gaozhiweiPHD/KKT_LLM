import numpy as np
import pytest
from kkt_agent.qp_generator import generate_qp
from kkt_agent.active_set_env import ActiveSetEnv,parse_action
from kkt_agent.expert_policy import trajectory

def test_stop_and_budget():
    q,o,_=generate_qp(5)
    e=ActiveSetEnv(q,o["active_set"]); assert not e.success
    e.step("STOP"); assert e.success
    with pytest.raises(RuntimeError): e.step("STOP")
    e=ActiveSetEnv(q); e.step("STOP"); assert not e.success and e.reason=="incorrect_stop"
    e=ActiveSetEnv(q,max_steps=1); e.step("ADD(0)"); assert not e.success and e.done

@pytest.mark.parametrize("action",["ADD(-1)","ADD(999)","DROP(0)","ADD(0) DROP(1)","STOP now","ADD(true)","__import__('os')"])
def test_invalid(action):
    q,_,_=generate_qp(4); e=ActiveSetEnv(q); before=e.solution.x.copy()
    e.step(action); assert e.invalid==1 and e.active==[]
    np.testing.assert_allclose(e.solution.x,before)

def test_transaction_and_cycle():
    q,_,_=generate_qp(4,n=2,m=6,active_size=1); q.A[1]=q.A[0]
    e=ActiveSetEnv(q); e.step("ADD(0)"); e.step("ADD(0)"); e.step("ADD(1)")
    assert e.active==[0] and e.invalid==2
    for _ in range(3):
        if e.done: break
        e.step("DROP(0)")
        if not e.done: e.step("ADD(0)")
    assert e.cycle

@pytest.mark.parametrize("mode",["cold","missing","extra","mixed","warm"])
def test_teacher(mode):
    q,o,_=generate_qp(44); e,rows=trajectory(q,o,mode,np.random.default_rng(1))
    assert e.success and e.invalid==0 and rows[-1]["action"]=="STOP"
    assert "x_star" not in str(rows) and "lambda_star" not in str(rows)
