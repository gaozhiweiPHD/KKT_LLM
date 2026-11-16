import pytest
from kkt_agent.qp_generator import generate_qp
from kkt_agent.active_set_env import ActiveSetEnv
from kkt_agent.rewards import reward
from kkt_agent.expert_policy import expert_action

def test_no_free_certificate_reward():
    q,o,_=generate_qp(19); e=ActiveSetEnv(q,o["active_set"])
    assert reward(e,"A")==0
    e.step("STOP"); assert reward(e,"A")==1 and reward(e,"B")==1 and reward(e,"C")<1

def test_failure_and_invalid_cost():
    q,_,_=generate_qp(19); e=ActiveSetEnv(q); e.step("DROP(999)")
    assert reward(e,"A")==0 and reward(e,"B")==0 and reward(e,"C")==pytest.approx(-.06)
    e.step("STOP"); assert reward(e,"A")==0

def test_progress_requires_stop():
    q,o,_=generate_qp(19); e=ActiveSetEnv(q)
    while not e.certificate.success: e.step(expert_action(e,o))
    assert reward(e,"A")==0 and 0<=reward(e,"B")<=.5
    e.step("STOP"); assert reward(e,"B")>=1
