import numpy as np
from kkt_agent.grpo import advantages
def test_group_relative_advantages():
    a=advantages([0,0,1,1])
    assert abs(a.mean())<1e-12 and a[0]<0 and a[2]>0
    np.testing.assert_allclose(advantages([1,1,1,1]),0)
def test_shift_invariance():
    np.testing.assert_allclose(advantages([.2,.3,.5,.8]),advantages([10.2,10.3,10.5,10.8]),atol=1e-12)
