import numpy as np
import torch
import kkt_agent.grpo as module
from kkt_agent.grpo import Turn

class TwoActionPolicy(torch.nn.Module):
    """Analytic test policy only; never used by an experiment."""
    def __init__(self):
        super().__init__(); self.logits=torch.nn.Parameter(torch.zeros(2))
    @property
    def device(self): return self.logits.device

def run_update(monkeypatch,rewards,winning_length=1):
    model=TwoActionPolicy()
    def logps(model,turns,temperature):
        lp=(model.logits/temperature).log_softmax(-1)
        return [lp[torch.tensor(t.completion_ids)] for t in turns]
    monkeypatch.setattr(module,"token_logprobs_batch",logps)
    trajectory0=[Turn([999],[0]*winning_length,torch.full((winning_length,),-np.log(2)),[-np.log(2)]*winning_length)]
    trajectory1=[Turn([999],[1],torch.tensor([-np.log(2)]),[-np.log(2)])]
    optimizer=torch.optim.SGD(model.parameters(),lr=.1)
    scaler=torch.amp.GradScaler("cuda",enabled=False)
    result=module.grpo_update(model,optimizer,scaler,[trajectory0,trajectory1],np.asarray(rewards),temperature=1.)
    return model.logits.detach(),result

def test_grpo_increases_rewarded_action_probability(monkeypatch):
    logits,result=run_update(monkeypatch,[1,0])
    assert logits[0]>logits[1]
    assert result["grad_norm"]>0 and result["initial_ratio_max_error"]==0

def test_equal_rewards_produce_no_policy_gradient(monkeypatch):
    logits,result=run_update(monkeypatch,[1,1])
    torch.testing.assert_close(logits,torch.zeros(2))
    assert result["grad_norm"]==0

def test_original_grpo_length_normalization(monkeypatch):
    one,_=run_update(monkeypatch,[1,0],1)
    three,_=run_update(monkeypatch,[1,0],3)
    torch.testing.assert_close(one,three)
