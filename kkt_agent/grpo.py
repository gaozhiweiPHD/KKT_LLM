"""Auditable multi-turn GRPO for a reactive text-action policy.

This backend uses actual on-policy Qwen token sampling and differentiable token
log probabilities. Only generated action tokens enter the policy objective.
Numerical observations are external state, never policy targets. No classifier
or deterministic policy is substituted for the causal LM.
"""
from dataclasses import dataclass
import time
import numpy as np
import torch
from .active_set_env import ActiveSetEnv
from .llm import prompt_text
from .rewards import reward

@dataclass
class Turn:
    prompt_ids: list
    completion_ids: list
    old_logprobs: torch.Tensor
    sampling_logprobs: list

def token_logprobs(model,prompt_ids,completion_ids,temperature):
    device=model.device
    ids=torch.tensor([prompt_ids+completion_ids],device=device)
    length=len(completion_ids)
    logits=model(input_ids=ids,attention_mask=torch.ones_like(ids),use_cache=False,
                 logits_to_keep=length+1).logits[:,:-1,:].float()/temperature
    targets=ids[:,-length:]
    return logits.log_softmax(-1).gather(-1,targets.unsqueeze(-1)).squeeze(0).squeeze(-1)

def group_rollout(model,tok,qp,initial_active,generations=4,max_actions=20,temperature=.7,max_new_tokens=12,reward_version="B"):
    model.eval()
    envs=[ActiveSetEnv(qp,initial_active,max_steps=max_actions) for _ in range(generations)]
    trajectories=[[] for _ in envs]
    for _ in range(max_actions):
        live=[i for i,e in enumerate(envs) if not e.done]
        if not live: break
        texts=[prompt_text(tok,envs[i]) for i in live]
        inputs=tok(texts,return_tensors="pt",padding=True,add_special_tokens=False).to(model.device)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.no_grad():
            sampled=model.generate(**inputs,return_dict_in_generate=True,output_scores=True,do_sample=True,temperature=temperature,top_p=1.,top_k=0,
                                     max_new_tokens=max_new_tokens,pad_token_id=tok.pad_token_id,use_cache=True)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        generated=sampled.sequences
        for j,i in enumerate(live):
            prompt_ids=inputs.input_ids[j][inputs.attention_mask[j].bool()].tolist()
            completion=generated[j,inputs.input_ids.shape[1]:].tolist()
            eos=model.generation_config.eos_token_id
            eos=set(eos if isinstance(eos,list) else [eos])
            eos.add(tok.eos_token_id)
            endings=[k for k,token in enumerate(completion) if token in eos]
            if endings: completion=completion[:endings[0]+1]
            # Recompute unpadded behavior probabilities before any update.
            # This avoids batch-padding roundoff affecting the initial ratio.
            with torch.no_grad():
                old=token_logprobs(model,prompt_ids,completion,temperature).detach().cpu()
            if not torch.isfinite(old).all(): raise FloatingPointError("nonfinite rollout probabilities")
            sampling=[float(sampled.scores[k][j].float().log_softmax(-1)[token]) for k,token in enumerate(completion)]
            trajectories[i].append(Turn(prompt_ids,completion,old,sampling))
            e=envs[i]; e.inference_time+=elapsed/len(live)
            e.input_tokens+=len(prompt_ids); e.output_tokens+=len(completion)
            text=tok.decode(completion,skip_special_tokens=True).strip()
            e.step(text)
    rewards=np.array([reward(e,reward_version) for e in envs],dtype=np.float64)
    return envs,trajectories,rewards

def advantages(rewards,epsilon=1e-4):
    values=np.asarray(rewards,dtype=float)
    return (values-values.mean())/(values.std(ddof=0)+epsilon)

def token_logprobs_batch(model,turns,temperature):
    """Left-pad full contexts, then compute only the final completion logits."""
    width=max(len(t.prompt_ids)+len(t.completion_ids) for t in turns)
    tail=max(len(t.completion_ids) for t in turns)
    ids=torch.zeros((len(turns),width),device=model.device,dtype=torch.long)
    attention=torch.zeros_like(ids)
    for j,t in enumerate(turns):
        values=t.prompt_ids+t.completion_ids
        ids[j,-len(values):]=torch.tensor(values,device=model.device)
        attention[j,-len(values):]=1
    positions=(attention.cumsum(-1)-1).clamp(min=0)
    logits=model(input_ids=ids,attention_mask=attention,position_ids=positions,use_cache=False,
                 logits_to_keep=tail+1).logits[:,:-1,:].float()/temperature
    all_logp=logits.log_softmax(-1).gather(-1,ids[:,-tail:].unsqueeze(-1)).squeeze(-1)
    return [all_logp[j,-len(t.completion_ids):] for j,t in enumerate(turns)]

def grpo_update(model,optimizer,scaler,trajectories,rewards,temperature=.7,clip_epsilon=.2,microbatch_size=4):
    """Per-trajectory token-mean clipped GRPO, beta_KL=0, one on-policy update."""
    model.eval()
    adv=advantages(rewards); optimizer.zero_grad(set_to_none=True)
    records=[]
    for trajectory,advantage in zip(trajectories,adv):
        total=sum(len(t.completion_ids) for t in trajectory)
        records.extend((t,float(advantage),1./(len(trajectories)*max(1,total))) for t in trajectory)
    loss_value=0.; max_ratio_error=0.; roundoff=0.; sampling_roundoff=0.; tokens=0
    for start in range(0,len(records),microbatch_size):
        chunk=records[start:start+microbatch_size]
        logps=token_logprobs_batch(model,[r[0] for r in chunk],temperature)
        loss=0.
        for logp,(turn,advantage,weight) in zip(logps,chunk):
            old=turn.old_logprobs.to(logp.device)
            # One on-policy update (mu=1): all behavior weights are still unchanged.
            # The official GRPO mu=1 form uses a stop-gradient denominator.
            # Replay in the same forward avoids batch-shape FP16 roundoff in r.
            roundoff=max(roundoff,float((logp.detach()-old).abs().max()))
            sampling_roundoff=max(sampling_roundoff,float((logp.detach()-torch.tensor(turn.sampling_logprobs,device=logp.device)).abs().max()))
            ratio=torch.exp(logp-logp.detach())
            max_ratio_error=max(max_ratio_error,float((ratio.detach()-1).abs().max()))
            objective=torch.minimum(ratio*advantage,ratio.clamp(1-clip_epsilon,1+clip_epsilon)*advantage)
            loss=loss-objective.sum()*weight
            tokens+=len(turn.completion_ids)
        if not torch.isfinite(loss): raise FloatingPointError("nonfinite GRPO loss")
        scaler.scale(loss).backward()
        loss_value+=float(loss.detach())
    # Numerical sampling/replay discrepancy is logged, not hidden by the ratio.
    if not np.isfinite(sampling_roundoff): raise FloatingPointError("sampling/replay logprob mismatch")
    scaler.unscale_(optimizer)
    grad=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
    if not torch.isfinite(grad): raise FloatingPointError("nonfinite gradient")
    scaler.step(optimizer); scaler.update()
    return {"loss":loss_value,"grad_norm":float(grad),"initial_ratio_max_error":max_ratio_error,
            "optimized_action_tokens":tokens,"advantages":adv.tolist(),"update_microbatch_size":microbatch_size,
            "replay_logprob_roundoff_max":roundoff,"sampling_logprob_roundoff_max":sampling_roundoff}
