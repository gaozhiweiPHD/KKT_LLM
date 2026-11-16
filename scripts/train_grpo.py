"""Real LoRA Qwen multi-turn GRPO with auditable on-policy groups and checkpoint resume."""
import argparse,json,time,random
from pathlib import Path
import numpy as np
import torch,yaml
from kkt_agent.llm import load_model,set_seed,trainable_snapshot,parameter_audit
from kkt_agent.grpo import group_rollout,grpo_update
from kkt_agent.io import load_qps,read_jsonl,dump_json

def save_checkpoint(model,tok,optimizer,scaler,rng,step,path,cfg):
    path=Path(path); path.mkdir(parents=True,exist_ok=True)
    model.save_pretrained(path); tok.save_pretrained(path)
    torch.save({"step":step,"optimizer":optimizer.state_dict(),"scaler":scaler.state_dict(),
                "numpy_rng":rng.bit_generator.state,"torch_rng":torch.get_rng_state(),
                "cuda_rng":torch.cuda.get_rng_state_all(),"python_rng":random.getstate(),"config":cfg},path/"training_state.pt")

def main():
    p=argparse.ArgumentParser(); p.add_argument("--config",default="configs/grpo.yaml")
    p.add_argument("--sft",default="checkpoints/sft/final"); p.add_argument("--steps",type=int)
    p.add_argument("--reward",choices=["A","B","C"],default="B"); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--output"); p.add_argument("--resume"); p.add_argument("--save-every",type=int,default=50)
    p.add_argument("--curriculum-level",type=int,choices=[0,1,2]); p.add_argument("--eval-every",type=int,default=50)
    p.add_argument("--eval-limit",type=int,default=8)
    a=p.parse_args(); cfg=yaml.safe_load(open(a.config))
    cfg["reward"]=a.reward; cfg["seed"]=a.seed
    if a.steps: cfg["max_steps"]=a.steps
    if cfg["num_generations"]<2: raise ValueError("GRPO needs multiple completions")
    out=Path(a.output or f"checkpoints/grpo_{a.reward}"); out.mkdir(parents=True,exist_ok=True)
    set_seed(a.seed); rng=np.random.default_rng(a.seed)
    model,tok=load_model(cfg["model"],a.resume or a.sft,trainable=True)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
    model.enable_input_require_grads()
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=cfg["learning_rate"],weight_decay=0.)
    scaler=torch.amp.GradScaler("cuda",init_scale=128.)
    first=0
    if a.resume:
        state=torch.load(Path(a.resume)/"training_state.pt",map_location="cpu",weights_only=False)
        if state["config"]["reward"]!=cfg["reward"]: raise ValueError("cannot change reward on resume")
        optimizer.load_state_dict(state["optimizer"]); scaler.load_state_dict(state["scaler"])
        rng.bit_generator.state=state["numpy_rng"]; torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"]); random.setstate(state["python_rng"]); first=state["step"]
    before=trainable_snapshot(model)
    assert all("lora_" in n for n in before)
    corpus={q.problem_id:q for q in load_qps("data/train.jsonl")}
    tasks=list(read_jsonl("data/curriculum.jsonl"))
    levels={i:[t for t in tasks if t["level"]==i] for i in range(3)}
    validation=load_qps("data/validation.jsonl",a.eval_limit)
    start=time.perf_counter(); varied=0; losses=[]
    with (out/"training.jsonl").open("a" if a.resume else "w") as log,(out/"rollouts.jsonl").open("a" if a.resume else "w") as traces:
        for step in range(first,cfg["max_steps"]):
            # Global step determines curriculum. Fixed level is for short smoke tests.
            level=a.curriculum_level if a.curriculum_level is not None else (0 if step<40 else 1 if step<100 else 2)
            horizon=[4,12,20][level]; task=levels[level][int(rng.integers(len(levels[level])))]
            q=corpus[task["problem_id"]]
            model.gradient_checkpointing_disable()
            envs,rollouts,rewards=group_rollout(model,tok,q,task["initial_active"],cfg["num_generations"],horizon,
                                               cfg["temperature"],cfg["max_new_tokens"],a.reward)
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
            update=grpo_update(model,optimizer,scaler,rollouts,rewards,cfg["temperature"],cfg["clip_epsilon"])
            varied+=int(rewards.std()>1e-8); losses.append(update["loss"])
            row={"step":step+1,"problem_id":q.problem_id,"initial_active":task["initial_active"],"level":level,"horizon":horizon,
                 "reward_version":a.reward,"rewards":rewards.tolist(),"reward_mean":float(rewards.mean()),
                 "reward_std":float(rewards.std()),"group_success_rate":float(np.mean([e.success for e in envs])),
                 "mean_actions":float(np.mean([e.steps for e in envs])),"invalid_actions":sum(e.invalid for e in envs),
                 "seconds":time.perf_counter()-start,**update}
            log.write(json.dumps(row)+"\n"); log.flush(); print(json.dumps(row),flush=True)
            for g,e in enumerate(envs):
                traces.write(json.dumps({"step":step+1,"group_member":g,"record":e.record(),"history":e.history,
                    "token_turns":[{"prompt_ids":t.prompt_ids,"completion_ids":t.completion_ids,
                      "old_replay_logprobs":t.old_logprobs.tolist(),"sampling_logprobs":t.sampling_logprobs} for t in rollouts[g]]})+"\n")
            traces.flush()
            if a.eval_every and (step+1)%a.eval_every==0:
                # Independent cold starts, no curriculum or oracle. Preserve train RNG.
                rng_t=torch.get_rng_state(); rng_c=torch.cuda.get_rng_state_all()
                torch.manual_seed(987); torch.cuda.manual_seed_all(987)
                model.gradient_checkpointing_disable()
                eval_rows=[]
                for vq in validation:
                    ve,_,_=group_rollout(model,tok,vq,[],1,20,cfg["temperature"],cfg["max_new_tokens"],a.reward)
                    eval_rows.append(ve[0].record())
                torch.set_rng_state(rng_t); torch.cuda.set_rng_state_all(rng_c)
                with (out/"validation.jsonl").open("a") as vf:
                    vf.write(json.dumps({"step":step+1,"n":len(eval_rows),"success_rate":float(np.mean([r["success"] for r in eval_rows])),"records":eval_rows})+"\n")
            if (step+1)%a.save_every==0:
                save_checkpoint(model,tok,optimizer,scaler,rng,step+1,out/f"checkpoint-{step+1}",cfg)
    save_checkpoint(model,tok,optimizer,scaler,rng,cfg["max_steps"],out/"final",cfg)
    audit=parameter_audit(model,before)
    audit.update(steps=cfg["max_steps"],resumed_from_step=first,seconds=time.perf_counter()-start,
                 groups_with_variance=varied,finite_losses=bool(np.isfinite(losses).all()),config=cfg,
                 engine="custom on-policy multi-turn token GRPO",checkpoint_resume=bool(a.resume),peak_gpu_memory_bytes=torch.cuda.max_memory_allocated())
    dump_json(out/("resume_audit.json" if a.resume else "parameter_audit.json"),audit)
    if varied and audit["changed_parameters"]==0: raise AssertionError("nonzero advantages but no LoRA update")
    print(json.dumps(audit,indent=2))
if __name__=="__main__": main()
