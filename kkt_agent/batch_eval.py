"""One-process batched GPU evaluation; independent environment per problem."""
from time import perf_counter
import torch
from .active_set_env import ActiveSetEnv
from .observation import messages,observation
from .llm import prompt_text

def evaluate_batches(policy,qps,batch_size=8):
    model,tok=policy.model,policy.tokenizer
    if policy.mode=="full" and batch_size!=1:
        raise ValueError("full-information evaluation currently requires batch_size=1")
    for offset in range(0,len(qps),batch_size):
        group_start=perf_counter()
        envs=[ActiveSetEnv(q) for q in qps[offset:offset+batch_size]]
        conversations=[messages(e,policy.mode) for e in envs]
        completed={}
        for _ in range(20):
            live=[i for i,e in enumerate(envs) if not e.done]
            if not live: break
            if policy.mode=="compact": texts=[prompt_text(tok,envs[i]) for i in live]
            else:
                for i in live:
                    if envs[i].steps:
                        conversations[i].append({"role":"user","content":"apply_action result:\n"+observation(envs[i])})
                texts=[tok.apply_chat_template(conversations[i],tokenize=False,add_generation_prompt=True,enable_thinking=False) for i in live]
            inputs=tok(texts,return_tensors="pt",padding=True,add_special_tokens=False).to(model.device)
            if inputs.input_ids.shape[1]+policy.max_new_tokens>32768: raise ValueError("context budget exceeded")
            if torch.cuda.is_available(): torch.cuda.synchronize()
            start=perf_counter()
            with torch.inference_mode():
                generated=model.generate(**inputs,max_new_tokens=policy.max_new_tokens,do_sample=True,
                           temperature=policy.temperature,top_p=1.,top_k=0,pad_token_id=tok.pad_token_id,use_cache=True)
            if torch.cuda.is_available(): torch.cuda.synchronize()
            elapsed=perf_counter()-start
            eos=model.generation_config.eos_token_id
            eos=set(eos if isinstance(eos,list) else [eos]); eos.add(tok.eos_token_id)
            for j,i in enumerate(live):
                ids=generated[j,inputs.input_ids.shape[1]:].tolist()
                endings=[k for k,t in enumerate(ids) if t in eos]
                if endings: ids=ids[:endings[0]+1]
                e=envs[i]; e.inference_time+=elapsed/len(live)
                e.input_tokens+=int(inputs.attention_mask[j].sum()); e.output_tokens+=len(ids)
                action=tok.decode(ids,skip_special_tokens=True).strip()
                if policy.mode=="full": conversations[i].append({"role":"assistant","content":action})
                e.step(action)
                if e.done: completed[i]=e.record()
        batch_wall=perf_counter()-group_start
        for i,e in enumerate(envs):
            rec=completed[i]
            e.history[-1]["candidate"]={"x":e.solution.x.tolist(),"lambda":e.solution.lam.tolist()}
            rec["episode_latency_seconds"]=rec["wall_seconds"]
            rec["amortized_wall_seconds"]=batch_wall/len(envs)
            rec["inference_batch_size"]=len(envs)
            yield rec,e.history
