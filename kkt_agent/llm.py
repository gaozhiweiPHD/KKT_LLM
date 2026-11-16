"""Actual Qwen causal LM inference/LoRA utilities; no heuristic action fallback."""
import os
from pathlib import Path
os.environ.setdefault("HF_HOME",str(Path(__file__).resolve().parents[1]/".cache/huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM","false")
_MODEL_CACHE=Path(os.environ.get("HF_HUB_CACHE",str(Path(os.environ["HF_HOME"])/"hub")))
os.environ.setdefault("HF_HUB_CACHE",str(_MODEL_CACHE))
import random,time,hashlib
import numpy as np
import torch
from transformers import AutoTokenizer,AutoModelForCausalLM
from peft import PeftModel,LoraConfig,get_peft_model
from .observation import messages,observation

def set_seed(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)

def load_model(model_name="Qwen/Qwen3-0.6B",adapter=None,trainable=False,rank=8,alpha=16):
    torch.set_num_threads(1)
    revision_file=Path(__file__).resolve().parents[1]/"configs/model_revision.txt"
    revision=None
    if model_name=="Qwen/Qwen3-0.6B":
        revision=revision_file.read_text().strip() if revision_file.exists() else "c1899de289a04d12100db370d81485cdf75e47ca"
    tok=AutoTokenizer.from_pretrained(model_name,revision=revision,cache_dir=str(_MODEL_CACHE))
    tok.padding_side="left"
    if tok.pad_token_id is None: tok.pad_token=tok.eos_token
    dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    model=AutoModelForCausalLM.from_pretrained(model_name,revision=revision,cache_dir=str(_MODEL_CACHE),dtype=dtype,attn_implementation="sdpa")
    if adapter:
        model=PeftModel.from_pretrained(model,adapter,is_trainable=trainable)
    elif trainable:
        model=get_peft_model(model,LoraConfig(r=rank,lora_alpha=alpha,lora_dropout=0.,
                             target_modules=["q_proj","k_proj","v_proj","o_proj"],task_type="CAUSAL_LM",bias="none"))
    if hasattr(model,"peft_config"):
        for config in model.peft_config.values(): config.revision=revision
    model.config.kkt_base_revision=revision
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model,tok

def prompt_text(tok,env,mode="compact"):
    # Reactive policy: the environment persists across turns; only current public
    # numerical state is re-encoded. Full mode includes matrices in the first prompt.
    return tok.apply_chat_template(messages(env,mode),tokenize=False,add_generation_prompt=True,enable_thinking=False)

def trainable_snapshot(model):
    return {n:p.detach().float().cpu().clone() for n,p in model.named_parameters() if p.requires_grad}

def parameter_audit(model,before):
    delta=0.; max_delta=0.; changed=0; h=hashlib.sha256()
    for n,p in model.named_parameters():
        if n in before:
            after=p.detach().float().cpu(); d=after-before[n]
            delta+=float(d.square().sum()); max_delta=max(max_delta,float(d.abs().max()))
            changed+=int(torch.count_nonzero(d)); h.update(after.numpy().tobytes())
    return {"trainable_parameters":sum(v.numel() for v in before.values()),
            "changed_parameters":changed,"delta_l2":delta**.5,"delta_max":max_delta,"sha256":h.hexdigest()}

class QwenPolicy:
    def __init__(self,model_name="Qwen/Qwen3-0.6B",adapter=None,mode="compact",temperature=.7,max_new_tokens=12):
        start=time.perf_counter(); self.model,self.tokenizer=load_model(model_name,adapter)
        self.model.eval(); self.mode,self.temperature,self.max_new_tokens=mode,temperature,max_new_tokens
        self.load_seconds=time.perf_counter()-start
        self.conversation=None

    def seed(self,seed): set_seed(seed)
    def reset(self,env): self.conversation=messages(env,self.mode)
    def act(self,env):
        tok=self.tokenizer
        if self.mode=="compact": text=prompt_text(tok,env)
        else:
            if env.steps: self.conversation.append({"role":"user","content":"apply_action result:\n"+observation(env)})
            text=tok.apply_chat_template(self.conversation,tokenize=False,add_generation_prompt=True,enable_thinking=False)
        inputs=tok(text,return_tensors="pt",add_special_tokens=False).to(self.model.device)
        if inputs.input_ids.shape[1]+self.max_new_tokens>32768: raise ValueError("context budget exceeded")
        if torch.cuda.is_available(): torch.cuda.synchronize()
        start=time.perf_counter()
        with torch.inference_mode():
            out=self.model.generate(**inputs,max_new_tokens=self.max_new_tokens,do_sample=True,temperature=self.temperature,
                                    top_p=1.,top_k=0,pad_token_id=tok.pad_token_id,use_cache=True)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        env.inference_time+=time.perf_counter()-start
        ids=out[0,inputs.input_ids.shape[1]:]
        action=tok.decode(ids,skip_special_tokens=True).strip()
        env.input_tokens+=inputs.input_ids.numel(); env.output_tokens+=ids.numel()
        if self.mode=="full": self.conversation.append({"role":"assistant","content":action})
        return action
