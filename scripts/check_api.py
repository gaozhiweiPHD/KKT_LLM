import inspect,json,importlib.metadata as md
from trl import SFTTrainer,GRPOTrainer
from kkt_agent.io import dump_json
info={"versions":{p:md.version(p) for p in ["torch","transformers","trl","peft","datasets","accelerate"]},
      "sft_signature":str(inspect.signature(SFTTrainer.__init__)),
      "grpo_signature":str(inspect.signature(GRPOTrainer.__init__))}
info["environment_factory"]="environment_factory" in inspect.signature(GRPOTrainer.__init__).parameters
dump_json("results/api_compatibility.json",info); print(json.dumps(info,indent=2))
