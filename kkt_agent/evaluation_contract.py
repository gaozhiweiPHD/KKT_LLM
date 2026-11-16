"""Prevent mislabeled base-model versus adapted-model comparisons."""
def validate_model_label(methods,adapter):
    llm=[method for method in methods if method.startswith("qwen")]
    if not llm: return
    if len(llm)!=1:
        raise ValueError("Evaluate one Qwen checkpoint per invocation.")
    if llm[0]=="qwen_zero" and adapter:
        raise ValueError("qwen_zero must not load a LoRA adapter.")
    if llm[0]!="qwen_zero" and not adapter:
        raise ValueError("SFT/GRPO evaluation requires --adapter; a base model cannot use this label.")
