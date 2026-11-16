import pytest
from kkt_agent.evaluation_contract import validate_model_label

@pytest.mark.parametrize("method,adapter",[("qwen_zero",None),("qwen_sft","adapter"),("qwen_grpo_A","adapter")])
def test_correct_model_labels(method,adapter):
    validate_model_label([method],adapter)

@pytest.mark.parametrize("methods,adapter",[
    (["qwen_sft"],None),(["qwen_grpo_B"],None),(["qwen_zero"],"adapter"),
    (["qwen_zero","qwen_sft"],"adapter")])
def test_reject_mislabeled_comparisons(methods,adapter):
    with pytest.raises(ValueError):
        validate_model_label(methods,adapter)
