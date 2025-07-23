import os
os.environ["HF_DATASETS_OFFLINE"] = "1"
import torch
from transformers import AutoTokenizer

from lm_eval.loggers.evaluation_tracker import EvaluationTracker
from lm_eval.models.huggingface import HFLM
from lm_eval.evaluator import simple_evaluate
from lm_eval.utils import make_table, setup_logging

from model import SkipLayersForCausalLM

BATCH_SIZE = 4
NUM_FEWSHOT = 5
LIMIT = 5
def main():
    model = SkipLayersForCausalLM.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-8B", trust_remote_code=True)
    tokenizer.padding_side = "right"

    model.set_skip_layers([0])

    setup_logging()

    lm = HFLM(
        pretrained=model, 
        tokenizer=tokenizer,
        use_prefix_caching=True,
        batch_size=BATCH_SIZE,
        device=model.device,
        dtype=model.dtype,
    )
    
    evaluation_tracker = EvaluationTracker(
        output_path="temp/outputs/qwen3-8b-skip-layers-mmlu.json",
    )

    results = simple_evaluate(
        model=lm,
        tasks=["mmlu"],
        num_fewshot=NUM_FEWSHOT,
        batch_size=BATCH_SIZE,
        limit=LIMIT,
        evaluation_tracker=evaluation_tracker,
    )

    if results is not None:
        evaluation_tracker.save_results_aggregated(
            results=results, samples=None
        )
        print(
            f"model: Qwen3-8B, limit: {LIMIT}, num_fewshot: {NUM_FEWSHOT}, "
            f"batch_size: {BATCH_SIZE}"
        )
        print(make_table(results))
        if "groups" in results:
            print(make_table(results, "groups"))

if __name__ == "__main__":
    main()