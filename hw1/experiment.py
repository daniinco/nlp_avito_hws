import argparse
import json
import math
import os
import shutil
import time
from pathlib import Path

import torch
from transformers import Qwen3ForCausalLM, AutoTokenizer, set_seed

import your_solution as solution


def save_json(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def train(config_path):
    config = json.loads(Path(config_path).read_text())
    scheduler_steps = config.pop("scheduler_steps", None)
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=False)
    solution.TRAINING_CONFIG.update(config)
    if solution.TRAINING_CONFIG['torch_compile']:
        os.environ.setdefault('TRITON_PTXAS_PATH', '/usr/local/cuda/bin/ptxas')
        os.environ['TORCHINDUCTOR_CACHE_DIR'] = str(output.resolve() / 'compile_cache' / 'inductor')
        os.environ['TRITON_CACHE_DIR'] = str(output.resolve() / 'compile_cache' / 'triton')

    class Trainer(solution.Trainer):
        def create_scheduler(self, num_training_steps, optimizer=None):
            return super().create_scheduler(scheduler_steps or num_training_steps, optimizer)

    class Callback(solution.TimeoutCallback):
        def on_step_end(self, args, state, control, **kwargs):
            self.training_seconds = time.time() - self.start_time
            return super().on_step_end(args, state, control, **kwargs)

        def on_log(self, args, state, control, logs=None, **kwargs):
            if self.start_time is not None and state.log_history:
                state.log_history[-1]["elapsed_seconds"] = round(time.time() - self.start_time, 3)

    solution.Trainer = Trainer
    solution.TimeoutCallback = Callback
    set_seed(solution.TRAINING_CONFIG.get("seed", 42))
    torch.cuda.reset_peak_memory_stats()
    save_json(output / "run_config.json", {
        "training": solution.TRAINING_CONFIG, "scheduler_steps": scheduler_steps,
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "duration_seconds": solution.MAX_TRAINING_TIME_SECONDS,
    })
    trainer = solution.train_model()
    metrics = dict(trainer.state.log_history[-1])
    timer = next(callback for callback in trainer.callback_handler.callbacks if isinstance(callback, Callback))
    batch = solution.TRAINING_CONFIG['per_device_train_batch_size']
    accumulation = solution.TRAINING_CONFIG['gradient_accumulation_steps']
    tokens = trainer.state.global_step * batch * accumulation * solution.MAX_LENGTH
    metrics.update({
        "perplexity": math.exp(metrics["eval_loss"]),
        "steps": trainer.state.global_step,
        "best_checkpoint": trainer.state.best_model_checkpoint,
        "peak_allocated_gb": torch.cuda.max_memory_allocated() / 1024 ** 3,
        "training_seconds": timer.training_seconds,
        "padded_tokens": tokens,
        "padded_tokens_per_second": tokens / timer.training_seconds,
        "eval_samples": len(trainer.eval_dataset),
    })
    save_json(output / "metrics.json", metrics)
    for checkpoint in output.glob("checkpoint-*"):
        shutil.rmtree(checkpoint)


def generate(model_path, prompt, output):
    set_seed(42)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = Qwen3ForCausalLM.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).to("cuda").eval()
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    results = []
    for sample in (False, True):
        kwargs = {"temperature": 0.8, "top_p": 0.9, "top_k": 50} if sample else {}
        with torch.inference_mode():
            ids = model.generate(
                **inputs, do_sample=sample, max_new_tokens=128,
                pad_token_id=tokenizer.pad_token_id, **kwargs,
            )
        results.append({
            "method": "sampling" if sample else "greedy", "settings": kwargs,
            "text": tokenizer.decode(ids[0], skip_special_tokens=True),
        })
    save_json(output, {"prompt": prompt, "seed": 42, "max_new_tokens": 128, "generations": results})
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("prepare")
    training = commands.add_parser("train")
    training.add_argument("config")
    generation = commands.add_parser("generate")
    generation.add_argument("--model", required=True)
    generation.add_argument("--prompt", required=True)
    generation.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        solution.prepare_dataset()
    elif args.command == "train":
        train(args.config)
    else:
        generate(args.model, args.prompt, args.output)
