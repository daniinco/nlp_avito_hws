import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    Qwen3Config,
    Qwen3ForCausalLM,
    Trainer,
    TrainingArguments,
    TrainerCallback
)
import torch
import time


# Don't change this parameter
MAX_TRAINING_TIME_SECONDS = 60 * 15
MAX_LENGTH = 512
INPUT_IDS = 'input_ids'
ATTENTION_MASK = 'attention_mask'
LABELS = 'labels'

# Don't change these parameters
TOKENIZER_NAME = "ai-forever/rugpt3small_based_on_gpt2"
OUTPUT_DIR = "./output_dir"
NUM_SHARDS = 32
VALIDATION_SIZE = 5000


# TODO: Configure training parameters
TRAINING_CONFIG = {
    'output_dir': f'{OUTPUT_DIR}/gpt2-1b-russian',
    'optim': 'adamw_torch_fused',
    'num_train_epochs': 1,
    'per_device_train_batch_size': 4,
    'save_steps': 100,
    'save_total_limit': 2,
    'learning_rate': 5e-5,
    'weight_decay': 0.01,
    'warmup_steps': 200,
    'logging_steps': 1,
    'eval_steps': 100,
    'eval_strategy': 'steps',
    'load_best_model_at_end': True,
    'metric_for_best_model': 'eval_loss',
    'bf16': True,
    'tf32': True,
    'gradient_checkpointing': False,
    'gradient_accumulation_steps': 1,
    'dataloader_num_workers': 4,
    'torch_compile': True,
    'report_to': 'none',
}


class TimeoutCallback(TrainerCallback):
    """Callback to stop training after a specified timeout."""
    def __init__(self, timeout_seconds):
        self.timeout_seconds = timeout_seconds
        self.start_time = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        if self.start_time is not None:
            elapsed = time.time() - self.start_time
            if elapsed > self.timeout_seconds:
                control.should_training_stop = True
                # Include the final weights in best-checkpoint selection.
                control.should_evaluate = True
                control.should_save = True
                print(f"Training stopped after {elapsed:.2f} seconds")
        return control


def prepare_tokenizer():
    """
    TODO: Implement tokenizer preparation.
    - Load the tokenizer from TOKENIZER_NAME
    - Set pad_token to eos_token
    - Return the tokenizer
    """
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def tokenize_function(examples, tokenizer):
    """
    TODO: Implement tokenization function.
    - Tokenize the text with truncation and padding to MAX_LENGTH
    - Create labels from input_ids
    - Return dictionary with 'labels', 'input_ids', and 'attention_mask'
    """
    result = tokenizer(examples['text'], truncation=True, padding='max_length', max_length=MAX_LENGTH)
    result[LABELS] = [
        [token if mask else -100 for token, mask in zip(ids, masks)]
        for ids, masks in zip(result[INPUT_IDS], result[ATTENTION_MASK])
    ]
    return result


def save_as_parquets(ds, output_dir=OUTPUT_DIR, num_shards=NUM_SHARDS):
    """
    TODO: Implement saving dataset as parquet shards.
    - Create output directory if it doesn't exist
    - Split dataset into num_shards shards
    - Save each shard as a parquet file with format: {output_dir}/{index:05d}.parquet
    """
    os.makedirs(output_dir, exist_ok=True)
    for index in range(num_shards):
        shard = ds.shard(num_shards=num_shards, index=index, contiguous=True)
        shard.to_parquet(os.path.join(output_dir, f'{index:05d}.parquet'))


def prepare_dataset():
    """
    TODO: Implement dataset preparation.
    - Load the Wikipedia dataset: "wikimedia/wikipedia", "20231101.ru", split="train"
    - Tokenize the dataset using tokenize_function
    - Save as parquet files
    """
    dataset = load_dataset("wikimedia/wikipedia", "20231101.ru", split="train")
    tokenizer = prepare_tokenizer()
    dataset = dataset.map(
        tokenize_function, fn_kwargs={'tokenizer': tokenizer},
        batched=True, batch_size=256, num_proc=4,
        remove_columns=dataset.column_names,
    )
    save_as_parquets(dataset)
    tokenizer.save_pretrained(os.path.join(OUTPUT_DIR, 'tokenizer'))
    with open(os.path.join(OUTPUT_DIR, 'ready'), 'w') as file:
        file.write(str(len(dataset)))



def load_tokenized_dataset(data_dir=OUTPUT_DIR):
    """
    TODO: Implement loading of tokenized dataset from parquet files.
    - List only parquet files in data_dir, sorted by filename
    - Load them using load_dataset('parquet', data_files=...)
    - Return the 'train' split
    """
    files = sorted(
        os.path.join(data_dir, name)
        for name in os.listdir(data_dir) if name.endswith('.parquet')
    )
    if not files or not os.path.isfile(os.path.join(data_dir, 'ready')):
        raise FileNotFoundError('Run prepare_dataset() first')
    dataset = load_dataset('parquet', data_files=files, split='train')
    with open(os.path.join(data_dir, 'ready')) as file:
        if len(dataset) != int(file.read()):
            raise ValueError('Incomplete tokenized dataset')
    return dataset


def split_dataset(dataset, validation_size=VALIDATION_SIZE):
    dataset_size = len(dataset)
    train_dataset = dataset.select(range(validation_size, dataset_size))
    eval_dataset = dataset.select(range(validation_size))
    
    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(eval_dataset)}")
    
    return train_dataset, eval_dataset


def create_model(tokenizer):
    # Don't change this parameter
    MODEL_CONFIG = {
        'hidden_size': 2048,
        'num_hidden_layers': 12,
        'num_attention_heads': 16,
        'num_key_value_heads': 8,
        'intermediate_size': 8192,
        'head_dim': 128,
        'hidden_act': 'silu',
        'initializer_range': 0.02,
        'scale_attn_weights': True,
        'use_cache': True,
    }

    config = Qwen3Config(
        vocab_size=tokenizer.vocab_size,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
        **MODEL_CONFIG
    )
    
    model = Qwen3ForCausalLM._from_config(
        config,
        attn_implementation='flash_attention_2',
        torch_dtype=torch.bfloat16
    )
    
    print(f"Model pad token id: {model.config.pad_token_id}")
    
    with torch.no_grad():
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Total params: {total_params:,}")
    
    return model


def train_model():
    """
    TODO: Implement the training pipeline.
    - Prepare tokenizer
    - Load tokenized dataset and split it
    - Create the model
    - Create TrainingArguments from TRAINING_CONFIG
    - Create Trainer with TimeoutCallback
    - Train the model
    - Run final evaluation and print results
    - Save metric history to trainer_state.json for local loss plots
    """
    tokenizer = prepare_tokenizer()
    train_dataset, eval_dataset = split_dataset(load_tokenized_dataset())
    torch.manual_seed(TRAINING_CONFIG.get('seed', 42))
    model = create_model(tokenizer)
    model.config.use_cache = False
    trainer = Trainer(
        model=model,
        args=TrainingArguments(**TRAINING_CONFIG),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        callbacks=[TimeoutCallback(timeout_seconds=MAX_TRAINING_TIME_SECONDS)] # dont change
        )
    trainer.train()
    print("Running final evaluation...")
    eval_results = trainer.evaluate()
    print(f"Final evaluation results: {eval_results}")
    trainer.save_state()
    trainer.save_metrics('eval', eval_results)
    model.config.use_cache = True
    trainer.save_model(os.path.join(TRAINING_CONFIG['output_dir'], 'model'))
    tokenizer.save_pretrained(os.path.join(TRAINING_CONFIG['output_dir'], 'model'))
    return trainer


if __name__ == "__main__":
    # Step 1: Prepare the dataset (run once)
    # prepare_dataset()
    
    # Step 2: Train the model
    train_model()
