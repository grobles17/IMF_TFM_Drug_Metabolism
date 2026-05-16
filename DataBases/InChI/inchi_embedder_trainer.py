import os
import math
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    RobertaConfig,
    RobertaForMaskedLM,
    PreTrainedTokenizerFast,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from tokenizers import ByteLevelBPETokenizer

def main():
    # ============================================================
    # 1. Reproducibility
    # ============================================================

    SEED = 17
    set_seed(SEED)
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ============================================================
    # 2. Paths
    # ============================================================

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

    VOCAB_PATH = os.path.join(SCRIPT_DIR, "inchi_tokenizer", "vocab.json")
    MERGES_PATH = os.path.join(SCRIPT_DIR, "inchi_tokenizer", "merges.txt")
    DB_PATH = os.path.join(SCRIPT_DIR, "inchi_output.txt")

    MODEL_SAVE_PATH = os.path.join(SCRIPT_DIR, "inchi_embedder_final")

    # ============================================================
    # 3. Load Tokenizer (SAFE VERSION – No internal truncation)
    # ============================================================

    tokenizer_bpe = ByteLevelBPETokenizer(
        vocab=VOCAB_PATH,
        merges=MERGES_PATH
    )

    # Ensure no internal truncation is enabled
    tokenizer_bpe.no_truncation()

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_bpe,
        bos_token="<s>",
        eos_token="</s>",
        unk_token="<unk>",
        pad_token="<pad>",
        mask_token="<mask>",
    )

    print("Special tokens:", fast_tokenizer.special_tokens_map)
    print("Tokenizer vocab size:", len(fast_tokenizer))

    vocab_size = len(fast_tokenizer)

    # ============================================================
    # 4. Model Configuration (Lightweight but Robust)
    # ============================================================

    MAX_SEQ_LENGTH = 512 

    config = RobertaConfig(
        vocab_size=vocab_size,
        max_position_embeddings= MAX_SEQ_LENGTH + 2, # Account for roBERTa's intenal buffering logic 
        hidden_size=384,              # Compact but expressive
        num_attention_heads=6,        # 384 / 6 = 64 per head
        num_hidden_layers=6,          # Increased from 4 → better depth
        intermediate_size=1536,       # 4x hidden_size
        type_vocab_size=1,
        pad_token_id=fast_tokenizer.pad_token_id,
        bos_token_id=fast_tokenizer.bos_token_id,
        eos_token_id=fast_tokenizer.eos_token_id,
        layer_norm_eps=1e-5,
    )

    model = RobertaForMaskedLM(config=config)
    model.resize_token_embeddings(len(fast_tokenizer))

    print(model.config.max_position_embeddings)
    print(f"Model parameters: {model.num_parameters():,}")

    # ============================================================
    # 5. Dataset Loading
    # ============================================================

    raw_dataset = load_dataset(
        "text",
        data_files={"data": DB_PATH},
    )

    dataset = raw_dataset["data"].train_test_split(
        test_size=0.1,
        seed=SEED
    )

    # ============================================================
    # 6. Tokenization (Manual truncation + Dynamic padding)
    # ============================================================

    def tokenize_function(examples):

        encodings = fast_tokenizer(
            examples["text"],
            padding=False,      # IMPORTANT: no fixed padding
            truncation=False,   # IMPORTANT: avoid internal truncation call
        )

        # Manual truncation
        encodings["input_ids"] = [
            ids[:MAX_SEQ_LENGTH] for ids in encodings["input_ids"]
        ]

        encodings["attention_mask"] = [
            mask[:MAX_SEQ_LENGTH] for mask in encodings["attention_mask"]
        ]

        return encodings


    tokenized_datasets = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=["text"],
        num_proc=4,
    )

    train_dataset = tokenized_datasets["train"]
    eval_dataset = tokenized_datasets["test"]

    # ============================================================
    # 7. Data Collator (Dynamic Padding + MLM)
    # ============================================================

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=fast_tokenizer,
        mlm=True,
        mlm_probability=0.15,
    )

    # ============================================================
    # 8. Training Steps Calculation
    # ============================================================

    BATCH_SIZE = 16     #Decreased due to limited VRAM (4GB)
    EPOCHS = 3          #limited by time constraints and GPU

    steps_per_epoch = math.ceil(len(train_dataset) / BATCH_SIZE)
    total_steps = steps_per_epoch * EPOCHS
    warmup_steps = int(0.06 * total_steps)

    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")

    print(torch.__version__)
    print(torch.version.cuda)
    print(torch.cuda.is_available())
    # ============================================================
    # 9. Training Arguments (Stable & Efficient)
    # ============================================================

    training_args = TrainingArguments(
        output_dir= os.path.join(SCRIPT_DIR, "inchi_embedder_checkpoints"),
        num_train_epochs=EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        eval_strategy="steps",
        eval_steps=5000,
        save_steps=5000,
        save_total_limit=2,
        logging_dir=os.path.join(SCRIPT_DIR, "logs"),
        logging_steps=500,
        learning_rate=5e-4,            # Slightly higher for small model
        warmup_steps=warmup_steps,
        weight_decay=0.01,
        max_grad_norm=1.0,
        fp16=True,                      # Mixed precision for efficiency
        dataloader_num_workers=4,
        load_best_model_at_end=True,
        report_to="none",
        seed=SEED,
        lr_scheduler_type="cosine",
        gradient_accumulation_steps=4,    # Effective batch size of 64
        dataloader_pin_memory=True,      
        )

    # ============================================================
    # 10. Trainer
    # ============================================================

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    # ============================================================
    # 11. Train
    # ============================================================

    trainer.train()

    # ============================================================
    # 12. Save Final Model
    # ============================================================

    trainer.save_model(MODEL_SAVE_PATH)
    fast_tokenizer.save_pretrained(MODEL_SAVE_PATH)

if __name__ == "__main__":
    main()