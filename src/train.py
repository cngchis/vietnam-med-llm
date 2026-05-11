import os
import unsloth
from transformers import TrainingArguments, EarlyStoppingCallback
from functools import partial
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from src.data_loader import load_my_dataset, split_dataset
from src.model_loader import load_model, apply_lora
from src.plot_metrics import plot_training_metrics

def make_formatter(tokenizer):

    def format_chat_template(row):
        instruction = """
        You are a Vietnamese medical assistant. 
        Answer accurately, safely, and avoid making medical diagnoses
        """

        messages = [
            {"role": "system", "content": instruction},
            {"role": "user", "content": row["question"]},
            {"role": "assistant", "content": row["answer"]},
        ]

        row["text"] = tokenizer.apply_chat_template(messages, tokenize=False)
        return row

    return format_chat_template

def main():
    # CONFIG
    base_model = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    output_dir = "outputs"
    new_model = "Llama-3.1-8B-Instruct-Medical"
    os.makedirs(output_dir, exist_ok=True)

    # LOAD MODEL
    model, tokenizer = load_model(base_model=base_model)

    model = apply_lora(model)

    # LOAD DATA
    dataset = load_my_dataset("data/raw/medicalqa.csv")

    # FORMAT DATA
    dataset = dataset.map(
        make_formatter(tokenizer),
        num_proc=4
    )

    # SPLIT
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    # Save Data Processed
    os.makedirs("data/processed", exist_ok=True)
    train_dataset.to_json("data/processed/train.jsonl", orient="records", lines=True, force_ascii=False)
    val_dataset.to_json(  "data/processed/val.jsonl",   orient="records", lines=True, force_ascii=False)
    test_dataset.to_json( "data/processed/test.jsonl",  orient="records", lines=True, force_ascii=False)

    # TRAINING CONFIG
    training_arguments = TrainingArguments(
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        eval_strategy="steps",
        eval_steps=1000,
        save_steps=1000,
        warmup_steps=50,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=100,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        remove_unused_columns=False,
        output_dir=output_dir,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
    )

    # TRAINER
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        dataset_text_field="text",
        max_seq_length=1024,
        dataset_num_proc=2,
        packing=True,
        args=training_arguments,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )

    # TRAIN
    print("Start training")
    trainer.train()

    # EVAL
    print("Start plot")
    plot_training_metrics(trainer, output_dir)

    # SAVE LoRA adapter
    lora_path = os.path.join(output_dir, new_model)
    model.save_pretrained(lora_path)
    tokenizer.save_pretrained(lora_path)
    print(f"LoRA adapter saved at {lora_path}")

    merged_path = os.path.join(output_dir, new_model + "-merged-fp16")
    print(f"Merging LoRA to full fp16 model, saving to {merged_path}...")
    model.save_pretrained_merged(
        merged_path,
        tokenizer,
        save_method="merged_16bit",
    )
    print(f"Merged fp16 saved at {merged_path}")
 
    print("\nTraining complete!")
    print(f"  LoRA adapter : {lora_path}")
    print(f"  Merged fp16  : {merged_path}")
    
if __name__ == "__main__":
    main()