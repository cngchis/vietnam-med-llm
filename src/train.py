import os
from transformers import TrainingArguments, EarlyStoppingCallback
from unsloth import is_bfloat16_supported
from trl import SFTTrainer
from src.data_loader import load_dataset, format_chat_template, split_dataset
from src.model_loader import load_model, apply_lora
from src.plot_metrics import plot_training_metrics


def main():
    # ====== CONFIG ======
    base_model = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"
    output_dir = "./src/output"
    new_model = "Llama-3.1-8B-Instruct-Medical"
    os.makedirs(output_dir, exist_ok=True)
    # LOAD MODEL
    model, tokenizer = load_model(base_model=base_model)

    model = apply_lora(model)

    # LOAD DATA
    dataset = load_dataset("./data/raw/medicalqa.csv")

    # FORMAT DATA
    dataset = dataset.map(format_chat_template,num_proc=4)

    # SPLIT
    train_dataset, val_dataset, test_dataset = split_dataset(dataset)

    # Save Data Processed
    os.makedirs("./data/processed", exist_ok=True)
    train_dataset.to_json("./data/processed/train.json")
    val_dataset.to_json("./data/processed/val.json")
    test_dataset.to_json("./data/processed/test.json")

    # TRAINING CONFIG
    training_arguments = TrainingArguments(
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        num_train_epochs=3,
        evaluation_strategy="steps",
        eval_steps=50,
        warmup_steps=50,
        learning_rate=2e-4,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        logging_steps=10,
        optim="adamw_8bit",
        weight_decay=0.01,
        lr_scheduler_type="linear",
        seed=3407,
        remove_unused_columns=False,
        output_dir=output_dir,
        save_steps=50,
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
    print("Start training...")
    trainer.train()

    # EVAL
    print("Start plot...")
    plot_training_metrics(trainer, output_dir)

    # SAVE MODEL
    model.save_pretrained(os.path.join(output_dir, new_model))
    tokenizer.save_pretrained(os.path.join(output_dir, new_model))

    print("Training complete & model saved!")
if __name__ == "__main__":
    main()