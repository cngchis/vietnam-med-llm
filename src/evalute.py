from unsloth import FastLanguageModel
from src.data_loader import load_dataset, format_chat_template
from trl import SFTTrainer

def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="./src/output/Llama-3.1-8B-Instruct-Medical",
        load_in_4bit=True
    )

    FastLanguageModel.for_inference(model)

    dataset = load_dataset("./data/test.csv")
    dataset = dataset.map(format_chat_template, num_proc=4)

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        eval_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=1024,
    )

    metrics = trainer.evaluate()
    print(metrics)

if __name__ == "__main__":
    main()