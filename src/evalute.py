from unsloth import FastLanguageModel
import pandas as pd
from datasets import Dataset
from trl import SFTTrainer

def main():
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="./src/output/Llama-3.1-8B-Instruct-Medical",
        load_in_4bit=True
    )

    FastLanguageModel.for_inference(model)

    df = pd.read_json('./data/processed/test.json')
    dataset = Dataset.from_pandas(df)

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