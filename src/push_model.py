from unsloth import FastLanguageModel
from huggingface_hub import login

def main():
    # 🔑 login (nếu chưa login CLI)
    login()

    # 📦 load model đã train
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="./output/Llama-3.1-8B-Instruct-Medical",
        load_in_4bit=True
    )

    # 🚀 push lên Hugging Face
    repo_id = "your/Llama-3.1-8B-Medical"

    model.push_to_hub(repo_id)
    tokenizer.push_to_hub(repo_id)

    print("✅ Push thành công!")

if __name__ == "__main__":
    main()