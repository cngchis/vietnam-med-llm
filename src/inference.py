import torch
from unsloth import FastLanguageModel
from src.model_loader import load_model

def generate_response(
    model,
    tokenizer,
    instruction,
    user_input,
    max_new_tokens=512,
    temperature=0.7,
    top_p=0.9
):
    FastLanguageModel.for_inference(model)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input}
    ]

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    inputs = tokenizer(
        prompt,
        return_tensors='pt',
        padding=True,
        truncation=True
    ).to(device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True,
        eos_token_id=tokenizer.eos_token_id,
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "<|assistant|>" in text:
        response = text.split("<|assistant|>")[-1].strip()
    else:
        response = text.strip()

    return response


if __name__ == "__main__":
    model, tokenizer = load_model(
        "./src/output/Llama-3-8B-Instruct-Medical",
        load_in_4bit=True
    )

    instruction = """
        You are a Vietnamese medical assistant. 
        Answer accurately, safely, and avoid making medical diagnoses
        """

    print("=== Doctor Assistant ===")
    print("Type 'exit' or 'quit' to exit.\n")

    while True:
        user_input = input("You: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        response = generate_response(model, tokenizer, instruction, user_input)

        print(f"\nDoctor: {response}\n")