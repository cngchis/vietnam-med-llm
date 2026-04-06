import torch
from unsloth import FastLanguageModel

def load_model(base_model, max_seq_length=1024, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=load_in_4bit,
    )
    return model, tokenizer


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
        "./src/output/Llama-3.1-8B-Instruct-Medical",
        load_in_4bit=True
    )

    instruction = """
    Bạn là một bác sĩ chăm sóc khách hàng tên Chis. 
    Hãy lịch sự với khách hàng và trả lời tất cả các câu hỏi của họ.
    """

    print("=== Bác sĩ Chis AI ===")
    print("Gõ 'exit' hoặc 'quit' để thoát.\n")

    while True:
        user_input = input("Bạn: ").strip()

        if not user_input:
            continue

        if user_input.lower() in ["exit", "quit"]:
            print("Tạm biệt!")
            break

        response = generate_response(model, tokenizer, instruction, user_input)

        print(f"\nBác sĩ Chis: {response}\n")