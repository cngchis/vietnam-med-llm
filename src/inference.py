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
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        do_sample=True
    )

    text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return text.split("assistant")[-1].strip() if "assistant" in text else text.strip()

if __name__ == "__main__":
    base_model = load_model("./models/base_model")
    model, tokenizer = load_model(base_model, load_in_4bit=True)
    instruction = ""