import torch
from unsloth import FastLanguageModel

def load_model(base_model, max_seq_length=2048, load_in_4bit=True):
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=base_model,
        max_seq_length=max_seq_length,
        dtype=torch.float16,
        load_in_4bit=False,
    )

    return model, tokenizer

def apply_lora(
    model,
    r=32,
    lora_alpha=64,
    lora_dropout=0.5,
):
    model = FastLanguageModel.get_peft_model(
        model,
        r=r,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj",
        ],
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=3407,
        use_rslora=True,
        loftq_config=None,
    )

    return model