from src.model_loader import load_model

base_model = load_model("./models/base_model")
model, tokenizer = load_model(base_model, load_in_4bit=True)
def load_dataset(path):
    dataset = load_dataset(
        "csv",
        data_files=path,
        split="all"
    )
    return dataset

def format_chat_template(row):
    instruction = """
    Bạn là một bác sĩ chăm sóc khách hàng tên Chis. 
    Hãy lịch sự với khách hàng và trả lời tất cả các câu hỏi của họ.
    """

    row_json = [{"role": "system", "content": instruction},
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["answer"]}]

    row["text"] = tokenizer.apply_chat_template(row_json, tokenize=False)
    return row

def split_dataset(dataset):
    split_1 = dataset.train_test_split(test_size=0.1)
    split_2 = split_1["train"].train_test_split(test_size=0.1667)

    train_dataset = split_2["train"]  # 75%
    val_dataset = split_2["test"]  # 15%
    test_dataset = split_1["test"]  # 10%

    return train_dataset, val_dataset, test_dataset



