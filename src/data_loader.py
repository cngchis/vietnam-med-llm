from datasets import load_dataset

def load_my_dataset(path):
    dataset = load_dataset(
        "csv",
        data_files=path,
        split="all"
    )
    return dataset

def split_dataset(dataset):
    split_1 = dataset.train_test_split(test_size=0.1)
    split_2 = split_1["train"].train_test_split(test_size=0.1667)

    train_dataset = split_2["train"]  # 75%
    val_dataset = split_2["test"]  # 15%
    test_dataset = split_1["test"]  # 10%

    return train_dataset, val_dataset, test_dataset



