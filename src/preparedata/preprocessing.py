import pandas as pd
import pyarrow.parquet as pq
import json
import re
from typing import Tuple
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import train_test_split
from src.utils.helper import get_env

def load_data(path):
    table = pq.read_table(path)
    df = table.to_pandas()
    for col in df.columns:
        df[col] = df[col].astype(str)
    return df

def find_boxplot_boundaries(
    col: pd.Series, whisker_coeff: float = 1.5
) -> Tuple[float, float]:
    """Findx minimum and maximum in boxplot.

    Args:
        col: a pandas serires of input.
        whisker_coeff: whisker coefficient in box plot
    """
    Q1 = col.quantile(0.25)
    Q3 = col.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - whisker_coeff * IQR
    upper = Q3 + whisker_coeff * IQR
    return lower, upper


class BoxplotOutlierClipper(BaseEstimator, TransformerMixin):
    def __init__(self, whisker_coeff: float = 1.5):
        self.whisker = whisker_coeff
        self.lower = None
        self.upper = None

    def fit(self, X: pd.Series):
        self.lower, self.upper = find_boxplot_boundaries(X, self.whisker)
        return self

    def transform(self, X):
        return X.clip(self.lower, self.upper)

def preprocess_data(df):
    df.dropna(inplace=True)
    df.drop_duplicates(subset='question', inplace=True)

    for col in df.columns:
        df[col] = df[col].apply(lambda x: re.sub(r'<.*?>','', str(x)))
        df[col] = df[col].apply(lambda x: x.replace('\n', ' ').strip())

    df['q_len'] = df['question'].apply(lambda x: len(str(x).split()))
    df['a_len'] = df['answer'].apply(lambda x: len(str(x).split()))

    cols = ['q_len', 'a_len']
    for col in cols:
        df[col] = BoxplotOutlierClipper().fit_transform(df[col])

    df = df[df['a_len'] > 3]


    return df

def split_data(df, train_ratio=0.2):
    train_df, val_df = train_test_split(df, test_size=train_ratio, random_state=42)
    return train_df, val_df
instruction = "Bạn là một bác sĩ am hiểu kiến thức y tế."
def convert_to_jsonl(df, path):
    with open(path, 'w', encoding='utf-8') as f_out:
        for idx, row in df.iterrows():
            data = {
                "messages": [
                    {"role": "system", "content": instruction},
                    {"role": "user", "content": row['question']},
                    {"role": "assistant", "content": row['answer']},
                ]
            }
            f_out.write(json.dumps(data, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    DATASET_PATH = get_env('DATASET_PATH')
    TRAIN_PATH = get_env('TRAIN_PATH')
    VAL_PATH = get_env('VAL_PATH')
    df = load_data(DATASET_PATH)
    df = preprocess_data(df)
    df.drop(["q_len", "a_len"], axis=1, inplace=True)
    df.to_csv("./data/medicalqa.csv", index=False, encoding="utf-8-sig")
    train_df, val_df = split_data(df)
    convert_to_jsonl(train_df, TRAIN_PATH)
    convert_to_jsonl(val_df, VAL_PATH)