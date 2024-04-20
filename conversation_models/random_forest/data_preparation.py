import pandas as pd
import torch
from transformers import RobertaTokenizer, RobertaModel
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime, timedelta


def calculate_time_delta(a, b):
    datetime_format = "%Y-%m-%d %H:%M:%S%z"
    a_datetime = datetime.strptime(a, datetime_format)
    b_datetime = datetime.strptime(b, datetime_format)
    time_delta = a_datetime - b_datetime
    return time_delta / timedelta(minutes=1)


def embedding_with_reoberta(messages):
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
    model = RobertaModel.from_pretrained("roberta-base")
    model.eval()

    embeddings = []
    with torch.no_grad():
        for i in range(len(messages)):
            input_tokens = tokenizer(
                messages[i], return_tensors="pt", padding=True, truncation=True
            )

            output = model(**input_tokens)

            embedding = output.last_hidden_state[:, 0, :]
            embeddings.append(embedding)
    return embeddings


def generate_dataset_from_labeled_data_with_sliding_window(df, window_size=5):
    embeddings = embedding_with_reoberta(df["text"])
    df["embedding"] = embeddings
    result_df = pd.DataFrame()
    for start_idx in range(len(df) - window_size - 1):
        record_df = df.iloc[start_idx : start_idx + window_size + 1]
        last_messages_conversation = record_df.iloc[-1]["conversation_id"]

        window_df = record_df.iloc[:-1]

        label = False
        if last_messages_conversation in window_df["conversation_id"].array:
            label = True

        train_record = {}
        for i in range(window_size):
            train_record |= {
                f"d{i}": cosine_similarity(
                    window_df.iloc[i]["embedding"], record_df.iloc[-1]["embedding"]
                )[0][0]
            }
            train_record |= {
                f"timegap{i}": calculate_time_delta(
                    record_df.iloc[-1]["date"], window_df.iloc[i]["date"]
                )
            }
            same_profile = False
            try:
                same_profile = int(record_df.iloc[-1]["sender"]) == int(
                    window_df.iloc[i]["sender"]
                )
            except:
                pass
            train_record |= {f"same_profile{i}": same_profile}
        train_record |= {"label": label}
        result_df = result_df._append(train_record, ignore_index=True)

    result_df = result_df.reset_index()
    return result_df
