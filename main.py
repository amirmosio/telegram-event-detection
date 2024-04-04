from data_collection.fetching_data import fetch_messages_from_client_add_to_the_datafram
import pandas as pd
import asyncio
from decouple import config


batch_size = config("BATCH_SIZE")
batch_size_increase_rate = config("BATCH_SIZE_INCREASE_RATE")
time_interval_to_fetch_messages = config("TIME_INTERVAL_TO_FETCH_MESSAGES")

df = pd.DataFrame()

groups_to_fetch_messages = ["https://t.me/+synza2388S80NWM0", "https://t.me/PoliGruppo"]
for chat in groups_to_fetch_messages:
    print(f"fetching from {chat}")
    group_df = asyncio.run(fetch_messages_from_client_add_to_the_datafram(chat, limit=batch_size, batch_size_rate=batch_size_increase_rate))
    df = pd.concat([df, group_df])

df.to_csv("./data/trial.csv")