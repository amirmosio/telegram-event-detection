from data_collection.fetching_data import fetch_messages_from_client_add_to_the_datafram
import pandas as pd
import asyncio
from decouple import config


batch_size = int(config("BATCH_SIZE"))
batch_size_increase_rate = int(config("BATCH_SIZE_INCREASE_RATE"))
time_interval_to_fetch_messages = int(config("TIME_INTERVAL_TO_FETCH_MESSAGES"))

df = pd.DataFrame()

groups_to_fetch_messages = ["https://t.me/joinchat/OOC4qk2QS1FM37aETHuzWQ",
                            "https://t.me/joinchat/RL4pXSkXipyuKDmd",
                            "https://t.me/joinchat/FNGD_0n6IpIbjfJBAZsuoA",
                            "https://t.me/joinchat/qyxbq_vZ5f4xYzg0",
                            "https://t.me/joinchat/rLRXuuItcHtkMTVk",
                            "https://t.me/joinchat/aiAC6RgOjBRkYjhk",
                            "https://t.me/PoliGruppo"]
for chat in groups_to_fetch_messages:
    print(f"fetching from {chat}")
    group_df = asyncio.run(fetch_messages_from_client_add_to_the_datafram(chat, limit=batch_size, batch_size_rate=batch_size_increase_rate))
    df = pd.concat([df, group_df])

df.to_csv("./data/trial.csv")


# Telegram User Client

# from decouple import config
# from user_telegram_client.user_client import UserClient
# import asyncio

# phone_number = config('TELEGRAM_CLIENT_PHONE_NUMBER')
# api_id = config('TELEGRAM_CLIENT_API_ID')
# api_hash = config('TELEGRAM_CLIENT_API_HASH')

# async def connect():
#     client = UserClient(phone_number, api_id, api_hash)
#     await client.connect(run_blocking=True)
# asyncio.run(connect())