from data_collection.fetching_data import fetch_messages_from_client_add_to_the_datafram
import pandas as pd
import asyncio
from decouple import config
from data_labeling.select_for_labeling import select_data_for_labeling
from decouple import config
from user_telegram_client.user_client import UserClient
import asyncio

from conversation_models.random_forest.data_preparation import (
    generate_dataset_from_labeled_data_with_sliding_window,
)

from conversation_models.random_forest.model_training import train_random_forest_model

commands = {
    "1": "Download dataset from telegram groups",
    "2": "Select labeling data",
    "3": "Train conversation model",
    "4": "Run Telegram client",
}
command = input(
    "Which phase:\n" + "\n".join([f"{key}-{commands[key]}" for key in commands]) + "\n"
)
commands[command]


if command == "1":
    batch_size = int(config("BATCH_SIZE"))
    batch_size_increase_rate = int(config("BATCH_SIZE_INCREASE_RATE"))
    time_interval_to_fetch_messages = int(config("TIME_INTERVAL_TO_FETCH_MESSAGES"))

    df = pd.DataFrame()

    groups_to_fetch_messages = [
        "https://t.me/joinchat/OOC4qk2QS1FM37aETHuzWQ",
        "https://t.me/joinchat/RL4pXSkXipyuKDmd",
        "https://t.me/joinchat/FNGD_0n6IpIbjfJBAZsuoA",
        "https://t.me/joinchat/qyxbq_vZ5f4xYzg0",
        "https://t.me/joinchat/rLRXuuItcHtkMTVk",
        "https://t.me/joinchat/aiAC6RgOjBRkYjhk",
        "https://t.me/PoliGruppo",
    ]
    for chat in groups_to_fetch_messages:
        print(f"fetching from {chat}")
        group_df = asyncio.run(
            fetch_messages_from_client_add_to_the_datafram(
                chat, limit=batch_size, batch_size_rate=batch_size_increase_rate
            )
        )
        df = pd.concat([df, group_df])

    df.to_csv("./data/trial.csv")

elif command == "2":
    raw_data = pd.read_csv("./data/trial.csv")
    df = select_data_for_labeling(raw_data)
    df.to_csv("./data/labeling_data.csv")

elif command == "3":
    raw_data = pd.read_csv(
        "./data/labeled_data.csv", sep=";", encoding="unicode_escape"
    )
    raw_data = raw_data.dropna(subset=["conversation_id"])
    raw_data = raw_data.reset_index(drop=True)

    train_df = generate_dataset_from_labeled_data_with_sliding_window(raw_data)
    acc, model = train_random_forest_model(train_df)
    print(acc)

elif command == "4":

    phone_number = config("TELEGRAM_CLIENT_PHONE_NUMBER")
    api_id = config("TELEGRAM_CLIENT_API_ID")
    api_hash = config("TELEGRAM_CLIENT_API_HASH")

    async def connect():
        client = UserClient(phone_number, api_id, api_hash)
        await client.connect(run_blocking=True)

    asyncio.run(connect())
