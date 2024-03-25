from data_collector_client import AnomalyDetectionClient
import asyncio
from decouple import config
import pandas as pd



phone_number = config('TELEGRAM_CLIENT_PHONE_NUMBER')
api_id = config('TELEGRAM_CLIENT_API_ID')
api_hash = config('TELEGRAM_CLIENT_API_HASH')

async def fetch_messages_from_client(chat):
    client = AnomalyDetectionClient(phone_number, api_id, api_hash)
    await client.connect(run_blocking=False)

    df = pd.DataFrame()

    messages = await client.get_messages(chat, 55)
    for message in messages:
        data = { "group" : chat, "sender" : message.sender_id, "text" : message.text, "date" : message.date}
        temp_df = pd.DataFrame(data, index=[1])
        df = pd.concat([df, temp_df])
    print(df.head())
    df.to_csv('trial.csv')
    
test_links = {"group": "https://t.me/joinchat/BlviDUvKluDK9m3p7OpIxQ", "private": "https://t.me/AmirMosio"}
chat = test_links['group']
asyncio.run(fetch_messages_from_client(chat))
