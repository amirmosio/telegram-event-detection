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
        data = { "group" : chat, "sender" : message.sender_id, "msg_id": message.id, "reply" : message.reply_to_msg_id,  "text" : message.text, "date" : message.date, "reactions" : message.reactions}
        temp_df = pd.DataFrame(data, index=[1])
        df = pd.concat([df, temp_df])
    print(df.head())
    df.to_csv('poli_msgs.csv')
    await client.disconnect()
    
test_links = {"group": "https://t.me/+synza2388S80NWM0", "poli_group": "https://t.me/PoliGruppo"}
chat = test_links['poli_group']
asyncio.run(fetch_messages_from_client(chat))
