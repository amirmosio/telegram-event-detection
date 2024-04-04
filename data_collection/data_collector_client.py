import telethon
from base_telegram_client.client import TelegramClient
class DataCollectorClient(TelegramClient):

    def __init__(self, phone_number, api_id, api_hash):
        super(DataCollectorClient, self).__init__(phone_number, api_id, api_hash)

    async def get_messages(self, chat_id, limit=None, offset_date=None):
        async with self.client.takeout() as takeout:
            return await takeout.get_messages(chat_id, limit=limit, offset_date=offset_date)  # wrapped through takeout (less limits)