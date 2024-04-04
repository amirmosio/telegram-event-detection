import telethon
from base_telegram_client.client import TelegramClient
from decouple import config
import asyncio
from .manager import QueryManager

class UserClient(TelegramClient):

    def __init__(self, phone_number, api_id, api_hash):
        super(UserClient, self).__init__(phone_number, api_id, api_hash)
        self.query_manager = QueryManager()

        @self.client.on(telethon.events.NewMessage())
        async def handler(event):
            if not event.is_group and not event.is_channel and event.is_private:
                a = event.from_id, event.peer_id, event.sender_id
                sender = await event.get_sender()
                sender.username, sender.last_name, sender.first_name, sender.phone, sender.id
                if self.query_manager.check_if_user_has_been_registered(event.sender_id):
                    # TODO
                    pass
                else:
                    await event.reply("Hello")
                print(event.chat_id)
                print(event.is_group, event.is_channel, event.is_private)
                print(event.raw_text)

    async def get_messages(self, chat_id, limit=None, offset_date=None):
        async with self.client.takeout() as takeout:
            return await takeout.get_messages(chat_id, limit=limit, offset_date=offset_date)  # wrapped through takeout (less limits)
        