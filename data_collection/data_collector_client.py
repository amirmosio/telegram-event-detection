from telethon import TelegramClient, events, errors

class AnomalyDetectionClient:

    def __init__(self, phone_number, api_id, api_hash):
        self.phone_number = phone_number
        self.client = TelegramClient("anomaly_detection", api_id, api_hash)

        @self.client.on(events.NewMessage())
        async def handler(event):
            if event.is_group or event.is_channel and not event.is_private:
                pass
            else:
                print(event.chat_id)
                print(event.is_group, event.is_channel, event.is_private)
                print(event.raw_text)

    async def get_messages(self, chat_id, limit=None, offset_date=None):
        async with self.client.takeout() as takeout:
            return await takeout.get_messages(chat_id, limit=limit, offset_date=offset_date)  # wrapped through takeout (less limits)


    async def connect(self, run_blocking=True):
        await self.client.connect()
        is_authorized = await self.client.is_user_authorized()
        if not is_authorized:
            await self.client.send_code_request(self.phone_number)
            await self.client.sign_in(self.phone_number, input('Enter the code: '))
        if run_blocking:
            print("client started...")
            await self.client.run_until_disconnected()

    async def disconnect(self):
        await self.client.disconnect()