import telethon

class TelegramClient:

    def __init__(self, phone_number, api_id, api_hash):
        self.phone_number = phone_number
        self.client = telethon.TelegramClient("telegram_client", api_id, api_hash)

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