import asyncio
import threading
import websockets
import json

class WebSocketClient:
    def __init__(self, uri):
        self.uri = uri
        self.ws = None
        self.loop = asyncio.new_event_loop()

    async def connect(self):
        self.ws = await websockets.connect(self.uri)

    async def send(self, data):
        if self.ws is not None:
            await self.ws.send(json.dumps(data))

    async def run(self):
        await self.connect()
        while True:
            await asyncio.sleep(1)

    def start(self):
        def runner():
            asyncio.set_event_loop(self.loop)
            self.loop.run_until_complete(self.run())
        thread = threading.Thread(target=runner, daemon=True)
        thread.start()

    def send_sync(self, data):
        asyncio.run_coroutine_threadsafe(self.send(data), self.loop)