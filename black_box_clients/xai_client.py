import os
from xai_sdk import Client
from xai_sdk.chat import user, system

class XAIClient:
    def __init__(self, api_key, model: str = "grok-3-fast"):
        self.client = Client(api_key=api_key, timeout=3600)
        self.model = model

    def generate_response(self, prompt: str):
        chat = self.client.create(model=self.model)
        chat.append(system("You are Grok, a highly intelligent, helpful AI assistant."))
        chat.append(user(prompt))
        response = chat.sample()
        return response