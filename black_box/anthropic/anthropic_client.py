import anthropic

class AnthropicClient:
    def __init__(self, api_key: str, model: str = "claude-sonnet-4-20250514"):
        self.client = anthropic.Client(api_key=api_key)
        self.model = model

    def generate_response(self, prompt: str, temperature = 1, max_tokens = None):
        request_kwargs = {
            'model': self.model,
            'max_tokens': max_tokens if max_tokens is not None else 1000,
            'temperature': temperature,
            'system': "You are a helpful assistant.",
            'messages': [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": prompt
                        }
                    ]
                }
            ]
        }
        response = self.client.messages.create(**request_kwargs)
        return response