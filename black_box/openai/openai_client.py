from openai import OpenAI
import time

class OpenAIClient:
    def __init__(self, api_key, model = 'gpt-4.1-nano-2025-04-14', retries = 3):
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.retries = retries

    def generate_response(self, prompt: str, temeprature = 1, max_output_tokens = None):
        retries = 3

        for attempt in range(retries):
            try:
                request_kwargs = {
                    'model': self.model,
                    'messages': [
                        {'role': 'system', 'content': 'You are a helpful assistant.'},
                        {'role': 'user', 'content': prompt}
                    ],
                    'temperature': temeprature,
                }

                if max_output_tokens is not None:
                    request_kwargs['max_tokens'] = max_output_tokens

                response = self.client.chat.completions.create(**request_kwargs)
                return response

            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(1)
                else:
                    raise e