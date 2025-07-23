from elevenlabs.client import ElevenLabs
from elevenlabs import Voice, VoiceSettings

class ElevenLabsClient:
    def __init__(self, api_key: str):
        self.client = ElevenLabs(api_key=api_key)

    def ttx(self, text: str, 
            voice_id: str, 
            model_id: str = "eleven_flash_v2_5"):
        audio = self.client.text_to_speech.convert(text=text, voice_id=voice_id, model_id=model_id)
        return audio
    
    def clone(self, name, 
              description, files=[]):
        voice = self.client.voices.ivc.create(
            name=name,
            description=description,
            files=files,
        )
        return voice