#google_asr.py-

from ..config import ALL_CONFIG
from google.cloud.speech_v2.types import cloud_speech
from google.api_core.client_options import ClientOptions
from google.cloud import speech_v1p1beta1 as speech_v1
from google.cloud import speech_v2
import json
import os

from src.utils.logger import get_logger
logger = get_logger(__name__)


class GoogleASRClient:
    def __init__(self):
        self.speech_client = speech_v2.SpeechClient(
            client_options=ClientOptions(api_endpoint=ALL_CONFIG['Urls']['google_asr'])
        )
        self.v1_client = speech_v1.SpeechClient()
        
        self.config = speech_v2.RecognitionConfig(
            auto_decoding_config=cloud_speech.AutoDetectDecodingConfig(),
            language_codes=["en-US"],
            model='latest_short'
        )
     
    async def transcribe(self, client):
        """
        Transcribes the audio file for a given client using Google's Speech API v2.
        """
        file_path = os.path.join(ALL_CONFIG['PATH']['audio_dir'], client.get_file_name())
        try:
            with open(file_path, "rb") as f:
                audio_content = f.read()

            request = cloud_speech.RecognizeRequest(
                recognizer= ALL_CONFIG['Credentials']['google_asr_recognizer'],
                config=self.config,
                content=audio_content
            )

            response = self.speech_client.recognize(request=request)
            transcriptions = [
                result.alternatives[0].transcript for result in response.results
            ]
            concatenated_transcription = " ".join(transcriptions).strip()
            
            if concatenated_transcription in ["", ".", ". ", "None", None]:
                concatenated_transcription = ""
            
            os.remove(file_path)
            return {"text": concatenated_transcription}
        
        except Exception as e:
            logger.error("Error in GOOGLE ASR pipeline: %s", e)
            return {"text": ""}
        
#src/config/google_credential.json-
{
  "type": "service_account",
  "project_id": "eci-ugi-digital-ccaipoc",
  "private_key_id": "xxxxxxxxxx",
  "private_key": "xxxxxxxxxxxxxxxxx",
  "client_email": "eci-ugi-digital-ccaipoc-crun@eci-ugi-digital-ccaipoc.iam.gserviceaccount.com",
  "client_id": "105631226810949087981",
  "auth_uri": "https://accounts.google.com/o/oauth2/auth",
  "token_uri": "https://oauth2.googleapis.com/token",
  "auth_provider_x509_cert_url": "https://www.googleapis.com/oauth2/v1/certs",
  "client_x509_cert_url": "https://www.googleapis.com/robot/v1/metadata/x509/eci-ugi-digital-ccaipoc-crun%40eci-ugi-digital-ccaipoc.iam.gserviceaccount.com",
  "universe_domain": "googleapis.com"
}


