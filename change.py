(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime# gcloud alpha ml speech recognizers create google-stt-default
--project="eci-ugi-digital-ccaipoc"
--location="us-central1"
--language-codes="en-US"
--model=latest_short
--enable-automatic-punctuation
--enable-word-confidence
--enable-word-time-offsets
ERROR: (gcloud.alpha.ml.speech.recognizers.create) argument --language-codes --model: Must be specified.
Usage: gcloud alpha ml speech recognizers create (RECOGNIZER : --location=LOCATION) --language-codes=[LANGUAGE_CODE,...] --model=MODEL [optional flags]
  optional flags may be  --async | --audio-channel-count | --display-name |
                         --enable-automatic-punctuation |
                         --enable-spoken-emojis | --enable-spoken-punctuation |
                         --enable-word-confidence | --enable-word-time-offsets |
                         --encoding | --help | --location | --max-alternatives |
                         --max-speaker-count | --min-speaker-count |
                         --profanity-filter | --sample-rate |
                         --separate-channel-recognition

For detailed information on this command and its flags, run:
  gcloud alpha ml speech recognizers create --help
--project=eci-ugi-digital-ccaipoc: command not found
--location=us-central1: command not found
--language-codes=en-US: command not found
--model=latest_short: command not found
--enable-automatic-punctuation: command not found
--enable-word-confidence: command not found
--enable-word-time-offsets: command not found
