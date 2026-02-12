gcloud alpha ml speech recognizers create google-stt-default \
  --project=eci-ugi-digital-ccaipoc \
  --location=us-central1 \
  --language-codes=en-US \
  --model=latest_short \
  --enable-automatic-punctuation \
  --enable-word-confidence \
  --enable-word-time-offsets
