docker build \
  --build-arg USE_PROXY=true \
  --build-arg HTTP_PROXY=http://163.116.128.80:8080 \
  --build-arg HTTPS_PROXY=http://163.116.128.80:8080 \
  -t asr-realtime:latest .
