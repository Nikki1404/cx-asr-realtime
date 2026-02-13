(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime/app# unset http_proxy
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime/app# unset https_proxy
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime/app# export AWS_ACCESS_KEY_ID=""
export AWS_SESSION_TOKEN=""
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime/app# kubectl get pods -n cx-speech
NAME                                       READY   STATUS    RESTARTS   AGE
asr-realtime-custom-vad-6b8b5f6c67-ctgd5   1/1     Running   0          57m
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime/app# kubectl scale deployment asr-realtime-custom-vad -n cx-speech --replicas=0
deployment.apps/asr-realtime-custom-vad scaled
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime/app# kubectl get pods -n cx-speech
No resources found in cx-speech namespace.
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime/app# kubectl set image deployment/asr-realtime-custom-vad asr-realtime-custom-vad-container=058264113403.dkr.ecr.us-east-1.amazonaws.com/cx-speech/asr-realtime-custom-vad:0.0.9 -n cx-speech
deployment.apps/asr-realtime-custom-vad image updated
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime/app# kubectl get pods -n cx-speech
No resources found in cx-speech namespace.
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime/app# kubectl get pods -n cx-speech
No resources found in cx-speech namespace.
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime/app# kubectl get pods -n cx-speech
No resources found in cx-speech namespace.
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime/app# kubectl scale deployment asr-realtime-custom-vad -n cx-speech --replicas=1
deployment.apps/asr-realtime-custom-vad scaled
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime/app# kubectl get pods -n cx-speech
NAME                                       READY   STATUS              RESTARTS   AGE
asr-realtime-custom-vad-684fdb44d5-h4s6s   0/1     ContainerCreating   0          6s
(base) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime/app#
