(asr_env) root@EC03-E01-AICOE1:/home/CORP/re_nikitav/bu-digital-cx-asr-realtime/benchmarking# python3 asr_realtime_s3_benchmarking.py --url wss://whisperstream.exlservice.com:3000/asr/realtime-custom-vad --max-folders 20
Benchmarking 15 folders...
Running: asr-realtime/benchmarking-data-3/1272_1281041272-128104-0000/
Running: asr-realtime/benchmarking-data-3/1272_1281041272-128104-0001/
Running: asr-realtime/benchmarking-data-3/1272_1281041272-128104-0002/
Traceback (most recent call last):
  File "/usr/lib/python3.10/asyncio/locks.py", line 214, in wait
    await fut
asyncio.exceptions.CancelledError

During handling of the above exception, another exception occurred:

Traceback (most recent call last):
  File "/usr/lib/python3.10/asyncio/tasks.py", line 456, in wait_for
    return fut.result()
asyncio.exceptions.CancelledError

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/home/CORP/re_nikitav/bu-digital-cx-asr-realtime/benchmarking/asr_realtime_s3_benchmarking.py", line 238, in <module>
    asyncio.run(main())
  File "/usr/lib/python3.10/asyncio/runners.py", line 44, in run
    return loop.run_until_complete(main)
  File "/usr/lib/python3.10/asyncio/base_events.py", line 649, in run_until_complete
    return future.result()
  File "/home/CORP/re_nikitav/bu-digital-cx-asr-realtime/benchmarking/asr_realtime_s3_benchmarking.py", line 226, in main
    row = await process_folder(s3, args.url, args.bucket, f)
  File "/home/CORP/re_nikitav/bu-digital-cx-asr-realtime/benchmarking/asr_realtime_s3_benchmarking.py", line 159, in process_folder
    results = await asyncio.gather(
  File "/home/CORP/re_nikitav/bu-digital-cx-asr-realtime/benchmarking/asr_realtime_s3_benchmarking.py", line 134, in transcribe
    await asyncio.wait_for(done.wait(), timeout=30)
  File "/usr/lib/python3.10/asyncio/tasks.py", line 458, in wait_for
    raise exceptions.TimeoutError() from exc
asyncio.exceptions.TimeoutError
