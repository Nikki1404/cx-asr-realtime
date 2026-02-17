INFO:     10.90.126.103:3818 - "GET / HTTP/1.1" 404 Not Found
INFO:     10.90.126.103:23536 - "WebSocket /asr/realtime-custom-vad" [accepted]
INFO:     connection open
2026-02-17 10:33:30,554 - INFO -  Using cached whisper engine (0ms latency!)
2026-02-17 10:33:30,554 - INFO - WS connected (whisper) Address(host='10.90.126.103', port=23536)
INFO:     10.90.126.16:49864 - "GET / HTTP/1.1" 404 Not Found
INFO:     10.90.126.103:23538 - "GET / HTTP/1.1" 404 Not Found
2026-02-17 10:33:34,495 - INFO - CLIENT: Address(host='10.90.126.103', port=23536), TEXT: Halló.
ERROR:    Exception in ASGI application
Traceback (most recent call last):
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/protocols/websockets/websockets_impl.py", line 244, in run_asgi
    result = await self.app(self.scope, self.asgi_receive, self.asgi_send)  # type: ignore[func-returns-value]
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/middleware/proxy_headers.py", line 60, in __call__
    return await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/applications.py", line 1138, in __call__
    await super().__call__(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/applications.py", line 107, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/errors.py", line 151, in __call__
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/middleware/exceptions.py", line 63, in __call__
    await wrap_app_handling_exceptions(self.app, conn)(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/middleware/asyncexitstack.py", line 18, in __call__
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 716, in __call__
    await self.middleware_stack(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 736, in app
    await route.handle(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/routing.py", line 364, in handle
    await self.app(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 147, in app
    await wrap_app_handling_exceptions(app, session)(scope, receive, send)
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 53, in wrapped_app
    raise exc
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 42, in wrapped_app
    await app(scope, receive, sender)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 144, in app
    await func(session)
  File "/usr/local/lib/python3.10/dist-packages/fastapi/routing.py", line 509, in app
    await dependant.call(**solved_result.values)
  File "/srv/app/main.py", line 243, in ws_asr
    await ws.close()
  File "/usr/local/lib/python3.10/dist-packages/starlette/websockets.py", line 181, in close
    await self.send({"type": "websocket.close", "code": code, "reason": reason or ""})
  File "/usr/local/lib/python3.10/dist-packages/starlette/websockets.py", line 86, in send
    await self._send(message)
  File "/usr/local/lib/python3.10/dist-packages/starlette/_exception_handler.py", line 39, in sender
    await send(message)
  File "/usr/local/lib/python3.10/dist-packages/uvicorn/protocols/websockets/websockets_impl.py", line 357, in asgi_send
    raise RuntimeError(msg % message_type)
RuntimeError: Unexpected ASGI message 'websocket.close', after sending 'websocket.close' or response already completed.
INFO:     connection closed
