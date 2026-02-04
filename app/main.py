import asyncio
import json
import time
import logging

from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.config import load_config, Config
from app.metrics import *
from app.vad import AdaptiveEnergyVAD
from app.factory import build_engine
from app.asr_engines.base import ASREngine

cfg = load_config()
logging.basicConfig(level=cfg.log_level)
log = logging.getLogger("asr_server")

app = FastAPI()

# cache engines per backend
ENGINE_CACHE: dict[str, ASREngine] = {}


def get_engine(backend: str) -> ASREngine:
    if backend in ENGINE_CACHE:
        return ENGINE_CACHE[backend]

    tmp_cfg = Config()
    tmp_cfg.asr_backend = backend
    tmp_cfg.model_name = ""
    tmp_cfg.device = cfg.device
    tmp_cfg.sample_rate = cfg.sample_rate
    tmp_cfg.context_right = cfg.context_right

    engine = build_engine(tmp_cfg)
    load_sec = engine.load()
    log.info(f"Loaded backend={backend} in {load_sec:.2f}s")

    ENGINE_CACHE[backend] = engine
    return engine


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    await ws.accept()

    # 🔑 FIRST MESSAGE MUST BE CONFIG
    init = await ws.receive_text()
    init_obj = json.loads(init)

    backend = init_obj.get("backend")
    if backend not in ("nemotron", "whisper"):
        await ws.close(code=4000)
        return

    engine = get_engine(backend)

    labels = (backend, engine.model_name)

    active_streams = ACTIVE_STREAMS.labels(*labels)
    partials_total = PARTIALS_TOTAL.labels(*labels)
    finals_total = FINALS_TOTAL.labels(*labels)
    utterances_total = UTTERANCES_TOTAL.labels(*labels)

    ttft_wall = TTFT_WALL.labels(*labels)
    ttf_wall = TTF_WALL.labels(*labels)

    infer_sec = INFER_SEC.labels(*labels)
    preproc_sec = PREPROC_SEC.labels(*labels)
    flush_sec = FLUSH_SEC.labels(*labels)

    audio_sec_hist = AUDIO_SEC.labels(*labels)
    rtf_hist = RTF.labels(*labels)
    backlog_ms_gauge = BACKLOG_MS.labels(*labels)

    active_streams.inc()
    log.info(f"WS connected ({backend}) {ws.client}")

    vad = AdaptiveEnergyVAD(
        cfg.sample_rate,
        cfg.vad_frame_ms,
        cfg.vad_start_margin,
        cfg.vad_min_noise_rms,
        cfg.pre_speech_ms,
    )

    session = engine.new_session(max_buffer_ms=cfg.max_utt_ms)

    frame_bytes = int(cfg.sample_rate * cfg.vad_frame_ms / 1000) * 2
    raw_buf = bytearray()

    utt_started = False
    utt_audio_ms = 0
    t_utt_start = None
    t_first_partial = None
    silence_ms = 0

    try:
        while True:
            msg = await ws.receive()
            if msg["type"] == "websocket.disconnect":
                break

            data = msg.get("bytes")
            if data is None:
                continue

            if data == b"":
                if utt_started:
                    final = session.finalize(cfg.post_speech_pad_ms)
                    await _emit_final(
                        ws,
                        session,
                        final,
                        utt_audio_ms,
                        t_utt_start,
                        t_first_partial,
                        "eos",
                        utterances_total,
                        finals_total,
                        ttf_wall,
                        audio_sec_hist,
                        rtf_hist,
                        engine,
                    )
                break

            raw_buf.extend(data)

            while len(raw_buf) >= frame_bytes:
                frame = bytes(raw_buf[:frame_bytes])
                del raw_buf[:frame_bytes]

                is_speech, pre = vad.push_frame(frame)
                silence_ms = 0 if is_speech else silence_ms + cfg.vad_frame_ms

                if pre and not utt_started:
                    utt_started = True
                    utt_audio_ms = 0
                    t_utt_start = time.time()
                    t_first_partial = None
                    silence_ms = 0
                    session.accept_pcm16(pre)

                if not utt_started:
                    continue

                session.accept_pcm16(frame)
                utt_audio_ms += cfg.vad_frame_ms

                if engine.caps.partials:
                    text = session.step_if_ready()
                    if text:
                        partials_total.inc()
                        if t_first_partial is None:
                            t_first_partial = time.time()
                            if engine.caps.ttft_meaningful:
                                ttft_wall.observe(t_first_partial - t_utt_start)
                        await ws.send_text(json.dumps({"type": "partial", "text": text}))

                if (
                    not is_speech
                    and utt_audio_ms >= cfg.min_utt_ms
                    and silence_ms >= cfg.end_silence_ms
                ):
                    final = session.finalize(cfg.post_speech_pad_ms)
                    await _emit_final(
                        ws,
                        session,
                        final,
                        utt_audio_ms,
                        t_utt_start,
                        t_first_partial,
                        "silence",
                        utterances_total,
                        finals_total,
                        ttf_wall,
                        audio_sec_hist,
                        rtf_hist,
                        engine,
                    )
                    vad.reset()
                    utt_started = False
                    utt_audio_ms = 0
                    silence_ms = 0

    finally:
        active_streams.dec()
        await ws.close()
        log.info("WS disconnected")


async def _emit_final(
    ws,
    session,
    final_text,
    audio_ms,
    t_start,
    t_first_partial,
    reason,
    utterances_total,
    finals_total,
    ttf_wall,
    audio_sec_hist,
    rtf_hist,
    engine,
):
    if not final_text:
        return

    utterances_total.inc()
    finals_total.inc()

    audio_sec = audio_ms / 1000.0
    ttf = time.time() - t_start

    ttf_wall.observe(ttf)
    audio_sec_hist.observe(audio_sec)

    compute_sec = session.utt_preproc + session.utt_infer + session.utt_flush
    if audio_sec > 0:
        rtf_hist.observe(compute_sec / audio_sec)

    await ws.send_text(json.dumps({
        "type": "final",
        "text": final_text,
        "reason": reason,
        "audio_ms": audio_ms,
        "ttf_ms": int(ttf * 1000),
        "ttft_ms": (
            int((t_first_partial - t_start) * 1000)
            if t_first_partial and engine.caps.ttft_meaningful
            else None
        ),
        "chunks": session.chunks,
        "model_preproc_ms": int(session.utt_preproc * 1000),
        "model_infer_ms": int(session.utt_infer * 1000),
        "model_flush_ms": int(session.utt_flush * 1000),
        "rtf": (compute_sec / audio_sec) if audio_sec > 0 else None,
    }))


