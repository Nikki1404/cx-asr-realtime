import asyncio
import json
import time
import logging

from fastapi import FastAPI, WebSocket
from fastapi.responses import Response
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST

from app.config import load_config
from app.metrics import *
from app.vad import AdaptiveEnergyVAD
from app.factory import build_engine
from app.asr_engines.base import ASREngine

cfg = load_config()
logging.basicConfig(level=cfg.log_level)
log = logging.getLogger("asr_server")

app = FastAPI()
engine: ASREngine | None = None


@app.on_event("startup")
async def startup():
    global engine
    engine = build_engine(cfg)
    load_sec = engine.load()
    log.info(
        f"ASR backend={cfg.asr_backend} "
        f"model={cfg.model_name} "
        f"loaded in {load_sec:.2f}s"
    )


@app.get("/metrics")
async def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.websocket("/ws/asr")
async def ws_asr(ws: WebSocket):
    assert engine is not None
    await ws.accept()

    # ===========================
    # 🔑 PROMETHEUS LABEL BINDING
    # ===========================
    labels = (cfg.asr_backend, cfg.model_name)

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
    log.info(f"WS connected: {ws.client}")

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

            # EOS
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
                        reason="eos",
                        utterances_total=utterances_total,
                        finals_total=finals_total,
                        ttf_wall=ttf_wall,
                        audio_sec_hist=audio_sec_hist,
                        rtf_hist=rtf_hist,
                    )
                break

            raw_buf.extend(data)

            while len(raw_buf) >= frame_bytes:
                frame = bytes(raw_buf[:frame_bytes])
                del raw_buf[:frame_bytes]

                is_speech, pre = vad.push_frame(frame)

                silence_ms = 0 if is_speech else silence_ms + cfg.vad_frame_ms

                # ---- utterance start ----
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
                backlog_ms_gauge.set(session.backlog_ms() if hasattr(session, "backlog_ms") else 0)

                # ---- partials ----
                if engine.caps.partials:
                    text = session.step_if_ready()
                    if text:
                        partials_total.inc()
                        if t_first_partial is None:
                            t_first_partial = time.time()
                            if engine.caps.ttft_meaningful:
                                ttft_wall.observe(t_first_partial - t_utt_start)
                        await ws.send_text(json.dumps({"type": "partial", "text": text}))

                # ---- endpoint ----
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
                        reason="silence",
                        utterances_total=utterances_total,
                        finals_total=finals_total,
                        ttf_wall=ttf_wall,
                        audio_sec_hist=audio_sec_hist,
                        rtf_hist=rtf_hist,
                    )
                    vad.reset()
                    utt_started = False
                    utt_audio_ms = 0
                    silence_ms = 0

                elif utt_audio_ms >= cfg.max_utt_ms:
                    final = session.finalize(cfg.post_speech_pad_ms)
                    await _emit_final(
                        ws,
                        session,
                        final,
                        utt_audio_ms,
                        t_utt_start,
                        t_first_partial,
                        reason="max_utt",
                        utterances_total=utterances_total,
                        finals_total=finals_total,
                        ttf_wall=ttf_wall,
                        audio_sec_hist=audio_sec_hist,
                        rtf_hist=rtf_hist,
                    )
                    vad.reset()
                    utt_started = False
                    utt_audio_ms = 0
                    silence_ms = 0

    finally:
        active_streams.dec()
        try:
            await ws.close()
        except Exception:
            pass
        log.info("WS disconnected")


async def _emit_final(
    ws: WebSocket,
    session,
    final_text: str,
    audio_ms: int,
    t_start: float,
    t_first_partial: float | None,
    reason: str,
    *,
    utterances_total,
    finals_total,
    ttf_wall,
    audio_sec_hist,
    rtf_hist,
):
    if not final_text:
        return

    utterances_total.inc()
    finals_total.inc()

    audio_sec = audio_ms / 1000.0
    ttf = time.time() - t_start if t_start else 0.0

    ttf_wall.observe(ttf)
    audio_sec_hist.observe(audio_sec)

    compute_sec = session.utt_preproc + session.utt_infer + session.utt_flush
    if audio_sec > 0:
        rtf_hist.observe(compute_sec / audio_sec)

    payload = {
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
    }

    await ws.send_text(json.dumps(payload))
