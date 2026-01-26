import argparse
import os
import uvicorn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["nemotron", "whisper"], default=os.getenv("ASR_BACKEND", "nemotron"))
    p.add_argument("--host", default=os.getenv("HOST", "0.0.0.0"))
    p.add_argument("--port", type=int, default=int(os.getenv("PORT", "8000")))
    p.add_argument("--workers", type=int, default=int(os.getenv("WORKERS", "1")))
    args = p.parse_args()

    os.environ["ASR_BACKEND"] = args.backend

    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
