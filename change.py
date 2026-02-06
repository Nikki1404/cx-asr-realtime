For latency i am thinking of sending audios of increasing length.. 1 seconds, 2 seconds and so on upto 20 seconds maybe 
For pauses it will be tricky... Maybe we can synthesise audios with custom pauses
One more important thing.  We expect to receive speech in realtime mostly via mic. So let's try to use reference audios which is as close to mic as possible (w.r.t. sample rate, file formats etc )

i already have this python script 
asr_benchmark_folder.py-
import os
import time
import csv
import requests
from jiwer import wer
from tqdm import tqdm


ASR_API_URL = "http://127.0.0.1:8002/asr/upload_file"

AUDIO_DIR = "/home/CORP/re_nikitav/bu-digital-cx-asr-whiperx/inspira_audio_wav"
REF_DIR = "references"

# Save CSV in a different folder (NOT named output)
RESULTS_DIR = "asr_results"
OUTPUT_CSV = os.path.join(RESULTS_DIR, "asr_batch_results.csv")

HEADERS = {
    "debug": "yes",
    "diarization": "true",
    "min-speakers": "1",
    "max-speakers": "4"
}

TIMEOUT = 600  # seconds


def ensure_dir(path: str):
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def load_reference_text(audio_filename):
    ref_path = os.path.join(
        REF_DIR, os.path.splitext(audio_filename)[0] + ".txt"
    )
    if os.path.exists(ref_path):
        with open(ref_path, "r", encoding="utf-8") as f:
            return f.read().strip()
    return None


def transcribe_audio(audio_path):
    with open(audio_path, "rb") as f:
        start_time = time.time()

        response = requests.post(
            ASR_API_URL,
            headers=HEADERS,
            files={"file": f},
            timeout=TIMEOUT
        )

        latency_ms = int((time.time() - start_time) * 1000)

    if response.status_code != 200:
        raise RuntimeError(response.text)

    result = response.json()

    segments = result.get("response", [])

    if isinstance(segments, list):
        transcript = " ".join(
            seg.get("sentence", "")
            for seg in segments
            if isinstance(seg, dict)
        ).strip()
    else:
        transcript = str(segments).strip()

    return transcript, latency_ms


def main():
    # Ensure required directories exist
    ensure_dir(AUDIO_DIR)
    ensure_dir(REF_DIR)
    ensure_dir(RESULTS_DIR)

    audio_files = [
        f for f in os.listdir(AUDIO_DIR)
        if f.lower().endswith((".wav", ".mp3", ".flac", ".m4a"))
    ]

    if not audio_files:
        print(f" No audio files found in '{AUDIO_DIR}'")
        return

    rows = []

    print(f"\n Starting transcription for {len(audio_files)} files\n")

    for audio_file in tqdm(audio_files):
        audio_path = os.path.join(AUDIO_DIR, audio_file)

        try:
            transcript, latency_ms = transcribe_audio(audio_path)
            reference = load_reference_text(audio_file)

            # WER empty if reference missing
            wer_score = ""
            if reference:
                wer_score = round(wer(reference, transcript), 4)

            rows.append([
                audio_file,
                latency_ms,
                wer_score,
                transcript
            ])

        except Exception as e:
            rows.append([
                audio_file,
                "",
                "",
                f"ERROR: {str(e)}"
            ])

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow([
            "file_name",
            "latency_ms",
            "wer",
            "transcription"
        ])
        writer.writerows(rows)

    print("\n Batch completed")
    print(f" Results saved to: {OUTPUT_CSV}\n")


if __name__ == "__main__":
    main()

whisperx_benchmark.py-
import time
import csv
import requests
from pathlib import Path
import jiwer
from concurrent.futures import ThreadPoolExecutor, as_completed


BASE_DIR = Path(__file__).resolve().parents[1]  # utils/

DATA_DIR = BASE_DIR / "datasets" / "data" / "wav"
RAW_LIBRISPEECH_DIR = BASE_DIR / "datasets" / "data" / "raw" / "LibriSpeech"


ASR_ENDPOINT = "http://127.0.0.1:8002/asr/upload_file"
OUTPUT_CSV = Path("/tmp/whisperx_benchmark_results.csv")

MAX_WORKERS = 4  # CPU: 2–4 | GPU: 2–4

MAX_FILES = None

transform = jiwer.Compose([
    jiwer.ToLowerCase(),
    jiwer.RemovePunctuation(),
    jiwer.RemoveMultipleSpaces(),
    jiwer.Strip(),
    jiwer.ReduceToListOfListOfWords(word_delimiter=" ")
])


def get_reference_text(wav_path: Path) -> str:
    """
    Extract reference transcription from LibriSpeech chapter-level *.trans.txt
    """

    utt_id = wav_path.stem

    # Path relative to wav root
    rel = wav_path.relative_to(DATA_DIR)
    subset = rel.parts[0]        # dev-clean / dev-other / test-clean / test-other
    speaker_id = rel.parts[1]
    chapter_id = rel.parts[2]

    trans_file = (
        RAW_LIBRISPEECH_DIR
        / subset
        / speaker_id
        / chapter_id
        / f"{speaker_id}-{chapter_id}.trans.txt"
    )

    if not trans_file.exists():
        return ""

    with open(trans_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith(utt_id):
                return line.strip().split(" ", 1)[1]

    return ""


def transcribe_via_api(wav_path: Path):
    """
    Always hits WhisperX API.
    Returns transcription and latency.
    """
    start = time.time()

    with open(wav_path, "rb") as f:
        response = requests.post(
            ASR_ENDPOINT,
            headers={
                "debug": "yes",
                "diarization": "true",
                "min-speakers": "1",
                "max-speakers": "4",
            },
            files={"file": f},
            timeout=300,
        )

    latency = time.time() - start
    response.raise_for_status()

    data = response.json()

    transcription = " ".join(
        seg["sentence"] for seg in data.get("response", [])
    )

    return transcription, latency


def process_single_wav(wav_path: Path):
    """
    1) Always call ASR API
    2) Then try reference lookup
    3) Compute WER only if reference exists
    """

    hyp_text, latency = transcribe_via_api(wav_path)

    ref_text = get_reference_text(wav_path)

    if ref_text:
        wer = jiwer.wer(
            ref_text,
            hyp_text,
            reference_transform=transform,
            hypothesis_transform=transform,
        )
        wer = round(wer, 4)
    else:
        wer = None

    subset = wav_path.relative_to(DATA_DIR).parts[0]

    return [
        subset,
        wav_path.name,
        ref_text,
        hyp_text,
        round(latency, 2),
        wer,
    ]

def main():
    if not DATA_DIR.exists():
        raise FileNotFoundError(f"WAV directory not found: {DATA_DIR}")

    if not RAW_LIBRISPEECH_DIR.exists():
        raise FileNotFoundError(f"LibriSpeech raw directory not found: {RAW_LIBRISPEECH_DIR}")

    # 🔴 UPDATED: limit to first 30 WAV files
    wav_files = list(DATA_DIR.rglob("*.wav"))[:MAX_FILES]
    print(f"Benchmarking {len(wav_files)} WAV files (limit={MAX_FILES})")

    results = []

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {
            executor.submit(process_single_wav, wav): wav
            for wav in wav_files
        }

        for future in as_completed(futures):
            wav = futures[future]
            try:
                row = future.result()
                results.append(row)
                print(f"[DONE] {wav.name}")
            except Exception as e:
                print(f"[ERROR] {wav}: {e}")

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "subset",
            "file",
            "reference_text",
            "predicted_text",
            "latency_sec",
            "wer",
        ])
        writer.writerows(results)

    print("\nParallel benchmark completed")
    print(f"Results saved to: {OUTPUT_CSV}")
    print(f"Total processed files: {len(results)}")

if __name__ == "__main__":
    main()
i have dataset in this manner 
e.g - 
"C:\Users\re_nikitav\Documents\utils\utils\datasets\data\wav"
"C:\Users\re_nikitav\Documents\utils\utils\datasets\data\wav\test-clean"
"C:\Users\re_nikitav\Documents\utils\utils\datasets\data\wav\test-other"
"C:\Users\re_nikitav\Documents\utils\utils\datasets\data\wav\dev-clean"
"C:\Users\re_nikitav\Documents\utils\utils\datasets\data\wav\dev-other"
and all these have multiple sub-folders 
example -
"C:\Users\re_nikitav\Documents\utils\utils\datasets\data\wav\dev-clean\84\121123\84-121123-0000.wav"

and we have reference text according to audio file name 
so we have reference text in this manner 
"C:\Users\re_nikitav\Documents\utils\utils\datasets\data\raw\LibriSpeech\dev-clean\84\121123"

which has transcriptions for all audio files available in 121123 folder in one txt file -
"C:\Users\re_nikitav\Documents\utils\utils\datasets\data\raw\LibriSpeech\dev-clean\84\121123\84-121123.trans.txt"
84-121123-0000 GO DO YOU HEAR
84-121123-0001 BUT IN LESS THAN FIVE MINUTES THE STAIRCASE GROANED BENEATH AN EXTRAORDINARY WEIGHT
84-121123-0002 AT THIS MOMENT THE WHOLE SOUL OF THE OLD MAN SEEMED CENTRED IN HIS EYES WHICH BECAME BLOODSHOT THE VEINS OF THE THROAT SWELLED HIS CHEEKS AND TEMPLES BECAME PURPLE AS THOUGH HE WAS STRUCK WITH EPILEPSY NOTHING WAS WANTING TO COMPLETE THIS BUT THE UTTERANCE OF A CRY
84-121123-0003 AND THE CRY ISSUED FROM HIS PORES IF WE MAY THUS SPEAK A CRY FRIGHTFUL IN ITS SILENCE
84-121123-0004 D'AVRIGNY RUSHED TOWARDS THE OLD MAN AND MADE HIM INHALE A POWERFUL RESTORATIVE
84-121123-0005 D'AVRIGNY UNABLE TO BEAR THE SIGHT OF THIS TOUCHING EMOTION TURNED AWAY AND VILLEFORT WITHOUT SEEKING ANY FURTHER EXPLANATION AND ATTRACTED TOWARDS HIM BY THE IRRESISTIBLE MAGNETISM WHICH DRAWS US TOWARDS THOSE WHO HAVE LOVED THE PEOPLE FOR WHOM WE MOURN EXTENDED HIS HAND TOWARDS THE YOUNG MAN
84-121123-0006 FOR SOME TIME NOTHING WAS HEARD IN THAT CHAMBER BUT SOBS EXCLAMATIONS AND PRAYERS
84-121123-0007 WHAT DO YOU MEAN SIR
84-121123-0008 OH YOU RAVE SIR EXCLAIMED VILLEFORT IN VAIN ENDEAVORING TO ESCAPE THE NET IN WHICH HE WAS TAKEN I RAVE
84-121123-0009 DO YOU KNOW THE ASSASSIN ASKED MORREL
84-121123-0010 NOIRTIER LOOKED UPON MORREL WITH ONE OF THOSE MELANCHOLY SMILES WHICH HAD SO OFTEN MADE VALENTINE HAPPY AND THUS FIXED HIS ATTENTION
84-121123-0011 SAID MORREL SADLY YES REPLIED NOIRTIER
84-121123-0012 THE OLD MAN'S EYES REMAINED FIXED ON THE DOOR
84-121123-0013 ASKED MORREL YES
84-121123-0014 MUST I LEAVE ALONE NO
84-121123-0015 BUT CAN HE UNDERSTAND YOU YES
84-121123-0016 GENTLEMEN HE SAID IN A HOARSE VOICE GIVE ME YOUR WORD OF HONOR THAT THIS HORRIBLE SECRET SHALL FOREVER REMAIN BURIED AMONGST OURSELVES THE TWO MEN DREW BACK
84-121123-0017 MY FATHER HAS REVEALED THE CULPRIT'S NAME MY FATHER THIRSTS FOR REVENGE AS MUCH AS YOU DO YET EVEN HE CONJURES YOU AS I DO TO KEEP THIS SECRET DO YOU NOT FATHER
84-121123-0018 MORREL SUFFERED AN EXCLAMATION OF HORROR AND SURPRISE TO ESCAPE HIM
84-121123-0019 THE OLD MAN MADE A SIGN IN THE AFFIRMATIVE
84-121123-0020 IT WAS SOMETHING TERRIBLE TO WITNESS THE SILENT AGONY THE MUTE DESPAIR OF NOIRTIER WHOSE TEARS SILENTLY ROLLED DOWN HIS CHEEKS
84-121123-0021 BUT HE STOPPED ON THE LANDING HE HAD NOT THE COURAGE TO AGAIN VISIT THE DEATH CHAMBER
84-121123-0022 THE TWO DOCTORS THEREFORE ENTERED THE ROOM ALONE
84-121123-0023 NOIRTIER WAS NEAR THE BED PALE MOTIONLESS AND SILENT AS THE CORPSE
84-121123-0024 THE DISTRICT DOCTOR APPROACHED WITH THE INDIFFERENCE OF A MAN ACCUSTOMED TO SPEND HALF HIS TIME AMONGST THE DEAD HE THEN LIFTED THE SHEET WHICH WAS PLACED OVER THE FACE AND JUST UNCLOSED THE LIPS
84-121123-0025 THE NEAREST SAID THE DISTRICT DOCTOR IS A GOOD ITALIAN ABBE WHO LIVES NEXT DOOR TO YOU SHALL I CALL ON HIM AS I PASS
84-121123-0026 D'AVRIGNY SAID VILLEFORT BE SO KIND I BESEECH YOU AS TO ACCOMPANY THIS GENTLEMAN HERE IS THE KEY OF THE DOOR SO THAT YOU CAN GO IN AND OUT AS YOU PLEASE YOU WILL BRING THE PRIEST WITH YOU AND WILL OBLIGE ME BY INTRODUCING HIM INTO MY CHILD'S ROOM DO YOU WISH TO SEE HIM
84-121123-0027 I ONLY WISH TO BE ALONE YOU WILL EXCUSE ME WILL YOU NOT
84-121123-0028 I AM GOING SIR AND I DO NOT HESITATE TO SAY THAT NO PRAYERS WILL BE MORE FERVENT THAN MINE


so firstly go through this ans understand and this is for now asr offline we were using for benchmarking 
