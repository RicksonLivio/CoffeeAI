"""Generates Misty's audio lesson (Lesson 1) from src/content/misty-lesson1.json.

Model: Kokoro-82M (hexgrad/Kokoro-82M, Apache 2.0), voices pf_dora (pt-BR) and af_heart (en-US).
It is a stock voice, not a clone of Misty's voice actress.

Setup, outside the repo (once):
    uv venv -p 3.11 ~/tts-kokoro/.venv
    uv pip install -p ~/tts-kokoro/.venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu
    uv pip install -p ~/tts-kokoro/.venv/bin/python "kokoro>=0.9.4" soundfile pip
Run from the repo root (needs ffmpeg):
    ~/tts-kokoro/.venv/bin/python scripts/misty_tts.py        # or: ... misty_tts.py pt

Writes public/audio/misty_lesson1_{pt,en}.mp3 (mono, 48 kbit/s) and src/content/misty-lesson1-audio.json with
the duration and the start time of each section, used by the page to jump to a section.
"""
import json
import pathlib
import re
import sys
import subprocess
import tempfile

import numpy as np
import soundfile as sf
from kokoro import KPipeline

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / 'src/content/misty-lesson1.json'
OUT = ROOT / 'public/audio'
SR = 24000
VOICES = {'pt': ('p', 'pf_dora'), 'en': ('a', 'af_heart')}
# Portuguese is read a little slower, so the calculations land at about the same pace as in English.
SPEED = {'pt': 0.92, 'en': 1.0}
PAUSE_PARAGRAPH = 0.6
# What the voice reads differs slightly from the transcript on the page:
# pt: "0,5" would be read "zero, cinco" (sounds like two numbers), so the decimal comma becomes "vírgula";
# en: fixed phonemes (Kokoro's [word](/phonemes/) syntax) for names the English G2P gets wrong.
EN_NAMES = {
    'Blastoise': '/blˈæstˌɔɪz/', 'Pokédex': '/pˈOkədˌɛks/', 'Poké Ball': '/pˈOki bˈɔl/',
    'Pokémon': '/pˈOkimˌɑn/', 'Venusaur': '/vˈinəsˌɔɹ/',
}


def spoken(text: str, lang: str) -> str:
    if lang == 'pt':
        return re.sub(r'(\d),(\d)', r'\1 vírgula \2', text)
    for name, ps in EN_NAMES.items():
        text = re.sub(rf'\b{re.escape(name)}\b', f'[{name}]({ps})', text)
    return text

PAUSE_SECTION = 1.2


def silence(seconds: float) -> np.ndarray:
    return np.zeros(int(SR * seconds), dtype=np.float32)


def main() -> None:
    sections = json.loads(SRC.read_text(encoding='utf-8'))['sections']
    OUT.mkdir(parents=True, exist_ok=True)
    meta_path = ROOT / 'src/content/misty-lesson1-audio.json'
    meta: dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({'model': 'hexgrad/Kokoro-82M', 'voices': {k: v[1] for k, v in VOICES.items()}})
    only = sys.argv[1:] or list(VOICES)  # e.g. `misty_tts.py pt` regenerates only Portuguese
    for li, (lang, (code, voice)) in enumerate(VOICES.items()):
        if lang not in only:
            continue
        pipe = KPipeline(lang_code=code, repo_id='hexgrad/Kokoro-82M')
        chunks: list[np.ndarray] = []
        starts: list[float] = []
        total = 0
        for s in sections:
            starts.append(round(total / SR, 2))
            for p in s['text'][li]:
                for _, _, audio in pipe(spoken(p, lang), voice=voice, speed=SPEED[lang], split_pattern=r'\n+'):
                    a = audio.numpy().astype(np.float32)
                    chunks.append(a)
                    total += len(a)
                pause = silence(PAUSE_PARAGRAPH)
                chunks.append(pause)
                total += len(pause)
            pause = silence(PAUSE_SECTION - PAUSE_PARAGRAPH)
            chunks.append(pause)
            total += len(pause)
        wav = np.concatenate(chunks)
        wav = 0.9 * wav / max(1e-6, float(np.abs(wav).max()))
        mp3 = OUT / f'misty_lesson1_{lang}.mp3'
        with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
            sf.write(tmp.name, wav, SR)
            subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp.name, '-ac', '1', '-b:a', '48k',
                            '-metadata', 'title=Misty explains the perceptron (CoffeeAI, Lesson 1)',
                            '-metadata', f'comment=Kokoro-82M, voice {voice}', str(mp3)], check=True)
        meta[lang] = {'duration': round(len(wav) / SR, 2), 'starts': starts}
        print(lang, voice, f'{len(wav) / SR / 60:.2f} min', mp3.stat().st_size // 1024, 'KiB')
    meta_path.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
