"""Generates Misty's audio lessons from src/content/misty-lesson{1,2,3}.json and checks every number.

Voice: Qwen3-TTS 1.7B CustomVoice (Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice, Apache 2.0), built-in speaker "vivian".
It is a stock voice, not a clone of Misty's voice actress.

Each paragraph is synthesized on its own and transcribed back with Whisper (large-v3-turbo). The paragraph is
accepted only if every number word of the text comes back unchanged (and the transcript is close to the text);
otherwise it is generated again with another seed. Accepted paragraphs are cached in ~/tts-qwen/work, so a
rerun only redoes what is missing or was edited.

Setup, outside the repo (once, needs an NVIDIA GPU with ~6 GB free):
    uv venv -p 3.11 ~/tts-qwen/.venv
    uv pip install -p ~/tts-qwen/.venv/bin/python torch torchaudio --index-url https://download.pytorch.org/whl/cu124
    uv pip install -p ~/tts-qwen/.venv/bin/python qwen-tts soundfile openai-whisper num2words
Run from the repo root (needs ffmpeg):
    ~/tts-qwen/.venv/bin/python scripts/misty_tts.py              # lesson 1, both languages
    ~/tts-qwen/.venv/bin/python scripts/misty_tts.py --lesson 2   # lesson 2; a language (pt, en) can follow
    ~/tts-qwen/.venv/bin/python scripts/misty_tts.py --lesson 3   # lesson 3 (the CNN)

Writes public/audio/misty_lesson{N}_{pt,en}.mp3 (mono, 48 kbit/s), src/content/misty-lesson{N}-audio.json (duration and
start of each section) and ~/tts-qwen/work/report_{lang}.json (lesson 1) or report_l{N}_{lang}.json (transcript and
check of every paragraph).
"""
import difflib
import gc
import hashlib
import json
import pathlib
import re
import subprocess
import sys
import tempfile
import unicodedata

import numpy as np
import soundfile as sf
import torch
from num2words import num2words

ROOT = pathlib.Path(__file__).resolve().parent.parent
OUT = ROOT / 'public/audio'
WORK = pathlib.Path.home() / 'tts-qwen/work'
LESSON = 1
TITLES = {1: 'Misty explains the perceptron (CoffeeAI, Lesson 1)', 2: 'Misty explains the MLP (CoffeeAI, Lesson 2)',
          3: 'Misty explains the CNN (CoffeeAI, Lesson 3)'}


def src_path() -> pathlib.Path:
    return ROOT / f'src/content/misty-lesson{LESSON}.json'


def meta_path() -> pathlib.Path:
    return ROOT / f'src/content/misty-lesson{LESSON}-audio.json'


def report_file(lang: str) -> pathlib.Path:
    # lesson 1 keeps its original report name, so its cache stays valid
    return WORK / (f'report_{lang}.json' if LESSON == 1 else f'report_l{LESSON}_{lang}.json')

MODEL = 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice'
SPEAKER = 'vivian'
LANGS = {'pt': ('Portuguese', 'pt_BR', 'Fale de forma calorosa, animada e conversada.'),
         'en': ('English', 'en', 'Warm, upbeat and conversational.')}
WHISPER = 'large-v3-turbo'
MAX_ROUNDS = 8
MAX_WER = 0.08
PAUSE_PARAGRAPH = 0.6
PAUSE_SECTION = 1.2
MAX_GAP = 0.45  # longer silences inside a paragraph are shortened to this


def squeeze(a: np.ndarray, sr: int) -> np.ndarray:
    """Shortens silences longer than MAX_GAP inside a paragraph (the voice sometimes pauses ~1 s mid-sentence)."""
    hop = int(sr * 0.02)
    frames = len(a) // hop
    rms = np.sqrt(np.mean(a[: frames * hop].reshape(frames, hop) ** 2, axis=1))
    quiet = rms < 0.03 * rms.max()
    keep = np.ones(len(a), dtype=bool)
    i, limit = 0, int(MAX_GAP / 0.02)
    while i < frames:
        if quiet[i]:
            j = i
            while j < frames and quiet[j]:
                j += 1
            if j - i > limit:
                cut0 = (i + limit // 2) * hop
                cut1 = (j - limit // 2) * hop
                keep[cut0:cut1] = False
            i = j
        else:
            i += 1
    return a[keep]

# ---------- what the voice reads: numbers and symbols written out, so nothing is left to guess ----------

LETTER_VARS = {'pt': {'y'}, 'en': set()}  # variable letters that must not go missing
LETTER = {'pt': {'x': 'xis', 'w': 'dáblio', 'h': 'agá'}, 'en': {'x': 'x', 'w': 'w', 'h': 'h'}}


def say_number(tok: str, lang: str) -> str:
    code = LANGS[lang][1]
    if lang == 'pt':
        v = tok.replace('.', '').replace(',', '.')
    else:
        v = tok.replace(',', '')
    num = float(v) if '.' in v else int(v)
    if lang == 'en' and isinstance(num, int) and 1900 <= num <= 2099:
        words = num2words(num, to='year')
    elif lang == 'pt' and isinstance(num, float):
        # "0,22" → "zero vírgula vinte e dois" (natural in Portuguese; digit by digit, "dois dois", is ambiguous)
        whole, frac = v.split('.')
        lead = len(frac) - len(frac.lstrip('0'))
        words = ' '.join([num2words(int(whole), lang=code), 'vírgula'] + ['zero'] * lead +
                         ([num2words(int(frac), lang=code)] if frac.strip('0') else []))
    else:
        words = num2words(num, lang=code)
    return words.replace(',', '').replace('-', ' ')


NUM_PT = r'\d{1,3}(?:\.\d{3})+(?:,\d+)?|\d+(?:,\d+)?'
NUM_EN = r'\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?'


def spoken(text: str, lang: str) -> str:
    """Text as the voice reads it (and as the check compares it): x1 → 'xis um', 15600,5 → words."""
    def var(m):
        return f"{LETTER[lang][m.group(1).lower()]} {say_number(m.group(2), lang)}"
    text = re.sub(r'\b([xwhXWH])\s*([0-9])\b', var, text)
    if lang == 'pt':
        text = re.sub(r'(?<![\w-])y(?![\w-])', 'ípsilon', text)  # the Portuguese voice tends to swallow a bare "y"
    return re.sub(NUM_PT if lang == 'pt' else NUM_EN, lambda m: say_number(m.group(0), lang), text)


# ---------- the check ----------

POKEMON = {'charmander', 'charmeleon', 'charizard', 'squirtle', 'wartortle', 'blastoise', 'bulbasaur', 'ivysaur', 'venusaur',
           'orbeetle', 'ekans', 'koffing'}

SUB = str.maketrans('₀₁₂₃₄₅₆₇₈₉', '0123456789')


def normalize(text: str, lang: str) -> list[str]:
    # Whisper writes "6 by 6" as "6x6" and "32 minus 5" as "32-5": read them back as words
    text = re.sub(r'(\d)\s*[x×]\s*(\d)', r'\1 ' + ('por' if lang == 'pt' else 'by') + r' \2', text)
    text = re.sub(r'(\d)\s*-\s*(\d)', r'\1 ' + ('menos' if lang == 'pt' else 'minus') + r' \2', text)
    text = text.translate(SUB).replace('θ', ' theta ').replace('η', ' eta ').replace('−', ' menos ' if lang == 'pt' else ' minus ')
    text = text.replace('%', ' por cento ' if lang == 'pt' else ' percent ')
    text = spoken(text, lang).lower()
    text = ''.join(c for c in unicodedata.normalize('NFD', text) if unicodedata.category(c) != 'Mn')
    text = re.sub(r"[^a-z0-9' ]", ' ', text)
    words = text.split()
    alias = {'pt': {'teta': 'theta', 'ipsilon': 'y', 'de': 'de', 'dablio': 'dablio', 'w': 'dablio', 'x': 'xis', 'h': 'aga', 'pokebola': 'pokebola'},
             'en': {'zeta': 'theta'}}[lang]
    return [alias.get(w, w) for w in words]


def number_vocab(lang: str) -> set[str]:
    code = LANGS[lang][1]
    vocab: set[str] = set()
    for v in list(range(0, 21)) + [30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1000000, 0.5]:
        vocab.update(num2words(v, lang=code).replace(',', '').replace('-', ' ').split())
    vocab.update(['virgula', 'point', 'menos', 'minus', 'mais', 'plus', 'e', 'and', 'cento', 'mil', 'thousand', 'hundred'])
    ignore = {'e', 'and', 'um', 'uma', 'a', 'one', 'mais', 'plus'}  # articles / joiners: only substitutions count
    vocab = {''.join(c for c in unicodedata.normalize('NFD', w) if unicodedata.category(c) != 'Mn') for w in vocab}
    return vocab, ignore


def check(expected: str, heard: str, lang: str, edges: bool = True) -> dict:
    a, b = normalize(expected, lang), normalize(heard, lang)
    vocab, ignore = number_vocab(lang)
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    errors, edits = [], 0
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == 'equal':
            continue
        exp, got = a[i1:i2], b[j1:j2]
        # Whisper spells Pokémon names its own way ("Ivisal", "Squirtail"): a replaced name is not a reading error
        names = sum(w in POKEMON for w in exp) if op == 'replace' else 0
        edits += max(len(exp) - names, len(got) - names, 0)
        # a number word changed (articles/joiners like "um", "e", "and" only count next to another number word)
        touched = [w for w in exp + got if w in vocab and w not in ignore]
        # a name heard as pieces ("Blastoise" → "blast dois"): the sound matches the name, no number was said
        if exp and all(w in POKEMON for w in exp) and difflib.SequenceMatcher(a=''.join(exp), b=''.join(got)).ratio() >= 0.6:
            touched = []
        touched += [w for w in exp if w in LETTER_VARS[lang]]
        if touched:
            errors.append(f"{' '.join(exp) or '∅'} → {' '.join(got) or '∅'}")
    ops = [o for o in sm.get_opcodes()]
    # an extra word before the first or after the last expected word is a sound the voice made up ("Tchutzi!")
    edge = [o for o in (ops[0], ops[-1]) if o[0] == 'insert'] if ops and edges else []
    if edge:
        errors.append('extra at edge: ' + ' / '.join(' '.join(b[o[3]:o[4]]) for o in edge))
    wer = edits / max(1, len(a))
    return {'ok': not errors and wer <= MAX_WER, 'wer': round(wer, 3), 'number_errors': errors}


# ---------- pipeline ----------

def paragraphs(lang: str):
    li = list(LANGS).index(lang)
    for s in json.loads(src_path().read_text(encoding='utf-8'))['sections']:
        yield s['id'], s['text'][li]


def key(text: str) -> str:
    return hashlib.sha1(f'{MODEL}|{SPEAKER}|{text}'.encode()).hexdigest()[:12]


def item_key(text: str, lang: str) -> str:
    """Cache key of a paragraph: changes when the text or the way it is read changes."""
    return key(spoken(text, lang))


def run(lang: str) -> dict:
    name, _, instruct = LANGS[lang]
    work = WORK / lang
    work.mkdir(parents=True, exist_ok=True)
    report_path = report_file(lang)
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    items = [(sid, j, p) for sid, ps in paragraphs(lang) for j, p in enumerate(ps)]
    def cached_ok(p: str) -> bool:
        r = report.get(item_key(p, lang))
        if not r or not (work / f'{item_key(p, lang)}.wav').exists():
            return False
        r.update(check(p, r['heard'], lang))  # re-check stored transcripts when the rules change
        return r['ok']
    pending = [it for it in items if not cached_ok(it[2])]
    for rnd in range(1, MAX_ROUNDS + 1):
        if not pending:
            break
        print(f'[{lang}] round {rnd}: generating {len(pending)} paragraph(s)', flush=True)
        from qwen_tts import Qwen3TTSModel
        tts = Qwen3TTSModel.from_pretrained(MODEL, device_map='cuda:0', dtype=torch.bfloat16)
        for sid, j, p in pending:
            torch.manual_seed(1000 * rnd + 17 * j + len(p))
            wavs, sr = tts.generate_custom_voice(text=spoken(p, lang), language=name, speaker=SPEAKER, instruct=instruct)
            sf.write(work / f'{item_key(p, lang)}.wav', wavs[0], sr)
        del tts
        gc.collect()
        torch.cuda.empty_cache()

        import whisper
        asr = whisper.load_model(WHISPER)
        still = []
        for sid, j, p in pending:
            heard = asr.transcribe(str(work / f'{item_key(p, lang)}.wav'), language=lang, temperature=0)['text'].strip()
            res = check(p, heard, lang)
            report[item_key(p, lang)] = {'section': sid, 'paragraph': j, 'round': rnd, 'heard': heard, **res}
            mark = 'ok ' if res['ok'] else 'BAD'
            print(f'  {mark} {sid}#{j} wer={res["wer"]} {res["number_errors"]}', flush=True)
            if not res['ok']:
                still.append((sid, j, p))
        del asr
        gc.collect()
        torch.cuda.empty_cache()
        report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
        pending = still
    if pending:
        sys.exit(f'[{lang}] {len(pending)} paragraph(s) still fail after {MAX_ROUNDS} rounds; see {report_path}')

    # assemble
    chunks, starts, spans, total, sr = [], [], [], 0, None
    last = None
    for sid, j, p in items:
        if sid != last:
            if last is not None:
                pad = np.zeros(int(sr * (PAUSE_SECTION - PAUSE_PARAGRAPH)), dtype=np.float32)
                chunks.append(pad)
                total += len(pad)
            starts.append(round(total / sr, 2) if sr else 0.0)
            last = sid
        a, sr = sf.read(work / f'{item_key(p, lang)}.wav', dtype='float32')
        if a.ndim > 1:
            a = a.mean(axis=1)
        a = squeeze(a, sr)
        spans.append((total / sr, (total + len(a)) / sr))
        chunks += [a, np.zeros(int(sr * PAUSE_PARAGRAPH), dtype=np.float32)]
        total += len(a) + int(sr * PAUSE_PARAGRAPH)
    wav = np.concatenate(chunks)
    wav = 0.9 * wav / max(1e-6, float(np.abs(wav).max()))
    mp3 = OUT / f'misty_lesson{LESSON}_{lang}.mp3'
    with tempfile.NamedTemporaryFile(suffix='.wav') as tmp:
        sf.write(tmp.name, wav, sr)
        subprocess.run(['ffmpeg', '-y', '-loglevel', 'error', '-i', tmp.name, '-ac', '1', '-b:a', '48k',
                        '-metadata', f'title={TITLES[LESSON]}',
                        '-metadata', f'comment=Qwen3-TTS CustomVoice, speaker {SPEAKER}', str(mp3)], check=True)
    final = verify_mp3(mp3, lang, starts, len(wav) / sr, spans)
    checked = [report[item_key(p, lang)] for _, _, p in items]
    print(f'{lang}: {len(wav) / sr / 60:.2f} min, {mp3.stat().st_size // 1024} KiB, {len(checked)} paragraphs checked, '
          f'max wer {max(c["wer"] for c in checked)}', flush=True)
    return {'duration': round(len(wav) / sr, 2), 'starts': starts,
            'checked': {'paragraphs': len(checked), 'number_errors': 0, 'max_wer': max(c['wer'] for c in checked),
                        'final_mp3_sections_ok': final}}


def verify_mp3(mp3: pathlib.Path, lang: str, starts: list[float], duration: float, spans: list[tuple[float, float]]) -> int:
    """Transcribes the finished MP3 chapter by chapter and checks every number of the chapter again.

    On a long chapter Whisper sometimes skips a sentence that is in the audio; then the chapter is transcribed again
    paragraph by paragraph (still from the finished MP3, using the paragraph positions) and passes only if all do."""
    import whisper
    asr = whisper.load_model(WHISPER)
    secs = list(paragraphs(lang))
    audio = whisper.load_audio(str(mp3))
    ok, first = 0, 0
    for k, (sid, ps) in enumerate(secs):
        end = starts[k + 1] if k + 1 < len(starts) else duration
        piece = audio[int(starts[k] * 16000): int(end * 16000)]
        heard = asr.transcribe(piece, language=lang, temperature=0, condition_on_previous_text=False)['text'].strip()
        # edge sounds are checked per paragraph; here Whisper may invent words over the silence at the cut
        res = check(' '.join(ps), heard, lang, edges=False)
        print(f'  final {lang} {sid}: {"ok" if not res["number_errors"] else "BAD"} wer={res["wer"]} {res["number_errors"]}', flush=True)
        if res['number_errors']:
            bad = []
            for j, p in enumerate(ps):
                t0, t1 = spans[first + j]
                piece = audio[max(0, int((t0 - 0.2) * 16000)): int((t1 + 0.2) * 16000)]
                heard = asr.transcribe(piece, language=lang, temperature=0, condition_on_previous_text=False)['text'].strip()
                r = check(p, heard, lang, edges=False)
                print(f'    paragraph {sid}#{j}: {"ok" if not r["number_errors"] else "BAD"} {r["number_errors"]}', flush=True)
                bad += r['number_errors']
            if bad:
                sys.exit(f'[{lang}] final MP3 check failed in {sid}: {bad}')
        first += len(ps)
        ok += 1
    del asr
    gc.collect()
    torch.cuda.empty_cache()
    return ok


def main() -> None:
    global LESSON
    args = sys.argv[1:]
    if args[:1] == ['--lesson']:
        LESSON = int(args[1])
        args = args[2:]
    if LESSON not in TITLES:
        sys.exit(f'unknown lesson {LESSON}')
    OUT.mkdir(parents=True, exist_ok=True)
    meta_file = meta_path()
    meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}
    meta.update({'model': MODEL, 'speaker': SPEAKER, 'asr_check': f'whisper {WHISPER}'})
    meta.pop('voices', None)
    for lang in (args or list(LANGS)):
        meta[lang] = run(lang)
        meta_file.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    main()
