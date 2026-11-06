import { useLang } from '../../i18n';
import script from '../../content/misty-lesson1.json';

function clock(s: number): string {
  const t = Math.max(0, Math.floor(s));
  return `${Math.floor(t / 60)}:${String(t % 60).padStart(2, '0')}`;
}

/** Chapters (click to jump) and the full text the audio reads, with the current chapter highlighted. */
export default function MistyTranscript({ current, starts, onSeek }: { current: number; starts: number[]; onSeek: (i: number) => void }) {
  const { lang } = useLang();
  const i = lang === 'pt' ? 0 : 1;
  return (
    <>
      <ol className="misty-chapters">
        {script.sections.map((s, k) => (
          <li key={s.id}>
            <button className={k === current ? 'on' : ''} onClick={() => onSeek(k)}>
              <span className="mono">{clock(starts[k] ?? 0)}</span> {s.title[i]}
            </button>
          </li>
        ))}
      </ol>
      <details className="xnote">
        <summary>{lang === 'pt' ? 'Texto completo do áudio' : 'Full text of the audio'}</summary>
        <div className="xbody misty-transcript">
          {script.sections.map((s, k) => (
            <div key={s.id} className={k === current ? 'on' : ''}>
              <h4>{s.title[i]}</h4>
              {s.text[i].map((p, j) => <p key={j}>{p}</p>)}
            </div>
          ))}
        </div>
      </details>
    </>
  );
}
