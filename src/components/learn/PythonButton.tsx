import type { MouseEvent } from 'react';
import { useLang } from '../../i18n';

/**
 * Retro "defeat" burst, in the style of the classic NES Mega Man death effect:
 * glowing yellow energy orbs fly out from the center in 8 directions, in two rings
 * (the inner ring slower than the outer one), flickering while they travel.
 * Pure DOM + CSS keyframes (see .mm-* in app.css); removed from the page when it ends.
 */
export function megaBurst(x: number, y: number) {
  if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  const root = document.createElement('div');
  root.className = 'mm-burst';
  root.setAttribute('aria-hidden', 'true');
  root.style.left = `${x}px`;
  root.style.top = `${y}px`;
  const reach = Math.max(window.innerWidth, window.innerHeight);
  const flash = document.createElement('span');
  flash.className = 'mm-flash';
  root.appendChild(flash);
  [0.32, 0.62].forEach((ring) => {
    for (let k = 0; k < 8; k++) {
      const a = (k * Math.PI) / 4;
      const orb = document.createElement('span');
      orb.className = 'mm-orb';
      orb.style.setProperty('--dx', `${Math.cos(a) * reach * ring}px`);
      orb.style.setProperty('--dy', `${Math.sin(a) * reach * ring}px`);
      orb.appendChild(document.createElement('i'));
      root.appendChild(orb);
    }
  });
  document.body.appendChild(root);
  window.setTimeout(() => root.remove(), 1500);
}

export interface CodeFile {
  file: string;
  label: [string, string];
  about: [string, string];
  run: string;
}

/** Download buttons for the lesson's Python simulations; each click also plays the burst. */
export default function PythonButton({ files }: { files: CodeFile[] }) {
  const { lang } = useLang();
  const pt = lang === 'pt';
  const i = pt ? 0 : 1;
  const onClick = (e: MouseEvent<HTMLAnchorElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    megaBurst(r.left + r.width / 2, r.top + r.height / 2);
  };
  return (
    <div className="panel panel--gold" style={{ marginTop: 24 }}>
      <span className="label">{pt ? 'Para rodar no seu computador' : 'To run on your computer'}</span>
      <h3 style={{ margin: '4px 0 12px' }}>{pt ? 'Exemplo de código Python' : 'Python code example'}</h3>
      <div className="grid grid-2">
        {files.map((f) => (
          <div key={f.file} className="py-panel" style={{ marginTop: 0, alignItems: 'flex-start', flexDirection: 'column' }}>
            <p className="small" style={{ margin: 0 }}><b className="gold">{f.label[i]}</b> · {f.about[i]}</p>
            <p className="small mono muted" style={{ margin: 0 }}>{f.run}</p>
            <a className="btn btn--gold btn--sm" href={`${import.meta.env.BASE_URL}code/${f.file}`} download={f.file} onClick={onClick}>
              ⬇ {pt ? 'Exemplo de código Python' : 'Python code example'} · {f.label[i]}
            </a>
          </div>
        ))}
      </div>
    </div>
  );
}
