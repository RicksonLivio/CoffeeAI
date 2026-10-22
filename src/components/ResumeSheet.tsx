import { assets } from '../api';
import { PROFILE } from '../content/profile';
import { useLang } from '../i18n';

export default function ResumeSheet() {
  const { lang } = useLang();
  const pt = lang === 'pt';
  return (
    <div className="sheet">
      <div className="section-head">
        <h2>{pt ? 'Currículo' : 'Resume'}</h2>
        <span className="label">Resume_updated_2026.pdf</span>
      </div>
      <div className="hero-actions" style={{ marginTop: 0, marginBottom: 18 }}>
        <a className="btn btn--gold" href={assets.resume} download="Rickson_Livio_Resume_2026.pdf">
          {pt ? 'Baixar PDF' : 'Download PDF'}
        </a>
        <a className="btn btn--ghost" href={assets.resume} target="_blank" rel="noopener noreferrer">
          {pt ? 'Abrir em nova aba' : 'Open in a new tab'} ↗
        </a>
        <a className="btn btn--ghost" href={PROFILE.linkedin} target="_blank" rel="noopener noreferrer">LinkedIn ↗</a>
      </div>
      <div className="panel" style={{ padding: 10 }}>
        <object className="pdf-frame" data={assets.resume} type="application/pdf" aria-label={pt ? 'Currículo em PDF' : 'Resume PDF'}>
          <p style={{ padding: 20 }}>
            {pt ? 'Seu navegador não mostra PDFs aqui. ' : 'Your browser cannot show PDFs here. '}
            <a href={assets.resume} download="Rickson_Livio_Resume_2026.pdf">{pt ? 'Baixe o arquivo.' : 'Download the file.'}</a>
          </p>
        </object>
      </div>
    </div>
  );
}
