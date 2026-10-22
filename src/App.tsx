import { useEffect, useState } from 'react';
import { assets } from './api';
import LearnSheet from './components/LearnSheet';
import PortfolioSheet from './components/PortfolioSheet';
import ResumeSheet from './components/ResumeSheet';
import { useLang } from './i18n';
import { PROFILE } from './content/profile';

export type Tab = 'portfolio' | 'learn' | 'resume';
const TABS: Tab[] = ['portfolio', 'learn', 'resume'];

// Hash routing (#portfolio, #learn, #resume): works on GitHub Pages with no server rewrite rules.
function tabFromHash(): Tab {
  const h = window.location.hash.replace('#', '') as Tab;
  return TABS.includes(h) ? h : 'portfolio';
}

export default function App() {
  const { lang, toggle } = useLang();
  const [tab, setTabState] = useState<Tab>(tabFromHash);

  useEffect(() => {
    const onHash = () => setTabState(tabFromHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  function setTab(t: Tab) {
    if (t !== tab) window.location.hash = t;
    setTabState(t);
    window.scrollTo({ top: 0 });
  }

  const label: Record<Tab, [string, string]> = {
    portfolio: ['01', lang === 'pt' ? 'Portfólio' : 'Portfolio'],
    learn: ['02', lang === 'pt' ? 'Aprenda comigo' : 'Learn with me'],
    resume: ['03', lang === 'pt' ? 'Currículo' : 'Resume'],
  };

  return (
    <>
      <header className="topbar">
        <div className="topbar-inner">
          <a className="brand" href="#portfolio" onClick={(e) => { e.preventDefault(); setTab('portfolio'); }}>
            <img className="brand-logo" src={`${import.meta.env.BASE_URL}favicon.svg`} alt="" />
            <span>
              <span className="brand-name">CoffeeAI</span>
              <span className="label">{PROFILE.short} · Cyber-Net</span>
            </span>
          </a>
          <nav className="tabs" aria-label={lang === 'pt' ? 'Seções' : 'Sections'}>
            {TABS.map((t) => (
              <button key={t} className={`tab ${tab === t ? 'active' : ''}`} aria-current={tab === t ? 'page' : undefined} onClick={() => setTab(t)}>
                <span className="key">{label[t][0]}</span>
                {label[t][1]}
              </button>
            ))}
          </nav>
          <div className="status">
            <a className="btn btn--gold btn--sm" href={assets.resume} download="Rickson_Livio_Resume_2026.pdf">
              ⬇ {lang === 'pt' ? 'Currículo' : 'Resume'}
            </a>
            <button className="btn btn--ghost btn--sm" onClick={toggle} aria-label={lang === 'pt' ? 'Switch to English' : 'Mudar para português'}>
              {lang === 'pt' ? 'EN' : 'PT'}
            </button>
          </div>
        </div>
      </header>

      <main className="shell">
        {tab === 'portfolio' && <PortfolioSheet onLearn={() => setTab('learn')} />}
        {tab === 'learn' && <LearnSheet />}
        {tab === 'resume' && <ResumeSheet />}

        <footer className="footer">
          <span>© 2026 {PROFILE.name}</span>
          <span className="mono">React · TypeScript · Vite · Recharts · GitHub Pages</span>
        </footer>
      </main>
    </>
  );
}
