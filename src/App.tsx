import { useEffect, useState } from 'react';
import { assets } from './api';
import LearnSheet from './components/LearnSheet';
import PortfolioSheet from './components/PortfolioSheet';
import ResumeSheet from './components/ResumeSheet';
import { useLang } from './i18n';
import { PROFILE } from './content/profile';

export type Tab = 'portfolio' | 'learn' | 'resume';
const TABS: Tab[] = ['portfolio', 'learn', 'resume'];

interface Route {
  tab: Tab;
  /** open lessons, from #learn/1 or #learn/1,2 */
  lessons: number[];
}

// Hash routing (#portfolio, #learn/1, #resume): works on GitHub Pages with no server rewrite rules.
function parseHash(): Route {
  const [head, rest = ''] = window.location.hash.replace('#', '').split('/');
  const tab = TABS.includes(head as Tab) ? (head as Tab) : 'portfolio';
  const lessons = rest.split(',').map(Number).filter((n) => Number.isInteger(n) && n > 0);
  return { tab, lessons };
}

function toHash(r: Route): string {
  return r.tab === 'learn' && r.lessons.length ? `learn/${r.lessons.join(',')}` : r.tab;
}

export default function App() {
  const { lang, toggle } = useLang();
  const [route, setRouteState] = useState<Route>(parseHash);

  useEffect(() => {
    const onHash = () => setRouteState(parseHash());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);

  function go(r: Route, scrollTop = true) {
    const h = toHash(r);
    if (window.location.hash.replace('#', '') !== h) window.history.replaceState(null, '', `#${h}`);
    setRouteState(r);
    if (scrollTop) window.scrollTo({ top: 0 });
  }

  const setTab = (t: Tab) => go({ tab: t, lessons: t === 'learn' ? route.lessons : [] });
  const toggleLesson = (n: number) =>
    go({ tab: 'learn', lessons: route.lessons.includes(n) ? route.lessons.filter((x) => x !== n) : [...route.lessons, n].sort() }, false);

  const tab = route.tab;
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
              <span className="label">{PROFILE.short}</span>
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
        {tab === 'portfolio' && <PortfolioSheet onLearn={() => go({ tab: 'learn', lessons: [1] })} />}
        {tab === 'learn' && <LearnSheet open={route.lessons} onToggle={toggleLesson} />}
        {tab === 'resume' && <ResumeSheet />}

        <footer className="footer">
          <span>© 2026 {PROFILE.name}</span>
          <span className="mono">React · TypeScript · Vite · Recharts · GitHub Pages</span>
        </footer>
      </main>
    </>
  );
}
