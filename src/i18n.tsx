import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';

export type Lang = 'en' | 'pt';

interface LangCtx {
  lang: Lang;
  toggle: () => void;
}

const Ctx = createContext<LangCtx>({ lang: 'en', toggle: () => {} });
const STORAGE_KEY = 'coffeeai-lang';

export function LangProvider({ children }: { children: ReactNode }) {
  const [lang, setLang] = useState<Lang>(() => {
    const saved = localStorage.getItem(STORAGE_KEY);
    if (saved === 'en' || saved === 'pt') return saved;
    return navigator.language.toLowerCase().startsWith('pt') ? 'pt' : 'en';
  });

  useEffect(() => {
    document.documentElement.lang = lang === 'pt' ? 'pt-BR' : 'en';
    localStorage.setItem(STORAGE_KEY, lang);
  }, [lang]);

  const value = useMemo(() => ({ lang, toggle: () => setLang((l) => (l === 'en' ? 'pt' : 'en')) }), [lang]);
  return <Ctx.Provider value={value}>{children}</Ctx.Provider>;
}

// eslint-disable-next-line react-refresh/only-export-components
export function useLang() {
  return useContext(Ctx);
}

/** Number as the lesson writes it: no thousands separator, locale decimal mark, up to 4 decimals. */
export function num(v: number, lang: Lang): string {
  if (!Number.isFinite(v)) return '—';
  const s = new Intl.NumberFormat(lang === 'pt' ? 'pt-BR' : 'en-US', { maximumFractionDigits: 4, useGrouping: false })
    .format(Math.abs(v));
  return v < 0 ? `−${s}` : s;
}

/** Same, wrapped in parentheses when negative, for use inside an expression: (0.5 · (−1)). */
export function par(v: number, lang: Lang): string {
  return v < 0 ? `(${num(v, lang)})` : num(v, lang);
}
