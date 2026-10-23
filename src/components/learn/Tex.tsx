import katex from 'katex';
import 'katex/dist/katex.min.css';
import { useMemo } from 'react';

/** Theme colours usable inside formulas: \gold{…}, \cy{…}, \vi{…}, \rd{…}. In macro bodies a literal # is written ##. */
const MACROS = {
  '\\gold': '{\\color{##ffd25e}{#1}}',
  '\\cy': '{\\color{##7df0ff}{#1}}',
  '\\vi': '{\\color{##b49bff}{#1}}',
  '\\rd': '{\\color{##ff6b6b}{#1}}',
  '\\R': '\\mathbb{R}',
};

/**
 * LaTeX formula rendered with KaTeX. Only static strings written in this repository are passed here,
 * so the generated HTML is trusted; `trust: false` keeps \href and similar commands disabled anyway.
 */
export default function Tex({ children, block = false }: { children: string; block?: boolean }) {
  const html = useMemo(
    () => katex.renderToString(children, { displayMode: block, throwOnError: false, strict: 'ignore', trust: false, macros: { ...MACROS }, output: 'html' }),
    [children, block],
  );
  return <span className={block ? 'tex-block' : 'tex'} dangerouslySetInnerHTML={{ __html: html }} />;
}
