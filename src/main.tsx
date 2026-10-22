import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import App from './App.tsx';
import { LangProvider } from './i18n.tsx';
import './styles/tokens.css';
import './styles/app.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <LangProvider>
      <App />
    </LangProvider>
  </StrictMode>,
);
