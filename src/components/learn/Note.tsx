import type { ReactNode } from 'react';

/** Short side explanation inside a lesson ("why?" boxes). */
export default function Note({ children, title }: { children: ReactNode; title?: ReactNode }) {
  return (
    <aside className="note">
      <span className="note-icon" aria-hidden="true">i</span>
      <div>
        {title && <b className="note-title">{title}</b>}
        {children}
      </div>
    </aside>
  );
}
