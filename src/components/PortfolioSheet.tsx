import { assets } from '../api';
import { ACADEMIC, CERTIFICATES, EDUCATION, JOBS, LEVEL_LABEL, LEVEL_SEGMENTS, PROFILE, SKILLS, STORY } from '../content/profile';
import { useLang } from '../i18n';
import Gauge from './Gauge';

export default function PortfolioSheet({ onLearn }: { onLearn: () => void }) {
  const { lang } = useLang();
  const pt = lang === 'pt';

  return (
    <div className="sheet">
      {/* ---------- Hero ---------- */}
      <section className="hero" aria-labelledby="hero-title">
        <div className="rings" aria-hidden="true" />
        <div className="rings inner" aria-hidden="true" />
        <div>
          <h1 id="hero-title" className="wordmark">CoffeeAI</h1>
          <p className="hero-name">{PROFILE.name}</p>
          <p className="hero-role"><span className="bracket">{PROFILE.role[lang]}</span></p>
          <p className="lead">{PROFILE.tagline[lang]}</p>
          <div className="hero-actions">
            <a className="btn btn--gold" href={assets.resume} download="Rickson_Gaspar_Resume_2026.pdf">
              {pt ? 'Baixar currículo' : 'Download resume'}
            </a>
            <button className="btn" onClick={onLearn}>{pt ? 'Aprenda comigo' : 'Learn with me'}</button>
            <a className="btn btn--ghost" href={PROFILE.github} target="_blank" rel="noopener noreferrer">GitHub ↗</a>
          </div>
        </div>
        <div className="hero-card">
          <img className="avatar" src={assets.avatar} alt={PROFILE.name} width={220} height={220} />
          <span className="navi-tag"><span className="led led--ok" /> {pt ? 'Operador conectado' : 'Operator online'} · {pt ? PROFILE.location.replace('Brazil', 'Brasil') : PROFILE.location}</span>
          <div className="stats">
            <div className="stat"><b>7+</b><span>{pt ? 'anos em dados & IA' : 'years in data & AI'}</span></div>
            <div className="stat"><b>7</b><span>{pt ? 'empresas' : 'companies'}</span></div>
            <div className="stat"><b>10% → 4%</b><span>{pt ? 'churn mensal' : 'monthly churn'}</span></div>
            <div className="stat"><b>80B+</b><span>{pt ? 'linhas em ETL' : 'rows in ETL'}</span></div>
          </div>
        </div>
      </section>

      {/* ---------- Story ---------- */}
      <section className="section" aria-labelledby="story">
        <div className="section-head">
          <h2 id="story">{pt ? 'Minha história' : 'My story'}</h2>
          <span className="label">{pt ? 'registro de missões' : 'quest log'}</span>
        </div>
        <div className="quest">
          {STORY.map((s, i) => (
            <article key={i} className={`panel ${i === STORY.length - 1 ? 'panel--gold' : ''}`}>
              <span className="lvl">LV.{String(i + 1).padStart(2, '0')}</span>
              <h3>{s.title[lang]}</h3>
              <p>{s.text[lang]}</p>
            </article>
          ))}
        </div>
      </section>

      {/* ---------- Work + Skills ---------- */}
      <div className="grid grid-side section">
        <section aria-labelledby="work">
          <div className="section-head">
            <h2 id="work">{pt ? 'Experiência' : 'Work history'}</h2>
          </div>
          <div className="panel">
            <div className="timeline">
              {JOBS.map((j) => (
                <article key={j.company + j.period.en} className="job">
                  <span className={`node ${j.current ? 'current' : ''}`} aria-hidden="true" />
                  <div className="job-head">
                    <span className="job-role">{j.role}</span>
                    <span className="job-company">{j.company}</span>
                    <span className="job-period">{j.period[lang]}</span>
                  </div>
                  <ul className="arrows">
                    {j.items.map((it, i) => <li key={i}>{it[lang]}</li>)}
                  </ul>
                  {j.wins?.map((w, i) => <div key={i} className="win">{w[lang]}</div>)}
                  <div className="chips">{j.stack.map((s) => <span key={s} className="chip">{s}</span>)}</div>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section aria-labelledby="skills">
          <div className="section-head">
            <h2 id="skills">{pt ? 'Habilidades' : 'Hard skills'}</h2>
          </div>
          <div className="panel">
            {SKILLS.map((g) => (
              <div key={g.title.en} className="skill-group">
                <h4>{g.title[lang]}</h4>
                {g.skills.map(([name, lvl]) => (
                  <div key={name} className="skill">
                    <span className="skill-name">{name}</span>
                    <Gauge level={LEVEL_SEGMENTS[lvl]} label={`${name}: ${LEVEL_LABEL[lvl][lang]}`} />
                    <span className="skill-level">{LEVEL_LABEL[lvl][lang]}</span>
                  </div>
                ))}
              </div>
            ))}
          </div>

          <div className="section-head" style={{ marginTop: 32 }}>
            <h2>{pt ? 'Certificados' : 'Certificates'}</h2>
          </div>
          <div className="panel panel--gold panel--tight">
            <ul className="arrows" style={{ margin: 0 }}>
              {CERTIFICATES.map((c) => <li key={c}>{c}</li>)}
            </ul>
          </div>
        </section>
      </div>

      {/* ---------- Academic experience ---------- */}
      <section className="section" aria-labelledby="academic">
        <div className="section-head">
          <h2 id="academic">{pt ? 'Experiência acadêmica' : 'Academic experience'}</h2>
          <span className="label">{pt ? 'pesquisa e publicações' : 'research and publications'}</span>
        </div>
        <div className="grid grid-side">
          <div className="panel panel--gold">
            <div className="timeline">
              {ACADEMIC.map((a) => (
                <article key={a.title} className="job">
                  <span className={`node ${a.current ? 'current' : ''}`} aria-hidden="true" />
                  <div className="job-head">
                    <span className="job-role">{a.kind[lang]}</span>
                    <span className="job-period">{a.period[lang]}</span>
                  </div>
                  <h3 className="acad-title">{a.title}</h3>
                  <span className="job-company">{a.where[lang]}</span>
                  <p className="small acad-summary">{a.summary[lang]}</p>
                  <ul className="arrows">
                    {a.items.map((it, i) => <li key={i}>{it[lang]}</li>)}
                  </ul>
                  {a.wins?.map((w, i) => <div key={i} className="win">{w[lang]}</div>)}
                  <div className="chips">{a.stack.map((s) => <span key={s} className="chip chip--gold">{s}</span>)}</div>
                  {(a.paper || a.links) && (
                    <div className="hero-actions doc-actions">
                      {a.paper && (
                        <>
                          <a className="btn btn--gold btn--sm" href={assets.paper} download="ICEIS2025_Predicting_B2B_Customer_Churn_Rickson_Gaspar.pdf">
                            ⬇ {pt ? 'Baixar o artigo (PDF)' : 'Download the paper (PDF)'}
                          </a>
                          <a className="btn btn--ghost btn--sm" href={assets.paper} target="_blank" rel="noopener noreferrer">{pt ? 'Ler' : 'Read'}</a>
                        </>
                      )}
                      {a.links?.map((l) => (
                        <a key={l.href} className="btn btn--ghost btn--sm" href={l.href} target="_blank" rel="noopener noreferrer">{l.label[lang]}</a>
                      ))}
                    </div>
                  )}
                </article>
              ))}
            </div>
          </div>
          <article className="panel">
            <h3>{pt ? 'Por que "Aprenda comigo"?' : 'Why "Learn with me"?'}</h3>
            <p>
              {pt
                ? 'A aplicação da minha dissertação mostra o treino neurônio a neurônio: os dados indo para a frente, o erro voltando e cada peso mudando. Esta seção usa a mesma ideia, começando pelo modelo mais simples.'
                : 'My dissertation app shows training neuron by neuron: data flowing forward, the error flowing back and each weight changing. This section uses the same idea, starting with the simplest model.'}
            </p>
            <ul className="arrows">
              <li>
                {pt
                  ? 'Aula 1: um perceptron aprende a separar Charizard (Fogo) de Blastoise (Água) usando Velocidade e Defesa, com todas as contas lado a lado.'
                  : 'Lesson 1: a perceptron learns to tell Charizard (Fire) from Blastoise (Water) using Speed and Defense, with every calculation side by side.'}
              </li>
              <li>
                {pt
                  ? 'Aula 2: um MLP com camada oculta e backpropagation reconhece o tipo Planta, que nenhum perceptron sozinho consegue isolar.'
                  : 'Lesson 2: an MLP with a hidden layer and backpropagation recognizes the Grass type, which no single perceptron can isolate.'}
              </li>
              <li>
                {pt
                  ? 'Aula 3: uma CNN olha a foto da carta e descobre o tipo, com convolução, pooling e backpropagation em blocos.'
                  : 'Lesson 3: a CNN looks at the card picture and finds the type, with convolution, pooling and backpropagation in blocks.'}
              </li>
            </ul>
            <button className="btn" onClick={onLearn}>{pt ? 'Começar a aula' : 'Start the lesson'} →</button>
          </article>
        </div>
      </section>

      {/* ---------- Education + Contact ---------- */}
      <div className="grid grid-2 section">
        <section aria-labelledby="edu">
          <div className="section-head"><h2 id="edu">{pt ? 'Formação' : 'Education'}</h2></div>
          <div className="panel">
            <div className="timeline">
              {EDUCATION.map((e) => (
                <article key={e.period} className="job">
                  <span className={`node ${e.current ? 'current' : ''}`} aria-hidden="true" />
                  <div className="job-head">
                    <span className="job-role">{e.degree[lang]}</span>
                  </div>
                  <div className="job-head">
                    <span className="job-company">{e.school[lang]}</span>
                    <span className="job-period">{e.period}</span>
                  </div>
                  {e.diploma && (
                    <div className="hero-actions doc-actions">
                      <span className="label">{pt ? 'Diploma' : 'Diploma'}</span>
                      <a className="btn btn--ghost btn--sm" href={assets.diploma} target="_blank" rel="noopener noreferrer">
                        {pt ? 'Ver' : 'View'}
                      </a>
                      <a className="btn btn--ghost btn--sm" href={assets.diploma} download="Diploma_Estatistica_Rickson_Gaspar.pdf">
                        ⬇ {pt ? 'Baixar' : 'Download'}
                      </a>
                    </div>
                  )}
                </article>
              ))}
            </div>
          </div>
        </section>

        <section aria-labelledby="contact">
          <div className="section-head"><h2 id="contact">{pt ? 'Contato' : 'Contact'}</h2></div>
          <div className="panel">
            <ul className="arrows" style={{ margin: 0 }}>
              <li><span className="label">E-mail</span>{' '}<a href={`mailto:${PROFILE.email}`}>{PROFILE.email}</a></li>
              <li><span className="label">GitHub</span>{' '}<a href={PROFILE.github} target="_blank" rel="noopener noreferrer">@RicksonLivio</a></li>
              <li><span className="label">LinkedIn</span>{' '}<a href={PROFILE.linkedin} target="_blank" rel="noopener noreferrer">Rickson Gaspar</a></li>
              <li><span className="label">{pt ? 'Local' : 'Location'}</span>{' '}{pt ? PROFILE.location.replace('Brazil', 'Brasil') : PROFILE.location}</li>
            </ul>
            <div className="hero-actions">
              <a className="btn btn--gold" href={assets.resume} download="Rickson_Gaspar_Resume_2026.pdf">
                {pt ? 'Baixar currículo (PDF)' : 'Download resume (PDF)'}
              </a>
            </div>
          </div>
        </section>
      </div>
    </div>
  );
}
