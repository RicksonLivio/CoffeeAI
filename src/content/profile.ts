// Portfolio content. Every fact here comes from public/Resume_updated_2026.pdf or, for the
// master's research, from the Neural Fourier Head repository README (dissertation v3).
import type { Lang } from '../i18n';

export type T = Record<Lang, string>;

export const PROFILE = {
  name: 'Rickson Livio de Souza Gaspar',
  short: 'Rickson Livio',
  role: { en: 'Staff AI & MLOps Engineer', pt: 'Staff AI & MLOps Engineer' } as T,
  tagline: {
    en: 'Building agents and production ML, from data lakes to GPU-optimized models. Always learning, always testing the latest tech, and a proud dog dad.',
    pt: 'Construo agentes e ML em produção, do data lake a modelos otimizados para GPU. Sempre aprendendo, sempre testando a tecnologia mais nova, e pai de pet com orgulho.',
  } as T,
  location: 'Fortaleza, Ceará, Brazil',
  email: 'rickson.gaspar@outlook.com',
  github: 'https://github.com/RicksonLivio',
  linkedin: 'https://www.linkedin.com/in/rickson-livio/',
};

/** Resume levels mapped onto the 12-segment gauge from tokens.css. */
export type Level = 'Intermediate' | 'Advanced' | 'Expert' | 'Native' | 'C2';
export const LEVEL_SEGMENTS: Record<Level, number> = { Intermediate: 7, Advanced: 10, Expert: 12, Native: 12, C2: 11 };
export const LEVEL_LABEL: Record<Level, T> = {
  Intermediate: { en: 'Intermediate', pt: 'Intermediário' },
  Advanced: { en: 'Advanced', pt: 'Avançado' },
  Expert: { en: 'Expert', pt: 'Especialista' },
  Native: { en: 'Fluent / Native', pt: 'Fluente / Nativo' },
  C2: { en: 'C2 Proficient', pt: 'C2 Proficiente' },
};

export interface SkillGroup {
  title: T;
  skills: [string, Level][];
}

export const SKILLS: SkillGroup[] = [
  {
    title: { en: 'Programming', pt: 'Programação' },
    skills: [['Python', 'Advanced'], ['R', 'Expert'], ['SQL', 'Advanced'], ['DAX', 'Advanced'], ['PySpark', 'Intermediate']],
  },
  {
    title: { en: 'Cloud', pt: 'Nuvem' },
    skills: [['Google Cloud Platform', 'Advanced'], ['Snowflake', 'Intermediate'], ['AWS', 'Intermediate'], ['Azure', 'Intermediate']],
  },
  {
    title: { en: 'Software & Frameworks', pt: 'Software & Frameworks' },
    skills: [
      ['Power BI', 'Advanced'], ['Docker', 'Intermediate'], ['Git / GitHub', 'Intermediate'],
      ['Alteryx', 'Intermediate'], ['Databricks', 'Intermediate'], ['Airflow', 'Intermediate'],
    ],
  },
  {
    title: { en: 'Analysis & Modelling', pt: 'Análise & Modelagem' },
    skills: [
      ['Clustering & Regression', 'Expert'], ['ETL / ELT', 'Expert'], ['Machine Learning', 'Advanced'],
      ['Time Series & Forecasting', 'Advanced'], ['Storytelling with Data', 'Advanced'], ['Data Architecture', 'Advanced'],
      ['Deep Learning', 'Intermediate'], ['Computer Vision', 'Intermediate'],
    ],
  },
  {
    title: { en: 'Languages', pt: 'Idiomas' },
    skills: [['Portuguese', 'Native'], ['English', 'C2']],
  },
];

export const CERTIFICATES: string[] = [
  'Computer Vision: Facial Analysis',
  'Data Warehouse - Snowflake',
  'Machine Learning Solutions on Vertex AI',
  'EF SET English Certificate (C1 Advanced)',
];

export interface Job {
  period: T;
  company: string;
  role: string;
  current?: boolean;
  items: T[];
  wins?: T[];
  stack: string[];
}

export const JOBS: Job[] = [
  {
    period: { en: 'Dec 2025 – Present', pt: 'dez 2025 – atual' },
    company: 'Nuvidio',
    role: 'Staff AI & MLOps Engineer',
    current: true,
    items: [
      { en: 'Trains models for spoofing detection, deepfake detection and lip-sync.', pt: 'Treina modelos de detecção de spoofing, de deepfake e de lip-sync.' },
      { en: 'Helped structure the AI and Data Science area, including a GCP data lake for CI/CD model training and team data.', pt: 'Auxílio na estruturação da área de IA e Ciência de Dados, incluindo um data lake no GCP para treino de modelos com CI/CD e dados do time.' },
      { en: 'Optimized model inference speed with CUDA and ONNX on state-of-the-art (SOTA) models.', pt: 'Otimizou a velocidade de inferência com CUDA e ONNX em modelos do estado da arte (SOTA).' },
    ],
    stack: ['CUDA', 'ONNX', 'GCP', 'CI/CD'],
  },
  {
    period: { en: 'Jul 2024 – Dec 2025', pt: 'jul 2024 – dez 2025' },
    company: 'Cognitivo.AI',
    role: 'Senior Data Scientist',
    items: [
      { en: 'Built demand forecasting per product and store for Petz Group (Prophet, ARIMA, deep learning with zero-inflated negative binomial).', pt: 'Construiu previsão de demanda por produto e loja para o Grupo Petz (Prophet, ARIMA, deep learning com binomial negativa inflada de zeros).' },
      { en: 'Developed WhatsApp product-search AI agents (LangGraph, Snowflake Cortex) and used Whisper to extract demand from customer audio.', pt: 'Desenvolveu agentes de IA de busca de produtos no WhatsApp (LangGraph, Snowflake Cortex) e usou o Whisper para extrair demanda de áudios de clientes.' },
      { en: 'Served models and LLM pipelines with FastAPI, built Streamlit dashboards, tracked code and models with GitHub and MLflow.', pt: 'Serviu modelos e pipelines de LLM com FastAPI, criou dashboards em Streamlit e versionou código e modelos com GitHub e MLflow.' },
    ],
    wins: [{ en: 'Saved R$250,000 per quarter in supply and increased WhatsApp conversion by 10%.', pt: 'Economia de R$ 250.000 por trimestre em suprimentos e conversão no WhatsApp 10% maior.' }],
    stack: ['PySpark', 'Databricks', 'Snowflake', 'LangGraph', 'FastAPI', 'MLflow'],
  },
  {
    period: { en: 'Oct 2024 – Jan 2025', pt: 'out 2024 – jan 2025' },
    company: 'Globo',
    role: 'Senior Data and ML Engineer',
    items: [
      { en: 'Fine-tuned Gemini (Vertex AI) to classify user comments, with an automated GCP pipeline (Docker, Cloud Run) for daily processing.', pt: 'Fez fine-tuning do Gemini (Vertex AI) para classificar comentários de usuários, com pipeline automatizado no GCP (Docker, Cloud Run) para o processamento diário.' },
    ],
    stack: ['Gemini', 'Vertex AI', 'Docker', 'Cloud Run'],
  },
  {
    period: { en: 'Apr 2023 – Jul 2024', pt: 'abr 2023 – jul 2024' },
    company: 'Sólides Tecnologia',
    role: 'Mid-level Data Scientist',
    items: [
      { en: 'Built ETL from PostgreSQL, MySQL (Psycopg2) and APIs (Requests) into GCP.', pt: 'Construiu ETL de PostgreSQL, MySQL (Psycopg2) e APIs (Requests) para o GCP.' },
      { en: 'Churn model with survival analysis (pysurvival), deployed with Docker and Streamlit with the MLOps team.', pt: 'Modelo de churn com análise de sobrevivência (pysurvival), implantado com Docker e Streamlit junto ao time de MLOps.' },
      { en: 'OpenAI-based virtual assistant (persona and prompt engineering) integrated into the company website.', pt: 'Assistente virtual com OpenAI (persona e engenharia de prompt) integrado ao site da empresa.' },
      { en: 'Facial analysis (OpenCV, TensorFlow, Keras) and speech analysis (SpaCy) models for HR hiring reports.', pt: 'Modelos de análise facial (OpenCV, TensorFlow, Keras) e de fala (SpaCy) para relatórios de contratação do RH.' },
    ],
    wins: [{ en: 'Cut monthly churn from 10% to 4%.', pt: 'Churn mensal reduzido de 10% para 4%.' }],
    stack: ['GCP', 'pysurvival', 'OpenAI', 'OpenCV', 'TensorFlow'],
  },
  {
    period: { en: 'Mar 2022 – Feb 2023', pt: 'mar 2022 – fev 2023' },
    company: 'Tecla T',
    role: 'Mid-level Data Engineer',
    items: [
      { en: 'ETL solutions for Grupo Mateus and Hapvida Group with Databricks and PySpark on 80+ billion row tables.', pt: 'Soluções de ETL para o Grupo Mateus e o Grupo Hapvida com Databricks e PySpark em tabelas de mais de 80 bilhões de linhas.' },
    ],
    stack: ['Databricks', 'PySpark'],
  },
  {
    period: { en: 'Aug 2021 – Feb 2022', pt: 'ago 2021 – fev 2022' },
    company: 'Conjecto.AI',
    role: 'Data Scientist',
    items: [
      { en: 'KPIs and Power BI analytics for Iguatemi mall from PostgreSQL on AWS and Google Calendar APIs.', pt: 'KPIs e análises em Power BI para o shopping Iguatemi, a partir de PostgreSQL na AWS e APIs do Google Calendar.' },
      { en: 'Revenue forecasting (Statsmodels, Prophet) and fraud outlier detection (Gaussian Mixture, Isolation Forest).', pt: 'Previsão de receita (Statsmodels, Prophet) e detecção de fraude por outliers (Gaussian Mixture, Isolation Forest).' },
    ],
    stack: ['Power BI', 'AWS', 'Prophet'],
  },
  {
    period: { en: 'Feb 2020 – Aug 2021', pt: 'fev 2020 – ago 2021' },
    company: 'Somapay',
    role: 'Data Scientist I',
    items: [
      { en: 'Migrated databases from MySQL to PostgreSQL on AWS and ran A/B tests for marketing.', pt: 'Migrou bancos de MySQL para PostgreSQL na AWS e conduziu testes A/B para o marketing.' },
      { en: 'Credit time-to-payment modeling with survival analysis.', pt: 'Modelagem do tempo até o pagamento de crédito com análise de sobrevivência.' },
    ],
    stack: ['AWS', 'PostgreSQL', 'A/B tests'],
  },
];

export interface School {
  period: string;
  school: T;
  degree: T;
  current?: boolean;
}

export const EDUCATION: School[] = [
  {
    period: '03/2025 – 03/2027',
    school: { en: 'Federal Institute of Ceará (IFCE)', pt: 'Instituto Federal do Ceará (IFCE)' },
    degree: { en: "Master's Degree in Computer Science Applied to AI", pt: 'Mestrado em Ciência da Computação aplicada à IA' },
    current: true,
  },
  {
    period: '03/2017 – 03/2023',
    school: { en: 'Federal University of Ceará (UFC)', pt: 'Universidade Federal do Ceará (UFC)' },
    degree: { en: "Bachelor's Degree in Statistics", pt: 'Bacharelado em Estatística' },
  },
];

/** The story, told from the resume: one paragraph per stage. */
export const STORY: { title: T; text: T }[] = [
  {
    title: { en: 'Level 1 · Statistics', pt: 'Fase 1 · Estatística' },
    text: {
      en: 'It started with a Bachelor’s in Statistics at the Federal University of Ceará. In 2020, still in college, I joined Somapay as Data Scientist I: database migration, A/B tests and survival models for credit.',
      pt: 'Começou com o bacharelado em Estatística na Universidade Federal do Ceará. Em 2020, ainda na graduação, entrei na Somapay como Cientista de Dados I: migração de bancos, testes A/B e modelos de sobrevivência para crédito.',
    },
  },
  {
    title: { en: 'Level 2 · Data at scale', pt: 'Fase 2 · Dados em escala' },
    text: {
      en: 'Forecasting and fraud detection at Conjecto.AI, then ETL on tables with more than 80 billion rows for Grupo Mateus and Hapvida at Tecla T.',
      pt: 'Previsão e detecção de fraude na Conjecto.AI, depois ETL em tabelas de mais de 80 bilhões de linhas para o Grupo Mateus e a Hapvida na Tecla T.',
    },
  },
  {
    title: { en: 'Level 3 · Models in production', pt: 'Fase 3 · Modelos em produção' },
    text: {
      en: 'At Sólides, a churn model cut monthly churn from 10% to 4%. At Cognitivo.AI, demand forecasting saved R$250,000 per quarter and WhatsApp agents raised conversion by 10%. At Globo, a fine-tuned Gemini classified user comments every day on Cloud Run.',
      pt: 'Na Sólides, um modelo de churn levou o churn mensal de 10% para 4%. Na Cognitivo.AI, a previsão de demanda economizou R$ 250.000 por trimestre e agentes no WhatsApp aumentaram a conversão em 10%. Na Globo, um Gemini ajustado classificou comentários de usuários todos os dias no Cloud Run.',
    },
  },
  {
    title: { en: 'Current stage · Staff AI & research', pt: 'Fase atual · Staff AI & pesquisa' },
    text: {
      en: 'Today I am Staff AI & MLOps Engineer at Nuvidio, where I help structure the AI and Data Science area (deepfake and spoofing detection, CUDA/ONNX inference), and I research explainable forecasting in my master’s at IFCE. This site is where I share what I learn.',
      pt: 'Hoje sou Staff AI & MLOps Engineer na Nuvidio, onde auxilio na estruturação da área de IA e Ciência de Dados (detecção de deepfake e spoofing, inferência com CUDA/ONNX), e pesquiso previsão explicável no mestrado do IFCE. Este site é onde compartilho o que aprendo.',
    },
  },
];

/** Master's research. Numbers from the NFH README (dissertation v3, Cap. 4 and Tables 6, 10, 13). */
export const THESIS = {
  title: 'Neural Fourier Head (NFH)',
  subtitle: {
    en: 'Intrinsic explainability through spectral decomposition for multi-step time-series forecasting, with formal guarantees from abductive explanations.',
    pt: 'Explicabilidade intrínseca via decomposição espectral para previsão de séries temporais multi-passo, com garantias formais por explicações abdutivas.',
  } as T,
  program: { en: 'PPGCC · IFCE', pt: 'PPGCC · IFCE' } as T,
  points: [
    {
      en: 'An output layer that replaces the final dense layer of any forecaster and splits each forecast exactly into trend + Fourier cycles + residual: ŷ = T + S + R, with energy shares adding up to 100%.',
      pt: 'Uma camada de saída que substitui a densa final de qualquer modelo de previsão e divide cada previsão, de forma exata, em tendência + ciclos de Fourier + resíduo: ŷ = T + S + R, com participações de energia somando 100%.',
    },
    {
      en: 'Same number of parameters as nn.Linear. A trained dense layer converts exactly into an NFH, and the safe selection never picks a model with worse validation RMSE than the dense one.',
      pt: 'O mesmo número de parâmetros de uma nn.Linear. A densa treinada é convertida exatamente em NFH, e a seleção segura nunca escolhe um modelo com RMSE de validação pior que o da densa.',
    },
    {
      en: 'In the dissertation: lower RMSE than the dense layer in 17 of 24 dataset–backbone pairs (median −0.81%, not significant: Wilcoxon p = 0.084); non-inferior within a 1% margin. On test, it was worse than the dense layer in 25 of 120 runs, by up to 11.9%.',
      pt: 'Na dissertação: RMSE menor que o da densa em 17 de 24 pares de conjunto e backbone (mediana −0,81%, sem significância: Wilcoxon p = 0,084); não inferior com margem de 1%. No teste, foi pior que a densa em 25 de 120 execuções, em até 11,9%.',
    },
    {
      en: 'The structural explanation comes with the forecast in 18.3 ms; Integrated Gradients per component cost 83 ms, 23 times less than KernelSHAP.',
      pt: 'A explicação estrutural sai com a previsão em 18,3 ms; os Integrated Gradients por componente custam 83 ms, 23 vezes menos que o KernelSHAP.',
    },
  ] as T[],
  stack: ['PyTorch', 'FastAPI', 'MLflow', 'React', 'TypeScript', 'Vite', 'Recharts'],
};
