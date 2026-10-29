import { useEffect, useState } from 'react'
import './App.css'
import Evaluations from './Evaluations'
import Experiments from './Experiments'
import Methodology from './Methodology'
import ProjectInfo from './ProjectInfo'

const navItems = [
  { id: 'Overview', icon: '◫' },
  { id: 'Experiments', icon: '⌘' },
  { id: 'Evaluations', icon: '↗' },
  { id: 'Methodology', icon: '◎' },
]

function App() {
  const [activePage, setActivePage] = useState('Overview')
  const [apiStatus, setApiStatus] = useState('Checking')
  const [data, setData] = useState(null)
  const [error, setError] = useState('')

  useEffect(() => {
    let active = true

    async function loadResults() {
      try {
        setApiStatus('Checking')
        const response = await fetch('/api/results')
        if (!response.ok) throw new Error(`Results API returned HTTP ${response.status}`)
        const result = await response.json()
        if (!active) return
        setData(result)
        setError('')
        setApiStatus('Connected')
      } catch (err) {
        if (!active) return
        setApiStatus('Unavailable')
        setError(err.message || 'Unable to load research results.')
      }
    }

    loadResults()
    return () => { active = false }
  }, [])

  const fixed = data?.summary?.fixed_success_rate ?? 0
  const adaptive = data?.summary?.adaptive_success_rate ?? 0
  const fixedSteps = data?.summary?.fixed_mean_steps ?? 0
  const adaptiveSteps = data?.summary?.adaptive_mean_steps ?? 0
  const successStats = data?.statistics?.find((item) => item.metric.startsWith('Success-rate'))
  const stepStats = data?.statistics?.find((item) => item.metric.startsWith('Episode-step'))

  const pct = (value) => `${(value * 100).toFixed(2)}%`
  const pp = (value) => `${value >= 0 ? '+' : ''}${value.toFixed(2)} pp`
  const steps = (value) => `${value.toFixed(2)} steps`

  return (
    <div className="app-shell">
      <aside className="sidebar">        
      <a
        className="brand"
        href="#overview"
        onClick={() => setActivePage('Overview')}
      >
        <span className="brand-copy">
          <strong>
            Adaptive-Horizon PPO with Dynamic Discount and GAE Scheduling
          </strong>
          <small>CINT-22 · CAPSTONE RESEARCH</small>
        </span>
      </a>

        <div className="workspace-label">WORKSPACE</div>
        <nav className="side-nav" aria-label="Main navigation">
          {navItems.map((item) => (
            <button key={item.id}
              className={`nav-item ${activePage === item.id ? 'active' : ''}`}
              onClick={() => setActivePage(item.id)}>
              <span className="nav-icon">{item.icon}</span>
              <span>{item.id}</span>
              {item.id === 'Evaluations' && <span className="nav-count">{data?.training_seed_pairs ?? '—'}</span>}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="project-chip">
            <span className="project-dot" />
            <div><strong>Adaptive-Horizon PPO</strong><small>Capstone project</small></div>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumbs"><span>Research</span><span className="crumb-separator">/</span><strong>{activePage}</strong></div>
          <div className="topbar-right">
            <span className={`api-pill ${apiStatus === 'Connected' ? 'online' : apiStatus === 'Checking' ? 'checking' : 'offline'}`}>
              <span className="status-dot" /> API {apiStatus}
            </span>
            <button className="icon-button" title="Reload research data" onClick={() => window.location.reload()}>↻</button>
            <button
              className="project-info-button"
              onClick={() => setActivePage('Project Info')}
            >
              <span className="project-info-icon">ⓘ</span>
              <span>Project Info</span>
            </button>
          </div>
        </header>

        <div className="page-content">
          {activePage === 'Evaluations' ? (
            <Evaluations comparisons={data?.comparisons ?? []} />
          ) : activePage === 'Experiments' ? (
            <Experiments comparisons={data?.comparisons ?? []} />
          ) : activePage === 'Methodology' ? (
            <Methodology />
          ) : activePage === 'Project Info' ? (
            <ProjectInfo />
          ) : (
            <>
          <section className="welcome-row">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> REINFORCEMENT LEARNING / RESEARCH</div>
              <h1>{activePage === 'Overview' ? 'Research overview' : activePage}</h1>
              <p className="page-subtitle">Monitor your experiments, compare policy performance, and track evaluation outcomes.</p>
            </div>
            <button className="primary-button" onClick={() => setActivePage('Experiments')}><span>＋</span> Explore experiments</button>
          </section>

          {error && <div className="notice"><span>ⓘ</span><div><strong>Could not load research results</strong><p>{error}. Check that FastAPI is running and try reloading.</p></div><button onClick={() => window.location.reload()}>Retry</button></div>}

          {activePage !== 'Overview' && (
            <div className="notice"><span>ⓘ</span><div><strong>{activePage} workspace</strong><p>Research data is loaded from your existing evaluation CSV files.</p></div><button onClick={() => setActivePage('Overview')}>Back to overview</button></div>
          )}

          <section className="section-heading">
            <div><h2>Study at a glance</h2><p>Live summary from your saved evaluation results</p></div>
            <span className="period-label"><span className="tiny-calendar">▦</span> {data?.environment ?? 'Loading results'}</span>
          </section>

          <section className="metrics-grid">
            <article className="metric-card">
              <div className="metric-top"><span>Training seeds</span><span className="metric-icon lavender">⌘</span></div>
              <div className="metric-value">{data?.training_seed_pairs ?? '—'} <span className="metric-unit">pairs</span></div>
              <div className="metric-foot"><span className="foot-dot green" /> Paired experimental design</div>
              <div className="metric-spark spark-one"><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /></div>
            </article>
            <article className="metric-card">
              <div className="metric-top"><span>Evaluation episodes</span><span className="metric-icon blue">◎</span></div>
              <div className="metric-value">{data ? data.evaluation_episodes.toLocaleString() : '—'}</div>
              <div className="metric-foot"><span className="foot-dot green" /> All evaluated model episodes</div>
              <div className="metric-spark spark-two"><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /></div>
            </article>
            <article className="metric-card">
              <div className="metric-top"><span>Success rate uplift</span><span className="metric-icon green-bg">↗</span></div>
              <div className="metric-value">{successStats ? pp(successStats.mean_difference) : '—'}</div>
              <div className="metric-foot"><span className="foot-dot green" /> Adaptive minus fixed PPO</div>
              <div className="metric-spark spark-three"><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /></div>
            </article>
            <article className="metric-card">
              <div className="metric-top"><span>Statistical significance</span><span className="metric-icon amber">⌁</span></div>
              <div className="metric-value">{successStats ? successStats.sign_flip_p_value.toFixed(5) : '—'}</div>
              <div className="metric-foot"><span className="foot-dot green" /> Exact two-sided sign-flip p-value</div>
              <div className="significance-track"><span /></div>
            </article>
          </section>

          <section className="analysis-grid">
            <article className="panel performance-panel">
              <div className="panel-heading"><div><h2>Policy performance</h2><p>Mean success rate across evaluated models</p></div><span className="tag tag-neutral">{data?.environment ?? 'Loading'}</span></div>
              <div className="chart-legend"><span><i className="legend-dot adaptive" /> Adaptive PPO</span><span><i className="legend-dot fixed" /> Fixed PPO</span></div>
              <div className="bar-chart" role="img" aria-label={`Adaptive PPO ${pct(adaptive)} and Fixed PPO ${pct(fixed)} mean success rate`}>
                <div className="chart-y-labels"><span>100%</span><span>75%</span><span>50%</span><span>25%</span><span>0%</span></div>
                <div className="chart-plot">
                  {[100, 75, 50, 25, 0].map((tick) => <div className="chart-gridline" key={tick} style={{ bottom: `${tick}%` }} />)}
                  <div className="bar-group">
                    <div className="bar-column"><span className="bar-value">{pct(adaptive)}</span><div className="bar adaptive-bar" style={{ height: `${Math.max(0, Math.min(100, adaptive * 100))}%` }} /><span className="bar-label">Adaptive PPO</span></div>
                    <div className="bar-column"><span className="bar-value">{pct(fixed)}</span><div className="bar fixed-bar" style={{ height: `${Math.max(0, Math.min(100, fixed * 100))}%` }} /><span className="bar-label">Fixed PPO</span></div>
                  </div>
                </div>
              </div>
              <div className="chart-insight"><span className="insight-icon">↗</span><div><strong>Adaptive PPO success rate: {pct(adaptive)} versus {pct(fixed)} for fixed PPO.</strong><p>Mean across the saved per-model results. Statistical inference uses paired training-seed differences.</p></div></div>
            </article>

            <article className="panel findings-panel">
              <div className="panel-heading"><div><h2>Key findings</h2><p>Paired-seed statistical analysis</p></div><span className="finding-symbol">✳</span></div>
              <div className="finding-row"><div className="finding-icon green-bg">↗</div><div className="finding-copy"><span>Success-rate difference</span><strong>{successStats ? pp(successStats.mean_difference) : '—'}</strong><small>{successStats ? `95% CI: ${pp(successStats.ci_95_low)} to ${pp(successStats.ci_95_high)}` : 'Loading statistics'}</small></div></div>
              <div className="finding-row"><div className="finding-icon blue">◷</div><div className="finding-copy"><span>Episode-step difference</span><strong>{stepStats ? steps(stepStats.mean_difference) : '—'}</strong><small>{stepStats ? `95% CI: ${steps(stepStats.ci_95_low)} to ${steps(stepStats.ci_95_high)}` : 'Loading statistics'}</small></div></div>
              <div className="finding-row"><div className="finding-icon lavender">∑</div><div className="finding-copy"><span>Sign-flip test</span><strong>{successStats ? `p = ${successStats.sign_flip_p_value.toFixed(8)}` : '—'}</strong><small>Two-sided exact test</small></div></div>
              <div className="finding-note"><span>ⓘ</span> Statistical values are read directly from sparse7_statistical_analysis.csv.</div>
            </article>
          </section>

          <section className="panel results-panel">
            <div className="panel-heading results-heading"><div><h2>Evaluation summary</h2><p>Mean performance across the saved training seeds</p></div><button className="text-button" onClick={() => setActivePage('Evaluations')}>View evaluations <span>→</span></button></div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>POLICY</th><th>MEAN SUCCESS RATE</th><th>MEAN REWARD</th><th>MEAN STEPS</th><th>STATUS</th></tr></thead>
                <tbody>
                  {data ? [
                    { method: 'Adaptive PPO', detail: 'Dynamic γ and λ schedules', success: adaptive, reward: adaptive, steps: adaptiveSteps },
                    { method: 'Fixed PPO', detail: 'Constant γ and λ', success: fixed, reward: fixed, steps: fixedSteps },
                  ].map((result) => (
                    <tr key={result.method}>
                      <td><div className="policy-cell"><span className={`policy-mark ${result.method === 'Adaptive PPO' ? 'mark-adaptive' : 'mark-fixed'}`}>{result.method === 'Adaptive PPO' ? 'A' : 'F'}</span><div><strong>{result.method}</strong><small>{result.detail}</small></div></div></td>
                      <td><div className="rate-cell"><strong>{pct(result.success)}</strong><div className="rate-track"><span style={{ width: `${result.success * 100}%` }} /></div></div></td>
                      <td>{result.reward.toFixed(3)}</td><td>{result.steps.toFixed(2)}</td><td><span className="completed-status"><span />Loaded from CSV</span></td>
                    </tr>
                  )) : <tr><td colSpan="5">Loading saved evaluation results…</td></tr>}
                </tbody>
              </table>
            </div>
            <div className="table-footer"><span>Deterministic evaluation · 100 episodes per model</span><span>Environment <strong>{data?.environment ?? '—'}</strong></span></div>
          </section>
          
          <footer className="page-footer">
            <span>
              CINT-22
              <span className="footer-divider"> / </span>
              Adaptive-Horizon PPO
            </span>
            <span>
              <span
                className={`footer-status ${
                  apiStatus === 'Connected' ? 'is-online' : ''
                }`}
              />
              API {apiStatus.toLowerCase()}
            </span>
          </footer>
          </>
          )}
        </div>
      </main>
    </div>
  )
}

export default App
