import { useEffect, useState } from 'react'
import './App.css'

const navItems = [
  { id: 'Overview', icon: '◫' },
  { id: 'Experiments', icon: '⌘' },
  { id: 'Evaluations', icon: '↗' },
  { id: 'Methodology', icon: '◎' },
]

const results = [
  { method: 'Adaptive PPO', detail: 'Dynamic γ and λ schedules', success: 87.1, reward: '—', steps: '—', status: 'Completed' },
  { method: 'Fixed PPO', detail: 'Constant γ and λ', success: 80.0, reward: '—', steps: '—', status: 'Completed' },
]

function App() {
  const [activePage, setActivePage] = useState('Overview')
  const [apiStatus, setApiStatus] = useState('Checking')
  const [lastUpdated, setLastUpdated] = useState('Not connected')

  useEffect(() => {
    let active = true

    fetch('/api/health')
      .then((response) => {
        if (!response.ok) throw new Error('API unavailable')
        return response.json()
      })
      .then((data) => {
        if (!active) return
        setApiStatus(data.status === 'ok' ? 'Connected' : 'Unavailable')
        setLastUpdated(new Date().toLocaleTimeString())
      })
      .catch(() => {
        if (active) {
          setApiStatus('Unavailable')
          setLastUpdated('Could not reach backend')
        }
      })

    return () => {
      active = false
    }
  }, [])

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <a className="brand" href="#overview" onClick={() => setActivePage('Overview')}>
          <span className="brand-mark">AH</span>
          <span className="brand-copy">
            <strong>Horizon Lab</strong>
            <small>RL RESEARCH WORKSPACE</small>
          </span>
        </a>

        <div className="workspace-label">WORKSPACE</div>
        <nav className="side-nav" aria-label="Main navigation">
          {navItems.map((item) => (
            <button
              key={item.id}
              className={`nav-item ${activePage === item.id ? 'active' : ''}`}
              onClick={() => setActivePage(item.id)}
            >
              <span className="nav-icon">{item.icon}</span>
              <span>{item.id}</span>
              {item.id === 'Evaluations' && <span className="nav-count">15</span>}
            </button>
          ))}
        </nav>

        <div className="sidebar-bottom">
          <div className="project-chip">
            <span className="project-dot" />
            <div>
              <strong>Adaptive-Horizon PPO</strong>
              <small>Capstone project</small>
            </div>
          </div>
          <div className="sidebar-footer">
            <span className="avatar">RL</span>
            <div className="profile-copy">
              <strong>Research workspace</strong>
              <small>Local development</small>
            </div>
            <span className="more-icon">···</span>
          </div>
        </div>
      </aside>

      <main className="main-content">
        <header className="topbar">
          <div className="breadcrumbs">
            <span>Research</span><span className="crumb-separator">/</span>
            <strong>{activePage}</strong>
          </div>
          <div className="topbar-right">
            <span className={`api-pill ${apiStatus === 'Connected' ? 'online' : apiStatus === 'Checking' ? 'checking' : 'offline'}`}>
              <span className="status-dot" />
              API {apiStatus}
            </span>
            <button className="icon-button" title="Refresh page" onClick={() => window.location.reload()}>↻</button>
            <span className="top-avatar">RL</span>
          </div>
        </header>

        <div className="page-content">
          <section className="welcome-row">
            <div>
              <div className="eyebrow"><span className="eyebrow-line" /> REINFORCEMENT LEARNING / RESEARCH</div>
              <h1>{activePage === 'Overview' ? 'Research overview' : activePage}</h1>
              <p className="page-subtitle">
                {activePage === 'Overview'
                  ? 'Monitor your experiments, compare policy performance, and track evaluation outcomes.'
                  : `Explore ${activePage.toLowerCase()} for the Adaptive-Horizon PPO study.`}
              </p>
            </div>
            <button className="primary-button" onClick={() => setActivePage('Experiments')}>
              <span>＋</span> Explore experiments
            </button>
          </section>

          {activePage !== 'Overview' && (
            <div className="notice">
              <span>ⓘ</span>
              <div><strong>{activePage} workspace</strong><p>This section is a UI foundation. We'll connect it to your experiment files in the next stages.</p></div>
              <button onClick={() => setActivePage('Overview')}>Back to overview</button>
            </div>
          )}

          <section className="section-heading">
            <div>
              <h2>Study at a glance</h2>
              <p>Current experiment configuration and evaluation coverage</p>
            </div>
            <span className="period-label"><span className="tiny-calendar">▦</span> Research summary</span>
          </section>

          <section className="metrics-grid">
            <article className="metric-card">
              <div className="metric-top"><span>Training seeds</span><span className="metric-icon lavender">⌘</span></div>
              <div className="metric-value">15 <span className="metric-unit">pairs</span></div>
              <div className="metric-foot"><span className="foot-dot green" /> Paired experimental design</div>
              <div className="metric-spark spark-one"><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /></div>
            </article>
            <article className="metric-card">
              <div className="metric-top"><span>Evaluation episodes</span><span className="metric-icon blue">◎</span></div>
              <div className="metric-value">3,000</div>
              <div className="metric-foot"><span className="foot-dot green" /> 30 models × 100 episodes</div>
              <div className="metric-spark spark-two"><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /></div>
            </article>
            <article className="metric-card">
              <div className="metric-top"><span>Success rate uplift</span><span className="metric-icon green-bg">↗</span></div>
              <div className="metric-value">+7.13<span className="metric-unit"> pp</span></div>
              <div className="metric-foot"><span className="foot-dot green" /> Adaptive minus fixed PPO</div>
              <div className="metric-spark spark-three"><i /><i /><i /><i /><i /><i /><i /><i /><i /><i /></div>
            </article>
            <article className="metric-card">
              <div className="metric-top"><span>Statistical significance</span><span className="metric-icon amber">⌁</span></div>
              <div className="metric-value">0.00012</div>
              <div className="metric-foot"><span className="foot-dot green" /> Exact sign-flip p-value</div>
              <div className="significance-track"><span /></div>
            </article>
          </section>

          <section className="analysis-grid">
            <article className="panel performance-panel">
              <div className="panel-heading">
                <div>
                  <h2>Policy performance</h2>
                  <p>Evaluation success rate by PPO method</p>
                </div>
                <span className="tag tag-neutral">SparseGridWorld-v1</span>
              </div>

              <div className="chart-legend">
                <span><i className="legend-dot adaptive" /> Adaptive PPO</span>
                <span><i className="legend-dot fixed" /> Fixed PPO</span>
              </div>

              <div className="bar-chart" role="img" aria-label="Success rate comparison: Adaptive PPO 87.13 percent and Fixed PPO 80 percent">
                <div className="chart-y-labels"><span>100%</span><span>75%</span><span>50%</span><span>25%</span><span>0%</span></div>
                <div className="chart-plot">
                  {[100, 75, 50, 25, 0].map((tick) => (
                    <div className="chart-gridline" key={tick} style={{ bottom: `${tick}%` }} />
                  ))}
                  <div className="bar-group">
                    <div className="bar-column">
                      <span className="bar-value">87.13%</span>
                      <div className="bar adaptive-bar" style={{ height: '87.13%' }} />
                      <span className="bar-label">Adaptive PPO</span>
                    </div>
                    <div className="bar-column">
                      <span className="bar-value">80.00%</span>
                      <div className="bar fixed-bar" style={{ height: '80%' }} />
                      <span className="bar-label">Fixed PPO</span>
                    </div>
                  </div>
                </div>
              </div>

              <div className="chart-insight">
                <span className="insight-icon">↗</span>
                <div><strong>Adaptive PPO improved success rate by 7.13 percentage points.</strong><p>Paired comparison across 15 training seeds. Confirm final values against your saved evaluation summary.</p></div>
              </div>
            </article>

            <article className="panel findings-panel">
              <div className="panel-heading">
                <div><h2>Key findings</h2><p>Paired-seed analysis</p></div>
                <span className="finding-symbol">✳</span>
              </div>
              <div className="finding-row">
                <div className="finding-icon green-bg">↗</div>
                <div className="finding-copy"><span>Success rate difference</span><strong>+7.13 pp</strong><small>95% CI: +2.73 to +11.53 pp</small></div>
              </div>
              <div className="finding-row">
                <div className="finding-icon blue">◷</div>
                <div className="finding-copy"><span>Episode steps difference</span><strong>−2.54 steps</strong><small>95% CI: −4.08 to −1.00 steps</small></div>
              </div>
              <div className="finding-row">
                <div className="finding-icon lavender">∑</div>
                <div className="finding-copy"><span>Sign-flip test</span><strong>p = 0.000122</strong><small>Two-sided exact test</small></div>
              </div>
              <div className="finding-note"><span>ⓘ</span> These are recorded project summary values. The dashboard is not yet reading the result files dynamically.</div>
            </article>
          </section>

          <section className="panel results-panel">
            <div className="panel-heading results-heading">
              <div><h2>Evaluation summary</h2><p>Comparison of baseline and adaptive policies</p></div>
              <button className="text-button" onClick={() => setActivePage('Evaluations')}>View evaluations <span>→</span></button>
            </div>
            <div className="table-wrap">
              <table>
                <thead><tr><th>POLICY</th><th>SUCCESS RATE</th><th>MEAN REWARD</th><th>MEAN STEPS</th><th>STATUS</th></tr></thead>
                <tbody>
                  {results.map((result) => (
                    <tr key={result.method}>
                      <td><div className="policy-cell"><span className={`policy-mark ${result.method === 'Adaptive PPO' ? 'mark-adaptive' : 'mark-fixed'}`}>{result.method === 'Adaptive PPO' ? 'A' : 'F'}</span><div><strong>{result.method}</strong><small>{result.detail}</small></div></div></td>
                      <td><div className="rate-cell"><strong>{result.success.toFixed(1)}%</strong><div className="rate-track"><span style={{ width: `${result.success}%` }} /></div></div></td>
                      <td>{result.reward}</td>
                      <td>{result.steps}</td>
                      <td><span className="completed-status"><span />{result.status}</span></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            <div className="table-footer"><span>Evaluation protocol: deterministic actions · 100 episodes per model</span><span>Environment <strong>SparseGridWorld-v1</strong></span></div>
          </section>

          <footer className="page-footer">
            <span>Horizon Lab <span className="footer-divider">/</span> Adaptive-Horizon PPO</span>
            <span><span className={`footer-status ${apiStatus === 'Connected' ? 'is-online' : ''}`} /> API {apiStatus.toLowerCase()} · {lastUpdated}</span>
          </footer>
        </div>
      </main>
    </div>
  )
}

export default App
