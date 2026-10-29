
import { useMemo, useState } from 'react'
import './Experiments.css'

function Experiments({ comparisons = [] }) {
  const [metric, setMetric] = useState('success')
  const [selectedSeed, setSelectedSeed] = useState(
    comparisons[0]?.training_seed ?? ''
  )

  const rows = useMemo(
    () => [...comparisons].sort(
      (a, b) => a.training_seed - b.training_seed
    ),
    [comparisons]
  )

  const selected = rows.find(
    (row) => String(row.training_seed) === String(selectedSeed)
  )

  const isSuccess = metric === 'success'
  const fixedKey = isSuccess ? 'fixed_success_rate' : 'fixed_mean_steps'
  const adaptiveKey = isSuccess ? 'adaptive_success_rate' : 'adaptive_mean_steps'

  const maxValue = isSuccess
    ? 1
    : Math.max(
        1,
        ...rows.flatMap((row) => [row[fixedKey], row[adaptiveKey]])
      ) * 1.12

  const formatValue = (value) => (
    isSuccess
      ? `${(value * 100).toFixed(1)}%`
      : Number(value).toFixed(2)
  )

  const difference = selected
    ? selected[adaptiveKey] - selected[fixedKey]
    : 0

  return (
    <div className="experiments-page">
      <section className="section-heading">
        <div>
          <h2>Experiment comparison</h2>
          <p>
            Explore performance across individual training seeds.
          </p>
        </div>
        <span className="period-label">
          {rows.length} paired seeds
        </span>
      </section>

      <section className="panel experiment-chart-panel">
        <div className="experiment-toolbar">
          <div>
            <h2>{isSuccess ? 'Success rate by seed' : 'Mean episode steps by seed'}</h2>
            <p>
              {isSuccess
                ? 'Higher is better.'
                : 'Lower is better when task success is maintained.'}
            </p>
          </div>

          <div className="experiment-controls">
            <label htmlFor="experiment-metric">Metric</label>
            <select
              id="experiment-metric"
              value={metric}
              onChange={(event) => setMetric(event.target.value)}
            >
              <option value="success">Success rate</option>
              <option value="steps">Mean episode steps</option>
            </select>
          </div>
        </div>

        <div className="experiment-legend">
          <span><i className="experiment-dot adaptive-dot" /> Adaptive PPO</span>
          <span><i className="experiment-dot fixed-dot" /> Fixed PPO</span>
        </div>

        {rows.length === 0 ? (
          <p className="experiment-empty">
            No paired comparisons were returned by the API.
          </p>
        ) : (
          <div className="seed-chart">
            {rows.map((row) => (
              <button
                type="button"
                key={row.training_seed}
                className={`seed-column ${String(row.training_seed) === String(selectedSeed) ? 'selected' : ''}`}
                onClick={() => setSelectedSeed(row.training_seed)}
                title={`Inspect training seed ${row.training_seed}`}
                aria-pressed={String(row.training_seed) === String(selectedSeed)}
              >
                <div className="seed-bars">
                  <div className="seed-bar-wrap">
                    <span className="seed-bar-value">{formatValue(row[adaptiveKey])}</span>
                    <div
                      className="seed-bar adaptive-seed-bar"
                      style={{ height: `${Math.max(2, row[adaptiveKey] / maxValue * 100)}%` }}
                    />
                  </div>
                  <div className="seed-bar-wrap">
                    <span className="seed-bar-value">{formatValue(row[fixedKey])}</span>
                    <div
                      className="seed-bar fixed-seed-bar"
                      style={{ height: `${Math.max(2, row[fixedKey] / maxValue * 100)}%` }}
                    />
                  </div>
                </div>
                <span className="seed-label">{row.training_seed}</span>
              </button>
            ))}
          </div>
        )}

        <p className="experiment-chart-note">
          Select any seed to inspect its results below. Bars are scaled to the
          selected metric; values are labelled directly.
        </p>
      </section>

      {selected && (
        <section className="panel selected-seed-panel">
          <div className="panel-heading">
            <div>
              <h2>Training seed {selected.training_seed}</h2>
              <p>Fixed PPO versus Adaptive PPO</p>
            </div>
            <div className="experiment-controls">
              <label htmlFor="experiment-seed">Select seed</label>
              <select
                id="experiment-seed"
                value={selectedSeed}
                onChange={(event) => setSelectedSeed(event.target.value)}
              >
                {rows.map((row) => (
                  <option key={row.training_seed} value={row.training_seed}>
                    Seed {row.training_seed}
                  </option>
                ))}
              </select>
            </div>
          </div>

          <div className="seed-detail-grid">
            <article className="seed-detail-card">
              <span>Adaptive PPO</span>
              <strong>{formatValue(selected[adaptiveKey])}</strong>
              <small>Mean success: {formatValue(selected.adaptive_success_rate)}</small>
              <small>Mean steps: {Number(selected.adaptive_mean_steps).toFixed(2)}</small>
            </article>
            <article className="seed-detail-card">
              <span>Fixed PPO</span>
              <strong>{formatValue(selected[fixedKey])}</strong>
              <small>Mean success: {formatValue(selected.fixed_success_rate)}</small>
              <small>Mean steps: {Number(selected.fixed_mean_steps).toFixed(2)}</small>
            </article>
            <article className="seed-detail-card">
              <span>{isSuccess ? 'Success-rate difference' : 'Episode-step difference'}</span>
              <strong className={difference < 0 ? 'difference-negative' : 'difference-positive'}>
                {isSuccess && difference > 0 ? '+' : ''}
                {isSuccess
                  ? `${(difference * 100).toFixed(1)} pp`
                  : difference.toFixed(2)}
              </strong>
              <small>Adaptive PPO minus Fixed PPO</small>
              <small>
                {isSuccess
                  ? difference > 0
                    ? 'Adaptive PPO has the higher success rate.'
                    : difference < 0
                      ? 'Fixed PPO has the higher success rate.'
                      : 'Both have equal success rates.'
                  : difference < 0
                    ? 'Adaptive PPO uses fewer steps on average.'
                    : difference > 0
                      ? 'Adaptive PPO uses more steps on average.'
                      : 'Both have equal mean steps.'}
              </small>
            </article>
          </div>
        </section>
      )}
    </div>
  )
}

export default Experiments
