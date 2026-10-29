
import { useMemo, useState } from 'react'

function formatPercent(value) {
  return `${(value * 100).toFixed(1)}%`
}

function formatDifference(value) {
  return `${value > 0 ? '+' : ''}${(value * 100).toFixed(1)} pp`
}

function Evaluations({ comparisons = [] }) {
  const [sortBy, setSortBy] = useState('training_seed')
  const [order, setOrder] = useState('asc')

  const sortedComparisons = useMemo(() => {
    return [...comparisons].sort((a, b) => {
      const first = a[sortBy]
      const second = b[sortBy]

      if (typeof first === 'number' && typeof second === 'number') {
        return order === 'asc' ? first - second : second - first
      }

      return order === 'asc'
        ? String(first).localeCompare(String(second))
        : String(second).localeCompare(String(first))
    })
  }, [comparisons, sortBy, order])

  function handleSort(field) {
    if (sortBy === field) {
      setOrder((current) => current === 'asc' ? 'desc' : 'asc')
    } else {
      setSortBy(field)
      setOrder('asc')
    }
  }

  const adaptiveWins = comparisons.filter(
    (row) => row.success_rate_difference > 0
  ).length

  const fixedWins = comparisons.filter(
    (row) => row.success_rate_difference < 0
  ).length

  const equalResults = comparisons.length - adaptiveWins - fixedWins

  const averageDifference = comparisons.length
    ? comparisons.reduce(
        (total, row) => total + row.success_rate_difference,
        0
      ) / comparisons.length
    : 0

  const columns = [
    { key: 'training_seed', label: 'Training seed' },
    { key: 'fixed_success_rate', label: 'Fixed success rate' },
    { key: 'adaptive_success_rate', label: 'Adaptive success rate' },
    { key: 'success_rate_difference', label: 'Difference (pp)' },
    { key: 'fixed_mean_steps', label: 'Fixed mean steps' },
    { key: 'adaptive_mean_steps', label: 'Adaptive mean steps' },
    { key: 'mean_steps_difference', label: 'Step difference' },
  ]

  return (
    <section className="evaluation-workspace">
      <div className="section-heading">
        <div>
          <h2>Paired-seed evaluations</h2>
          <p>
            Compare Fixed PPO and Adaptive PPO on each training seed.
          </p>
        </div>
        <span className="period-label">
          {comparisons.length} seed pairs
        </span>
      </div>

      <section className="metrics-grid">
        <article className="metric-card">
          <div className="metric-top">
            <span>Adaptive success wins</span>
          </div>
          <div className="metric-value">{adaptiveWins}</div>
          <div className="metric-foot">
            Seeds where adaptive PPO has a higher success rate
          </div>
        </article>

        <article className="metric-card">
          <div className="metric-top">
            <span>Fixed success wins</span>
          </div>
          <div className="metric-value">{fixedWins}</div>
          <div className="metric-foot">
            Seeds where fixed PPO has a higher success rate
          </div>
        </article>

        <article className="metric-card">
          <div className="metric-top">
            <span>Equal success rates</span>
          </div>
          <div className="metric-value">{equalResults}</div>
          <div className="metric-foot">
            Seed pairs with no success-rate difference
          </div>
        </article>

        <article className="metric-card">
          <div className="metric-top">
            <span>Average improvement</span>
          </div>
          <div className="metric-value">
            {formatDifference(averageDifference)}
          </div>
          <div className="metric-foot">
            Mean paired difference across seed pairs
          </div>
        </article>
      </section>

      <section className="panel results-panel">
        <div className="panel-heading results-heading">
          <div>
            <h2>Seed-by-seed comparison</h2>
            <p>Select a column heading to sort the results.</p>
          </div>
        </div>

        <div className="table-wrap">
          <table>
            <thead>
              <tr>
                {columns.map((column) => (
                  <th key={column.key}>
                    <button
                      className="sort-button"
                      onClick={() => handleSort(column.key)}
                    >
                      {column.label}
                      {sortBy === column.key
                        ? order === 'asc' ? ' ↑' : ' ↓'
                        : ''}
                    </button>
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sortedComparisons.map((row) => (
                <tr key={row.training_seed}>
                  <td><strong>{row.training_seed}</strong></td>
                  <td>{formatPercent(row.fixed_success_rate)}</td>
                  <td>{formatPercent(row.adaptive_success_rate)}</td>
                  <td>
                    <span
                      className={
                        row.success_rate_difference > 0
                          ? 'completed-status'
                          : row.success_rate_difference < 0
                            ? 'difference-negative'
                            : 'tag tag-neutral'
                      }
                    >
                      {formatDifference(row.success_rate_difference)}
                    </span>
                  </td>
                  <td>{Number(row.fixed_mean_steps).toFixed(2)}</td>
                  <td>{Number(row.adaptive_mean_steps).toFixed(2)}</td>
                  <td>
                    {Number(row.mean_steps_difference).toFixed(2)}
                  </td>
                </tr>
              ))}
              {comparisons.length === 0 && (
                <tr>
                  <td colSpan={7}>
                    No paired evaluation results are available.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>

        <div className="table-footer">
          <span>
            Positive success-rate differences favor Adaptive PPO.
          </span>
          <span>
            Success-rate differences are shown in percentage points.
          </span>
        </div>
      </section>
    </section>
  )
}

export default Evaluations
