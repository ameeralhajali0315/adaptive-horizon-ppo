
import './Methodology.css'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
} from 'recharts'

function Methodology() {
  const seeds = [
    42, 123, 456, 789, 1024,
    2026, 2027, 2028, 2029, 2030,
    2031, 2032, 2033, 2034, 2035,
  ]

  const scheduleData = Array.from({ length: 100 }, (_, index) => {
  const progress = index / 99

    return {
            update: index + 1,
            gamma: 0.90 + progress * (0.99 - 0.90),
            lambda: 0.80 + progress * (0.95 - 0.80),
        }
  })

  return (
    <div className="methodology-page">
      <section className="section-heading">
        <div>
          <h2>Research methodology</h2>
          <p>
            Algorithm design, experimental protocol, and statistical evaluation.
          </p>
        </div>
        <span className="period-label">Research protocol</span>
      </section>

      <section className="method-hero">
        <span className="method-eyebrow">CAPSTONE RESEARCH</span>
        <h2>Adaptive-Horizon Proximal Policy Optimization</h2>
        <p>
          This study compares standard PPO with a variant that schedules
          the discount factor and Generalized Advantage Estimation parameter
          during training.
        </p>
        <div className="method-tags">
          <span>PPO</span>
          <span>Adaptive γ</span>
          <span>Adaptive λ</span>
          <span>SparseGridWorld-v1</span>
        </div>
      </section>

      <section className="method-section">
        <div className="method-section-heading">
          <span className="method-number">01</span>
          <div>
            <h2>Algorithm design</h2>
            <p>Two policy-optimization configurations are compared.</p>
          </div>
        </div>

        <div className="method-card-grid">
          <article className="method-card">
            <div className="method-card-icon fixed-icon">F</div>
            <h3>Fixed PPO</h3>
            <p>
              The baseline uses constant discount-factor and GAE settings
              throughout training.
            </p>
            <span className="method-card-label">Baseline policy</span>
          </article>

          <article className="method-card adaptive-method-card">
            <div className="method-card-icon adaptive-icon">A</div>
            <h3>Adaptive-Horizon PPO</h3>
            <p>
              The adaptive variant changes γ and λ according to a schedule
              as training progresses.
            </p>
            <span className="method-card-label">Adaptive policy</span>
          </article>
        </div>

        <div className="method-explanation">
          <h3>What do γ and λ control?</h3>
          <div className="parameter-row">
            <strong>Discount factor (γ)</strong>
            <p>
              Controls how strongly future rewards contribute to the
              learning objective. Larger values place more weight on
              longer-term rewards.
            </p>
          </div>
          <div className="parameter-row">
            <strong>GAE parameter (λ)</strong>
            <p>
              Controls the bias-variance trade-off in Generalized Advantage
              Estimation, which estimates the advantages used to update
              the policy.
            </p>
          </div>
        </div>
      </section>

      <section className="method-section">
        <div className="method-section-heading">
          <span className="method-number">02</span>
          <div>
            <h2>Scheduling strategy</h2>
            <p>Illustration of a linear schedule for the adaptive parameters.</p>
          </div>
        </div>

        <div className="schedule-chart-card">
        <div className="schedule-chart-heading">
            <div>
            <h3>Linear scheduling over training</h3>
            <p>Parameter values across 100 training updates</p>
            </div>
            <span className="chart-badge">100 updates</span>
        </div>

        <div className="schedule-chart">
            <ResponsiveContainer width="100%" height={320}>
            <LineChart
                data={scheduleData}
                margin={{ top: 12, right: 20, left: 4, bottom: 8 }}
            >
                <CartesianGrid strokeDasharray="3 3" stroke="#e9e7f2" />
                <XAxis
                dataKey="update"
                type="number"
                domain={[1, 100]}
                ticks={[1, 20, 40, 60, 80, 100]}
                label={{ value: 'Training update', position: 'insideBottom', offset: -4 }}
                />
                <YAxis
                domain={[0.75, 1]}
                tickFormatter={(value) => value.toFixed(2)}
                />
                <Tooltip
                labelFormatter={(value) => `Update ${value}`}
                formatter={(value, name) => [
                    Number(value).toFixed(4),
                    name === 'Discount factor (γ)'
                    ? 'Discount factor (γ)'
                    : name === 'GAE parameter (λ)'
                        ? 'GAE parameter (λ)'
                        : name,
                ]}
                />
                <Legend verticalAlign="top" height={36} />
                <Line
                type="linear"
                dataKey="gamma"
                name="Discount factor (γ)"
                stroke="#7967e8"
                strokeWidth={3}
                dot={false}
                activeDot={{ r: 5 }}
                />
                <Line
                type="linear"
                dataKey="lambda"
                name="GAE parameter (λ)"
                stroke="#4d9ce8"
                strokeWidth={3}
                dot={false}
                activeDot={{ r: 5 }}
                />
            </LineChart>
            </ResponsiveContainer>
        </div>

        </div>
        
        <div className="schedule-formula-card">
        <h3>Linear scheduling equations</h3>
        <p>
            Both parameters follow a linear schedule over the 100 training updates.
        </p>

        <div className="formula-block">
            <span>Normalized progress</span>
            <code>pₜ = (t − 1) / (T − 1)</code>
        </div>

        <div className="formula-block">
            <span>Discount factor</span>
            <code>γₜ = 0.90 + pₜ × (0.99 − 0.90)</code>
        </div>

        <div className="formula-block">
            <span>GAE parameter</span>
            <code>λₜ = 0.80 + pₜ × (0.95 − 0.80)</code>
        </div>

        <p className="formula-note">
            Here, t is the current update and T = 100. The schedule starts at
            update 1 and reaches its final values at update 100.
        </p>
        </div>



        <div className="method-note">
        <strong>Reproducibility note</strong>
        <p>
            The adaptive PPO configuration uses linear scheduling over 100 updates.
            The discount factor γ increases from 0.90 to 0.99, while the GAE
            parameter λ increases from 0.80 to 0.95. Both schedules use the same
            normalized progress function, with values evaluated at each training
            update.
        </p>
        </div>
      </section>

      <section className="method-section">
        <div className="method-section-heading">
          <span className="method-number">03</span>
          <div>
            <h2>Experimental protocol</h2>
            <p>
                Fixed and adaptive PPO are compared across the same 15 random seeds,
                with 100 evaluation episodes per model.
            </p>
          </div>
        </div>

        <div className="protocol-stats">
          <article>
            <strong>15</strong>
            <span>Paired random seeds</span>
          </article>
          <article>
            <strong>100</strong>
            <span>Evaluation episodes per model</span>
          </article>
          <article>
            <strong>30</strong>
            <span>Evaluated model configurations</span>
          </article>
          <article>
            <strong>3,000</strong>
            <span>Total evaluation episodes</span>
          </article>
        </div>

        <div className="seed-list-card">
          <h3>Training seeds</h3>
          <p>
            Both algorithm variants are evaluated using the same set of
            training seed values.
          </p>
          <div className="seed-chip-list">
            {seeds.map((seed) => (
              <span key={seed}>{seed}</span>
            ))}
          </div>
        </div>
      </section>

      <section className="method-section">
        <div className="method-section-heading">
          <span className="method-number">04</span>
          <div>
            <h2>Evaluation metrics</h2>
            <p>Three complementary measures describe policy performance.</p>
          </div>
        </div>

        <div className="metric-definition-grid">
          <article>
            <h3>Success rate</h3>
            <p>
              The proportion of evaluation episodes in which the task
              succeeds. Higher values indicate more reliable task completion.
            </p>
          </article>
          <article>
            <h3>Mean reward</h3>
            <p>
              Average reward per episode. This provides a reward-based view
              of performance alongside the success rate.
            </p>
          </article>
          <article>
            <h3>Mean episode steps</h3>
            <p>
              Average number of steps per episode. Fewer steps can indicate
              greater efficiency when success is maintained.
            </p>
          </article>
        </div>
      </section>

      <section className="method-section">
        <div className="method-section-heading">
          <span className="method-number">05</span>
          <div>
            <h2>Statistical analysis</h2>
            <p>Compare the algorithms using paired seed-level differences.</p>
          </div>
        </div>

        <div className="method-explanation">
          <div className="parameter-row">
            <strong>Paired differences</strong>
            <p>
              For each seed, compare Adaptive PPO with Fixed PPO. The
              direction of the difference depends on the metric: positive
              success-rate differences favor Adaptive PPO, while negative
              step differences favor it.
            </p>
          </div>
          <div className="parameter-row">
            <strong>95% confidence interval</strong>
            <p>
              Summarizes uncertainty around the estimated mean paired
              difference. An interval that excludes zero indicates that
              zero is not included in that interval under the chosen method.
            </p>
          </div>
          <div className="parameter-row">
            <strong>Exact sign-flip test</strong>
            <p>
              Assesses the paired differences by evaluating sign assignments
              under the null hypothesis. The reported p-value should be
              interpreted alongside the effect size and confidence interval.
            </p>
          </div>
        </div>
      </section>

      <footer className="method-footer">
        <strong>Reproducibility</strong>
        <p>
          This page documents the experimental design and evaluation
          framework. Consult the saved configuration files, training code,
          and statistical-analysis CSV for the authoritative implementation
          details and numerical results.
        </p>
      </footer>
    </div>
  )
}

export default Methodology
