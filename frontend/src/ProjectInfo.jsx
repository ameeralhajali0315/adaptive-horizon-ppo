
import './ProjectInfo.css'

const teamMembers = [
  { name: 'AMEER ALSABAGH ALHAJALI', id: '2300033922', branch: 'CSE 4' },
  { name: 'Team Member 2', id: 'Add student ID', branch: 'CSE 4' },
  { name: 'Team Member 3', id: 'Add student ID', branch: 'CSE 4' },
  { name: 'Team Member 4', id: 'Add student ID', branch: 'CSE 4' },
]

export default function ProjectInfo() {
  return (
    <div className="project-info-page">
      <section className="project-info-hero">
        <span className="project-info-eyebrow">CAPSTONE PROJECT</span>
        <h1>CINT-22</h1>
        <h2>
          Adaptive-Horizon PPO with Dynamic Discount and GAE Scheduling
        </h2>
        <p className="project-info-description">
          This project investigates an adaptive variant of Proximal Policy
          Optimization (PPO) for reinforcement learning.
          It dynamically schedules the discount factor and Generalized
          Advantage Estimation (GAE) parameter during training.
          The approach is compared with fixed-parameter PPO under a
          controlled experimental protocol.
          Performance is evaluated using success rate, reward, and
          episode length across paired random seeds.
          Statistical analysis is used to assess differences between
          the two approaches on the SparseGridWorld-v1 environment.
        </p>
      </section>

      <section className="project-info-section">
        <div className="project-info-section-heading">
          <span>01</span>
          <div>
            <h3>Project team</h3>
            <p>Team members and academic details</p>
          </div>
        </div>

        <div className="team-grid">
          {teamMembers.map((member, index) => (
            <article className="team-card" key={index}>
              <div className="team-member-number">
                {String(index + 1).padStart(2, '0')}
              </div>
              <h4>{member.name}</h4>
              <p><strong>Student ID:</strong> {member.id}</p>
              <p><strong>Branch:</strong> {member.branch}</p>
            </article>
          ))}
        </div>
      </section>

      <section className="project-info-section">
        <div className="project-info-section-heading">
          <span>02</span>
          <div>
            <h3>Project mentor</h3>
            <p>Faculty supervision details</p>
          </div>
        </div>

        <article className="mentor-card">
          <div>
            <span>Name</span>
            <strong>To be provided</strong>
          </div>
          <div>
            <span>Department</span>
            <strong>To be provided</strong>
          </div>
          <div>
            <span>Faculty ID</span>
            <strong>To be provided</strong>
          </div>
        </article>
      </section>
    </div>
  )
}
