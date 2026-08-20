import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const DEMO_ID = 100001;

function App() {
  const [id, setId] = useState(DEMO_ID);
  const [decision, setDecision] = useState(null);
  const [timeline, setTimeline] = useState(null);
  const [factors, setFactors] = useState([]);
  const [loading, setLoading] = useState(false);

  async function loadApplicant() {
    setLoading(true);
    try {
      const res = await fetch("/api/v1/decision", {
        method: "POST",
        headers: {"Content-Type": "application/json"},
        body: JSON.stringify({ applicantId: Number(id)})
      });
      const data = await res.json();
      setDecision(data);

      const [t, f] = await Promise.all([
        fetch(`http://localhost:8001/api/v1/applicants/${id}/timeline`),
        fetch(`http://localhost:8001/api/v1/applicants/${id}/explanation-factors`)
      ]);
      setTimeline(await t.json());
      setFactors((await f.json()).factors || []);
    } catch (e) {
      alert("Could not load applicant. Make sure ML service, backend and data/model are running.");
    } finally {
      setLoading(false);
    }
  }

  async function simulateBehavior() {
    const res = await fetch(`http://localhost:8001/api/v1/applicants/${id}/behavior/update`, {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        timestamp: new Date().toISOString(),
        paymentAmount: 8000,
        scheduledAmount: 12000,
        balance: 84000,
        daysPastDue: 7
      })
    });
    setDecision(await res.json());
  }

  return (
    <div className="app">
      <header>
        <div>
          <p className="eyebrow">CREDLENS</p>
          <h1>Proactive, contextual credit decisioning</h1>
          <p className="sub">Risk + Evidence Confidence + Behavioral Trajectory + Integrity</p>
        </div>
        <span className="live">● Prototype</span>
      </header>

      <main>
        <section className="toolbar">
          <label>Applicant ID</label>
          <input value={id} onChange={e => setId(e.target.value)} />
          <button onClick={loadApplicant} disabled={loading}>
            {loading ? "Loading..." : "Open Applicant 360"}
          </button>
        </section>

        {decision && (
          <>
            <section className="grid">
              <Card title="Risk" value={`${(decision.riskScore * 100).toFixed(1)}%`} />
              <Card title="Evidence Confidence" value={`${decision.confidence}%`} />
              <Card title="Integrity" value={decision.integrityStatus} />
              <Card title="Decision" value={decision.decision} />
            </section>

            <section className="panel">
              <div className="panelHead">
                <h2>Decision context</h2>
                <span className="badge">LIVE SCORE</span>
              </div>
              <div className="audit">
                <div><b>Model</b><span>{decision.modelVersion}</span></div>
                <div><b>Policy</b><span>{decision.policyVersion}</span></div>
                <div><b>Mode</b><span>{decision.mode}</span></div>
              </div>
            </section>

            <section className="two">
              <div className="panel">
                <h2>Why?</h2>
                {factors.map((f, i) => (
                  <div className="factor" key={i}>
                    <span>{f.direction === "INCREASED_RISK" ? "+" : "−"}</span>
                    {f.factor}
                  </div>
                ))}
              </div>

              <div className="panel">
                <div className="panelHead">
                  <h2>Risk Journey</h2>
                  <span className="badge muted">PRECOMPUTED</span>
                </div>
                {timeline?.points?.map((p, i) => (
                  <div className="timeline" key={i}>
                    <span>{p.label}</span>
                    <div className="bar"><div style={{width: `${p.riskScore * 100}%`}} /></div>
                    <b>{(p.riskScore * 100).toFixed(1)}%</b>
                  </div>
                ))}
              </div>
            </section>

            <section className="panel demo">
              <h2>Live behavioral update</h2>
              <p>Inject a new payment-delay event and recompute the applicant state live.</p>
              <button onClick={simulateBehavior}>Receive new behavioral event</button>
              <span className="hint">No Kafka/Kinesis required for the prototype.</span>
            </section>
          </>
        )}
      </main>
    </div>
  );
}

function Card({title, value}) {
  return (
    <div className="card">
      <span>{title}</span>
      <strong>{value}</strong>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);
