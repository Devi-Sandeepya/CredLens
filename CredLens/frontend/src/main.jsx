import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const DEMO_ID = 100002;

// --------------------------------------------------------
// Fixed demo personas (Section 30a of the plan).
// Fill in personB / personC applicant IDs once identified
// from your own data exploration — do not browse live on
// stage. 100002 is confirmed to return DECLINE.
// --------------------------------------------------------
const PERSONAS = [
  { label: "Thin-file / good", id: 100044, hint: "confirmed APPROVE" },
  { label: "Deteriorating", id: 100002, hint: "confirmed DECLINE" },
  { label: "Suspicious", id: 100082, hint: "confirmed UNUSUAL / REFER" },
];

const DECISION_COLOR = {
  APPROVE: "var(--mint)",
  REFER: "var(--amber)",
  DECLINE: "var(--coral)",
};

function App() {
  const [id, setId] = useState(DEMO_ID);
  const [decision, setDecision] = useState(null);
  const [prevDecision, setPrevDecision] = useState(null);
  const [timeline, setTimeline] = useState(null);
  const [factors, setFactors] = useState([]);
  const [modelFactors, setModelFactors] = useState(null);
  const [loading, setLoading] = useState(false);
  const [updating, setUpdating] = useState(false);

  async function loadApplicant(applicantId) {
    const targetId = applicantId ?? id;
    if (!targetId) return;
    setLoading(true);
    setPrevDecision(null);
    try {
      const res = await fetch("/api/v1/decision", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ applicantId: Number(targetId) }),
      });
      const data = await res.json();
      setDecision(data);

      const [t, f] = await Promise.all([
        fetch(`http://localhost:8001/api/v1/applicants/${targetId}/timeline`),
        fetch(
          `http://localhost:8001/api/v1/applicants/${targetId}/explanation-factors`
        ),
      ]);
      setTimeline(await t.json());
      const explanationData = await f.json();
      setFactors(explanationData.contextualFactors || []);
      setModelFactors(explanationData.modelExplanation || null);
    } catch (e) {
      alert(
        "Could not load applicant. Make sure ML service, backend and data/model are running."
      );
    } finally {
      setLoading(false);
    }
  }

  async function simulateBehavior() {
    setUpdating(true);
    setPrevDecision(decision);
    try {
      const res = await fetch(
        `http://localhost:8001/api/v1/applicants/${id}/behavior/update`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            timestamp: new Date().toISOString(),
            paymentAmount: 8000,
            scheduledAmount: 12000,
            balance: 84000,
            daysPastDue: 7,
          }),
        }
      );
      setDecision(await res.json());
    } catch (e) {
      alert("Could not reach the ML service for the live update.");
    } finally {
      setUpdating(false);
    }
  }

  const riskPct = decision ? Math.round(decision.riskScore * 100) : 0;
  const confPct = decision ? Math.round(decision.confidence) : 0;
  const decisionColor = decision
    ? DECISION_COLOR[decision.decision] || "var(--ink-3)"
    : "var(--ink-3)";
  const isLive = decision?.mode === "LIVE_BEHAVIOR_UPDATE";

  return (
    <div className="app">
      <header className="hero">
        <div className="heroTop">
          <span className="eyebrow">CREDLENS</span>
          <span className="protoBadge">
            <i className="dot" /> Prototype
          </span>
        </div>
        <h1>Proactive, contextual credit decisioning</h1>
        <p className="sub">
          Risk · Evidence Confidence · Behavioral Trajectory · Integrity
        </p>

        <BeforeAfter />
      </header>

      <div className="layout">
        <aside className="rail">
          <div className="railLabel">Audit Rail</div>
          <RailRow label="Applicant" value={decision ? `#${decision.applicantId}` : "—"} />
          <RailRow label="Model" value={decision?.modelVersion || "—"} />
          <RailRow label="Policy" value={decision?.policyVersion || "—"} />
          <RailRow
            label="Mode"
            value={
              decision ? (
                <span className={`modeTag ${isLive ? "live" : "batch"}`}>
                  <i className="dot" />
                  {decision.mode}
                </span>
              ) : (
                "—"
              )
            }
          />
          {decision?.policyThresholds && (
            <div className="thresholds">
              <div className="railLabel small">Policy thresholds</div>
              <RailRow
                label="Approve below"
                value={`${(decision.policyThresholds.approveBelow * 100).toFixed(0)}%`}
              />
              <RailRow
                label="Decline at/above"
                value={`${(decision.policyThresholds.declineAtOrAbove * 100).toFixed(0)}%`}
              />
              <RailRow
                label="Min. confidence"
                value={`${decision.policyThresholds.minimumConfidence.toFixed(0)}%`}
              />
            </div>
          )}
        </aside>

        <main>
          <section className="toolbar">
            <div className="personas">
              {PERSONAS.map((p) => (
                <button
                  key={p.label}
                  className="personaChip"
                  disabled={!p.id || loading}
                  title={p.hint}
                  onClick={() => {
                    setId(p.id);
                    loadApplicant(p.id);
                  }}
                >
                  {p.label}
                  {!p.id && <span className="chipTodo">TBD</span>}
                </button>
              ))}
            </div>
            <div className="manual">
              <label>Applicant ID</label>
              <input value={id} onChange={(e) => setId(e.target.value)} />
              <button
                className="primaryBtn"
                onClick={() => loadApplicant()}
                disabled={loading}
              >
                {loading ? "Loading…" : "Open Applicant 360"}
              </button>
            </div>
          </section>

          {decision && (
            <>
              <section className="scoreRow">
                <RiskGauge value={riskPct} color={decisionColor} />
                <ConfidenceRing value={confPct} />
                <StatusCard
                  label="Integrity"
                  value={decision.integrityStatus}
                  tone={decision.integrityStatus === "NORMAL" ? "mint" : "coral"}
                />
                <StatusCard
                  label="Decision"
                  value={decision.decision}
                  tone={
                    decision.decision === "APPROVE"
                      ? "mint"
                      : decision.decision === "DECLINE"
                      ? "coral"
                      : "amber"
                  }
                  big
                />
              </section>

              {prevDecision && (
                <section className="deltaBanner">
                  <span className="deltaLabel">Live behavioral event applied</span>
                  <span className="deltaRow">
                    Risk{" "}
                    <s>{Math.round(prevDecision.riskScore * 100)}%</s>
                    <span className="arrow">→</span>
                    <b style={{ color: decisionColor }}>{riskPct}%</b>
                  </span>
                  <span className="deltaRow">
                    Decision{" "}
                    <s>{prevDecision.decision}</s>
                    <span className="arrow">→</span>
                    <b style={{ color: decisionColor }}>{decision.decision}</b>
                  </span>
                </section>
              )}

              <section className="two">
                <div className="panel">
                  <div className="panelHead">
                    <h2>Why?</h2>
                  </div>

                  {modelFactors && (
                    <div className="factorGroup">
                      <div className="factorGroupLabel">
                        SHAP · {modelFactors.model}
                      </div>
                      {modelFactors.topRiskFactors?.slice(0, 3).map((f, i) => (
                        <FactorRow
                          key={`r${i}`}
                          text={`${f.label}: ${f.displayValue}`}
                          direction="INCREASED_RISK"
                        />
                      ))}
                      {modelFactors.topProtectiveFactors
                        ?.slice(0, 3)
                        .map((f, i) => (
                          <FactorRow
                            key={`p${i}`}
                            text={`${f.label}: ${f.displayValue}`}
                            direction="REDUCED_RISK"
                          />
                        ))}
                    </div>
                  )}

                  <div className="factorGroup">
                    <div className="factorGroupLabel">Contextual evidence</div>
                    {factors.map((f, i) => (
                      <FactorRow key={i} text={f.factor} direction={f.direction} />
                    ))}
                  </div>
                </div>

                <div className="panel">
                  <div className="panelHead">
                    <h2>Risk Journey</h2>
                    <span className="badge muted">
                      <i className="dot" /> PRECOMPUTED
                    </span>
                  </div>
                  {timeline?.points?.map((p, i) => {
                    const value =
                      p.riskScore !== undefined
                        ? p.riskScore * 100
                        : (p.onTimeRatio ?? 0) * 100;
                    return (
                      <div className="journeyRow" key={i}>
                        <span className="journeyLabel">{p.label}</span>
                        <div className="journeyBar">
                          <div
                            className="journeyFill"
                            style={{
                              width: `${value}%`,
                              background:
                                p.riskScore !== undefined ? decisionColor : "var(--ink-3)",
                            }}
                          />
                        </div>
                        <b className="journeyValue">{value.toFixed(1)}%</b>
                      </div>
                    );
                  })}
                  {timeline?.behaviorTrend && (
                    <div className="trendTag">
                      Trend: <b>{timeline.behaviorTrend}</b>
                    </div>
                  )}
                </div>
              </section>

              <section className="panel liveUpdate">
                <div className="panelHead">
                  <h2>Live behavioral update</h2>
                  <span className={`badge ${isLive ? "live" : "muted"}`}>
                    <i className="dot" /> {isLive ? "LIVE" : "READY"}
                  </span>
                </div>
                <p>
                  Simulate a new payment-delay event arriving for this applicant and
                  recompute their state through the real M5 model — end to end, live.
                  No Kafka/Kinesis required for the prototype.
                </p>
                <button
                  className="primaryBtn"
                  onClick={simulateBehavior}
                  disabled={updating}
                >
                  {updating ? "Recomputing…" : "Receive new behavioral event"}
                </button>
              </section>
            </>
          )}
        </main>
      </div>
    </div>
  );
}

function RailRow({ label, value }) {
  return (
    <div className="railRow">
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}

function FactorRow({ text, direction }) {
  const sign =
    direction === "INCREASED_RISK" ? "+" : direction === "REDUCED_RISK" ? "−" : "·";
  const tone =
    direction === "INCREASED_RISK"
      ? "coral"
      : direction === "REDUCED_RISK"
      ? "mint"
      : "amber";
  return (
    <div className="factor">
      <span className={`factorSign ${tone}`}>{sign}</span>
      {text}
    </div>
  );
}

function StatusCard({ label, value, tone, big }) {
  return (
    <div className={`statusCard tone-${tone} ${big ? "big" : ""}`}>
      <span className="statusLabel">{label}</span>
      <strong className="statusValue">{value}</strong>
    </div>
  );
}

function RiskGauge({ value, color }) {
  const r = 46;
  const c = 2 * Math.PI * r;
  const offset = c - (value / 100) * c;
  return (
    <div className="gaugeCard">
      <span className="statusLabel">Risk</span>
      <svg viewBox="0 0 120 120" className="gaugeSvg">
        <circle cx="60" cy="60" r={r} className="gaugeTrack" />
        <circle
          cx="60"
          cy="60"
          r={r}
          className="gaugeFill"
          style={{
            stroke: color,
            strokeDasharray: c,
            strokeDashoffset: offset,
          }}
        />
      </svg>
      <strong className="gaugeValue">{value}%</strong>
    </div>
  );
}

function ConfidenceRing({ value }) {
  const r = 46;
  const c = 2 * Math.PI * r;
  const offset = c - (value / 100) * c;
  return (
    <div className="gaugeCard signature">
      <span className="statusLabel">Evidence Confidence</span>
      <svg viewBox="0 0 120 120" className="gaugeSvg">
        <circle cx="60" cy="60" r={r} className="gaugeTrack" />
        <circle
          cx="60"
          cy="60"
          r={r}
          className="gaugeFill"
          style={{
            stroke: "var(--gold)",
            strokeDasharray: c,
            strokeDashoffset: offset,
          }}
        />
      </svg>
      <strong className="gaugeValue">{value}%</strong>
    </div>
  );
}

function BeforeAfter() {
  return (
    <div className="beforeAfter">
      <div className="baCol baTraditional">
        <span className="baLabel">Traditional approach</span>
        <div className="baLine">Income</div>
        <div className="baLine">Loan amount</div>
        <div className="baLine">Basic credit history</div>
        <div className="baArrow">↓</div>
        <div className="baLine baResult">Risk Score</div>
        <div className="baArrow">↓</div>
        <div className="baLine baUnknown">?</div>
      </div>
      <div className="baDivider" />
      <div className="baCol baCredlens">
        <span className="baLabel">CredLens</span>
        <div className="baLine">Risk <b>34%</b></div>
        <div className="baLine">Evidence Confidence <b>47%</b></div>
        <div className="baLine">Trajectory <b>↑ Deteriorating</b></div>
        <div className="baLine">Integrity <b>⚠ Unusual</b></div>
        <div className="baLine">Decision <b>REFER</b></div>
        <div className="baLine">Explanation <b>Available</b></div>
      </div>
    </div>
  );
}

function Card({ title, value }) {
  return (
    <div className="card">
      <span>{title}</span>
      <strong>{value}</strong>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);