import React, { useState } from "react";
import { createRoot } from "react-dom/client";
import "./styles.css";

const DEMO_ID = "";

const PERSONAS = [
  {
    label: "Thin-file / good",
    id: 100044,
    expected: "APPROVE",
    desc: "Low risk, high evidence confidence, no integrity flag.",
  },
  {
    label: "Deteriorating",
    id: 100002,
    expected: "DECLINE",
    desc: "Recent payment timing trending later than history.",
  },
  {
    label: "Suspicious",
    id: 100082,
    expected: "REFER",
    desc: "Moderate risk, but flagged UNUSUAL by the integrity layer.",
  },
];

const DECISION_COLOR = {
  APPROVE: "var(--mint)",
  REFER: "var(--amber)",
  DECLINE: "var(--coral)",
};

const TABS = ["Overview", "Risk Journey", "Why?", "Audit Trail"];

const LIMITATIONS = [
  "Trained on public anonymized historical data",
  "Not trained on Indian Account Aggregator data",
  "Anomaly layer is unsupervised — no fraud ground truth exists",
  "Confidence is a documented heuristic, not statistically calibrated",
  "Production deployment would require regulatory/fairness validation",
  "Human review remains necessary for REFER cases",
];

function formatCurrency(value) {
  if (value === undefined || value === null || isNaN(value)) return "—";
  return "₹" + Math.round(value).toLocaleString("en-IN");
}

function formatEmployment(daysEmployed) {
  if (daysEmployed === undefined || daysEmployed === null) return "—";
  if (daysEmployed > 0) return "Not currently employed";
  const years = Math.abs(daysEmployed) / 365;
  return years < 1 ? `${Math.round(years * 12)} months` : `${years.toFixed(1)} years`;
}

function formatAge(daysBirth) {
  if (daysBirth === undefined || daysBirth === null) return "—";
  return Math.floor(Math.abs(daysBirth) / 365) + " years";
}

function formatGender(code) {
  if (!code) return "—";
  return code === "M" ? "Male" : code === "F" ? "Female" : code;
}

function formatText(value) {
  if (value === undefined || value === null || value === "") return "—";
  return String(value).replace(/_/g, " ");
}

function App() {
  const [id, setId] = useState(DEMO_ID);
  const [decision, setDecision] = useState(null);
  const [beforeState, setBeforeState] = useState(null);
  const [event, setEvent] = useState(null);
  const [timeline, setTimeline] = useState(null);
  const [factors, setFactors] = useState([]);
  const [modelFactors, setModelFactors] = useState(null);
  const [aiExplanation, setAiExplanation] = useState(null);
  const [profile, setProfile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [updating, setUpdating] = useState(false);
  const [tab, setTab] = useState("Overview");

  const [user, setUser] = useState(null);
  const [authMode, setAuthMode] = useState("login");
  const [authForm, setAuthForm] = useState({
    username: "",
    password: "",
    fullName: "",
    age: "",
    role: "UNDERWRITER",
  });
  const [authError, setAuthError] = useState("");
  const [authLoading, setAuthLoading] = useState(false);

  async function loadApplicant(applicantId) {
    const targetId = applicantId ?? id;
    if (!targetId) return;
    setLoading(true);
    setBeforeState(null);
    setEvent(null);
    setTab("Overview");
    try {
      const res = await fetch("/api/v1/decision", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ applicantId: Number(targetId) }),
      });
      const data = await res.json();
      setDecision(data);

      const [t, f, p, feat] = await Promise.all([
        fetch(`http://localhost:8001/api/v1/applicants/${targetId}/timeline`),
        fetch(
          `http://localhost:8001/api/v1/applicants/${targetId}/explanation-factors`
        ),
        fetch(
          `http://localhost:8001/api/v1/applicants/${targetId}/policy-context`
        ),
        fetch(`http://localhost:8001/api/v1/applicants/${targetId}/features`),
      ]);
      setTimeline(await t.json());
      const explanationData = await f.json();
      setFactors(explanationData.contextualFactors || []);
      setModelFactors(explanationData.modelExplanation || null);
      const policyData = await p.json();
      setAiExplanation(policyData.aiExplanation || null);
      const featureData = await feat.json();
      setProfile(featureData.features || null);
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
    setBeforeState(decision);
    const payload = {
      timestamp: new Date().toISOString(),
      paymentAmount: 8000,
      scheduledAmount: 12000,
      balance: 84000,
      daysPastDue: 7,
    };
    setEvent(payload);
    try {
      const res = await fetch(
        `http://localhost:8001/api/v1/applicants/${id}/behavior/update`,
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload),
        }
      );
      setDecision(await res.json());
    } catch (e) {
      alert("Could not reach the ML service for the live update.");
    } finally {
      setUpdating(false);
    }
  }

  async function handleAuth(e) {
    e.preventDefault();
    setAuthError("");
    setAuthLoading(true);
    try {
      const endpoint = authMode === "login" ? "login" : "register";
      const body =
        authMode === "login"
          ? { username: authForm.username, password: authForm.password }
          : {
              username: authForm.username,
              password: authForm.password,
              fullName: authForm.fullName,
              age: authForm.age ? Number(authForm.age) : null,
              role: authForm.role,
            };
      const res = await fetch(`/api/v1/auth/${endpoint}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      const data = await res.json();
      if (!res.ok) {
        setAuthError(data.error || "Something went wrong");
        return;
      }
      if (authMode === "register") {
        setAuthMode("login");
        setAuthForm({ ...authForm, password: "", fullName: "", age: "", role: "UNDERWRITER" });
        setAuthError("Account created. Please log in.");
      } else {
        setUser(data);
      }
    } catch (err) {
      setAuthError("Could not reach the server.");
    } finally {
      setAuthLoading(false);
    }
  }

  function logout() {
    setUser(null);
  }

  const riskPct = decision ? Math.round(decision.riskScore * 100) : 0;
  const confPct = decision ? Math.round(decision.confidence) : 0;
  const decisionColor = decision
    ? DECISION_COLOR[decision.decision] || "var(--ink-3)"
    : "var(--ink-3)";
  const isLive = decision?.mode === "LIVE_BEHAVIOR_UPDATE";

  if (!user) {
    return (
      <div className="authScreen">
        <div className="authCard">
          <span className="eyebrow">CREDLENS</span>
          <h1 className="authTitle">Staff sign-in</h1>
          <p className="authSub">
            CredLens is an internal tool for bank underwriters and admins.
            Applicants are never users of this system.
          </p>

          <div className="authTabs">
            <button
              className={authMode === "login" ? "authTab active" : "authTab"}
              onClick={() => {
                setAuthMode("login");
                setAuthError("");
              }}
            >
              Log in
            </button>
            <button
              className={authMode === "register" ? "authTab active" : "authTab"}
              onClick={() => {
                setAuthMode("register");
                setAuthError("");
              }}
            >
              Register
            </button>
          </div>

          <form onSubmit={handleAuth} className="authForm">
            <label>Username</label>
            <input
              value={authForm.username}
              onChange={(e) => setAuthForm({ ...authForm, username: e.target.value })}
              required
            />

            <label>Password</label>
            <input
              type="password"
              value={authForm.password}
              onChange={(e) => setAuthForm({ ...authForm, password: e.target.value })}
              required
            />

            {authMode === "register" && (
              <>
                <label>Full name</label>
                <input
                  value={authForm.fullName}
                  onChange={(e) => setAuthForm({ ...authForm, fullName: e.target.value })}
                  required
                />

                <label>Age</label>
                <input
                  type="number"
                  value={authForm.age}
                  onChange={(e) => setAuthForm({ ...authForm, age: e.target.value })}
                />

                <label>Role</label>
                <select
                  value={authForm.role}
                  onChange={(e) => setAuthForm({ ...authForm, role: e.target.value })}
                >
                  <option value="UNDERWRITER">Underwriter</option>
                  <option value="ADMIN">Admin</option>
                </select>
              </>
            )}

            {authError && (
              <p
                className={`authError ${
                  authMode === "login" && authError.includes("created") ? "success" : ""
                }`}
              >
                {authError}
              </p>
            )}

            <button className="primaryBtn authSubmit" disabled={authLoading}>
              {authLoading
                ? "Please wait…"
                : authMode === "login"
                ? "Log in"
                : "Create account"}
            </button>
          </form>
        </div>
      </div>
    );
  }

  return (
    <div className="app">
      <header className="hero">
        <div className="heroTop">
          <span className="eyebrow">CREDLENS</span>
          <div className="heroTopRight">
            <span className="protoBadge">
              <i className="dot" /> Prototype
            </span>
            <div className="userBadge">
              <div className="userBadgeInfo">
                <b>{user.fullName}</b>
                <span>
                  {user.role} · {user.age ? `${user.age} yrs` : "—"}
                </span>
              </div>
              <button className="logoutBtn" onClick={logout}>
                Log out
              </button>
            </div>
          </div>
        </div>
        <h1>Proactive, contextual credit decisioning</h1>
        <p className="sub">
          Risk · Evidence Confidence · Behavioral Trajectory · Integrity
        </p>

        <section className="toolbar heroToolbar">
          <div className="manual">
            <label>Applicant ID</label>
            <input
              value={id}
              placeholder="Type an applicant ID…"
              onChange={(e) => setId(e.target.value)}
            />
            <button
              className="primaryBtn"
              onClick={() => loadApplicant()}
              disabled={loading || !id}
            >
              {loading ? "Loading…" : "Open Applicant 360"}
            </button>
          </div>
        </section>

        <BeforeAfter
          decision={decision}
          riskPct={riskPct}
          confPct={confPct}
          trend={timeline?.behaviorTrend}
        />
      </header>

      <div className="layout">
        <aside className="rail">
          <div className="railLabel">Audit Rail</div>
          {decision?.decisionId && (
            <RailRow label="Decision ID" value={decision.decisionId} />
          )}
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

          <div className="railLabel small">Prototype limitations</div>
          <ul className="limitList">
            {LIMITATIONS.map((l, i) => (
              <li key={i}>{l}</li>
            ))}
          </ul>
        </aside>

        <main>
          {decision && (
            <>
              {profile && (
                <section className="panel profilePanel">
                  <div className="panelHead">
                    <h2>Applicant Profile</h2>
                    <span className="badge muted">
                      <i className="dot" /> FROM APPLICATION DATA
                    </span>
                  </div>
                  <div className="profileGrid">
                    <ProfileCard label="Age" value={formatAge(profile.DAYS_BIRTH)} />
                    <ProfileCard label="Gender" value={formatGender(profile.CODE_GENDER)} />
                    <ProfileCard
                      label="Income"
                      value={formatCurrency(profile.AMT_INCOME_TOTAL)}
                    />
                    <ProfileCard
                      label="Loan amount"
                      value={formatCurrency(profile.AMT_CREDIT)}
                    />
                    <ProfileCard
                      label="Employment"
                      value={formatEmployment(profile.DAYS_EMPLOYED)}
                    />
                    <ProfileCard
                      label="Family status"
                      value={formatText(profile.NAME_FAMILY_STATUS)}
                    />
                    <ProfileCard
                      label="Education"
                      value={formatText(profile.NAME_EDUCATION_TYPE)}
                    />
                    <ProfileCard
                      label="Housing"
                      value={formatText(profile.NAME_HOUSING_TYPE)}
                    />
                    <ProfileCard
                      label="Children"
                      value={
                        profile.CNT_CHILDREN !== undefined
                          ? String(profile.CNT_CHILDREN)
                          : "—"
                      }
                    />
                  </div>
                </section>
              )}

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

              <nav className="tabs">
                {TABS.map((t) => (
                  <button
                    key={t}
                    className={`tabBtn ${tab === t ? "active" : ""}`}
                    onClick={() => setTab(t)}
                  >
                    {t}
                  </button>
                ))}
              </nav>

              {tab === "Overview" && (
                <section className="panel liveUpdate">
                  <div className="panelHead">
                    <h2>Live behavioral update</h2>
                    <span className={`badge ${isLive ? "live" : "muted"}`}>
                      <i className="dot" /> {isLive ? "LIVE" : "READY"}
                    </span>
                  </div>
                  <p>
                    Simulate a new payment-delay event arriving for this applicant
                    and recompute their state through the real M5 model — end to
                    end, live. No Kafka/Kinesis required for the prototype.
                  </p>
                  <button
                    className="primaryBtn"
                    onClick={simulateBehavior}
                    disabled={updating}
                  >
                    {updating ? "Recomputing…" : "Receive new behavioral event"}
                  </button>

                  {beforeState && event && (
                    <div className="diffGrid">
                      <div className="diffCol">
                        <span className="diffLabel">Before event</span>
                        <DiffRow label="Risk" value={`${Math.round(beforeState.riskScore * 100)}%`} />
                        <DiffRow label="Confidence" value={`${Math.round(beforeState.confidence)}%`} />
                        <DiffRow label="Decision" value={beforeState.decision} />
                      </div>
                      <div className="diffCol diffEvent">
                        <span className="diffLabel">New event</span>
                        <DiffRow label="Payment" value={`₹${event.paymentAmount.toLocaleString()}`} />
                        <DiffRow label="Scheduled" value={`₹${event.scheduledAmount.toLocaleString()}`} />
                        <DiffRow label="Days past due" value={event.daysPastDue} />
                      </div>
                      <div className="diffCol diffAfter">
                        <span className="diffLabel">After event</span>
                        <DiffRow label="Risk" value={`${riskPct}%`} highlight={decisionColor} />
                        <DiffRow label="Confidence" value={`${confPct}%`} />
                        <DiffRow label="Decision" value={decision.decision} highlight={decisionColor} />
                      </div>
                    </div>
                  )}
                </section>
              )}

              {tab === "Risk Journey" && (
                <section className="panel">
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
                </section>
              )}

              {tab === "Why?" && (
                <section className="panel">
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

                  {aiExplanation && (
                    <div className="factorGroup aiExplanationBox">
                      <div className="factorGroupLabel">
                        AI Explanation
                        <span className={`aiSourceTag ${aiExplanation.source}`}>
                          {aiExplanation.source === "llm"
                            ? `Groq · ${aiExplanation.model}`
                            : "Fallback"}
                        </span>
                      </div>
                      <p className="aiExplanationText">{aiExplanation.explanation}</p>
                      <p className="aiExplanationNote">
                        Grounded in retrieved policy text via pgvector semantic
                        search. This explanation does not influence the decision —
                        it is generated after the deterministic Policy Engine
                        has already decided.
                      </p>
                    </div>
                  )}
                </section>
              )}

              {tab === "Audit Trail" && (
                <section className="panel">
                  <div className="panelHead">
                    <h2>Decision Audit Trail</h2>
                    {decision.decisionId && (
                      <span className="badge muted">{decision.decisionId}</span>
                    )}
                  </div>
                  <div className="auditGrid">
                    <RailRow label="Decision ID" value={decision.decisionId || "—"} />
                    <RailRow
                      label="Persisted at"
                      value={
                        decision.persistedAt
                          ? new Date(decision.persistedAt).toLocaleString()
                          : "—"
                      }
                    />
                    <RailRow label="Applicant" value={`#${decision.applicantId}`} />
                    <RailRow label="Risk model" value={decision.modelVersion} />
                    <RailRow label="Policy" value={decision.policyVersion} />
                    <RailRow label="Mode" value={decision.mode} />
                    <RailRow label="Risk" value={`${riskPct}%`} />
                    <RailRow label="Confidence" value={`${confPct}%`} />
                    <RailRow label="Integrity" value={decision.integrityStatus} />
                    <RailRow label="Decision" value={decision.decision} />
                  </div>
                  <p className="auditNote">
                    Every decision is written to PostgreSQL with a unique Decision
                    ID, making it fully reproducible and auditable after the fact.
                  </p>
                </section>
              )}
            </>
          )}

          <section className="personaGrid personaGridBottom">
            {PERSONAS.map((p) => (
              <button
                key={p.label}
                className={`personaCard ${id === p.id ? "active" : ""}`}
                disabled={loading}
                onClick={() => {
                  setId(p.id);
                  loadApplicant(p.id);
                }}
              >
                <div className="personaHead">
                  <span className="personaName">{p.label}</span>
                  <span
                    className="personaOutcome"
                    style={{ color: DECISION_COLOR[p.expected] }}
                  >
                    {p.expected}
                  </span>
                </div>
                <p>{p.desc}</p>
                <span className="personaId">#{p.id}</span>
              </button>
            ))}
          </section>
        </main>
      </div>
    </div>
  );
}

function ProfileCard({ label, value }) {
  return (
    <div className="profileCard">
      <span className="profileCardLabel">{label}</span>
      <b className="profileCardValue">{value}</b>
    </div>
  );
}

function DiffRow({ label, value, highlight }) {
  return (
    <div className="diffRow">
      <span>{label}</span>
      <b style={highlight ? { color: highlight } : undefined}>{value}</b>
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
          style={{ stroke: color, strokeDasharray: c, strokeDashoffset: offset }}
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
          style={{ stroke: "var(--gold)", strokeDasharray: c, strokeDashoffset: offset }}
        />
      </svg>
      <strong className="gaugeValue">{value}%</strong>
    </div>
  );
}

function BeforeAfter({ decision, riskPct, confPct, trend }) {
  const live = !!decision;
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
        <span className="baLabel">
          CredLens {live ? `· Applicant #${decision.applicantId} (live)` : "· Search an applicant ID above"}
        </span>
        <div className="baLine">
          Risk <b>{live ? `${riskPct}%` : "—"}</b>
        </div>
        <div className="baLine">
          Evidence Confidence <b>{live ? `${confPct}%` : "—"}</b>
        </div>
        <div className="baLine">
          Trajectory <b>{live && trend ? trend : "—"}</b>
        </div>
        <div className="baLine">
          Integrity{" "}
          <b>{live ? (decision.integrityStatus === "UNUSUAL" ? "⚠ Unusual" : "✓ Normal") : "—"}</b>
        </div>
        <div className="baLine">
          Decision{" "}
          <b style={live ? { color: DECISION_COLOR[decision.decision] } : undefined}>
            {live ? decision.decision : "—"}
          </b>
        </div>
        <div className="baLine">Explanation <b>{live ? "Available" : "—"}</b></div>
      </div>
    </div>
  );
}

createRoot(document.getElementById("root")).render(<App />);