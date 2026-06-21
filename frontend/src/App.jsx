import { useEffect, useMemo, useRef, useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

const MODULES = [
  {
    key: "vehicle",
    title: "Vehicle Detection",
    short: "Vehicles",
    description: "Locate and classify cars, buses, trucks and motorcycles.",
  },
  {
    key: "license_plate",
    title: "License Plate Recognition",
    short: "License plates",
    description: "Detect visible plates and extract registration text with OCR.",
  },
  {
    key: "helmet",
    title: "Helmet Compliance",
    short: "Helmets",
    description: "Review detected riders for helmet use.",
  },
  {
    key: "seatbelt",
    title: "Seat Belt Compliance",
    short: "Seat belts",
    description: "Inspect supported vehicle regions for visible seat belt use.",
  },
  {
    key: "triple_riding",
    title: "Triple Riding Detection",
    short: "Triple riding",
    description: "Screen two-wheelers for three or more overlapping riders.",
  },
  {
    key: "wrong_side",
    title: "Wrong-Side Screening",
    short: "Wrong-side",
    description: "Flag vehicles inside the configured opposing traffic zone.",
  },
  {
    key: "illegal_parking",
    title: "Illegal Parking Review",
    short: "Illegal parking",
    description: "Review vehicles inside a marked restricted parking area.",
  },
  {
    key: "redlight",
    title: "Traffic Signal & Red-Light",
    short: "Traffic signal",
    description: "Detect signal state and assess stop-line crossing with vehicles.",
  },
];

const MODULE_MAP = Object.fromEntries(MODULES.map((item) => [item.key, item]));

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [selectedModules, setSelectedModules] = useState(MODULES.map((item) => item.key));
  const [confidence, setConfidence] = useState(0.25);
  const [stopline, setStopline] = useState(0.72);
  const [preprocessing, setPreprocessing] = useState("auto");
  const [parkingZone, setParkingZone] = useState([0.65, 0.35, 0.98, 0.95]);
  const [legalTrafficSide, setLegalTrafficSide] = useState("left");
  const [loading, setLoading] = useState(false);
  const [dragging, setDragging] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [activeFilter, setActiveFilter] = useState("all");
  const [search, setSearch] = useState("");
  const [analytics, setAnalytics] = useState(null);
  const [evaluation, setEvaluation] = useState(null);
  const inputRef = useRef(null);

  useEffect(() => () => preview && URL.revokeObjectURL(preview), [preview]);
  useEffect(() => {
    Promise.all([
      fetch(`${API_BASE}/analytics`).then((response) => response.ok ? response.json() : null),
      fetch(`${API_BASE}/evaluation`).then((response) => response.ok ? response.json() : null),
    ]).then(([analyticsData, evaluationData]) => {
      setAnalytics(analyticsData);
      setEvaluation(evaluationData);
    }).catch(() => {});
  }, []);

  const allSelected = selectedModules.length === MODULES.length;
  const canAssessRedLight = selectedModules.includes("redlight") && selectedModules.includes("vehicle");

  const setSelectedFile = (selected) => {
    if (!selected) return;
    if (!selected.type.startsWith("image/")) {
      setError("Please select a JPG, PNG, WebP or BMP image.");
      return;
    }
    if (preview) URL.revokeObjectURL(preview);
    setFile(selected);
    setPreview(URL.createObjectURL(selected));
    setResult(null);
    setError("");
  };

  const removeFile = () => {
    if (preview) URL.revokeObjectURL(preview);
    setFile(null);
    setPreview(null);
    setResult(null);
    setError("");
    if (inputRef.current) inputRef.current.value = "";
  };

  const toggleModule = (key) => {
    setSelectedModules((current) =>
      current.includes(key) ? current.filter((item) => item !== key) : [...current, key],
    );
    setResult(null);
    setError("");
  };

  const toggleAll = () => {
    setSelectedModules(allSelected ? [] : MODULES.map((item) => item.key));
    setResult(null);
    setError("");
  };

  const analyze = async () => {
    if (!file) {
      setError("Add a traffic image before starting the analysis.");
      return;
    }
    if (selectedModules.length === 0) {
      setError("Select at least one analysis module.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setActiveFilter("all");
    setSearch("");

    const form = new FormData();
    form.append("file", file);
    form.append("conf", String(confidence));
    form.append("stopline_y_ratio", String(stopline));
    form.append("modules", selectedModules.join(","));
    form.append("preprocessing", preprocessing);
    form.append("no_parking_zone", parkingZone.join(","));
    form.append("legal_traffic_side", legalTrafficSide);

    try {
      const response = await fetch(`${API_BASE}/analyze`, { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok || !data.success) {
        throw new Error(data.detail || data.error || "The analysis could not be completed.");
      }
      setResult(data);
      fetch(`${API_BASE}/analytics`).then((response) => response.json()).then(setAnalytics).catch(() => {});
      window.setTimeout(() => document.getElementById("results")?.scrollIntoView({ behavior: "smooth" }), 80);
    } catch (requestError) {
      setError(requestError.message || "Unable to connect to the analysis service.");
    } finally {
      setLoading(false);
    }
  };

  const metadata = result?.meta || [];
  const filteredMetadata = useMemo(() => {
    const normalizedSearch = search.trim().toLowerCase();
    return metadata.filter((row) => {
      const rowModule = moduleKeyForRow(row.module);
      const matchesModule = activeFilter === "all" || rowModule === activeFilter;
      const matchesSearch = !normalizedSearch || Object.values(row).some((value) =>
        String(value ?? "").toLowerCase().includes(normalizedSearch),
      );
      return matchesModule && matchesSearch;
    });
  }, [metadata, activeFilter, search]);

  const annotatedImage = result?.annotated_image_url ? `${API_BASE}${result.annotated_image_url}` : null;
  const processedImage = result?.processed_image_url ? `${API_BASE}${result.processed_image_url}` : null;

  const downloadReport = () => {
    const blob = new Blob([JSON.stringify(result, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement("a");
    anchor.href = url;
    anchor.download = `traffic-analysis-${result.analysis_id}.json`;
    anchor.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="RoadSight home">
          <span className="brand-mark"><Icon type="signal" /></span>
          <span><b>RoadSight</b><small>Traffic intelligence</small></span>
        </a>
        <nav className="main-nav" aria-label="Main navigation">
          <a href="#analyze">Analyze</a>
          <a href="#analytics">Analytics</a>
          <a href="#evaluation">Evaluation</a>
        </nav>
        <div className="system-state"><span /> Analysis service ready</div>
      </header>

      <main id="top">
        <section className="hero" id="analyze">
          <div className="hero-copy">
            <div className="eyebrow light">Configurable vision analysis</div>
            <h1>See the road.<br /><em>Review what matters.</em></h1>
            <p>
              Select individual safety checks or run the complete traffic analysis suite.
              Every result includes annotated evidence and structured detection data.
            </p>
          </div>
          <div className="hero-metric" aria-label="Five analysis modules available">
            <strong>08</strong>
              <span>challenge-aligned<br />review modules</span>
            <div className="signal-lights"><i /><i /><i /></div>
          </div>
        </section>

        <section className="workspace" aria-label="Configure traffic analysis">
          <div className="config-panel panel">
            <SectionHeading number="01" title="Choose analysis modules" subtitle="Select one check, a custom group, or the complete suite." />

            <div className="selection-toolbar">
              <span>{selectedModules.length} of {MODULES.length} selected</span>
              <button className="text-button" type="button" onClick={toggleAll}>
                {allSelected ? "Clear selection" : "Select all modules"}
              </button>
            </div>

            <div className="module-grid">
              {MODULES.map((module) => {
                const checked = selectedModules.includes(module.key);
                return (
                  <label className={`module-option ${checked ? "selected" : ""}`} key={module.key}>
                    <input
                      type="checkbox"
                      checked={checked}
                      onChange={() => toggleModule(module.key)}
                    />
                    <span className="module-icon"><Icon type={module.key} /></span>
                    <span className="module-copy">
                      <b>{module.title}</b>
                      <small>{module.description}</small>
                    </span>
                    <span className="checkmark" aria-hidden="true">✓</span>
                  </label>
                );
              })}
            </div>

            {selectedModules.includes("redlight") && !canAssessRedLight && (
              <div className="context-note">
                <Icon type="info" />
                <span>Traffic signal state will be detected. Add <b>Vehicle Detection</b> to also assess stop-line violations.</span>
              </div>
            )}
          </div>

          <div className="input-panel panel">
            <SectionHeading number="02" title="Add traffic evidence" subtitle="Upload a clear road image up to 15 MB." />

            {!preview ? (
              <div
                className={`dropzone ${dragging ? "dragging" : ""}`}
                onDragEnter={(event) => { event.preventDefault(); setDragging(true); }}
                onDragOver={(event) => event.preventDefault()}
                onDragLeave={() => setDragging(false)}
                onDrop={(event) => {
                  event.preventDefault();
                  setDragging(false);
                  setSelectedFile(event.dataTransfer.files?.[0]);
                }}
                onClick={() => inputRef.current?.click()}
                role="button"
                tabIndex="0"
                onKeyDown={(event) => event.key === "Enter" && inputRef.current?.click()}
              >
                <input
                  ref={inputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp,image/bmp"
                  onChange={(event) => setSelectedFile(event.target.files?.[0])}
                  hidden
                />
                <span className="upload-icon"><Icon type="upload" /></span>
                <b>Drop a traffic image here</b>
                <span>or click to browse your files</span>
                <small>JPG, PNG, WebP or BMP · maximum 15 MB</small>
              </div>
            ) : (
              <div className="file-preview">
                <img src={preview} alt="Selected traffic scene" />
                {selectedModules.includes("illegal_parking") && (
                  <span
                    className="parking-zone-overlay"
                    style={{
                      left: `${parkingZone[0] * 100}%`,
                      top: `${parkingZone[1] * 100}%`,
                      width: `${(parkingZone[2] - parkingZone[0]) * 100}%`,
                      height: `${(parkingZone[3] - parkingZone[1]) * 100}%`,
                    }}
                  >Restricted zone</span>
                )}
                <div className="file-details">
                  <div><span>Ready for analysis</span><b>{file.name}</b><small>{formatBytes(file.size)}</small></div>
                  <button type="button" onClick={removeFile} aria-label="Remove selected image">Remove</button>
                </div>
              </div>
            )}

            <div className="settings-block">
              <div className="setting-row preprocessing-row">
                <div><b>Image preprocessing</b><small>Improve difficult lighting, noise, blur or contrast before detection.</small></div>
                <select value={preprocessing} onChange={(event) => setPreprocessing(event.target.value)}>
                  <option value="none">None</option>
                  <option value="auto">Auto enhance</option>
                  <option value="low_light">Low light</option>
                  <option value="denoise">Denoise</option>
                  <option value="sharpen">Sharpen</option>
                  <option value="contrast">Contrast</option>
                </select>
              </div>

              <div className="setting-row">
                <div><b>Detection confidence</b><small>Higher values show fewer, more certain detections.</small></div>
                <output>{Math.round(confidence * 100)}%</output>
              </div>
              <input
                className="range"
                type="range"
                min="0.1"
                max="0.8"
                step="0.05"
                value={confidence}
                onChange={(event) => setConfidence(Number(event.target.value))}
                aria-label="Detection confidence"
              />

              {selectedModules.includes("wrong_side") && (
                <div className="special-setting">
                  <div><b>Expected traffic side</b><small>Vehicles in the opposite half are marked for human direction review.</small></div>
                  <div className="segmented-control">
                    <button type="button" className={legalTrafficSide === "left" ? "active" : ""} onClick={() => setLegalTrafficSide("left")}>Left side</button>
                    <button type="button" className={legalTrafficSide === "right" ? "active" : ""} onClick={() => setLegalTrafficSide("right")}>Right side</button>
                  </div>
                </div>
              )}

              {selectedModules.includes("illegal_parking") && (
                <div className="special-setting">
                  <div><b>Restricted parking zone</b><small>Choose the road area where parked vehicles require review.</small></div>
                  <div className="zone-presets">
                    <button type="button" onClick={() => setParkingZone([0.02, 0.35, 0.35, 0.95])}>Left curb</button>
                    <button type="button" onClick={() => setParkingZone([0.65, 0.35, 0.98, 0.95])}>Right curb</button>
                    <button type="button" onClick={() => setParkingZone([0.15, 0.62, 0.85, 0.98])}>Lower road</button>
                  </div>
                </div>
              )}

              {canAssessRedLight && (
                <>
                  <div className="setting-row stopline-setting">
                    <div><b>Stop-line position</b><small>Vertical position used for red-light crossing assessment.</small></div>
                    <output>{Math.round(stopline * 100)}%</output>
                  </div>
                  <input
                    className="range danger-range"
                    type="range"
                    min="0.4"
                    max="0.9"
                    step="0.02"
                    value={stopline}
                    onChange={(event) => setStopline(Number(event.target.value))}
                    aria-label="Stop-line position"
                  />
                </>
              )}
            </div>

            <button
              className="primary-button"
              type="button"
              onClick={analyze}
              disabled={loading || !file || selectedModules.length === 0}
            >
              {loading ? <><span className="spinner" /> Processing selected modules…</> : <>Run traffic analysis <span>→</span></>}
            </button>

            {loading && <p className="loading-note">The first run may take longer while model assets initialize.</p>}
            {error && <div className="error-message"><Icon type="warning" /><span>{error}</span></div>}
          </div>
        </section>

        {!result && (
          <section className="process-strip" aria-label="How the analysis works">
            <div><b>1</b><span><strong>Configure</strong>Choose focused checks or all modules.</span></div>
            <i>→</i>
            <div><b>2</b><span><strong>Analyze</strong>Models process only your selection.</span></div>
            <i>→</i>
            <div><b>3</b><span><strong>Review</strong>Inspect visual and structured evidence.</span></div>
          </section>
        )}

        {result && (
          <section id="results" className="results-section">
            <SectionHeading number="03" title="Analysis results" subtitle="Review AI-assisted findings alongside the annotated evidence." />

            <OutcomeBanner result={result} onDownload={downloadReport} />
            <ViolationReportCard result={result} />

            <div className="module-results">
              {result.selected_modules.map((key) => (
                <ModuleResult key={key} moduleKey={key} data={result.module_results[key]} />
              ))}
            </div>

            <SummaryGrid summary={result.summary} selected={result.selected_modules} />

            <div className={`comparison-grid ${processedImage ? "three-up" : ""}`}>
              <figure className="evidence-card">
                <figcaption><span>Source evidence</span><small>Original upload</small></figcaption>
                <div className="image-stage"><img src={preview} alt="Original traffic scene" /></div>
              </figure>
              {processedImage && (
                <figure className="evidence-card processed-card">
                  <figcaption><span>Enhanced evidence</span><small>{humanize(result.summary.preprocessing)}</small></figcaption>
                  <div className="image-stage"><img src={processedImage} alt="Preprocessed traffic evidence" /></div>
                </figure>
              )}
              <figure className="evidence-card annotated-card">
                <figcaption>
                  <span>Annotated evidence</span>
                  <small>{result.selected_modules.length} module{result.selected_modules.length === 1 ? "" : "s"} applied</small>
                </figcaption>
                <div className="image-stage"><img src={annotatedImage} alt="Traffic scene with selected detections annotated" /></div>
              </figure>
            </div>

            <section className="data-panel">
              <div className="data-header">
                <div>
                  <div className="eyebrow">Structured output</div>
                  <h2>Detection evidence</h2>
                  <p>{metadata.length} records generated by the selected analysis.</p>
                </div>
                <label className="search-field">
                  <Icon type="search" />
                  <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Search evidence…" />
                </label>
              </div>

              <div className="filter-tabs" role="tablist" aria-label="Filter detection data">
                <button className={activeFilter === "all" ? "active" : ""} onClick={() => setActiveFilter("all")}>All records</button>
                {result.selected_modules.map((key) => (
                  <button key={key} className={activeFilter === key ? "active" : ""} onClick={() => setActiveFilter(key)}>
                    {MODULE_MAP[key].short}
                  </button>
                ))}
              </div>

              <EvidenceTable rows={filteredMetadata} />
              <div className="data-footer">
                <span>Showing {filteredMetadata.length} of {metadata.length} records</span>
                <span>AI-assisted review · verify consequential findings manually</span>
              </div>
            </section>
          </section>
        )}

        <AnalyticsDashboard data={analytics} />
        <EvaluationPanel data={evaluation} />
      </main>

      <footer><span>RoadSight Traffic Intelligence</span><small>Visual findings support review and are not a legal determination.</small></footer>
    </div>
  );
}

function SectionHeading({ number, title, subtitle }) {
  return (
    <div className="section-heading">
      <span>{number}</span>
      <div><h2>{title}</h2><p>{subtitle}</p></div>
    </div>
  );
}

function OutcomeBanner({ result, onDownload }) {
  const status = result.summary.final_status;
  const tone = toneFor(status);
  const content = {
    violation_detected: ["Potential violation detected", "Review the highlighted evidence and structured findings carefully."],
    review_required: ["Manual review recommended", "One or more selected checks could not be assessed conclusively."],
    analysis_complete: ["Selected analysis complete", "No clear violation was identified by the selected modules."],
  }[status] || [humanize(status), "The selected analysis modules completed."];

  return (
    <div className={`outcome-banner ${tone}`}>
      <span className="outcome-icon"><Icon type={tone === "danger" ? "warning" : tone === "warning" ? "review" : "check"} /></span>
      <div><small>Overall assessment</small><h2>{content[0]}</h2><p>{content[1]}</p></div>
      <div className="outcome-actions">
        <code title={result.analysis_id}>ID {result.analysis_id.slice(0, 8).toUpperCase()}</code>
        <a href={`${API_BASE}/analyses/${result.analysis_id}/report.pdf`}><Icon type="download" /> PDF</a>
        <a href={`${API_BASE}/analyses/${result.analysis_id}/report.csv`}><Icon type="download" /> CSV</a>
        <button type="button" onClick={onDownload}><Icon type="download" /> JSON</button>
      </div>
    </div>
  );
}

function ViolationReportCard({ result }) {
  const summary = result.summary;
  const score = summary.risk_score ?? 0;
  return (
    <section className="report-card">
      <div className="report-card-heading">
        <div>
          <div className="eyebrow">Enforcement review card</div>
          <h2>Case {result.analysis_id.slice(0, 8).toUpperCase()}</h2>
          <p>{result.original_filename} · {formatDate(result.created_at)}</p>
        </div>
        <div className={`risk-dial ${toneFor(summary.severity)}`} style={{ "--risk": `${score * 3.6}deg` }}>
          <div><strong>{score}</strong><small>/100 risk</small></div>
        </div>
      </div>
      <div className="report-grid">
        <div><small>Final decision</small><b><StatusPill value={summary.final_status} /></b></div>
        <div><small>Severity</small><b className={`severity ${summary.severity}`}>{humanize(summary.severity)}</b></div>
        <div><small>Processing time</small><b>{formatDuration(summary.processing_time_ms)}</b></div>
        <div><small>Plate evidence</small><b>{summary.recognized_plates?.join(", ") || "No readable plate"}</b></div>
      </div>
      <div className="decision-rationale">
        <h3>Decision rationale</h3>
        <ul>{(summary.decision_reasons || []).map((reason) => <li key={reason}>{reason}</li>)}</ul>
        <p><b>Recommendation:</b> {summary.recommendation}</p>
      </div>
    </section>
  );
}

function ModuleResult({ moduleKey, data = {} }) {
  const module = MODULE_MAP[moduleKey];
  const assessment = data.assessment || (data.violation === true ? "violation_detected" : data.signal) || data.status;
  return (
    <article className={`module-result ${toneFor(assessment)}`}>
      <div className="module-result-top"><span><Icon type={moduleKey} /></span><StatusPill value={assessment} /></div>
      <h3>{module.title}</h3>
      <strong>{data.detections ?? 0}<small> detection{data.detections === 1 ? "" : "s"}</small></strong>
      <p>{data.message || "Analysis complete."}</p>
    </article>
  );
}

function SummaryGrid({ summary, selected }) {
  const items = [];
  if (selected.includes("vehicle")) items.push(["Vehicles detected", summary.vehicle_count, "vehicle"]);
  if (selected.includes("license_plate")) {
    items.push(["License plates", summary.plate_count, "license_plate"]);
    if (summary.recognized_plates?.length) items.push(["Plate text", summary.recognized_plates.join(", "), "license_plate"]);
  }
  if (selected.includes("helmet")) items.push(["Helmet assessment", humanize(summary.helmet_status), "helmet"]);
  if (selected.includes("seatbelt")) items.push(["Seat belt assessment", humanize(summary.seatbelt_status), "seatbelt"]);
  if (selected.includes("triple_riding")) items.push(["Triple riding", summary.triple_riding_count, "triple_riding"]);
  if (selected.includes("wrong_side")) items.push(["Wrong-side reviews", summary.wrong_side_review_count, "wrong_side"]);
  if (selected.includes("illegal_parking")) items.push(["Parking reviews", summary.illegal_parking_review_count, "illegal_parking"]);
  if (selected.includes("redlight")) items.push(["Traffic signal", humanize(summary.traffic_signal_status), "redlight"]);
  if (selected.includes("redlight") && selected.includes("vehicle")) {
    items.push(["Stop-line crossings", summary.crossed_vehicle_count, "redlight"]);
    items.push(["Red-light violation", summary.redlight_violation ? "Detected" : "Not detected", "redlight"]);
  }
  items.push(["Risk score", `${summary.risk_score ?? 0}/100`, "risk"]);

  return (
    <div className="summary-grid">
      {items.map(([label, value, icon], index) => (
        <div className="summary-item" key={`${label}-${index}`}>
          <span><Icon type={icon} /></span><div><small>{label}</small><b>{value ?? "Not assessed"}</b></div>
        </div>
      ))}
    </div>
  );
}

function AnalyticsDashboard({ data }) {
  const violations = Object.entries(data?.violation_types || {});
  const maxViolation = Math.max(1, ...violations.map(([, count]) => count));
  return (
    <section id="analytics" className="insight-section">
      <SectionHeading number="04" title="Operations analytics" subtitle="Persisted trends across the latest 1,000 analyses." />
      {!data ? <div className="insight-loading">Loading analytics…</div> : (
        <>
          <div className="analytics-kpis">
            <InsightKpi label="Images processed" value={data.total_analyses} icon="analytics" />
            <InsightKpi label="Potential violations" value={data.status_counts.violation_detected || 0} icon="warning" />
            <InsightKpi label="Manual reviews" value={data.status_counts.review_required || 0} icon="review" />
            <InsightKpi label="Clear analyses" value={`${data.clear_rate}%`} icon="check" />
            <InsightKpi label="Average latency" value={formatDuration(data.average_processing_ms)} icon="timer" />
          </div>
          <div className="analytics-grid">
            <article className="chart-card">
              <div className="card-title"><div><span>Violation profile</span><small>Detected and review-required findings</small></div><Icon type="analytics" /></div>
              <div className="bar-chart">
                {violations.map(([label, count]) => (
                  <div className="bar-row" key={label}>
                    <span>{label}</span><div><i style={{ width: `${(count / maxViolation) * 100}%` }} /></div><b>{count}</b>
                  </div>
                ))}
              </div>
            </article>
            <article className="chart-card">
              <div className="card-title"><div><span>Recent case activity</span><small>Latest persisted analyses</small></div><Icon type="history" /></div>
              <div className="recent-cases">
                {(data.recent_cases || []).slice(0, 6).map((item) => (
                  <div key={item.id}>
                    <code>{item.id.slice(0, 8).toUpperCase()}</code>
                    <span><b>{item.original_filename}</b><small>{formatDate(item.created_at)}</small></span>
                    <StatusPill value={item.final_status} />
                    <strong>{item.risk_score || 0}</strong>
                  </div>
                ))}
                {!data.recent_cases?.length && <p className="no-data">No analysis history yet.</p>}
              </div>
            </article>
          </div>
        </>
      )}
    </section>
  );
}

function InsightKpi({ label, value, icon }) {
  return <div className="insight-kpi"><span><Icon type={icon} /></span><div><strong>{value ?? "—"}</strong><small>{label}</small></div></div>;
}

function EvaluationPanel({ data }) {
  return (
    <section id="evaluation" className="insight-section evaluation-section">
      <SectionHeading number="05" title="Model evaluation & transparency" subtitle="Measured operations are separated from quality metrics that require labeled ground truth." />
      {!data ? <div className="insight-loading">Loading evaluation status…</div> : (
        <div className="evaluation-layout">
          <article className="evaluation-runtime">
            <div className="eyebrow">Measured in this deployment</div>
            <strong>{formatDuration(data.runtime.average_processing_ms)}</strong>
            <span>average end-to-end latency</span>
            <small>Based on {data.runtime.measured_samples} recorded sample{data.runtime.measured_samples === 1 ? "" : "s"}</small>
          </article>
          <article className="evaluation-table-card">
            <table className="evaluation-table">
              <thead><tr><th>Model</th><th>Artifact</th><th>Precision</th><th>Recall</th><th>F1</th><th>mAP@50</th></tr></thead>
              <tbody>{data.models.map((model) => (
                <tr key={model.module}>
                  <td><b>{humanize(model.module)}</b><small>{model.model_available ? "Model ready" : "Unavailable"}</small></td>
                  <td>{model.artifact_size_mb ? `${model.artifact_size_mb} MB` : "—"}</td>
                  <td colSpan={4}><span className="ground-truth-needed">Ground-truth dataset required — not fabricated</span></td>
                </tr>
              ))}</tbody>
            </table>
          </article>
          <div className="methodology-note"><Icon type="info" /><div><b>Evaluation protocol</b>{data.methodology.map((item) => <p key={item}>{item}</p>)}</div></div>
        </div>
      )}
    </section>
  );
}

function EvidenceTable({ rows }) {
  if (!rows.length) {
    return <div className="empty-table"><Icon type="search" /><b>No matching records</b><span>Adjust the module filter or search term.</span></div>;
  }
  return (
    <div className="table-wrap">
      <table>
        <thead><tr><th>Module</th><th>Detection</th><th>Confidence</th><th>Assessment</th><th>Recognized text</th><th>Location</th></tr></thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={`${row.module}-${index}`}>
              <td><span className="module-cell"><Icon type={moduleKeyForRow(row.module)} />{moduleLabel(row.module)}</span></td>
              <td><b>{humanize(row.class_name)}</b><small>{humanize(row.rule)}</small></td>
              <td>{row.confidence === null || row.confidence === undefined ? <span className="muted">—</span> : <span className="confidence"><i style={{ width: `${Math.round(row.confidence * 100)}%` }} />{Math.round(row.confidence * 100)}%</span>}</td>
              <td><StatusPill value={row.status} /></td>
              <td>{row.ocr_text ? <code className="plate-text">{row.ocr_text}</code> : <span className="muted">—</span>}</td>
              <td><code className="bbox">{formatBox(row.bbox)}</code></td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function StatusPill({ value }) {
  return <span className={`status-pill ${toneFor(value)}`}>{humanize(value)}</span>;
}

function moduleKeyForRow(module) {
  return {
    vehicle_detection: "vehicle",
    license_plate_ocr: "license_plate",
    helmet_detection: "helmet",
    seatbelt_detection: "seatbelt",
    triple_riding_detection: "triple_riding",
    wrong_side_screening: "wrong_side",
    illegal_parking_screening: "illegal_parking",
    redlight_detection: "redlight",
    analysis_summary: "summary",
  }[module] || "summary";
}

function moduleLabel(module) {
  const key = moduleKeyForRow(module);
  return key === "summary" ? "Overall assessment" : MODULE_MAP[key]?.short || humanize(module);
}

function humanize(value) {
  if (value === null || value === undefined || value === "") return "Not assessed";
  const labels = {
    analysis_complete: "Analysis complete",
    violation_detected: "Potential violation",
    review_required: "Review required",
    compliant: "Compliant",
    not_detected: "Not detected",
    no_supported_vehicle: "No supported vehicle",
    model_unavailable: "Model unavailable",
    red_signal: "Red signal",
    green_signal: "Green signal",
    yellow_signal: "Yellow signal",
    detected_text_unclear: "Text unclear",
    stop_line_crossing: "Stop-line crossing",
    selected_module_assessment: "Selected module assessment",
  };
  return labels[value] || String(value).replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function toneFor(value = "") {
  const text = String(value).toLowerCase();
  if (text.includes("violation") || text === "high") return "danger";
  if (text.includes("review") || text === "medium" || text === "red" || text.includes("yellow") || text.includes("unclear")) return "warning";
  if (text.includes("compliant") || text.includes("complete") || text === "low" || text === "green" || text === "recognized") return "success";
  return "neutral";
}

function formatBox(value) {
  if (!Array.isArray(value) || value.length !== 4) return "—";
  return `${value[0]},${value[1]} → ${value[2]},${value[3]}`;
}

function formatBytes(bytes) {
  if (!bytes) return "0 KB";
  if (bytes < 1024 * 1024) return `${Math.ceil(bytes / 1024)} KB`;
  return `${(bytes / 1024 / 1024).toFixed(1)} MB`;
}

function formatDate(value) {
  if (!value) return "Timestamp unavailable";
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(value));
}

function formatDuration(value) {
  if (value === null || value === undefined) return "Not measured";
  return value >= 1000 ? `${(value / 1000).toFixed(2)} s` : `${Math.round(value)} ms`;
}

function Icon({ type }) {
  const common = { viewBox: "0 0 24 24", fill: "none", stroke: "currentColor", strokeWidth: "1.8", strokeLinecap: "round", strokeLinejoin: "round", "aria-hidden": "true" };
  const paths = {
    vehicle: <><path d="M3 14l1.8-5.2A2 2 0 016.7 7.5h10.6a2 2 0 011.9 1.3L21 14"/><path d="M4 14h16v4H4z"/><path d="M7 18v2M17 18v2M6.5 14h.01M17.5 14h.01"/></>,
    license_plate: <><rect x="3" y="6" width="18" height="12" rx="2"/><path d="M7 10h10M7 14h6M17 14h.01"/></>,
    helmet: <><path d="M4 15a8 8 0 0116 0H4z"/><path d="M12 7v8M3 15h18M16 15v3h4"/></>,
    seatbelt: <><circle cx="8" cy="6" r="2"/><path d="M9.5 8l6 11M6 10l5 4M7 8v11M13 15h5v4h-3"/></>,
    triple_riding: <><circle cx="7" cy="6" r="2"/><circle cx="12" cy="5" r="2"/><circle cx="17" cy="6" r="2"/><path d="M4 13h16M6 10l2 3M11 9l1 4M16 10l-1 3M5 18h14"/></>,
    wrong_side: <><path d="M4 8h13M13 4l4 4-4 4M20 16H7M11 12l-4 4 4 4"/></>,
    illegal_parking: <><circle cx="12" cy="12" r="9"/><path d="M9 17V7h4a3 3 0 010 6H9M5 5l14 14"/></>,
    redlight: <><rect x="7" y="2" width="10" height="20" rx="3"/><circle cx="12" cy="7" r="2"/><circle cx="12" cy="12" r="2"/><circle cx="12" cy="17" r="2"/></>,
    signal: <><path d="M6 3h12v18H6z"/><circle cx="12" cy="8" r="2"/><circle cx="12" cy="16" r="2"/></>,
    upload: <><path d="M12 16V4M7 9l5-5 5 5"/><path d="M4 15v5h16v-5"/></>,
    info: <><circle cx="12" cy="12" r="9"/><path d="M12 11v5M12 8h.01"/></>,
    warning: <><path d="M12 3L2.5 20h19L12 3z"/><path d="M12 9v5M12 17h.01"/></>,
    review: <><circle cx="11" cy="11" r="7"/><path d="M20 20l-4-4M11 8v3l2 2"/></>,
    check: <><circle cx="12" cy="12" r="9"/><path d="M8 12l2.5 2.5L16 9"/></>,
    download: <><path d="M12 3v12M7 10l5 5 5-5"/><path d="M4 20h16"/></>,
    search: <><circle cx="10.5" cy="10.5" r="6.5"/><path d="M16 16l4 4"/></>,
    summary: <><path d="M5 3h14v18H5z"/><path d="M8 8h8M8 12h8M8 16h5"/></>,
    risk: <><path d="M12 3l8 3v5c0 5-3.4 8.2-8 10-4.6-1.8-8-5-8-10V6l8-3z"/><path d="M12 8v5M12 16h.01"/></>,
    analytics: <><path d="M4 20V10M10 20V4M16 20v-7M22 20H2"/></>,
    history: <><path d="M3 12a9 9 0 109-9 9 9 0 00-7 3.3L3 8"/><path d="M3 3v5h5M12 7v5l3 2"/></>,
    timer: <><circle cx="12" cy="13" r="8"/><path d="M12 9v4l3 2M9 2h6"/></>,
  };
  return <svg {...common}>{paths[type] || paths.summary}</svg>;
}
