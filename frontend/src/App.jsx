import { useState } from "react";

const API_BASE = import.meta.env.VITE_API_BASE || "http://127.0.0.1:8000";

export default function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [conf, setConf] = useState(0.25);
  const [stopline, setStopline] = useState(0.72);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");

  const onFileChange = (e) => {
    const selected = e.target.files?.[0];
    setFile(selected || null);
    setResult(null);
    setError("");
    if (selected) {
      setPreview(URL.createObjectURL(selected));
    } else {
      setPreview(null);
    }
  };

  const analyze = async () => {
    if (!file) {
      setError("Please upload an image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);

    const form = new FormData();
    form.append("file", file);
    form.append("conf", String(conf));
    form.append("stopline_y_ratio", String(stopline));

    try {
      const res = await fetch(`${API_BASE}/analyze`, {
        method: "POST",
        body: form,
      });

      const data = await res.json();

      if (!res.ok || !data.success) {
        throw new Error(data.error || "Analysis failed");
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  const imageUrl = result?.annotated_image_url
    ? `${API_BASE}${result.annotated_image_url}`
    : null;

  const summary = result?.summary || null;
  const meta = result?.meta || [];

  return (
    <div className="page">
      <header className="header">
        <div>
          <h1>Traffic Analysis AI</h1>
          <p>Upload a traffic image and check vehicle, plate, helmet, seatbelt and red-light rules.</p>
        </div>
      </header>

      <main className="grid">
        <section className="card controls">
          <h2>Upload Image</h2>
          <input type="file" accept="image/*" onChange={onFileChange} />

          <label>
            Confidence: <b>{conf}</b>
            <input
              type="range"
              min="0.1"
              max="0.8"
              step="0.05"
              value={conf}
              onChange={(e) => setConf(Number(e.target.value))}
            />
          </label>

          <label>
            Stop-line position: <b>{stopline}</b>
            <input
              type="range"
              min="0.4"
              max="0.9"
              step="0.02"
              value={stopline}
              onChange={(e) => setStopline(Number(e.target.value))}
            />
          </label>

          <button onClick={analyze} disabled={loading || !file}>
            {loading ? "Analyzing..." : "Run Analysis"}
          </button>

          {error && <p className="error">{error}</p>}
        </section>

        <section className="card">
          <h2>Original Image</h2>
          {preview ? <img className="result-img" src={preview} alt="preview" /> : <p>No image selected.</p>}
        </section>

        <section className="card wide">
          <h2>Annotated Output</h2>
          {imageUrl ? <img className="result-img" src={imageUrl} alt="annotated result" /> : <p>Output will appear here.</p>}
        </section>

        <section className="card wide">
          <h2>Rule Summary</h2>
          {summary ? (
            <div className="summary-grid">
              <Info label="Vehicles" value={summary.vehicle_count} />
              <Info label="Plates" value={summary.plate_count} />
              <Info label="Helmet" value={summary.helmet_status} />
              <Info label="Seatbelt" value={summary.seatbelt_status} />
              <Info label="Red Signal" value={String(summary.red_signal)} />
              <Info label="Crossed Vehicles" value={summary.crossed_vehicle_count} />
              <Info label="Red-light Violation" value={String(summary.redlight_violation)} />
              <Info label="Final Status" value={summary.final_status} />
            </div>
          ) : (
            <p>No summary yet.</p>
          )}
        </section>

        <section className="card wide">
          <h2>Metadata</h2>
          {meta.length > 0 ? <MetaTable rows={meta} /> : <p>No metadata yet.</p>}
        </section>
      </main>
    </div>
  );
}

function Info({ label, value }) {
  return (
    <div className="info">
      <span>{label}</span>
      <b>{value}</b>
    </div>
  );
}

function MetaTable({ rows }) {
  const cols = ["module", "class_name", "confidence", "status", "ocr_text", "bbox"];

  return (
    <div className="table-wrap">
      <table>
        <thead>
          <tr>
            {cols.map((c) => (
              <th key={c}>{c}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, idx) => (
            <tr key={idx}>
              {cols.map((c) => (
                <td key={c}>{formatCell(row[c])}</td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function formatCell(value) {
  if (value === null || value === undefined) return "-";
  if (Array.isArray(value)) return `[${value.join(", ")}]`;
  if (typeof value === "number") return Number(value).toFixed(3);
  return String(value);
}
