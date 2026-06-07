import type { StepResult } from "@/lib/types";

const pct = (n: number) => `${(n * 100).toFixed(3)}%`;
const ms = (n: number) => {
  const s = Math.floor(n / 1000), cs = Math.round((n % 1000) / 10);
  return `00:00:${String(s).padStart(2, "0")}:${String(cs).padStart(2, "0")}`;
};

function frameSrc(ts: number) {
  return ts <= 4200 ? "/frames/goal-4000.jpg" : "/frames/scorer-4625.jpg";
}

export default function EvidencePanel({ step }: { step: StepResult | null }) {
  const ev = step?.evidence;
  const ts = ev?.frame_ts_ms ?? 0;
  const overlays = ev?.overlays ?? [];
  const conf = step?.confidence;
  // A step with ts 0 and no overlays has no frame-grounded evidence (e.g. count, or an ungrounded
  // answer). Free-text programs may run partway, so degrade gracefully instead of showing a
  // misleading hero still as if it were this step's evidence.
  const hasFrame = ts > 0 || overlays.length > 0;
  const shownTs = hasFrame && ts > 0 ? ts : 4625; // default the placeholder background to a real still

  return (
    <section className="panel evidence" aria-label="Evidence — inspectable">
      <div className="panel-head">
        <div className="panel-title">Evidence <span className="sub">(inspectable)</span></div>
        <div className="panel-meta">{hasFrame ? `frame ${ms(shownTs)}` : "no evidence frame"}</div>
      </div>
      <div className="evi-body">
        <div className={`frame ${hasFrame ? "" : "no-evi"}`}>
          {/* eslint-disable-next-line @next/next/no-img-element */}
          <img src={frameSrc(shownTs)} alt={hasFrame ? "analyzed video frame" : "no evidence for this step"} />
          <div className="tc">{ms(shownTs)}</div>
          {!hasFrame && <div className="no-evi-tag">This step produced no frame-level evidence.</div>}
          {overlays.map((o, i) => (
            <div
              key={i}
              className={`ov ${o.tone}`}
              style={{ left: pct(o.box.x), top: pct(o.box.y), width: pct(o.box.w), height: pct(o.box.h) }}
            >
              {o.label && (
                <span className="ov-tag">
                  {o.label}
                  {o.confidence != null && <span className="conf"> {o.confidence.toFixed(2)}</span>}
                </span>
              )}
            </div>
          ))}
        </div>

        <p className="evi-cap">
          Every result is <b>traceable to a frame</b> — the box above is what the active step
          {step ? <> (<code>{step.producer ?? step.op}</code>)</> : ""} produced.
        </p>

        <div className="readout">
          <div className="row"><div className="k">Producer step</div><div className="v">{step?.producer ?? step?.op ?? "—"}</div></div>
          <div className="row"><div className="k">Input</div><div className="v">{step?.input_label ?? "—"}</div></div>
          <div className="row">
            <div className="k">Output</div>
            <div className="v">
              <span className={`pill ${step?.source === "pinned" ? "pinned" : ""}`}>{step?.output_label ?? "—"}</span>
              {step?.source && <span className={`src-tag ${step.source}`}>{step.source}</span>}
            </div>
          </div>
          <div className="row">
            <div className="k">Confidence</div>
            <div className="v">
              <div className="conf-bar"><div className="conf-fill" style={{ width: conf != null ? pct(conf) : "0%" }} /></div>
              <span>{conf != null ? conf.toFixed(2) : "—"}</span>
            </div>
          </div>
        </div>
        {step?.note && <div className="note">{step.note}</div>}
      </div>
    </section>
  );
}
