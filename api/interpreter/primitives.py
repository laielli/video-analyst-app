"""
Op implementations. Each is fn(step, env, cache) -> (binding, step_result):
  - binding: the value stored under step["id"] for later steps to reference.
  - step_result: the run-doc trace entry (status, provenance, EVIDENCE overlays).

Replay mode: detect/read_text read from the cache (real Azure outputs + pins);
crop/filter/count/temporal_order/answer are local compute. No live calls.

Binding kinds and item shapes:
  frames      items=[{frame_ts_ms}]
  detections  items=[{det_id,cls,confidence,box,frame_ts_ms}]
  crops       items=[{crop_id,det_id,box,person_box,person_conf,frame_ts_ms}]
  texts       items=[{det_id,text,confidence,source,note?,crop_box,person_box,frame_ts_ms}]
  number      value=int
  ordered     value={events,first,subject_dets,subject_label,subject_is_first_scorer}
  answer      value={answer,verdict,question,yes,grounded,reason?}  (answer/verdict null when ungrounded)
"""
from __future__ import annotations


def _evi(frame_ts_ms, overlays):
    return {"frame_ts_ms": int(frame_ts_ms), "overlays": overlays}


def op_sample_frames(step, env, cache):
    a = step["args"]
    start, end, fps = a["start_ms"], a["end_ms"], a["fps"]
    stride = max(1, round(1000 / fps))
    ts = list(range(start, end + 1, stride))
    binding = {"kind": "frames", "items": [{"frame_ts_ms": t} for t in ts]}
    result = {
        "id": step["id"], "op": "sample_frames", "producer": "sample_frames",
        "status": "done", "source": "live", "inputs": [],
        "input_label": f"clip {start}-{end}ms",
        "output_label": f"{len(ts)} frames @ {fps}fps", "confidence": None,
        "evidence": _evi(ts[len(ts) // 2] if ts else start, []),
    }
    return binding, result


def op_detect(step, env, cache):
    a = step["args"]
    frames = env[a["frames"]]["items"]
    classes = [c.lower() for c in a["classes"]]
    dets = []
    for f in frames:
        for d in cache.detections_at(f["frame_ts_ms"]):
            if d["cls"].lower() in classes:
                dets.append({**d, "frame_ts_ms": f["frame_ts_ms"]})
    focus_ts = dets[-1]["frame_ts_ms"] if dets else (frames[-1]["frame_ts_ms"] if frames else 0)
    overlays = [
        {"box": d["box"], "label": f"{d['cls']} {d['confidence']:.2f}",
         "confidence": d["confidence"], "tone": "azure", "kind": "box"}
        for d in dets if d["frame_ts_ms"] == focus_ts
    ]
    noun = "people" if classes == ["person"] else "objects"
    result = {
        "id": step["id"], "op": "detect", "producer": "Azure AI Vision · detect",
        "status": "done" if dets else "empty", "source": "cached",
        "inputs": [a["frames"]], "input_label": f"{len(frames)} frames",
        "output_label": f"{len(dets)} {noun}",
        "confidence": max((d["confidence"] for d in dets), default=None),
        "evidence": _evi(focus_ts, overlays),
    }
    if not dets:
        result["note"] = "no detections in the sampled frames"
    return {"kind": "detections", "items": dets}, result


def op_crop(step, env, cache):
    a = step["args"]
    dets = env[a["detections"]]["items"]
    region = a["region"]
    crops = []
    for i, d in enumerate(dets):
        b = d["box"]
        if region == "jersey":
            box = {"x": round(b["x"] + 0.22 * b["w"], 4), "y": round(b["y"] + 0.10 * b["h"], 4),
                   "w": round(0.56 * b["w"], 4), "h": round(0.22 * b["h"], 4)}
        else:
            box = dict(b)
        crops.append({"crop_id": f"c{i}", "det_id": d["det_id"], "box": box,
                      "person_box": b, "person_conf": d.get("confidence"), "frame_ts_ms": d["frame_ts_ms"]})
    focus_ts = crops[-1]["frame_ts_ms"] if crops else 0
    overlays = [{"box": c["box"], "label": "crop_jersey" if region == "jersey" else f"crop_{region}",
                 "tone": "teal", "kind": "crop"} for c in crops if c["frame_ts_ms"] == focus_ts]
    result = {
        "id": step["id"], "op": "crop", "producer": f"crop_{region}",
        "status": "done" if crops else "empty", "source": "live",
        "inputs": [a["detections"]], "input_label": f"{len(dets)} boxes",
        "output_label": f"{len(crops)} {region} crops", "confidence": None,
        "evidence": _evi(focus_ts, overlays),
    }
    return {"kind": "crops", "items": crops}, result


def op_read_text(step, env, cache):
    a = step["args"]
    crops = env[a["crops"]]["items"]
    texts, sources, notes = [], set(), []
    for c in crops:
        rt = cache.read_text_for(c["det_id"])
        if rt and rt.get("text"):
            texts.append({"det_id": c["det_id"], "text": rt["text"], "confidence": rt.get("confidence"),
                          "source": rt.get("source", "live"), "note": rt.get("note"),
                          "crop_box": c["box"], "person_box": c["person_box"], "frame_ts_ms": c["frame_ts_ms"]})
            sources.add(rt.get("source", "live"))
            if rt.get("note"):
                notes.append(rt["note"])
    step_source = "pinned" if "pinned" in sources else ("cached" if "cached" in sources else "live")
    focus_ts = texts[-1]["frame_ts_ms"] if texts else (crops[-1]["frame_ts_ms"] if crops else 0)
    overlays = [{"box": t["crop_box"], "label": f"read_text -> {t['text']}",
                 "confidence": t.get("confidence"), "tone": "teal", "kind": "crop"}
                for t in texts if t["frame_ts_ms"] == focus_ts]
    result = {
        "id": step["id"], "op": "read_text", "producer": "read_text",
        "status": "done" if texts else "empty", "source": step_source,
        "inputs": [a["crops"]], "input_label": f"{len(crops)} crops",
        "output_label": ", ".join(sorted({t["text"] for t in texts})) or "(none legible)",
        "confidence": next((t["confidence"] for t in texts if t["confidence"] is not None), None),
        "evidence": _evi(focus_ts, overlays),
    }
    if notes:
        result["note"] = notes[0]
    elif not texts:
        result["note"] = "no legible text in the jersey crops (read returned nothing)"
    return {"kind": "texts", "items": texts}, result


def op_filter(step, env, cache):
    a = step["args"]
    src = env[a["items"]]
    field, val = a["where"]["field"], a["where"]["equals"]
    kept = [it for it in src["items"] if str(it.get(field)) == str(val)]
    overlays, focus_ts = [], 0
    for it in kept:
        pb = it.get("person_box") or it.get("box")
        if pb:
            overlays.append({"box": pb, "label": f"#{val}", "tone": "teal", "kind": "box"})
            focus_ts = it.get("frame_ts_ms", focus_ts)
    result = {
        "id": step["id"], "op": "filter", "producer": "filter",
        "status": "done" if kept else "empty", "source": "live",
        "inputs": [a["items"]], "input_label": f"{src['kind']} where {field} == {val}",
        "output_label": f"{len(kept)} match" + (f" · #{val}" if kept else ""), "confidence": None,
        "evidence": _evi(focus_ts, overlays),
    }
    if not kept:
        result["note"] = f"no items with {field} == {val}"
    # Carry the filter predicate forward so a downstream temporal_order/answer can derive the
    # subject label (#<val>) even when the filter matched nothing — the subject is what was ASKED
    # FOR (the filter value), derived from data, never parsed from the question text.
    return {"kind": src["kind"], "items": kept,
            "filter_field": field, "filter_value": val}, result


def op_count(step, env, cache):
    a = step["args"]
    src = env[a["items"]]
    n = len(src["items"])
    result = {
        "id": step["id"], "op": "count", "producer": "count", "status": "done", "source": "live",
        "inputs": [a["items"]], "input_label": src["kind"], "output_label": str(n),
        "confidence": None, "evidence": _evi(0, []),
    }
    return {"kind": "number", "value": n}, result


def _subject_label(binding) -> str | None:
    """Derive the subject label '#<n>' (Phase 0: data-derived, never the literal '#10'). Prefer the
    filter predicate the binding carries (`filter_value`, on a text/number field) so the subject is
    whatever the question filtered for — robust even when the filter matched nothing. Fall back to
    the read text on a kept item."""
    fv = binding.get("filter_value")
    if fv is not None and binding.get("filter_field") in ("text", "number", "cls", "class"):
        return f"#{fv}"
    for it in binding.get("items") or []:
        t = it.get("text")
        if t:
            return f"#{t}"
    return None


def op_temporal_order(step, env, cache):
    a = step["args"]
    subj = env[a["events"]]
    items = subj.get("items", [])
    subject_dets = {it.get("det_id") for it in items if it.get("det_id")}
    subject_label = _subject_label(subj)         # '#7', '#10', … derived from the filter, not hardcoded
    goals = cache.goal_events()
    first = goals[0] if goals else None
    subject_is_first = bool(first and first.get("scorer_det") in subject_dets)
    overlays, focus_ts = [], 0
    if first:
        focus_ts = first["ts_ms"]
        if first.get("box"):
            overlays.append({"box": first["box"], "label": f"goal {first['ts_ms']}ms", "tone": "green", "kind": "goal"})
    result = {
        "id": step["id"], "op": "temporal_order", "producer": "temporal_order",
        "status": "done" if first else "empty", "source": "live",
        "inputs": [a["events"]], "input_label": f"{len(subject_dets)} subject(s) + {len(goals)} goal(s)",
        "output_label": (f"goal @ {first['ts_ms']}ms (first)" if first else "no goal events"),
        "confidence": None, "evidence": _evi(focus_ts, overlays),
    }
    value = {"events": goals, "first": first, "subject_dets": sorted(subject_dets),
             "subject_label": subject_label, "subject_is_first_scorer": subject_is_first}
    return {"kind": "ordered", "value": value}, result


def _answer_temporal(src, q):
    """Boolean/temporal answer: did the subject score the first goal? Subject is DERIVED from the
    filter's matched value (subject_label), never the literal '#10'."""
    v = src["value"]
    subj = v.get("subject_label") or "the subject"
    yes = bool(v.get("subject_is_first_scorer"))
    answer = "Yes" if yes else "No"
    verdict = (f"Yes — {subj} scored the first goal" if yes
               else f"No — the first goal was not scored by {subj}")
    return answer, verdict, yes


def _answer_count(src, q):
    """Count answer: 'N matches'. The question carries the noun; the answer carries the number."""
    n = src["value"]
    if n == 0:
        return None, None, None              # honest: nothing to count -> ungrounded
    answer = str(n)
    verdict = f"{n} match" if n == 1 else f"{n} matches"
    return answer, verdict, True


def _answer_text(src, q):
    """Text-readout answer: report the legible values read off the crops."""
    vals = sorted({it.get("text") for it in src["items"] if it.get("text")})
    if not vals:
        return None, None, None              # nothing legible -> ungrounded
    answer = ", ".join(vals)
    verdict = f"Read: {answer}"
    return answer, verdict, True


def _answer_presence(src, q):
    """Existence answer: did any item survive to here? ('Yes — N found' / 'No — none found')."""
    n = len(src["items"])
    yes = n > 0
    answer = "Yes" if yes else "No"
    verdict = f"Yes — {n} found" if yes else "No — none found"
    return answer, verdict, yes


def _answer_overlays(src, cache):
    """Best-effort evidence overlay for the answer step. For a temporal answer, highlight the
    scorer det that grounds it; other kinds carry no extra overlay (the supporting step's
    overlays already speak for the evidence)."""
    overlays, focus_ts = [], 0
    if src["kind"] == "ordered":
        v = src["value"]
        first = v.get("first") or {}
        scorer = first.get("scorer_det")
        label = (v.get("subject_label") or "subject") + " scored"
        if scorer:
            for ts in cache.analyzed_frames():
                for d in cache.detections_at(ts):
                    if d["det_id"] == scorer:
                        overlays.append({"box": d["box"], "label": label, "tone": "green", "kind": "box"})
                        focus_ts = ts
    return overlays, focus_ts


def op_answer(step, env, cache):
    """Generalized answer (Phase 0): dispatch on the KIND of the consumed binding and synthesize a
    deterministic, honest answer that addresses the question actually asked — never the hardcoded
    '#10' verdict. When the binding can't be synthesized (empty filter, count 0, unsupported kind)
    emit the ungrounded state (answer/verdict null + reason) per the resolved Q1 decision, NOT a
    fabricated verdict. Stays replay-mode (no live call)."""
    a = step["args"]
    src = env[a["from"]]
    q = a["question"]
    kind = src["kind"]

    if kind == "ordered":
        answer, verdict, yes = _answer_temporal(src, q)
    elif kind == "number":
        answer, verdict, yes = _answer_count(src, q)
    elif kind == "texts":
        answer, verdict, yes = _answer_text(src, q)
    elif kind in ("detections", "crops", "frames"):
        answer, verdict, yes = _answer_presence(src, q)
    else:
        answer, verdict, yes = None, None, None    # unsupported kind -> ungrounded

    grounded = answer is not None
    overlays, focus_ts = _answer_overlays(src, cache) if grounded else ([], 0)

    if grounded:
        binding = {"kind": "answer", "value": {
            "answer": answer, "verdict": verdict, "question": q, "yes": bool(yes), "grounded": True}}
        out_label = str(answer)[:24]
    else:
        binding = {"kind": "answer", "value": {
            "answer": None, "verdict": None, "question": q, "yes": None,
            "grounded": False, "reason": "answer-not-synthesizable"}}
        out_label = "ungrounded"

    result = {
        "id": step["id"], "op": "answer", "producer": "answer_question",
        "status": "done" if grounded else "empty", "source": "live", "inputs": [a["from"]],
        "input_label": f"{kind} binding", "output_label": out_label,
        "confidence": None, "evidence": _evi(focus_ts, overlays),
    }
    if not grounded:
        result["note"] = "could not synthesize an evidence-grounded answer from this binding"
    return binding, result


OPS = {
    "sample_frames": op_sample_frames,
    "detect": op_detect,
    "crop": op_crop,
    "read_text": op_read_text,
    "filter": op_filter,
    "count": op_count,
    "temporal_order": op_temporal_order,
    "answer": op_answer,
}
