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
  answer      value={answer,verdict,question,yes,grounded,reason}
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
    return {"kind": src["kind"], "items": kept}, result


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


def _subject_label(subj) -> str | None:
    """Derive the '#<n>' subject framing from the matched value of a prior filter step, NOT a
    literal. A temporal_order/answer fed a filtered `texts` collection (the load-bearing
    detect->crop->read_text->filter chain) carries the jersey number on each item's `text`."""
    for it in subj.get("items", []):
        t = it.get("text")
        if t not in (None, ""):
            return f"#{t}"
    return None


def op_temporal_order(step, env, cache):
    a = step["args"]
    subj = env[a["events"]]
    subject_dets = {it.get("det_id") for it in subj.get("items", []) if it.get("det_id")}
    subject_label = _subject_label(subj)
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


# ---- answer synthesizers (Phase 0): deterministic structural templating per binding kind. ----
# Each returns (answer, verdict, yes, overlays, focus_ts). `yes` is the boolean verdict where
# meaningful (temporal/presence), or None where it is not (count/text readout). None of these
# fabricate the hero "#10" string — the subject is always derived from the data in hand.

def _answer_temporal(src, q, cache):
    """ordered binding -> 'did <subject> score the first goal?'. Subject derived from the filter."""
    val = src["value"]
    yes = bool(val.get("subject_is_first_scorer"))
    subject = val.get("subject_label") or "the subject"
    if yes:
        verdict = f"Yes — {subject} scored the first goal"
    else:
        verdict = f"No — the first goal was not scored by {subject}"
    overlays, focus_ts = [], 0
    first = val.get("first") or {}
    scorer = first.get("scorer_det")
    if scorer:
        for ts in cache.analyzed_frames():
            for d in cache.detections_at(ts):
                if d["det_id"] == scorer:
                    label = f"{subject} scored" if subject != "the subject" else "scorer"
                    overlays.append({"box": d["box"], "label": label, "tone": "green", "kind": "box"})
                    focus_ts = ts
    return ("Yes" if yes else "No"), verdict, yes, overlays, focus_ts


def _answer_count(src, q, cache):
    """number binding -> a count readout. n == 0 grounds out (handled by the dispatcher)."""
    n = src["value"]
    answer = str(n)
    verdict = f"{n} " + ("match" if n == 1 else "matches")
    return answer, verdict, None, [], 0


def _answer_text(src, q, cache):
    """texts binding -> read the OCR'd value(s) back. Overlay the crop the value came from."""
    items = src.get("items", [])
    values = [it["text"] for it in items if it.get("text")]
    if not values:
        return None, None, None, [], 0
    uniq = sorted(dict.fromkeys(values))
    answer = ", ".join(uniq)
    verdict = f"Read: {answer}"
    overlays, focus_ts = [], 0
    last = items[-1]
    box = last.get("crop_box") or last.get("box")
    if box:
        overlays.append({"box": box, "label": f"read_text -> {last['text']}", "tone": "teal", "kind": "crop"})
        focus_ts = last.get("frame_ts_ms", 0)
    return answer, verdict, None, overlays, focus_ts


def _answer_presence(src, q, cache):
    """detections/crops/frames binding -> existence: any items => present."""
    items = src.get("items", [])
    yes = len(items) > 0
    verdict = "Yes — present in the analyzed frames" if yes else "No — not found in the analyzed frames"
    overlays, focus_ts = [], 0
    if items:
        last = items[-1]
        box = last.get("box") or last.get("person_box")
        if box:
            overlays.append({"box": box, "label": "found", "tone": "green", "kind": "box"})
            focus_ts = last.get("frame_ts_ms", 0)
    return ("Yes" if yes else "No"), verdict, yes, overlays, focus_ts


def _ungrounded_answer(q, reason):
    """Honest 'couldn't ground' answer binding (Q1 -> A). No fabricated verdict."""
    binding = {"kind": "answer", "value": {
        "answer": None, "verdict": None, "question": q, "yes": None,
        "grounded": False, "reason": reason,
    }}
    result = {
        "id": "__answer__", "op": "answer", "producer": "answer_question",
        "status": "empty", "source": "live", "input_label": "(no groundable evidence)",
        "output_label": "ungrounded", "confidence": None, "evidence": _evi(0, []),
        "note": "could not ground an answer from the available evidence",
    }
    return binding, result


def op_answer(step, env, cache):
    a = step["args"]
    src = env[a["from"]]
    q = a["question"]
    kind = src["kind"]

    if kind == "ordered":
        ans, verdict, yes, overlays, focus_ts = _answer_temporal(src, q, cache)
    elif kind == "number":
        # n == 0 has no groundable subject -> honest ungrounded (Q1 -> A).
        if src["value"] == 0:
            binding, result = _ungrounded_answer(q, "no-grounded-answer")
            result["id"] = step["id"]
            result["inputs"] = [a["from"]]
            return binding, result
        ans, verdict, yes, overlays, focus_ts = _answer_count(src, q, cache)
    elif kind == "texts":
        ans, verdict, yes, overlays, focus_ts = _answer_text(src, q, cache)
        if ans is None:  # nothing legible -> ungrounded
            binding, result = _ungrounded_answer(q, "no-grounded-answer")
            result["id"] = step["id"]
            result["inputs"] = [a["from"]]
            return binding, result
    elif kind in ("detections", "crops", "frames"):
        ans, verdict, yes, overlays, focus_ts = _answer_presence(src, q, cache)
    else:
        # Unsupported binding kind handed to answer -> honest ungrounded, never a fake verdict.
        binding, result = _ungrounded_answer(q, "unsupported-binding")
        result["id"] = step["id"]
        result["inputs"] = [a["from"]]
        return binding, result

    result = {
        "id": step["id"], "op": "answer", "producer": "answer_question",
        "status": "done", "source": "live", "inputs": [a["from"]],
        "input_label": f"{kind} binding", "output_label": ans,
        "confidence": None, "evidence": _evi(focus_ts, overlays),
    }
    binding = {"kind": "answer", "value": {
        "answer": ans, "verdict": verdict, "question": q, "yes": yes,
        "grounded": True, "reason": None,
    }}
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
