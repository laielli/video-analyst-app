"""
Tree-walking interpreter: executes a (pre-validated) DSL program against a Cache and
emits a run-doc (run_doc.schema.json). Maintains a binding environment id -> value;
each op also produces its run-doc trace entry. Replay mode = zero live calls.
"""
from __future__ import annotations

from .primitives import OPS


class Interpreter:
    def __init__(self, cache):
        self.cache = cache

    def run(self, program: list[dict], query: str | None = None) -> dict:
        env: dict[str, dict] = {}
        trace: list[dict] = []
        answer_binding: dict | None = None

        for step in program:
            fn = OPS.get(step["op"])
            if fn is None:
                raise ValueError(f"unknown op '{step['op']}' in step '{step.get('id')}'")
            for ref in self._refs(step):
                if ref not in env:
                    raise ValueError(f"step '{step['id']}' references undefined binding '{ref}'")
            binding, result = fn(step, env, self.cache)
            env[step["id"]] = binding
            trace.append(result)
            if step["op"] == "answer":
                answer_binding = binding

        if query is None:
            query = next((s["args"].get("question") for s in program if s["op"] == "answer"), "") or ""

        return {
            "schema_version": "1",
            "query": query,
            "clip": self.cache.clip,
            "program": program,
            "trace": trace,
            "findings": self._findings(program, answer_binding),
        }

    @staticmethod
    def _refs(step: dict) -> list[str]:
        """Binding ids this step's args reference (string args, excluding literals like region/by/where)."""
        args = step.get("args", {})
        ref_fields = {"frames", "detections", "crops", "items", "events", "from"}
        return [v for k, v in args.items() if k in ref_fields and isinstance(v, str)]

    @staticmethod
    def _findings(program: list[dict], answer_binding: dict | None) -> dict:
        if answer_binding and answer_binding["kind"] == "answer":
            v = answer_binding["value"]
            # Phase 0 / Resolved Q1 -> A: an answer binding that couldn't ground carries
            # grounded:false; surface the honest discriminator instead of a fake verdict.
            if v.get("grounded") is False:
                return {"answer": None, "verdict": None, "partial": False,
                        "grounded": False, "reason": "no-grounded-answer"}
            supporting = (next((s["id"] for s in program if s["op"] == "temporal_order"), None)
                          or next((s["id"] for s in program if s["op"] == "answer"), None))
            out = {"answer": v["answer"], "verdict": v["verdict"], "partial": False, "grounded": True}
            if supporting:
                out["supporting_step"] = supporting
            return out
        return {"answer": "Unknown", "verdict": "Could not determine an answer", "partial": True}
