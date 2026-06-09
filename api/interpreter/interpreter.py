"""
Tree-walking interpreter: executes a (pre-validated) DSL program against a Cache and
emits a run-doc (run_doc.schema.json). Maintains a binding environment id -> value;
each op also produces its run-doc trace entry. Replay mode = zero live calls.
"""
from __future__ import annotations

from limits import MAX_STEPS, ProgramLimitExceeded

from .primitives import OPS


class Interpreter:
    def __init__(self, cache):
        self.cache = cache

    def run(self, program: list[dict], query: str | None = None) -> dict:
        # Runtime step cap (trust boundary): the residual guard for any caller that reaches the
        # interpreter without going through validation_errors first (the server path validates
        # before calling run, but a future/direct caller might not). Cheap len() comparison —
        # no per-step instrumentation, since steps<=32 and frames<=2000/call bound the work.
        if len(program) > MAX_STEPS:
            raise ProgramLimitExceeded(f"program has {len(program)} steps; max is {MAX_STEPS}")

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
        # Grounded path: a real answer synthesized from the answer binding (any question shape).
        if answer_binding and answer_binding["kind"] == "answer" and answer_binding["value"].get("grounded"):
            v = answer_binding["value"]
            supporting = (next((s["id"] for s in program if s["op"] == "temporal_order"), None)
                          or next((s["id"] for s in program if s["op"] == "answer"), None))
            out = {"answer": v["answer"], "verdict": v["verdict"],
                   "grounded": True, "reason": None, "partial": False}
            if supporting:
                out["supporting_step"] = supporting
            return out
        # Ungrounded path (Q1 -> A): the answer step couldn't ground, or there was no answer
        # binding at all. Carry the explicit discriminator — never a fabricated "Unknown" verdict.
        reason = "no-answer-step"
        if answer_binding and answer_binding["kind"] == "answer":
            reason = answer_binding["value"].get("reason") or "no-grounded-answer"
        return {"answer": None, "verdict": None, "grounded": False, "reason": reason, "partial": True}
