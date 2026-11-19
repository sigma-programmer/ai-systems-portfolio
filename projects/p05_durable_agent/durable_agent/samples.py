"""Sample workflows and a simulated external system.

* :class:`Ledger` is an *external* side-effect target (its own SQLite file,
  as a payment or email provider would be). It deduplicates on idempotency
  keys, as real providers do, and it records every invocation so tests can
  count re-invocations as well as effects.
* :func:`order_workflow` is a small reserve → charge → ship pipeline in which
  every step has a side effect. The crash-recovery tests use it.
* :func:`research_workflow` is the sample agent:
  plan (LLM) → search (flaky tool) → draft (LLM) → human review → publish.
"""
from __future__ import annotations

import json
import random
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Callable

from aikit import LLM, MockLLM, get_llm

from .engine import StepContext
from .errors import RetryableError
from .retry import RetryPolicy
from .workflow import END, Workflow, branch

# ============================================================================ ledger


class Ledger:
    """Simulated external system with idempotency-key deduplication."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False, timeout=30)
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode=WAL")
        self._conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS effects (
                id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT UNIQUE NOT NULL,
                run_id TEXT, name TEXT, visit INTEGER, payload TEXT, ts REAL);
            CREATE TABLE IF NOT EXISTS calls (id INTEGER PRIMARY KEY AUTOINCREMENT, key TEXT, ts REAL);
            """
        )

    def apply(self, key: str, name: str, payload: Any = None) -> dict[str, Any]:
        """Perform effect ``name`` once per ``key``; returns a receipt."""
        run_id, _step, visit, _ = key.split("/", 3)
        with self._lock:
            self._conn.execute("INSERT INTO calls (key, ts) VALUES (?, ?)", (key, time.time()))
            cur = self._conn.execute(
                "INSERT OR IGNORE INTO effects (key, run_id, name, visit, payload, ts) VALUES (?,?,?,?,?,?)",
                (key, run_id, name, int(visit), json.dumps(payload), time.time()),
            )
            receipt = self._conn.execute("SELECT id FROM effects WHERE key=?", (key,)).fetchone()[0]
        return {"receipt": receipt, "deduplicated": cur.rowcount == 0}

    def count(self, name: str | None = None, run_id: str | None = None) -> int:
        sql, params = "SELECT COUNT(*) FROM effects WHERE 1=1", []
        if name:
            sql += " AND name=?"
            params.append(name)
        if run_id:
            sql += " AND run_id=?"
            params.append(run_id)
        with self._lock:
            return self._conn.execute(sql, params).fetchone()[0]

    def calls(self) -> int:
        with self._lock:
            return self._conn.execute("SELECT COUNT(*) FROM calls").fetchone()[0]

    def duplicates(self) -> int:
        """Logical effects (run, name, visit) applied more than once. Must be 0."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT COUNT(*) FROM (SELECT run_id, name, visit FROM effects GROUP BY run_id, name, visit HAVING COUNT(*) > 1)"
            ).fetchone()
        return rows[0]

    def effects(self, run_id: str | None = None) -> list[tuple[str, str]]:
        with self._lock:
            if run_id:
                rows = self._conn.execute("SELECT run_id, name FROM effects WHERE run_id=? ORDER BY id", (run_id,)).fetchall()
            else:
                rows = self._conn.execute("SELECT run_id, name FROM effects ORDER BY id").fetchall()
        return [(r[0], r[1]) for r in rows]


# ============================================================================ order pipeline

ORDER_EFFECTS = ("reserve", "charge", "ship")


def order_workflow(
    ledger: Ledger,
    hang_after: str | None = None,
    on_effect: Callable[[str], None] | None = None,
    flaky: Callable[[str, StepContext], None] | None = None,
    retry: RetryPolicy | None = None,
) -> Workflow:
    """reserve → charge → ship, each step performing one external side effect.

    Args:
        hang_after: step name after whose effect the step blocks forever
            (used to SIGKILL a worker process at a precise point).
        on_effect: callback invoked right after each effect (e.g. print a marker).
        flaky: hook called at the start of each step (inject transient errors).
        retry: retry policy for all steps.
    """
    wf = Workflow("order", start="reserve")
    policy = retry or RetryPolicy(max_attempts=5, base_delay=0.1, max_delay=2.0, retry_on=(ConnectionError, TimeoutError))

    def make(name: str, nxt: str | None) -> None:
        def handler(ctx: StepContext) -> dict[str, Any]:
            if flaky is not None:
                flaky(name, ctx)
            receipt = ctx.effect(name, lambda key: ledger.apply(key, name, {"order": ctx.input}))
            if on_effect is not None:
                on_effect(name)
            if hang_after == name:
                while True:  # wait to be killed
                    time.sleep(0.05)
            return {"receipt": receipt["receipt"]}

        handler.__name__ = name
        wf.step(name, next=nxt, retry=policy)(handler)

    make("reserve", "charge")
    make("charge", "ship")
    make("ship", END)
    return wf


# ============================================================================ research agent

CORPUS: dict[str, str] = {
    "durable execution": "Durable execution engines persist workflow progress so that a crash resumes from the last checkpoint instead of restarting.",
    "event sourcing": "Event sourcing stores every state change as an immutable event; current state is a fold over the log.",
    "idempotency": "Idempotency keys let a client safely retry a request: the server performs the operation at most once per key.",
    "exponential backoff": "Exponential backoff with full jitter spreads retries uniformly in a growing window, preventing thundering herds.",
    "human in the loop": "Human-in-the-loop gates pause automated workflows until a person approves high-impact actions.",
    "sqlite wal": "SQLite in WAL mode allows concurrent readers with one writer and survives crashes once a transaction commits.",
}


def search_corpus(query: str, k: int = 2) -> list[dict[str, str]]:
    """Tiny keyword search tool over :data:`CORPUS`."""
    q = set(re.findall(r"[a-z]+", query.lower()))
    scored = []
    for title, text in CORPUS.items():
        words = set(re.findall(r"[a-z]+", (title + " " + text).lower()))
        overlap = len(q & words)
        if overlap:
            scored.append((overlap, title, text))
    scored.sort(key=lambda x: (-x[0], x[1]))
    return [{"title": t, "snippet": s} for _, t, s in scored[:k]]


class FlakySearch:
    """Search tool that fails transiently with probability ``fail_rate`` (seeded)."""

    def __init__(self, fail_rate: float = 0.0, seed: int = 0) -> None:
        self.fail_rate = fail_rate
        self.rng = random.Random(seed)
        self.calls = 0
        self.failures = 0

    def __call__(self, query: str) -> list[dict[str, str]]:
        self.calls += 1
        if self.rng.random() < self.fail_rate:
            self.failures += 1
            raise ConnectionError("search backend unavailable (simulated)")
        return search_corpus(query)


def research_mock_llm() -> MockLLM:
    """Deterministic offline LLM for the research agent."""

    def plan(prompt: str, system: str | None) -> str:
        topic = re.search(r"TOPIC: (.*)", prompt)
        t = topic.group(1).strip() if topic else "agents"
        return json.dumps({"queries": [f"{t} durable execution", "idempotency keys", "exponential backoff"]})

    def draft(prompt: str, system: str | None) -> str:
        topic = re.search(r"TOPIC: (.*)", prompt)
        feedback = re.search(r"REVIEWER FEEDBACK: (.*)", prompt)
        sources = re.findall(r"- \[(.*?)\]", prompt)
        body = (
            f"# Report: {topic.group(1).strip() if topic else 'topic'}\n\n"
            f"Production agents need durable state. Findings draw on {len(sources)} sources: {', '.join(sources)}.\n"
            "Checkpoint after every step, make side effects idempotent, and retry with jittered backoff."
        )
        if feedback:
            body += f"\n\nRevision note: addressed feedback ({feedback.group(1).strip()})."
        return body

    return MockLLM().on(r"Plan research", plan).on(r"Write a report", draft)


def research_workflow(publisher: Ledger) -> Workflow:
    """plan → search → draft → review (human gate) → publish → notify."""
    wf = Workflow("research_report", start="plan", version="1")
    llm_retry = RetryPolicy(max_attempts=4, base_delay=1.0, max_delay=20.0)
    tool_retry = RetryPolicy(max_attempts=6, base_delay=0.5, max_delay=10.0)

    @wf.step(next="search", retry=llm_retry, timeout=120)
    def plan(ctx: StepContext) -> dict[str, Any]:
        """Ask the LLM for search queries."""
        llm: LLM = ctx.llm
        resp = llm.complete(f"Plan research. Return JSON {{\"queries\": [...]}} with 3 queries.\nTOPIC: {ctx.input['topic']}", json_mode=True)
        try:
            data = resp.json()
        except ValueError as exc:  # malformed model output: sampling again usually fixes it
            raise RetryableError(f"planner returned invalid JSON: {exc}") from exc
        if not isinstance(data, dict) or not data.get("queries"):
            raise RetryableError("planner returned no queries")
        return {"queries": [str(q) for q in data["queries"]][:5]}

    @wf.step(next="draft", retry=tool_retry, timeout=60)
    def search(ctx: StepContext) -> dict[str, Any]:
        """Run every planned query through the (flaky) search tool."""
        tool = ctx.services.get("search", search_corpus)
        seen: dict[str, dict[str, str]] = {}
        for q in ctx.outputs["plan"]["queries"]:
            for hit in tool(q):
                seen.setdefault(hit["title"], hit)
        return {"results": list(seen.values())}

    @wf.step(next="review", retry=llm_retry, timeout=120)
    def draft(ctx: StepContext) -> dict[str, Any]:
        """Write the report, incorporating reviewer feedback on revisions."""
        sources = "\n".join(f"- [{r['title']}] {r['snippet']}" for r in ctx.outputs["search"]["results"])
        prompt = f"Write a report.\nTOPIC: {ctx.input['topic']}\nSOURCES:\n{sources}"
        review = ctx.approvals.get("review")
        if review and review["decision"] == "rejected":
            prompt += f"\nREVIEWER FEEDBACK: {review['comment']}"
        text = ctx.llm.complete(prompt).text
        return {"title": f"Report: {ctx.input['topic']}", "body": text, "revision": ctx.visit}

    wf.approval(
        "review",
        on_approve="publish",
        on_reject="draft",  # revision loop
        approvers={"editor"},
        timeout=24 * 3600,
        escalate_to={"managing_editor"},
        escalation_timeout=12 * 3600,
        on_timeout="fail",
        prepare=lambda ctx: {"title": ctx.outputs["draft"]["title"], "preview": ctx.outputs["draft"]["body"][:200]},
        max_visits=5,
    )

    @wf.step(next=branch(lambda out, ctx: bool(ctx.input.get("notify")), {True: "notify", False: END}))
    def publish(ctx: StepContext) -> dict[str, Any]:
        """Publish exactly once to the external CMS (idempotency key)."""
        d = ctx.outputs["draft"]
        receipt = ctx.effect("publish", lambda key: publisher.apply(key, "publish", {"title": d["title"]}))
        return {"published": True, "receipt": receipt["receipt"]}

    @wf.step(next=END)
    def notify(ctx: StepContext) -> dict[str, Any]:
        """Enqueue a notification through a transactional outbox in the engine DB."""

        def enqueue(conn: Any, key: str) -> dict[str, Any]:
            conn.execute("CREATE TABLE IF NOT EXISTS outbox (key TEXT PRIMARY KEY, topic TEXT, body TEXT)")
            conn.execute("INSERT INTO outbox VALUES (?,?,?)", (key, "reports", ctx.outputs["draft"]["title"]))
            return {"queued": key}

        return ctx.effect("notify", enqueue, transactional=True)

    return wf


def default_llm() -> LLM:
    """Real Claude when credentials exist and AIKIT_OFFLINE is unset, else the mock."""
    return get_llm(research_mock_llm())
