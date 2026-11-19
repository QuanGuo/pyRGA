"""In-memory, single-worker jobs for the local experiment service.

Each computation runs in its own spawned process. This isolates NumPy random
state and allows cancellation of an expensive numerical solve.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
import multiprocessing
from multiprocessing.connection import Connection
import threading
from typing import Any, Literal
from uuid import uuid4

from pydantic import BaseModel, Field


JobKind = Literal["generate_field", "run_pumping_test", "run_inverse_model"]
JobState = Literal["queued", "running", "succeeded", "failed", "cancelled"]
TERMINAL_STATES = {"succeeded", "failed", "cancelled"}


class JobError(BaseModel):
    code: str
    message: str


class JobStatus(BaseModel):
    job_id: str
    kind: JobKind
    status: JobState = "queued"
    progress: float = Field(default=0.0, ge=0, le=1)
    stage: str = "queued"
    message: str = "Waiting for the compute worker"
    detail: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    error: JobError | None = None


@dataclass
class _Job:
    status: JobStatus
    payload: dict[str, Any]
    result: dict[str, Any] | None = None


def _now() -> datetime:
    return datetime.now(timezone.utc)


def validate_job_payload(kind: JobKind, payload: dict[str, Any]) -> dict[str, Any]:
    """Validate before queueing and pass only JSON-compatible values to workers."""
    from .experiment_models import GenerateFieldRequest, InverseRequest, PumpingTestRequest

    model = {
        "generate_field": GenerateFieldRequest,
        "run_pumping_test": PumpingTestRequest,
        "run_inverse_model": InverseRequest,
    }[kind]
    return model.model_validate(payload).model_dump(mode="json")


def _compute(kind: JobKind, payload: dict[str, Any], connection: Connection, cancel_event) -> None:
    """Spawn-safe entrypoint; communication contains JSON-compatible values only."""
    try:
        from .experiment_models import GenerateFieldRequest, InverseRequest, PumpingTestRequest
        from .experiments import generate_field, run_pumping_test
        from .inverse import run_inverse

        def report(update: dict[str, Any]) -> None:
            connection.send(("progress", update))

        function, model = {
            "generate_field": (generate_field, GenerateFieldRequest),
            "run_pumping_test": (run_pumping_test, PumpingTestRequest),
            "run_inverse_model": (run_inverse, InverseRequest),
        }[kind]
        result = function(model.model_validate(payload), progress=report, cancelled=cancel_event.is_set)
        connection.send(("result", result.model_dump(mode="json")))
    except Exception as error:
        connection.send(("error", {"code": "computation_failed", "message": str(error)}))
    finally:
        connection.close()


class JobManager:
    """Bounded FIFO queue with one isolated compute process at a time."""

    def __init__(self, *, queue_limit: int = 16, retained_limit: int = 64):
        self.queue_limit = queue_limit
        self.retained_limit = retained_limit
        self._context = multiprocessing.get_context("spawn")
        self._jobs: dict[str, _Job] = {}
        self._queue: deque[str] = deque()
        self._finished: deque[str] = deque()
        self._running: tuple[str, Any, Connection, Any] | None = None
        self._lock = threading.RLock()
        self._wake = threading.Event()
        self._closed = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        with self._lock:
            if self._closed:
                raise RuntimeError("Job manager is closed")
            if self._thread is None:
                self._thread = threading.Thread(target=self._monitor, name="pyrga-jobs", daemon=True)
                self._thread.start()

    def submit(self, kind: JobKind, payload: dict[str, Any]) -> JobStatus:
        validated = validate_job_payload(kind, payload)
        with self._lock:
            self.start()
            if len(self._queue) >= self.queue_limit:
                raise OverflowError("Compute queue is full; retry after a job finishes")
            status = JobStatus(job_id=str(uuid4()), kind=kind, created_at=_now())
            self._jobs[status.job_id] = _Job(status=status, payload=validated)
            self._queue.append(status.job_id)
            self._wake.set()
            return status.model_copy(deep=True)

    def status(self, job_id: str) -> JobStatus:
        with self._lock:
            return self._jobs[job_id].status.model_copy(deep=True)

    def result(self, job_id: str) -> dict[str, Any]:
        with self._lock:
            job = self._jobs[job_id]
            if job.status.status != "succeeded":
                raise ValueError(f"Job is {job.status.status}; no successful result is available")
            assert job.result is not None
            return job.result

    def cancel(self, job_id: str) -> JobStatus:
        with self._lock:
            job = self._jobs[job_id]
            if job.status.status in TERMINAL_STATES:
                return job.status.model_copy(deep=True)
            if job_id in self._queue:
                self._queue.remove(job_id)
            if self._running is not None and self._running[0] == job_id:
                _, process, _, cancel_event = self._running
                cancel_event.set()
                process.terminate()
                process.join(timeout=2)
            self._finish(job_id, "cancelled", "Cancelled by the client")
            self._wake.set()
            return job.status.model_copy(deep=True)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
            for job_id in list(self._queue):
                self._finish(job_id, "cancelled", "Service is shutting down")
            self._queue.clear()
            if self._running is not None:
                job_id, process, connection, cancel_event = self._running
                cancel_event.set()
                if process.is_alive():
                    process.terminate()
                process.join(timeout=2)
                connection.close()
                self._running = None
                if self._jobs[job_id].status.status not in TERMINAL_STATES:
                    self._finish(job_id, "cancelled", "Service is shutting down")
                self._prune_finished()
            self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout=2)

    def _finish(self, job_id: str, state: JobState, message: str) -> None:
        job = self._jobs[job_id]
        if job.status.status in TERMINAL_STATES:
            return
        job.status.status = state
        job.status.stage = state
        job.status.message = message
        job.status.finished_at = _now()
        if state == "succeeded":
            job.status.progress = 1.0
        self._finished.append(job_id)
        self._prune_finished()

    def _prune_finished(self) -> None:
        while len(self._finished) > self.retained_limit:
            # Keep the record until the worker is reaped, then retry pruning.
            # Removing only the deque entry here would leak a retained record.
            if self._running is not None and self._running[0] == self._finished[0]:
                break
            self._jobs.pop(self._finished.popleft(), None)

    def _monitor(self) -> None:
        while True:
            with self._lock:
                if self._closed:
                    return
                if self._running is not None:
                    self._poll_running()
                if self._running is None and self._queue:
                    self._launch(self._queue.popleft())
            self._wake.wait(timeout=0.03)
            self._wake.clear()

    def _launch(self, job_id: str) -> None:
        job = self._jobs[job_id]
        parent, child = self._context.Pipe(duplex=False)
        cancel_event = self._context.Event()
        process = self._context.Process(
            target=_compute,
            args=(job.status.kind, job.payload, child, cancel_event),
            name=f"pyrga-{job_id}",
            daemon=True,
        )
        try:
            process.start()
        except Exception as error:
            parent.close()
            job.status.error = JobError(code="worker_start_failed", message=str(error))
            self._finish(job_id, "failed", "Unable to start the compute worker")
        else:
            job.status.status = "running"
            job.status.stage = "starting"
            job.status.message = "Starting the compute worker"
            job.status.started_at = _now()
            self._running = (job_id, process, parent, cancel_event)
        finally:
            child.close()

    def _poll_running(self) -> None:
        job_id, process, connection, _ = self._running
        job = self._jobs[job_id]
        try:
            while connection.poll():
                message_type, value = connection.recv()
                if job.status.status in TERMINAL_STATES:
                    continue
                if message_type == "progress":
                    job.status.progress = max(job.status.progress, min(1, max(0, value.get("progress", 0))))
                    job.status.stage = value.get("stage", "computing")
                    job.status.message = value.get("message", "Computing")
                    job.status.detail = {key: item for key, item in value.items() if key not in {"progress", "stage", "message"}}
                elif message_type == "result":
                    job.result = value
                    self._finish(job_id, "succeeded", "Computation completed")
                elif message_type == "error":
                    job.status.error = JobError.model_validate(value)
                    self._finish(job_id, "failed", value["message"])
        except (EOFError, OSError):
            pass
        if not process.is_alive():
            process.join(timeout=0.2)
            connection.close()
            self._running = None
            if job.status.status not in TERMINAL_STATES:
                job.status.error = JobError(code="worker_exited", message=f"Compute worker exited with code {process.exitcode}")
                self._finish(job_id, "failed", job.status.error.message)
            self._prune_finished()
