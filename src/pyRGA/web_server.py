"""Optional local HTTP adapter for the shared pyRGA experiment workflows."""

from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import os
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import Field

from .experiment_models import (
    ExperimentExport,
    ExperimentModel,
    GeneratedField,
    GenerateFieldRequest,
    InverseRequest,
    InverseResult,
    PumpingTestRequest,
    PumpingTestResult,
)
from .jobs import JobManager, JobStatus


class GenerateFieldJob(ExperimentModel):
    kind: Literal["generate_field"]
    payload: GenerateFieldRequest


class PumpingTestJob(ExperimentModel):
    kind: Literal["run_pumping_test"]
    payload: PumpingTestRequest


class InverseJob(ExperimentModel):
    kind: Literal["run_inverse_model"]
    payload: InverseRequest


JobRequest = Annotated[GenerateFieldJob | PumpingTestJob | InverseJob, Field(discriminator="kind")]


def create_app(*, manager: JobManager | None = None) -> FastAPI:
    """Create an app without starting processes until its lifespan begins."""
    jobs = manager if manager is not None else JobManager()

    @asynccontextmanager
    async def lifespan(app):
        jobs.start()
        try:
            yield
        finally:
            jobs.close()

    app = FastAPI(
        title="pyRGA local experiment API",
        version="1.0",
        description="Local, in-memory steady-state experiments. Job state is lost on restart.",
        lifespan=lifespan,
    )
    app.state.jobs = jobs
    origins = ["http://localhost:3000", "http://127.0.0.1:3000"]
    if origin := os.environ.get("PYRGA_PORTAL_ORIGIN"):
        if origin not in origins:
            origins.append(origin)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_methods=["GET", "POST", "DELETE"],
        allow_headers=["Content-Type"],
    )

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, error):
        # Exception objects in model-validator contexts and nonfinite raw input
        # are not JSON-safe; report the public error type/location/message only.
        details = [
            {key: item[key] for key in ("type", "loc", "msg") if key in item}
            for item in error.errors()
        ]
        return JSONResponse(
            status_code=422,
            content={"error": {"code": "validation_error", "message": "Invalid request", "details": details}},
        )

    @app.exception_handler(HTTPException)
    async def http_error(request, error):
        detail = error.detail if isinstance(error.detail, dict) else {
            "code": "http_error", "message": str(error.detail)
        }
        return JSONResponse(status_code=error.status_code, content={"error": detail})

    @app.get("/api/v1/health")
    def health():
        from . import __version__

        return {"status": "ok", "schema_version": "1.0", "solver_version": __version__}

    @app.get("/api/v1/schema")
    def schemas():
        from .api_contract import experiment_schemas
        return experiment_schemas()

    @app.post("/api/v1/jobs", response_model=JobStatus, status_code=202)
    def submit(request: JobRequest):
        try:
            return jobs.submit(request.kind, request.payload.model_dump(mode="json"))
        except OverflowError as error:
            raise HTTPException(429, {"code": "queue_full", "message": str(error)}) from error

    @app.get("/api/v1/jobs/{job_id}", response_model=JobStatus)
    def status(job_id: str):
        try:
            return jobs.status(job_id)
        except KeyError as error:
            raise HTTPException(404, {"code": "job_not_found", "message": "Unknown or expired job"}) from error

    @app.get(
        "/api/v1/jobs/{job_id}/result",
        response_model=GeneratedField | PumpingTestResult | InverseResult,
    )
    def result(job_id: str):
        try:
            return jobs.result(job_id)
        except KeyError as error:
            raise HTTPException(404, {"code": "job_not_found", "message": "Unknown or expired job"}) from error
        except ValueError as error:
            raise HTTPException(409, {"code": "result_unavailable", "message": str(error)}) from error

    @app.delete("/api/v1/jobs/{job_id}", response_model=JobStatus)
    def cancel(job_id: str):
        try:
            return jobs.cancel(job_id)
        except KeyError as error:
            raise HTTPException(404, {"code": "job_not_found", "message": "Unknown or expired job"}) from error

    @app.post("/api/v1/experiments/validate", response_model=ExperimentExport)
    def validate_experiment(experiment: ExperimentExport):
        return experiment

    @app.post("/api/v1/experiments/inverse-request", response_model=InverseRequest)
    def inverse_request(experiment: ExperimentExport):
        try:
            return experiment.inverse_request()
        except ValueError as error:
            raise HTTPException(422, {"code": "invalid_inverse_selection", "message": str(error)}) from error

    return app


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="Run the local pyRGA experiment API")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    uvicorn.run(create_app(), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
