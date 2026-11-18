"""Abstract base class for job runners."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Literal


@dataclass
class Job:
    """Represents a pipeline job."""

    object_name: str
    method_name: str
    stage: Literal["cleanup", "evaluate", "variant_gt"]  # variant_gt: per object, no method
    config_path: str


class JobRunner(ABC):
    """Abstract base class for job execution."""

    @abstractmethod
    def submit(self, job: Job) -> str:
        """Submit a job for execution.

        Args:
            job: Job to submit.

        Returns:
            Job ID string.
        """
        pass

    @abstractmethod
    def submit_with_dependency(self, job: Job, depends_on: list[str]) -> str:
        """Submit a job with dependencies.

        Args:
            job: Job to submit.
            depends_on: List of job IDs that must complete first.

        Returns:
            Job ID string.
        """
        pass

    @abstractmethod
    def wait(self, job_ids: list[str]) -> dict[str, bool]:
        """Wait for jobs to complete.

        Args:
            job_ids: List of job IDs to wait for.

        Returns:
            Dictionary mapping job_id to success status.
        """
        pass

    def submit_variant_gt(self, config, objects: list[str]) -> dict[str, str]:
        """Jobs deriving the GT of a custom view set, as {object: job id}; the evaluate jobs wait for them.

        Empty by default: the caller derived it already.
        """
        return {}

    def run_pipeline(
        self,
        config,
        objects: list[str] | None = None,
        methods: list[str] | None = None,
        stages: list[str] = ["cleanup", "evaluate"],
    ) -> dict[str, bool]:
        """Run the full pipeline for specified objects and methods.

        Args:
            config: Pipeline configuration.
            objects: Objects to process (default: all from config).
            methods: Methods to process (default: all from config).
            stages: Pipeline stages to run.

        Returns:
            Dictionary mapping job descriptions to success status.
        """
        objects = objects or config.dataset.objects
        methods = methods or config.dataset.methods

        cleanup_jobs = {}
        eval_jobs = {}

        if "cleanup" in stages:
            for obj in objects:
                for method in methods:
                    job = Job(
                        object_name=obj,
                        method_name=method,
                        stage="cleanup",
                        config_path=str(config._config_path) if hasattr(config, "_config_path") else "",
                    )
                    job_id = self.submit(job)
                    cleanup_jobs[f"{obj}/{method}/cleanup"] = job_id

        gt_jobs = self.submit_variant_gt(config, objects) if "evaluate" in stages else {}

        if "evaluate" in stages:
            for obj in objects:
                for method in methods:
                    job = Job(
                        object_name=obj,
                        method_name=method,
                        stage="evaluate",
                        config_path=str(config._config_path) if hasattr(config, "_config_path") else "",
                    )

                    cleanup_key = f"{obj}/{method}/cleanup"
                    depends_on = [jobs[key] for jobs, key in ((cleanup_jobs, cleanup_key), (gt_jobs, obj)) if key in jobs]
                    if depends_on:
                        job_id = self.submit_with_dependency(job, depends_on)
                    else:
                        job_id = self.submit(job)

                    eval_jobs[f"{obj}/{method}/evaluate"] = job_id

        all_jobs = {**cleanup_jobs, **{f"{obj}/variant_gt": job_id for obj, job_id in gt_jobs.items()}, **eval_jobs}
        results = self.wait(list(all_jobs.values()))

        return {desc: results.get(job_id, False) for desc, job_id in all_jobs.items()}
