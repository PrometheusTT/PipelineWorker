"""Regression tests for dispatching archived images to Slurm."""

import asyncio
import os
import subprocess
import struct
import tempfile
import unittest
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import HTTPException

import pipeline_worker as module


def tiny_v3draw():
    return (
        struct.pack("<24s", b"raw_image_stack_by_hpeng")
        + b"L"
        + struct.pack("<h", 1)
        + struct.pack("<4I", 2, 1, 1, 1)
        + b"\x01\x02"
    )


class ArchivedPipelineTests(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.old_cwd = os.getcwd()
        os.chdir(self.temp.name)
        self.root = Path(self.temp.name) / "pool"
        self.root.mkdir()
        self.h5 = self.root / "folder" / "nested" / "sample.pyramid.h5"
        self.h5.parent.mkdir(parents=True)
        self.h5.write_bytes(b"verified H5")
        self.config_patch = patch.object(module.cfg, "ProcessingStorageRoot", str(self.root))
        self.config_patch.start()
        self.worker = module.PipelineWorker(worker_id="test")

    async def asyncTearDown(self):
        await self.worker.http_client.aclose()
        self.config_patch.stop()
        os.chdir(self.old_cwd)
        self.temp.cleanup()

    def job(self, step_name="h5_to_v3draw"):
        return module.JobInfo(
            job_id="42",
            pipeline_id="pipeline-1",
            step_name=step_name,
            submit_time=datetime.now(),
            last_check_time=datetime.now(),
            h5_image_name=self.h5.name,
            source_relative_path="folder/nested/sample.pyramid.h5",
        )

    async def test_resolves_nested_pool_input_and_rejects_escape(self):
        self.assertEqual(
            self.worker._resolve_h5_path("folder/nested/sample.pyramid.h5", self.h5.name),
            os.path.realpath(self.h5),
        )
        for unsafe in ("../outside.pyramid.h5", "/tmp/outside.pyramid.h5"):
            with self.assertRaises(ValueError):
                self.worker._resolve_h5_path(unsafe, self.h5.name)
        (self.root / "escape").symlink_to(Path(self.temp.name))
        with self.assertRaises(ValueError):
            self.worker._resolve_h5_path("escape/outside.pyramid.h5", self.h5.name)

    async def test_sbatch_uses_archived_path_and_checks_dependent_input(self):
        script_dir = Path(self.temp.name) / "slurm"
        with patch.object(module.cfg, "SlurmScriptPath", str(script_dir)), patch.object(
            module.subprocess, "run",
            return_value=SimpleNamespace(stdout="Submitted batch job 123\n"),
        ):
            job_id = await self.worker._submit_sbatch_job(
                "pipeline-1", "h5_to_v3draw", self.h5.name,
                "folder/nested/sample.pyramid.h5",
            )
        self.assertEqual(job_id, "123")
        script = next(script_dir.glob("sbatch_*.sh")).read_text()
        self.assertIn(str(self.h5), script)
        self.assertNotIn("transfer_temp", script)
        self.assertIsNone(
            await self.worker._submit_sbatch_job(
                "pipeline-1", "bit_conversion", self.h5.name,
                "folder/nested/sample.pyramid.h5",
            )
        )

    async def test_completion_checks_output_in_archived_directory(self):
        output = self.h5.with_name("sample.v3draw")
        output.write_bytes(tiny_v3draw())
        self.worker.http_client.post = AsyncMock(
            return_value=SimpleNamespace(status_code=200)
        )
        self.assertTrue(await self.worker._handle_job_completion(self.job(), "COMPLETED"))
        self.worker.http_client.post.assert_awaited_once()

        self.worker.http_client.post.reset_mock()
        output.unlink()
        self.worker._notify_step_failed = AsyncMock(return_value=True)
        self.assertTrue(await self.worker._handle_job_completion(self.job(), "COMPLETED"))
        self.worker._notify_step_failed.assert_awaited_once()
        self.worker.http_client.post.assert_not_awaited()

    async def test_completion_callback_failure_retains_job_for_retry(self):
        self.h5.with_name("sample.v3draw").write_bytes(tiny_v3draw())
        self.worker.http_client.post = AsyncMock(
            return_value=SimpleNamespace(status_code=503)
        )
        self.assertFalse(await self.worker._handle_job_completion(self.job(), "COMPLETED"))

    async def test_failed_callback_failure_retains_job_for_retry(self):
        job = self.job()
        job.retry_count = 2
        self.worker._notify_step_failed = AsyncMock(return_value=False)
        self.worker._get_job_error_details = AsyncMock(return_value=None)
        self.assertFalse(await self.worker._handle_job_completion(job, "FAILED"))

    async def test_recovery_keeps_job_until_callback_acknowledged(self):
        job = self.job()
        key = "pipeline-1_h5_to_v3draw"
        self.worker.running_jobs[key] = job
        self.worker._check_job_status = AsyncMock(return_value="COMPLETED")
        self.worker._handle_job_completion = AsyncMock(return_value=False)
        with patch.object(module.asyncio, "sleep", new=AsyncMock()):
            await self.worker._verify_recovered_jobs()
        self.assertIn(key, self.worker.running_jobs)
        self.assertEqual(self.worker.running_jobs[key].job_id, "42")

    async def test_running_jobs_state_is_published_atomically(self):
        self.worker.running_jobs["pipeline-1_h5_to_v3draw"] = self.job()
        self.worker._save_running_jobs()
        self.assertTrue(Path("running_jobs.json").is_file())
        self.assertEqual(
            __import__("json").loads(Path("running_jobs.json").read_text())[
                "pipeline-1_h5_to_v3draw"
            ]["source_relative_path"],
            "folder/nested/sample.pyramid.h5",
        )
        self.assertEqual(list(Path(".").glob(".running_jobs-*")), [])

    async def test_running_jobs_state_can_live_outside_checkout(self):
        state_file = Path(self.temp.name) / "state" / "running_jobs.json"
        state_file.parent.mkdir()
        with patch.object(module.cfg, "WorkerStatePath", str(state_file)):
            worker = module.PipelineWorker(worker_id="test-external-state")
        try:
            self.assertEqual(worker.running_jobs_file, str(state_file))
            self.assertEqual(worker.running_jobs, {})
            worker.running_jobs["pipeline-1_h5_to_v3draw"] = self.job()
            worker._save_running_jobs()
            self.assertTrue(state_file.is_file())
            self.assertFalse(Path("running_jobs.json").exists())
            with patch.object(module.cfg, "WorkerStatePath", str(state_file)):
                recovered = module.PipelineWorker(worker_id="test-recovered-state")
            try:
                self.assertIn("pipeline-1_h5_to_v3draw", recovered.running_jobs)
            finally:
                await recovered.http_client.aclose()
        finally:
            await worker.http_client.aclose()

    async def test_slurm_query_timeout_does_not_mark_job_failed(self):
        with patch.object(
            module.subprocess, "run", side_effect=subprocess.TimeoutExpired("squeue", 10)
        ):
            self.assertEqual(await self.worker._check_job_status(self.job()), "UNKNOWN")

    async def test_cancelled_sacct_state_is_not_truncated(self):
        def run(command, **_kwargs):
            if command[0] == "squeue":
                return SimpleNamespace(returncode=0, stdout="")
            self.assertIn("State%32", command)
            return SimpleNamespace(returncode=0, stdout="CANCELLED by 1000\n")

        with patch.object(module.subprocess, "run", side_effect=run):
            self.assertEqual(await self.worker._check_job_status(self.job()), "CANCELLED")

    async def test_cancelled_job_is_not_retried(self):
        job = self.job()
        self.worker._submit_sbatch_job = AsyncMock(return_value="43")
        self.worker._notify_step_failed = AsyncMock(return_value=True)
        self.worker._get_job_error_details = AsyncMock(return_value=None)
        self.assertTrue(await self.worker._handle_job_completion(job, "CANCELLED"))
        self.worker._submit_sbatch_job.assert_not_awaited()
        self.worker._notify_step_failed.assert_awaited_once()

    async def test_retry_preserves_archived_source(self):
        job = self.job()
        self.worker._submit_sbatch_job = AsyncMock(return_value="43")
        self.worker._notify_step_started = AsyncMock()
        with patch.object(self.worker, "_save_running_jobs"):
            self.assertFalse(await self.worker._handle_job_completion(job, "FAILED"))
        self.worker._submit_sbatch_job.assert_awaited_once_with(
            "pipeline-1", "h5_to_v3draw", self.h5.name,
            "folder/nested/sample.pyramid.h5",
        )
        self.assertEqual(job.job_id, "43")

    async def test_start_endpoint_requires_token_and_passes_source_path(self):
        request = module.StartStepRequest(
            pipeline_id="pipeline-1",
            step_name="h5_to_v3draw",
            h5_image_name=self.h5.name,
            source_relative_path="folder/nested/sample.pyramid.h5",
        )
        fake_worker = SimpleNamespace(
            start_step=AsyncMock(return_value={"status": "submitted", "job_id": "123"})
        )
        with patch.object(module.cfg, "WorkerToken", "secret"), patch.object(
            module, "worker", fake_worker
        ):
            with self.assertRaises(HTTPException) as error:
                await module.start_step_endpoint(request, "wrong")
            self.assertEqual(error.exception.status_code, 403)
            response = await module.start_step_endpoint(request, "secret")
        self.assertEqual(response["job_id"], "123")
        fake_worker.start_step.assert_awaited_once_with(
            "pipeline-1", "h5_to_v3draw",
            h5_image_name=self.h5.name,
            source_relative_path="folder/nested/sample.pyramid.h5",
            force_restart=False,
        )


if __name__ == "__main__":
    unittest.main()
