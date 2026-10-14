import os

CoreServerURL = os.environ.get("HNDB_CORE_SERVER_URL", "http://127.0.0.1:8000")
WorkerToken = os.environ.get("HNDB_PIPELINE_WORKER_TOKEN", "")

HNDBRootDirectory = os.environ.get("HNDB_ROOT", "/mnt/nfs/hndb")
SamplePreparationDirectory = os.environ.get(
    "HNDB_SAMPLE_PREPARATION_ROOT",
    os.path.join(HNDBRootDirectory, "HNDB_files", "SamplePreparation"),
)
Cell_ImagesDirectory = os.environ.get(
    "HNDB_CELL_IMAGE_ROOT", os.path.join(HNDBRootDirectory, "Cell_Image")
)
Cell_MIPsDirectory = os.environ.get(
    "HNDB_CELL_MIP_ROOT", os.path.join(HNDBRootDirectory, "Cell_MIP")
)
PTRSB_DBDirectory = os.path.join(HNDBRootDirectory, "PTRSB_DB")
ProcessingStorageRoot = os.environ.get(
    "HNDB_PROCESSING_STORAGE_ROOT", PTRSB_DBDirectory
)

# Optional compatibility path for manually created historical pipelines.
LegacyInputRoot = os.environ.get("HNDB_LEGACY_PROCESSING_INPUT_ROOT", "")

SlurmScriptPath = os.environ.get(
    "HNDB_SLURM_SCRIPT_PATH", "/home/power/Desktop/PipelineWorker/slurm_scripts"
)
SlurmPartition = os.environ.get("HNDB_SLURM_PARTITION", "normal")
SlurmQos = os.environ.get("HNDB_SLURM_QOS", "pbstorageaccess")
SlurmMemory = os.environ.get("HNDB_SLURM_MEMORY", "180G")
ProcessingPython = os.environ.get(
    "HNDB_PROCESSING_PYTHON",
    "/home/power/anaconda3/envs/HumanDatabaseTools/bin/python3",
)
Delete16BitAfter8Bit = os.environ.get(
    "HNDB_DELETE_16BIT_AFTER_8BIT", "true"
).strip().lower() in {"1", "true", "yes", "on"}
