from pathlib import Path

from onnxruntime import (
    InferenceSession,
    SessionOptions,  # ty: ignore[possibly-missing-import]  # onnxruntime imports it inside a try block
)


def cpu_session(model: Path) -> InferenceSession:
    """A single-threaded CPU session: these models are small and run once per audio frame or turn."""
    options = SessionOptions()
    options.inter_op_num_threads = 1
    options.intra_op_num_threads = 1
    options.log_severity_level = 3
    return InferenceSession(model, sess_options=options, providers=["CPUExecutionProvider"])
