"""Launch the BSTS kernel while preserving native TensorFlow diagnostics off-notebook."""

from __future__ import annotations

import os

from ipykernel.kernelapp import launch_new_instance


def main() -> None:
    """Redirect native stderr to a diagnostic log, then start ipykernel."""
    log_path = os.environ.get("ML4T_BSTS_KERNEL_LOG", "/tmp/ml4t-bsts-kernel.stderr")
    stderr_fd = os.open(log_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    os.dup2(stderr_fd, 2)
    os.close(stderr_fd)
    launch_new_instance()


if __name__ == "__main__":
    main()
