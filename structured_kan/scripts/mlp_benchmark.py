"""Backward-compatible MLP entry point for the common neural runner."""
from .neural_benchmark import inspect_config, fit_job, run_queue, main

if __name__ == '__main__':
    main()
