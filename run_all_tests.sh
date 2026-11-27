#!/usr/bin/env bash
# Runs every correctness test, each against 4 simulated CPU devices.
set -euo pipefail
cd "$(dirname "$0")"

export XLA_FLAGS="--xla_force_host_platform_device_count=4"

python3 -m tests.test_model
echo
python3 -m tests.test_data_parallel
echo
python3 -m tests.test_tensor_parallel
echo
python3 -m tests.test_pipeline_parallel
echo
python3 -m tests.test_ring_allreduce
echo
echo "ALL TESTS PASSED"
