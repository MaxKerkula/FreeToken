from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from types import ModuleType


def load_benchmark_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "benchmarks" / "bench_decode_moe.py"
    spec = importlib.util.spec_from_file_location("bench_decode_moe_test_module", script_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def command_value(command: list[str], option: str) -> str:
    return command[command.index(option) + 1]


def test_exact_qwen38_geometry_reaches_server_command(tmp_path: Path) -> None:
    benchmark = load_benchmark_module()
    args = benchmark.parse_args(
        [
            "--model",
            "C:/Models/Qwen3.8-Flash-Next-NVFP4-FTW",
            "--backend",
            "offload",
            "--nvfp4-backend",
            "triton",
            "--ple-backend",
            "mmap",
            "--expert-load",
            "serial",
            "--attention-backend",
            "qsa_sparse",
            "--cache",
            "2048",
            "--max-seq-len",
            "131072",
            "--kv-reserve-tokens",
            "131072",
            "--decode",
            "256",
            "--temp-dir",
            str(tmp_path),
        ]
    )

    command = benchmark.serve_cmd(args, "offload", 8123)

    assert command_value(command, "--max-seq-len-override") == "131072"
    assert command_value(command, "--kv-reserve-tokens") == "131072"
    assert command_value(command, "--moe-cache-size") == "2048"
    assert command_value(command, "--ple-backend") == "mmap"
    assert command_value(command, "--expert-load") == "serial"
    assert command_value(command, "--attention-backend") == "qsa_sparse"
    assert command_value(command, "--nvfp4-backend") == "triton"
    assert "--moe-cache-auto" not in command


def test_default_geometry_and_auto_cache_are_preserved() -> None:
    benchmark = load_benchmark_module()
    args = benchmark.parse_args(["--model", "model", "--decode", "256"])

    command = benchmark.serve_cmd(args, "offload", 8123)

    assert benchmark.resolved_max_seq_len(args) == 8448
    assert command_value(command, "--max-seq-len-override") == "8448"
    assert "--kv-reserve-tokens" not in command
    assert "--moe-cache-auto" in command


def test_server_environment_pins_source_and_temp_paths(tmp_path: Path) -> None:
    benchmark = load_benchmark_module()
    environment = benchmark.server_environment(tmp_path)
    expected_source = str(Path(benchmark.__file__).resolve().parents[1] / "python")

    assert environment["PYTHONPATH"].split(os.pathsep)[0] == expected_source
    assert environment["PYTHONUNBUFFERED"] == "1"
    assert environment["TEMP"] == str(tmp_path)
    assert environment["TMP"] == str(tmp_path)


def test_resolve_temp_dir_creates_requested_directory(tmp_path: Path) -> None:
    benchmark = load_benchmark_module()
    requested = tmp_path / "nested" / "bench"

    resolved = benchmark.resolve_temp_dir(str(requested))

    assert resolved == requested.resolve()
    assert requested.is_dir()


def test_start_server_captures_output() -> None:
    benchmark = load_benchmark_module()
    command = [sys.executable, "-c", "print('BENCH_CHILD_OK')"]

    process = benchmark.start_server(command, os.environ.copy())
    output, _ = process.communicate(timeout=10)

    assert process.returncode == 0
    assert output.strip() == b"BENCH_CHILD_OK"
