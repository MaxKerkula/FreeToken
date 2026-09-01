"""Single-stream (bs=1) decode benchmark for any MoE model on any offload backend.

Measures through the real serving path: for each backend the bench spawns ``ft serve``,
sends a warmed chat request over /v1/chat/completions with ``stream=true``, and
timestamps every SSE event as it arrives. Numbers therefore include the scheduler,
detokenizer, and HTTP/SSE hop -- what a client actually sees -- not bare engine forwards.

Method -- at bs=1 the server emits one delta event per decode step, and the final chunk
(``stream_options.include_usage``) reports exact token counts, so

    decode_tok_s = (completion_tokens - 1) / (t_last_event - t_first_event)

which stays correct even when the detokenizer coalesces a few tokens into one event
(multibyte characters): the window is still anchored on the first and last token's
arrival. ``ignore_eos`` keeps the step count at exactly ``D`` regardless of sampling.
TTFT is the measured run's warm first-token latency (template rendering + prefill
included). Engine-internal diagnostics (expert-cache miss rate, hybrid fetch split) are
not exposed over the API and are not reported; VRAM is the server's live /v1/stats figure.

Prompt: an AIME-25 problem sent as a chat message with thinking enabled -- a real
reasoning workload, so expert routing is representative. The server renders the chat
template (including checkpoint-shipped encoders like DSV4's ``encoding_dsv4.py``). The
problems come from the ``math-ai/aime25`` dataset on the Hub, downloaded into the usual
HF cache on first run; ``--aime`` points at a local jsonl instead.

Sampling: the checkpoint's recommended params (``generation_config.json``), falling back
to temperature 1.0 / top_p 0.95 / top_k 64 for fields the checkpoint does not specify --
resolved here and sent explicitly, because the server's own unspecified-field defaults
are greedy and would silently degrade the routing workload for checkpoints without a
full sampling recommendation. The generated text is per-server-process deterministic
(fresh server, fixed request sequence), so one text sha1 per backend is a real
cross-backend check; token ids are not visible over the API, so this is a weaker
invariant than the old in-process id hash. ``--greedy`` sends temperature 0 for the
stricter comparison.

Run (one backend):
    CUDA_VISIBLE_DEVICES=0 PYTHONPATH=python python benchmarks/bench_decode_moe.py \
        --model /path/to/model

Exact Windows Qwen3.8 research run:
    python benchmarks/bench_decode_moe.py \
        --model C:\\Models\\Qwen3.8-Flash-Next-NVFP4-FTW \
        --backend offload --nvfp4-backend triton --ple-backend mmap \
        --expert-load serial --cache 2048 --max-seq-len 131072 \
        --kv-reserve-tokens 131072 --decode 256 --greedy \
        --temp-dir .bench-temp --json qwen38-128k.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

FALLBACK_SAMPLING = {"temperature": 1.0, "top_p": 0.95, "top_k": 64}
AIME_REPO = "math-ai/aime25"
AIME_FILE = "test.jsonl"
BOXED_INSTRUCTION = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True, help="checkpoint dir (or .ftw)")
    parser.add_argument(
        "--backend",
        default="offload",
        help="comma list of offload|cpu|hybrid; one server per backend",
    )
    parser.add_argument(
        "--aime",
        default=os.environ.get("FREETOKEN_AIME25_JSONL"),
        help=f"local jsonl instead of downloading {AIME_REPO}; default $FREETOKEN_AIME25_JSONL",
    )
    parser.add_argument("--problem", type=int, default=0, help="0-based AIME problem index")
    parser.add_argument("--decode", type=int, default=256, help="decode tokens to measure")
    parser.add_argument(
        "--cache",
        type=int,
        default=0,
        help="GPU expert cache slots; 0 = auto-size from free VRAM",
    )
    parser.add_argument(
        "--cache-rate", type=float, default=None, help="cache slots as a fraction of L*E"
    )
    parser.add_argument(
        "--hybrid-fetch",
        type=int,
        default=-1,
        help="hybrid: maximum PCIe fetches per layer; -1 = auto",
    )
    parser.add_argument("--mem-ratio", type=float, default=0.9, help="target VRAM utilization")
    parser.add_argument("--gpu", default=None, help="GPU UUID or nvidia-smi index")
    parser.add_argument("--no-graph", action="store_true", help="use eager decode")
    parser.add_argument(
        "--greedy",
        action="store_true",
        help="force temperature 0 so output is comparable",
    )
    parser.add_argument(
        "--server-timeout",
        type=float,
        default=1800,
        help="seconds to wait for the spawned server",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=0,
        help="physical sequence capacity; 0 keeps the historical 8192+decode value",
    )
    parser.add_argument(
        "--kv-reserve-tokens",
        type=int,
        default=0,
        help="KV tokens reserved before automatic expert-cache sizing; 0 omits the flag",
    )
    parser.add_argument(
        "--ple-backend",
        choices=("pinned", "mmap"),
        default="pinned",
        help="Qwen3.8 PLE table storage",
    )
    parser.add_argument(
        "--expert-load",
        choices=("auto", "serial", "parallel"),
        default="auto",
        help="host expert-bank load strategy",
    )
    parser.add_argument("--attention-backend", default="auto")
    parser.add_argument(
        "--nvfp4-backend",
        choices=("auto", "triton", "marlin", "flashinfer"),
        default="auto",
    )
    parser.add_argument(
        "--temp-dir",
        default=os.environ.get("FREETOKEN_BENCH_TEMP"),
        help="writable directory for logs and child-process temporary files",
    )
    parser.add_argument("--json", dest="json_out", default=None, help="append result rows here")
    return parser.parse_args(argv)


def load_problem(path: str | None, index: int) -> tuple[str, str]:
    """Load one AIME-25 problem and its expected answer."""
    if not path:
        from huggingface_hub import hf_hub_download

        try:
            path = hf_hub_download(AIME_REPO, AIME_FILE, repo_type="dataset")
        except Exception as error:
            sys.exit(
                f"could not fetch {AIME_REPO}/{AIME_FILE} ({error}); "
                "pass --aime <local jsonl>"
            )
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    if not 0 <= index < len(rows):
        sys.exit(f"--problem {index} out of range ({len(rows)} problems available)")
    row = rows[index]
    text = row.get("problem") or row["prompt"]
    if "boxed" not in text:
        text = f"{text}\n{BOXED_INSTRUCTION}"
    return text, str(row.get("answer", ""))


def resolve_sampling(model_path: str, greedy: bool) -> tuple[dict, str]:
    """Return explicit sampling parameters and their source."""
    if greedy:
        return {"temperature": 0.0, "top_p": 1.0, "top_k": -1}, "greedy (--greedy)"
    recommended: dict = {}
    config_path = Path(model_path) / "generation_config.json"
    if config_path.is_file():
        raw = json.loads(config_path.read_text())
        recommended = {key: raw[key] for key in FALLBACK_SAMPLING if raw.get(key) is not None}
        if raw.get("do_sample") is False or recommended.get("temperature") == 0.0:
            return {"temperature": 0.0, "top_p": 1.0, "top_k": -1}, "checkpoint (greedy)"
    params = {**FALLBACK_SAMPLING, **recommended}
    if params["top_k"] == 0:
        params["top_k"] = -1
    selected = sorted(recommended)
    source = f"checkpoint{selected} + fallback" if selected else "fallback (no generation_config)"
    return params, source


def get_json(url: str, timeout: float = 10) -> dict:
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return json.load(response)


def free_port() -> int:
    with socket.socket() as server_socket:
        server_socket.bind(("127.0.0.1", 0))
        return server_socket.getsockname()[1]


def resolved_max_seq_len(args: argparse.Namespace) -> int:
    if args.max_seq_len > 0:
        return args.max_seq_len
    return 8192 + args.decode


def resolve_temp_dir(raw_path: str | None) -> Path | None:
    if raw_path is None:
        return None
    temp_dir = Path(raw_path).resolve()
    temp_dir.mkdir(parents=True, exist_ok=True)
    return temp_dir


def server_environment(temp_dir: Path | None) -> dict[str, str]:
    environment = os.environ.copy()
    source_python = Path(__file__).resolve().parents[1] / "python"
    current_python_path = environment.get("PYTHONPATH")
    python_paths = [str(source_python)]
    if current_python_path:
        python_paths.append(current_python_path)
    environment["PYTHONPATH"] = os.pathsep.join(python_paths)
    environment["PYTHONUNBUFFERED"] = "1"
    if temp_dir is not None:
        environment["TEMP"] = str(temp_dir)
        environment["TMP"] = str(temp_dir)
    return environment


def serve_cmd(args: argparse.Namespace, backend: str, port: int) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "freetoken.cli",
        "serve",
        "--model",
        args.model,
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--moe-backend",
        backend,
        "--max-running-requests",
        "1",
        "--max-seq-len-override",
        str(resolved_max_seq_len(args)),
        "--memory-ratio",
        str(args.mem_ratio),
        "--cuda-graph-max-bs",
        "0" if args.no_graph else "1",
        "--moe-hybrid-max-fetch",
        str(args.hybrid_fetch),
        "--attention-backend",
        args.attention_backend,
        "--nvfp4-backend",
        args.nvfp4_backend,
        "--expert-load",
        args.expert_load,
        "--ple-backend",
        args.ple_backend,
    ]
    if args.kv_reserve_tokens > 0:
        command += ["--kv-reserve-tokens", str(args.kv_reserve_tokens)]
    if args.gpu:
        command += ["--gpu", args.gpu]
    if args.cache > 0:
        command += ["--moe-cache-size", str(args.cache)]
    elif args.cache_rate is not None:
        command += ["--moe-cache-rate", str(args.cache_rate)]
    else:
        command.append("--moe-cache-auto")
    return command


def die_with_log(message: str, log_path: str) -> None:
    tail = "".join(
        Path(log_path).read_text(errors="replace").splitlines(keepends=True)[-30:]
    )
    sys.exit(f"[bench] {message}\n[bench] server log tail ({log_path}):\n{tail}")


def wait_ready(origin: str, process: subprocess.Popen, log_path: str, timeout: float) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            die_with_log(
                f"server exited with code {process.returncode} during startup", log_path
            )
        try:
            health = get_json(f"{origin}/health", timeout=5)
        except (OSError, ValueError):
            time.sleep(1.0)
            continue
        if health.get("status") == "error":
            die_with_log(f"server reported startup error: {health}", log_path)
        if health.get("maintenance") == "serving":
            return
        time.sleep(1.0)
    die_with_log(f"server not ready after {timeout:.0f}s", log_path)


def pump_output(source, log_file) -> None:
    """Mirror server bytes to the terminal and the persistent log."""
    for chunk in iter(lambda: source.read1(65536), b""):
        log_file.write(chunk)
        log_file.flush()
        sys.stdout.buffer.write(chunk)
        sys.stdout.flush()


def start_server(command: list[str], environment: dict[str, str]) -> subprocess.Popen:
    common = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "env": environment,
    }
    if os.name == "nt":
        return subprocess.Popen(
            command,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            **common,
        )
    return subprocess.Popen(command, start_new_session=True, **common)


def wait_for_exit(process: subprocess.Popen, timeout: float) -> bool:
    try:
        process.wait(timeout=timeout)
        return True
    except subprocess.TimeoutExpired:
        return False


def stop_windows_process_tree(process: subprocess.Popen) -> None:
    try:
        process.send_signal(signal.CTRL_BREAK_EVENT)
    except (OSError, ValueError):
        pass
    if wait_for_exit(process, 20):
        return
    subprocess.run(
        ["taskkill", "/PID", str(process.pid), "/T", "/F"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    wait_for_exit(process, 30)


def stop_posix_process_group(process: subprocess.Popen) -> None:
    for process_signal, wait_seconds in ((signal.SIGTERM, 90), (signal.SIGKILL, 30)):
        try:
            os.killpg(process.pid, process_signal)
        except ProcessLookupError:
            return
        if wait_for_exit(process, wait_seconds):
            return


def stop_server(process: subprocess.Popen) -> None:
    """Stop only the process tree that this benchmark started."""
    if process.poll() is None:
        if os.name == "nt":
            stop_windows_process_tree(process)
        else:
            stop_posix_process_group(process)
    if process.poll() is None:
        process.kill()
        wait_for_exit(process, 10)
    time.sleep(3)


def stream_generate(
    origin: str,
    model_id: str,
    problem: str,
    sampling: dict,
    args: argparse.Namespace,
) -> dict:
    """Run one streamed completion and return arrival times, text, and usage."""
    body = {
        "model": model_id,
        "messages": [{"role": "user", "content": problem}],
        "max_tokens": args.decode,
        "ignore_eos": True,
        "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": True},
        **sampling,
    }
    request = urllib.request.Request(
        f"{origin}/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    stamps: list[float] = []
    pieces: list[str] = []
    usage: dict | None = None
    start_time = time.perf_counter()
    try:
        response = urllib.request.urlopen(request, timeout=1800)
    except urllib.error.HTTPError as error:
        sys.exit(f"[bench] request failed: HTTP {error.code}: {error.read()[:500]!r}")
    with response:
        for raw in response:
            line = raw.strip()
            if not line or not line.startswith(b"data:"):
                continue
            payload = line[len(b"data:") :].strip()
            if payload == b"[DONE]":
                break
            now = time.perf_counter()
            chunk = json.loads(payload)
            if chunk.get("usage"):
                usage = chunk["usage"]
            for choice in chunk.get("choices", []):
                delta = choice.get("delta") or {}
                text = delta.get("reasoning_content") or delta.get("content")
                if text:
                    stamps.append(now)
                    pieces.append(text)
    if usage is None:
        sys.exit("[bench] stream ended without a usage chunk")
    return {"t0": start_time, "stamps": stamps, "text": "".join(pieces), "usage": usage}


def run_one(args: argparse.Namespace, backend: str) -> dict:
    problem, answer = load_problem(args.aime, args.problem)
    sampling, sampling_source = resolve_sampling(args.model, args.greedy)
    port = free_port()
    origin = f"http://127.0.0.1:{port}"
    temp_dir = resolve_temp_dir(args.temp_dir)
    fd, log_path = tempfile.mkstemp(
        prefix=f"bench-serve-{backend}-",
        suffix=".log",
        dir=str(temp_dir) if temp_dir is not None else None,
    )
    command = serve_cmd(args, backend, port)

    print(
        f"[bench] model={args.model}\n"
        f"[bench] backend={backend} cache={args.cache or args.cache_rate or 'auto'} "
        f"max_seq_len={resolved_max_seq_len(args)} kv_reserve={args.kv_reserve_tokens or 'default'}\n"
        f"[bench] ple={args.ple_backend} expert_load={args.expert_load} "
        f"attention={args.attention_backend} nvfp4={args.nvfp4_backend}\n"
        f"[bench] mem_ratio={args.mem_ratio} decode={args.decode} graph={not args.no_graph}\n"
        f"[bench] sampling={sampling} <- {sampling_source}\n"
        f"[bench] server log: {log_path}",
        flush=True,
    )

    with os.fdopen(fd, "wb") as log_file:
        process = start_server(command, server_environment(temp_dir))
        if process.stdout is None:
            raise RuntimeError("server stdout pipe was not created")
        pump = threading.Thread(
            target=pump_output, args=(process.stdout, log_file), daemon=True
        )
        pump.start()
        try:
            wait_ready(origin, process, log_path, args.server_timeout)
            model_id = get_json(f"{origin}/v1/models")["data"][0]["id"]
            print(f"[bench] model_id={model_id}", flush=True)
            print(f"[bench] AIME25 #{args.problem} (answer {answer})", flush=True)
            stream_generate(origin, model_id, problem, sampling, args)
            result = stream_generate(origin, model_id, problem, sampling, args)
            stats = get_json(f"{origin}/v1/stats")
        finally:
            stop_server(process)
            pump.join(timeout=10)

    stamps = result["stamps"]
    usage = result["usage"]
    if len(stamps) < 2:
        sys.exit(f"[bench] need at least 2 token events, got {len(stamps)}")
    completion = usage["completion_tokens"]
    if completion != args.decode:
        print(
            f"[bench] WARNING: completion_tokens={completion} != --decode {args.decode}",
            flush=True,
        )
    steps = completion - 1
    decode_time = stamps[-1] - stamps[0]
    gaps = sorted((end - start) * 1e3 for start, end in zip(stamps, stamps[1:]))
    row = {
        "model": args.model,
        "backend": backend,
        "problem": args.problem,
        "prompt_tokens": usage["prompt_tokens"],
        "decode_steps": steps,
        "decode_tok_s": steps / decode_time if decode_time > 0 else 0.0,
        "ms_per_token": decode_time / steps * 1e3 if steps > 0 else 0.0,
        "event_ms_p50": gaps[len(gaps) // 2],
        "event_ms_p99": gaps[min(len(gaps) - 1, int(len(gaps) * 0.99))],
        "ttft_ms": (stamps[0] - result["t0"]) * 1e3,
        "events": len(stamps),
        "completion_tokens": completion,
        "vram_gib": stats.get("vram_bytes", 0) / 2**30,
        "sampling": sampling,
        "output_sha1": hashlib.sha1(result["text"].encode()).hexdigest()[:12],
        "server_log": log_path,
        "max_seq_len": resolved_max_seq_len(args),
        "kv_reserve_tokens": args.kv_reserve_tokens,
        "moe_cache_size": args.cache,
        "ple_backend": args.ple_backend,
        "expert_load": args.expert_load,
        "attention_backend": args.attention_backend,
        "nvfp4_backend": args.nvfp4_backend,
    }

    print(f"\n==== decode bs=1 [{backend}] via /v1/chat/completions ====", flush=True)
    print(f"  decode throughput : {row['decode_tok_s']:8.2f} tok/s  ({row['ms_per_token']:.3f} ms/token)")
    print(f"  TTFT (warm)       : {row['ttft_ms']:8.1f} ms  (prompt {row['prompt_tokens']} tok)")
    print(
        f"  decode measured   : {steps} steps in {decode_time:.3f} s  "
        f"(event p50 {row['event_ms_p50']:.3f} / p99 {row['event_ms_p99']:.3f} ms, "
        f"{len(stamps)} events)"
    )
    print(f"  vram (server)     : {row['vram_gib']:8.2f} GiB")
    sha_note = "greedy" if args.greedy else "sampled, per-server deterministic"
    print(f"  output sha1       : {row['output_sha1']}  ({sha_note})")
    print(f"  output sample     : {result['text'][:240]!r}")
    return row


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    backends = [backend.strip() for backend in args.backend.split(",") if backend.strip()]
    unsupported = [
        backend for backend in backends if backend not in ("offload", "cpu", "hybrid")
    ]
    if unsupported:
        sys.exit(f"unsupported backend(s): {unsupported}")

    failed: list[str] = []
    for backend in backends:
        try:
            row = run_one(args, backend)
        except (SystemExit, Exception) as error:
            if len(backends) == 1:
                raise
            print(f"\n[bench] backend {backend} failed: {error!r}", flush=True)
            failed.append(backend)
            continue
        if args.json_out:
            with open(args.json_out, "a", encoding="utf-8") as output_file:
                output_file.write(json.dumps(row) + "\n")
    if failed:
        print(f"\n[bench] backends that failed: {failed}", flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
