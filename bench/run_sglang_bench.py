"""Launch an SGLang server and benchmark it.

Handles server lifecycle: launch, health-check, benchmark, cleanup.
The benchmark client (sglang_eval_client.py) sends requests and logs metrics.

Usage:
    python run_sglang_bench.py --llama                     # SD, Llama 70B
    python run_sglang_bench.py --qwen                      # SD, Qwen 32B
    python run_sglang_bench.py --llama --mode ar            # autoregressive baseline
    python run_sglang_bench.py --llama --wandb --name myrun # log to wandb

Set model paths via env vars (BENCH_LLAMA_70B, etc.) or edit bench_paths.py.
"""
import os
import sys
import time
import signal
import argparse
import subprocess
import requests

sys.path.insert(0, os.path.dirname(__file__))


def get_server_cmd(args):
    # Lazy import so `--help` works without SSD_HF_CACHE.
    from bench_helpers import get_model_paths
    _, target, draft = get_model_paths(args)

    cmd = [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", target,
        "--tp", str(args.tp),
        "--mem-fraction-static", str(args.mem_frac),
        "--max-running-requests", "1",
        "--disable-radix-cache",
        "--log-level", "warning",
        "--port", str(args.port),
    ]

    if args.mode == "sd":
        # Speculative decoding with standalone draft model.
        # Default: k=5 (num_steps=4, num_draft_tokens=5).
        cmd += [
            "--speculative-algorithm", "STANDALONE",
            "--speculative-draft-model-path", draft,
            "--speculative-num-steps", str(args.num_steps),
            "--speculative-eagle-topk", "1",
            "--speculative-num-draft-tokens", str(args.num_draft_tokens),
        ]
    # mode == "ar": no speculative flags, just serve the target model.

    return cmd, target


def wait_for_server(port, timeout=900, interval=5):
    url = f"http://localhost:{port}/health"
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            if requests.get(url, timeout=2).status_code == 200:
                return True
        except requests.ConnectionError:
            pass
        time.sleep(interval)
    return False


def kill_server(proc):
    if proc.poll() is None:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.wait()


def main():
    parser = argparse.ArgumentParser(description="Launch SGLang server and benchmark it")
    parser.add_argument("--llama", action="store_true", default=True)
    parser.add_argument("--qwen", action="store_true")
    parser.add_argument("--gemma", action="store_true")
    parser.add_argument("--vicuna", action="store_true")
    parser.add_argument("--vicuna13b_160m", action="store_true")
    parser.add_argument("--size", type=str, choices=["0.6", "1.7", "4", "8", "14", "32", "1", "3", "70"], default="70")
    parser.add_argument("--draft", type=str, default=None,
                        help="Draft model size or path (bench.py semantics)")
    parser.add_argument("--mode", choices=["ar", "sd"], default="sd",
                        help="ar = autoregressive, sd = speculative decoding (default)")
    parser.add_argument("--tp", type=int, default=4)
    parser.add_argument("--port", type=int, default=40010)
    parser.add_argument("--mem_frac", type=float, default=0.70)
    parser.add_argument("--num_steps", type=int, default=4, help="draft chain depth (k = num_steps + 1)")
    parser.add_argument("--num_draft_tokens", type=int, default=5)
    # Pass-through to eval client
    parser.add_argument("--numseqs", type=int, default=128)
    parser.add_argument("--b", type=int, default=1, help="Batch size for eval client requests")
    parser.add_argument("--output_len", type=int, default=512)
    parser.add_argument("--temp", type=float, default=0.0)
    parser.add_argument("--input_len", type=int, default=128)
    parser.add_argument("--humaneval", action="store_true")
    parser.add_argument("--alpaca", action="store_true")
    parser.add_argument("--c4", action="store_true")
    parser.add_argument("--ultrafeedback", action="store_true")
    parser.add_argument("--aime2025", action="store_true")
    parser.add_argument("--livecodebench", action="store_true")
    parser.add_argument("--codeelo", action="store_true")
    parser.add_argument("--math500", action="store_true")
    parser.add_argument("--govreport", action="store_true")
    parser.add_argument("--random", action="store_true")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--example", action="store_true")
    parser.add_argument("--eagle", action="store_true")
    parser.add_argument("--chat-template", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--group", type=str, default=None)
    parser.add_argument("--name", type=str, default=None)
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    _n_hf_preset = int(bool(args.qwen)) + int(bool(args.gemma)) + int(bool(args.vicuna)) + int(bool(args.vicuna13b_160m))
    if _n_hf_preset > 1:
        parser.error("Use at most one of --qwen --gemma --vicuna --vicuna13b_160m")
    if args.qwen or args.gemma or args.vicuna or args.vicuna13b_160m:
        args.llama = False

    server_cmd, target = get_server_cmd(args)
    print(f"Mode: {args.mode}, Target: {target}")
    print(f"Server cmd: {' '.join(server_cmd)}")

    # Kill stale sglang processes
    subprocess.run(["pkill", "-9", "-f", "sglang.launch_server"],
                   capture_output=True)
    time.sleep(2)

    proc = subprocess.Popen(server_cmd, preexec_fn=os.setsid)
    try:
        print("Waiting for server...")
        if not wait_for_server(args.port):
            print("Server failed to start"); sys.exit(1)
        print("Server ready")

        # Build eval client command
        bench_dir = os.path.dirname(__file__)
        eval_cmd = [
            sys.executable, os.path.join(bench_dir, "sglang_eval_client.py"),
            "--size", str(args.size),
            "--numseqs", str(args.numseqs),
            "--input_len", str(args.input_len),
            "--output_len", str(args.output_len),
            "--temp", str(args.temp),
            "--b", str(args.b),
            "--port", str(args.port),
        ]
        if args.llama:
            eval_cmd.append("--llama")
        elif args.qwen:
            eval_cmd.append("--qwen")
        elif args.gemma:
            eval_cmd.append("--gemma")
        elif args.vicuna:
            eval_cmd.append("--vicuna")
        elif args.vicuna13b_160m:
            eval_cmd.append("--vicuna13b_160m")
        if args.humaneval:
            eval_cmd.append("--humaneval")
        if args.alpaca:
            eval_cmd.append("--alpaca")
        if args.c4:
            eval_cmd.append("--c4")
        if args.ultrafeedback:
            eval_cmd.append("--ultrafeedback")
        if args.aime2025:
            eval_cmd.append("--aime2025")
        if args.livecodebench:
            eval_cmd.append("--livecodebench")
        if args.codeelo:
            eval_cmd.append("--codeelo")
        if args.math500:
            eval_cmd.append("--math500")
        if args.govreport:
            eval_cmd.append("--govreport")
        if args.random:
            eval_cmd.append("--random")
        if args.all:
            eval_cmd.append("--all")
        if args.example:
            eval_cmd.append("--example")
        if args.eagle:
            eval_cmd.append("--eagle")
        if args.chat_template:
            eval_cmd.append("--chat-template")
        if args.verbose:
            eval_cmd.append("--verbose")
        if args.mode == "sd":
            if args.draft is not None:
                eval_cmd += ["--draft", str(args.draft)]
        if args.wandb:
            eval_cmd += ["--wandb"]
            if args.group:
                eval_cmd += ["--group", args.group]
            if args.name:
                eval_cmd += ["--name", args.name]

        print(f"Eval cmd: {' '.join(eval_cmd)}")
        subprocess.run(eval_cmd, check=True, cwd=bench_dir)
    finally:
        kill_server(proc)
        print("Server stopped")


if __name__ == "__main__":
    main()
