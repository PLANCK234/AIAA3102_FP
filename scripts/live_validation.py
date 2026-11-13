"""Explicit, billable live development validation; never run by pytest.

Supply SUPERAGENT_API_KEY via environment, or --key-stdin for a non-echoing
automation pipe. The key is propagated only in process memory.
"""
import argparse
import os
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--key-stdin", action="store_true")
    parser.add_argument("--loop-only", action="store_true", help="Use only after both single-call probes")
    parser.add_argument("--loop-model", default="Qwen3.8-27B")
    parser.add_argument("--max-llm-calls", type=int, default=4)
    parser.add_argument("--max-experiments", type=int, default=3)
    args = parser.parse_args()
    if args.key_stdin:
        print("READY_FOR_KEY", flush=True)
        if sys.stdin.isatty():
            import getpass
            os.environ["SUPERAGENT_API_KEY"] = getpass.getpass("API key: ").strip()
        else:
            os.environ["SUPERAGENT_API_KEY"] = sys.stdin.readline().strip()
    if not os.environ.get("SUPERAGENT_API_KEY"):
        print("SKIP: SUPERAGENT_API_KEY is not configured")
        return 2
    os.environ.setdefault("SUPERAGENT_BASE_URL", "https://aigc-api.hkust-gz.edu.cn/api/v2")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    root = Path(__file__).resolve().parent.parent
    if not args.loop_only:
        for model in ("Qwen3.8-27B", "DeepSeek-V4.1-Flash"):
            print(f"Starting one-call probe: {model}", flush=True)
            subprocess.run([sys.executable, "main.py", "llm-test", "--model", model], cwd=root, check=True)
    print(f"Starting limited {args.loop_model} autonomous development run", flush=True)
    os.environ["SUPERAGENT_MODEL"] = args.loop_model
    subprocess.run([sys.executable, "main.py", "run", "--task", "configs/coupon_development.yaml",
                    "--planner", "llm-dev", "--max-llm-calls", str(args.max_llm_calls), "--max-experiments", str(args.max_experiments),
                    "--llm-max-output-tokens", "2048"], cwd=root, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
