import argparse
import subprocess
import shlex
import time
import sys
import os
import signal

def test_batch(batch, cfg_path, timeout=45, dtype="bfloat16", use_amp=True):
    cmd = (
        f'uv run python -m examples.video_jepa.main '
        f'--fname {shlex.quote(cfg_path)} '
        f'data.batch_size={batch} '
        f'training.use_amp={str(use_amp).lower()} '
        f'training.dtype={dtype} '
        f'optim.epochs=1'
    )
    print(f"Testing batch={batch} -> {cmd}")
    p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, preexec_fn=os.setsid)
    try:
        _, err = p.communicate(timeout=timeout)
        ret = p.returncode
        stderr = err.decode(errors="ignore").lower()
    except subprocess.TimeoutExpired:
        # Assume allocation succeeded (process alive); kill it and treat as success
        os.killpg(os.getpgid(p.pid), signal.SIGTERM)
        try:
            _, err = p.communicate(timeout=5)
        except Exception:
            pass
        print(f"Timeout ({timeout}s) reached — treating as SUCCESS for batch {batch}")
        return True, "timeout_assumed_success"
    # If process exited with OOM or CUDA OOM in stderr -> fail
    if "out of memory" in stderr or "cuda out of memory" in stderr or "oom" in stderr:
        return False, stderr
    if ret != 0:
        # non-zero exit but not OOM -> treat as failure but show stderr
        return False, stderr
    # exit code 0 -> success
    return True, stderr

def binary_search(cfg_path, lo=1, hi=256, dtype="bfloat16", use_amp=True):
    best = lo
    while lo <= hi:
        mid = (lo + hi) // 2
        ok, msg = test_batch(mid, cfg_path, dtype=dtype, use_amp=use_amp)
        if ok:
            best = mid
            lo = mid + 1
        else:
            print(f"Batch {mid} failed: {msg.splitlines()[-1] if msg else msg}")
            hi = mid - 1
    return best

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", default="examples/video_jepa/cfgs/autoregressive.yaml")
    ap.add_argument("--min", type=int, default=1)
    ap.add_argument("--max", type=int, default=256)
    ap.add_argument("--dtype", choices=["float32","float16","bfloat16"], default="bfloat16")
    ap.add_argument("--no-amp", action="store_true", help="Disable AMP")
    ap.add_argument("--timeout", type=int, default=45)
    args = ap.parse_args()

    print("Probe config:", args)
    best = binary_search(args.cfg, lo=args.min, hi=args.max, dtype=args.dtype, use_amp=not args.no_amp)
    print(f"Max batch size found (approx): {best}")
    sys.exit(0)