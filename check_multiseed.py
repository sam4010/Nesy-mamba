"""
Check status and fetch results from multi-seed Kaggle runs.

Usage:
    python check_multiseed.py              # check status of all 12 runs
    python check_multiseed.py --fetch      # also download outputs
    python check_multiseed.py --summarize  # print mean ± std table
"""

import requests, sys, json, os
from base64 import b64encode

ACCOUNTS = {
    "sanskrutib01": {
        "headers": {
            "Authorization": "Bearer KGAT_0647377ef4036100da56ddfa2c1f97b3",
            "Content-Type": "application/json",
        },
        "username": "sanskrutib01",
    },
    "samarthbhalera0234": {
        "headers": {
            "Authorization": "Bearer KGAT_a8bf8b602bbfa6a6dd8cb8e19da46c8f",
            "Content-Type": "application/json",
        },
        "username": "samarthbhalera0234",
    },
    "sahildamke07": {
        "headers": {
            "Authorization": "Bearer KGAT_ee7571bcabb39a9f28804af56e5b0571",
            "Content-Type": "application/json",
        },
        "username": "sahildamke07",
    },
    "manojkalasgonda7": {
        "headers": {
            "Authorization": "Basic " + b64encode(b"manojkalasgonda7:872b3b4cc86e23c3e4f40175a5e68e76").decode(),
            "Content-Type": "application/json",
        },
        "username": "manojkalasgonda7",
    },
    "avdhootpimparkar010": {
        "headers": {
            "Authorization": "Basic " + b64encode(b"avdhootpimparkar010:83ef77feafe8ce92605d30bea5880572").decode(),
            "Content-Type": "application/json",
        },
        "username": "avdhootpimparkar010",
    },
}

CONFIGS = ["base", "v14_nosup", "lam03", "v14_type"]
SEEDS = [42, 123, 456]

# Same split as push_multiseed.py
BATCHES = {
    "sanskrutib01":       [("base", 42), ("base", 123)],
    "samarthbhalera0234": [("lam03", 42)],
    "sahildamke07":       [("v14_nosup", 42), ("v14_nosup", 123)],
    "manojkalasgonda7":   [("v14_type", 42), ("v14_type", 123)],
    "avdhootpimparkar010": [
        ("base", 456), ("v14_nosup", 456),
        ("lam03", 123), ("lam03", 456),
        ("v14_type", 456),
    ],
}


def get_slug(config, seed):
    return f"nesy-mamba-v14d-{config}-s{seed}-gpu".lower().replace("_", "-")


def check_status(acct_name, config, seed):
    """Check kernel status."""
    acct = ACCOUNTS[acct_name]
    slug = get_slug(config, seed)
    url = f"https://www.kaggle.com/api/v1/kernels/status"
    params = {"userName": acct["username"], "kernelSlug": slug}

    r = requests.get(url, headers=acct["headers"], params=params)
    if r.status_code == 200:
        return r.json().get("status", "unknown")
    return f"error-{r.status_code}"


def fetch_output(acct_name, config, seed, out_dir="multiseed_results"):
    """Fetch kernel output log."""
    acct = ACCOUNTS[acct_name]
    slug = get_slug(config, seed)
    url = f"https://www.kaggle.com/api/v1/kernels/output"
    params = {"userName": acct["username"], "kernelSlug": slug}

    r = requests.get(url, headers=acct["headers"], params=params)
    if r.status_code != 200:
        return None

    os.makedirs(out_dir, exist_ok=True)
    data = r.json()
    raw_log = data.get("log", "")

    # Kaggle logs are JSON arrays of {"stream_name","time","data"} entries
    try:
        entries = json.loads(raw_log)
        log = "".join(e.get("data", "") for e in entries)
    except (json.JSONDecodeError, TypeError):
        log = raw_log  # fallback: treat as plain text

    out_path = os.path.join(out_dir, f"{config}_seed{seed}.log")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(log)
    return log


def parse_results_from_log(log_text):
    """Extract key metrics from kernel output log."""
    result = {}
    for line in log_text.split("\n"):
        line = line.strip()
        if "Val Accuracy:" in line:
            try:
                result["val_acc"] = float(line.split(":")[-1].strip())
            except:
                pass
        if "Best Epoch:" in line:
            try:
                result["best_epoch"] = int(line.split(":")[-1].strip())
            except:
                pass
        # Parse per-depth lines like "  Depth-0: 0.9335"
        if "Depth-" in line and ":" in line:
            try:
                parts = line.split(":")
                depth_part = parts[0].strip()
                depth = depth_part.split("Depth-")[-1].strip()
                val = float(parts[1].strip())
                result[f"acc_d{depth}"] = val
            except:
                pass
        # Parse "Best: 0.40 -> 0.8066" threshold sweep
        if line.startswith("Best:") and "->" in line:
            try:
                result["best_thresh_acc"] = float(line.split("->")[-1].strip())
            except:
                pass
    return result


def summarize(all_results):
    """Compute mean ± std across seeds."""
    import numpy as np

    print(f"\n{'='*80}")
    print(f"  MULTI-SEED SUMMARY (mean ± std across {len(SEEDS)} seeds)")
    print(f"{'='*80}")

    metrics = ["val_acc", "acc_d0", "acc_d1", "acc_d2", "acc_d3", "acc_d4", "acc_d5"]
    header = f"{'Config':<12} | {'Seeds':>5} | {'Val Acc':>14} | {'D0':>14} | {'D1':>14} | {'D2':>14} | {'D3':>14}"
    print(header)
    print("-" * len(header))

    for config in CONFIGS:
        seeds_data = [all_results.get((config, s), {}) for s in SEEDS]
        seeds_data = [d for d in seeds_data if "val_acc" in d]
        n = len(seeds_data)

        if n == 0:
            print(f"{config:<12} | {n:>5} | {'N/A':>14}")
            continue

        vals = {}
        for m in metrics:
            v = [d[m] for d in seeds_data if m in d]
            if v:
                vals[m] = (np.mean(v), np.std(v))

        row = f"{config:<12} | {n:>5}"
        for m in ["val_acc", "acc_d0", "acc_d1", "acc_d2", "acc_d3"]:
            if m in vals:
                mean, std = vals[m]
                row += f" | {mean:.4f}±{std:.4f}"
            else:
                row += f" | {'':>14}"
        print(row)

    print()

    # Also format for LaTeX
    print("LaTeX table rows:")
    for config in CONFIGS:
        seeds_data = [all_results.get((config, s), {}) for s in SEEDS]
        seeds_data = [d for d in seeds_data if "val_acc" in d]
        if not seeds_data:
            continue

        acc_vals = [d["val_acc"] for d in seeds_data if "val_acc" in d]
        mean_acc = np.mean(acc_vals)
        std_acc = np.std(acc_vals)
        print(f"  {config} & ${100*mean_acc:.1f} \\pm {100*std_acc:.1f}$")


def main():
    do_fetch = "--fetch" in sys.argv
    do_summarize = "--summarize" in sys.argv

    print("=" * 60)
    print("  Multi-Seed Status Check")
    print("=" * 60)

    all_results = {}
    all_complete = True

    for acct_name, batch_jobs in BATCHES.items():
        print(f"\n{acct_name}:")
        for config, seed in batch_jobs:
            status = check_status(acct_name, config, seed)
            emoji = "✓" if status == "complete" else "…" if status == "running" else "✗"
            print(f"  {emoji} {config:12s} seed={seed}: {status}")

            if status != "complete":
                all_complete = False
                continue

            # Fetch if requested
            if do_fetch or do_summarize:
                log = fetch_output(acct_name, config, seed)
                if log:
                    result = parse_results_from_log(log)
                    all_results[(config, seed)] = result
                    print(f"    → val_acc={result.get('val_acc', '?')}")

    if do_summarize and all_results:
        summarize(all_results)
    elif do_summarize:
        print("\nNo results to summarize yet.")

    if all_complete and not do_fetch:
        print("\nAll runs complete! Use --fetch or --summarize to get results.")


if __name__ == "__main__":
    main()
