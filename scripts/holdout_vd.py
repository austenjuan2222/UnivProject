"""完全沒參加訓練的 VD 能不能直接用全域模型：32 台分 8 折，每折拿掉 4 台（partition % 8 == 折號）只用 28 台訓練，
再把全域模型套到被拿掉的 4 台的測試時段。每台 VD 都會被拿掉一次，和「有參加訓練」（主表 hmac 組，同 seed）配對比較。

用法：
  python scripts/holdout_vd.py make <原資料夾>                       # 建 8 個 28 台的資料夾
  powershell -File scripts/run_holdout_vd.ps1                          # 訓練
  python scripts/holdout_vd.py eval <原資料夾> <holdout 結果> <hmac 結果 ...>
被拿掉的 VD 用自己前 80% 的資料算標準化參數（新裝的 VD 要先累積一段歷史才能用，這裡等於有約 22 天）。
"""
import csv
import json
import shutil
import sys
from pathlib import Path

FOLDS = 8


def held_out(fold: int, n: int = 32) -> list[int]:
    return [p for p in range(n) if p % FOLDS == fold]


def make(src: Path) -> None:
    man = json.loads((src / "manifest.json").read_text(encoding="utf-8"))
    for f in range(FOLDS):
        out = src.parent / f"{src.name}_hold{FOLDS}_f{f}"
        out.mkdir(exist_ok=True)
        keep = [c for c in man["clients"] if c["partition"] not in held_out(f, len(man["clients"]))]
        for i, c in enumerate(keep):
            shutil.copy2(src / c["file"], out / c["file"])
        clients = [{**c, "partition": i, "orig_partition": c["partition"]} for i, c in enumerate(keep)]
        (out / "manifest.json").write_text(json.dumps({**man, "clients": clients}, ensure_ascii=False, indent=2),
                                           encoding="utf-8")
        print(out, len(clients))


def _runs(root: Path) -> dict[int, Path]:
    """seed → 執行資料夾（含 final_model.pt、clients.csv、config.json）。"""
    return {int(d.name.rsplit("seed", 1)[1]): d for d in root.rglob("dp_sigma1.1_seed*") if (d / "config.json").exists()}


def _in_training(roots: list[Path]) -> dict[tuple[int, int], float]:
    """(seed, partition) → 有參加訓練時最後一輪的全域模型 MAE。"""
    out = {}
    for root in roots:
        for seed, d in _runs(root).items():
            rows = list(csv.DictReader((d / "clients.csv").open(encoding="utf-8")))
            last = max(int(r["round"]) for r in rows)
            for r in rows:
                if int(r["round"]) == last and r["mae_kmh"]:
                    out[(seed, int(r["partition"]))] = float(r["mae_kmh"])
    return out


def evaluate_holdout(src: Path, hold_root: Path) -> list[dict]:
    import torch
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from data import load_client_arrays, n_calendar
    from task import TrafficModel, evaluate
    n = len(json.loads((src / "manifest.json").read_text(encoding="utf-8"))["clients"])
    rows = []
    for f in range(FOLDS):
        for seed, d in _runs(hold_root / f"f{f}").items():
            c = json.loads((d / "config.json").read_text(encoding="utf-8"))["run_config"]
            model = TrafficModel(c["lookback"], c["hidden-size"], c["residual"], n_calendar(c["calendar"]) + int(c["lag-day"]))
            model.load_state_dict(torch.load(d / "final_model.pt", map_location="cpu", weights_only=True), strict=True)
            for p in held_out(f, n):
                a = load_client_arrays(str(src), p, n, c["lookback"], c["train-frac"], c["horizon"], c["calendar"], c["lag-day"])
                with torch.no_grad():
                    r = evaluate(model, a["X_test"], a["y_test"], a["mean"], a["std"])
                rows.append({"fold": f, "seed": seed, "partition": p, "vdid": a["vdid"],
                             "held_out_mae_kmh": r["mae_kmh"], "naive_mae_kmh": r["naive_mae_kmh"]})
    return rows


def main(argv: list[str]) -> None:
    if argv[0] == "make":
        return make(Path(argv[1]))
    from seed_stats import paired_summary
    src, hold_root, base_roots = Path(argv[1]), Path(argv[2]), [Path(a) for a in argv[3:]]
    rows, base = evaluate_holdout(src, hold_root), _in_training(base_roots)
    for r in rows:
        r["in_training_mae_kmh"] = base.get((r["seed"], r["partition"]), float("nan"))
    out = hold_root / "holdout_vd.csv"
    with out.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    rows = [r for r in rows if r["in_training_mae_kmh"] == r["in_training_mae_kmh"]]
    seeds = sorted({r["seed"] for r in rows})
    mean = lambda xs: sum(xs) / len(xs)  # noqa: E731
    by_seed = [(s, mean([r["in_training_mae_kmh"] for r in rows if r["seed"] == s]),
                mean([r["held_out_mae_kmh"] for r in rows if r["seed"] == s])) for s in seeds]
    vds = sorted({r["partition"] for r in rows})
    by_vd = [(p, mean([r["in_training_mae_kmh"] for r in rows if r["partition"] == p]),
              mean([r["held_out_mae_kmh"] for r in rows if r["partition"] == p])) for p in vds]
    naive = mean([r["naive_mae_kmh"] for r in rows])
    print(f"{len(vds)} 台 VD × {len(seeds)} seed；有參加訓練 {mean([b for _, b, _ in by_seed]):.3f}、"
          f"沒參加 {mean([h for _, _, h in by_seed]):.3f}、naive {naive:.3f}（{out}）")
    for name, pairs in (("以 seed 為單位（各 seed 先平均 32 台）", by_seed), ("以 VD 為單位（各台先平均 seed）", by_vd)):
        s = paired_summary(pairs)
        print(f"{name}：沒參加 − 有參加 = {s['mean_diff']:+.4f} [{s['ci_low']:+.4f}, {s['ci_high']:+.4f}]，p = {s['p_t']:.3g}，n = {len(pairs)}")
    worse = sum(h > b for _, b, h in by_vd)
    print(f"沒參加比有參加差的 VD：{worse}／{len(by_vd)}；沒參加仍贏 naive 的 VD："
          f"{sum(mean([r['held_out_mae_kmh'] for r in rows if r['partition'] == p]) < mean([r['naive_mae_kmh'] for r in rows if r['partition'] == p]) for p in vds)}／{len(vds)}")


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))
    main(sys.argv[1:])
