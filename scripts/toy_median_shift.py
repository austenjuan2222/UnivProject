"""玩具模型：3 台攻擊者把更新縮小 α 倍時，平均／截尾平均／中位數各被推動多少（README「確認：中位數對縮小的攻擊幾乎不敏感」）。
每個參數：29 份誠實更新 ~ N(0, 1)，3 份攻擊更新 ~ α·(d + 0.3·N(0, 1))；印出 2 萬個參數的平均位移。"""
import numpy as np
from scipy import stats


def shifts(d: float, a: float, n_params: int = 20000, seed: int = 0) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    h = rng.standard_normal((29, n_params))
    x = np.vstack([h, a * (d + 0.3 * rng.standard_normal((3, n_params)))])
    tm = lambda z: stats.trim_mean(z, 0.2, axis=0)  # 和 Flower FedTrimmedAvg 預設 beta=0.2 相同
    return (float(np.mean(x.mean(0) - h.mean(0))), float(np.mean(tm(x) - tm(h))),
            float(np.mean(np.median(x, 0) - np.median(h, 0))))


if __name__ == "__main__":
    for d in (1.0, 2.0):
        for a in (1.0, 0.5, 0.25, 0.125):
            m, t, md = shifts(d, a)
            print(f"d={d} α={a}: 平均 {m:.3f}  截尾平均 {t:.3f}  中位數 {md:.3f}")
    m1, _, md1 = shifts(2.0, 1.0)
    m8, _, md8 = shifts(2.0, 0.125)
    assert m1 / m8 > 7 and md1 / md8 < 2  # 縮小 8 倍：平均的位移跟著小 8 倍，中位數不到 2 倍
