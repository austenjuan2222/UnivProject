"""把 edge_int8.quantize 的結果輸出成一個獨立的 C 檔（只用 int8／int32／int64，沒有浮點）。

vd_predict(x) 吃已經量化成 int8 的輸入（x_int8 = round(標準化車速 / s_x)，可在 VD 上用定點數算），
回傳 int32，乘 s_x / OUT_FRAC 就是標準化尺度的預測（和 predict_int8 同一個數）。
加 -DVD_MAIN 編譯成可執行檔：從 stdin 每行讀 n_in 個整數、印一個整數，測試用。
限制：右移負數在 C 標準裡由編譯器決定，主流編譯器（gcc、clang、armcc）都是算術右移，這裡照這個假設。
"""
from __future__ import annotations

import numpy as np

from edge_int8 import OUT_FRAC


def _arr(ctype: str, name: str, a) -> str:
    vals = ", ".join(str(int(v)) for v in np.asarray(a).ravel())
    return f"static const {ctype} {name}[{np.asarray(a).size}] = {{{vals}}};\n"


def to_c(q: dict, residual: bool = True) -> str:
    Ls = q["layers"]
    dims = [Ls[0]["w"].shape[1]] + [L["w"].shape[0] for L in Ls]
    src = ["#include <stdint.h>\n", f"#define N_IN {dims[0]}\n", f"#define OUT_FRAC {OUT_FRAC}\n",
           f"/* 輸出乘 {q['s_x'] / OUT_FRAC!r} 是標準化尺度的預測 */\n"]
    for i, L in enumerate(Ls):
        src += [_arr("int8_t", f"W{i}", L["w"]), _arr("int32_t", f"B{i}", L["b"]),
                _arr("int32_t", f"M{i}", L["m"]), _arr("uint8_t", f"S{i}", L["s"])]
    src.append("""
static int32_t rq(int32_t acc, int32_t m, uint8_t s) {
    return (int32_t)(((int64_t)acc * m + ((int64_t)1 << (s - 1))) >> s);
}

static void dense_relu(const int8_t *x, int n_in, const int8_t *w, const int32_t *b,
                       const int32_t *m, const uint8_t *s, int n_out, int8_t *y) {
    for (int o = 0; o < n_out; o++) {
        int32_t acc = b[o];
        for (int i = 0; i < n_in; i++) acc += (int32_t)x[i] * w[o * n_in + i];
        if (acc < 0) acc = 0;
        int32_t v = rq(acc, m[o], s[o]);
        y[o] = (int8_t)(v > 127 ? 127 : v);
    }
}
""")
    src.append(f"""
int32_t vd_predict(const int8_t *x) {{
    int8_t h1[{dims[1]}], h2[{dims[2]}];
    dense_relu(x, {dims[0]}, W0, B0, M0, S0, {dims[1]}, h1);
    dense_relu(h1, {dims[1]}, W1, B1, M1, S1, {dims[2]}, h2);
    int32_t acc = B2[0];
    for (int i = 0; i < {dims[2]}; i++) acc += (int32_t)h2[i] * W2[i];
    return rq(acc, M2[0], S2[0]){" + (int32_t)x[N_IN - 1] * OUT_FRAC" if residual else ""};
}}

#ifdef VD_MAIN
#include <stdio.h>
int main(void) {{
    int8_t x[N_IN];
    for (;;) {{
        for (int i = 0; i < N_IN; i++) {{ int v; if (scanf("%d", &v) != 1) return 0; x[i] = (int8_t)v; }}
        printf("%ld\\n", (long)vd_predict(x));
    }}
}}
#endif
""")
    return "".join(src)
