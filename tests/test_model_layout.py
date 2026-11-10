"""VD 收到全域模型先比對名稱、形狀、型別，再做簽章驗證（講義 Edge 資源＋攻擊類型：阻斷服務）。

沒有伺服器私鑰的中間人塞一個超大陣列，VD 原本要先把整個陣列雜湊完才發現簽章不對；
float64 的陣列轉成 float32 後雜湊值一樣，簽章也擋不住型別被換掉。
"""
import unittest
from unittest import mock

import numpy as np
from flwr.app import Array, ArrayRecord, ConfigRecord, Message, RecordDict

import client_app
from guard import AUTH
from task import TrafficModel

CFG = {"lookback": 12, "hidden": 16, "residual": True, "calendar": "", "lag_day": False,
       "sign_models": True, "key_dir": "unused"}


def msg(arrays: dict) -> Message:
    rec = ArrayRecord({k: Array(np.ascontiguousarray(v)) for k, v in arrays.items()})
    return Message(content=RecordDict({"arrays": rec, "config": ConfigRecord({"server-round": 1}), AUTH: ConfigRecord({"sig": "00"})}),
                   dst_node_id=1, message_type="train")


def good() -> dict:
    return {k: v.numpy() for k, v in TrafficModel(12, 16, True, 0).state_dict().items()}


class ModelLayoutTest(unittest.TestCase):
    def check(self, arrays):
        with mock.patch.object(client_app, "verify_model", return_value=True) as v, \
                mock.patch.object(client_app, "load_server_public_key", return_value=None):
            client_app._check_global_signature(CFG, msg(arrays), 1, 7)
        return v

    def test_matching_layout_goes_on_to_signature_check(self):
        self.assertTrue(self.check(good()).called)

    def test_bad_layouts_are_rejected_before_hashing(self):
        extra = {**good(), "junk": np.zeros(1_000_000, np.float32)}
        p = good()
        k = next(iter(p))
        reshaped = {**p, k: np.zeros(p[k].size + 1, np.float32)}
        wide = {**p, k: p[k].astype(np.float64)}
        missing = dict(list(p.items())[1:])
        for arrays in (extra, reshaped, wide, missing):
            with mock.patch.object(client_app, "verify_model", return_value=True) as v, \
                    mock.patch.object(client_app, "load_server_public_key", return_value=None):
                with self.assertRaises(RuntimeError):
                    client_app._check_global_signature(CFG, msg(arrays), 1, 7)
                self.assertFalse(v.called)


if __name__ == "__main__":
    unittest.main()
