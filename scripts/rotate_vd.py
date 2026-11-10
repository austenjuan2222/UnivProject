"""管理者定期換新 VD 金鑰：python scripts/rotate_vd.py <key-dir> <VD 編號>…（HMAC 和 Ed25519 都換，舊的下次啟動伺服器起無效）。
新的 vd-<編號>.key、vd-<編號>.ed25519 要再安裝到那台 VD 上；已撤銷的 VD 不能用這個加回來。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from security import rotate_vd  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    for pid in sys.argv[2:]:
        rotate_vd(sys.argv[1], int(pid))
        print(f"已換新 VD {pid} 的金鑰")
