"""管理者撤銷 VD：python scripts/revoke_vd.py <key-dir> <VD 編號>…（寫進 <key-dir>/server/revoked.json，下次啟動伺服器起生效）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from security import revoke_vd  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    for pid in sys.argv[2:]:
        revoke_vd(sys.argv[1], int(pid))
        print(f"已撤銷 VD {pid}")
