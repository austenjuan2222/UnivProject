"""管理者換新伺服器簽章金鑰：python scripts/rotate_server.py <key-dir>
舊私鑰簽新公鑰寫進 server_chain.json，VD 不必重新安裝 server.pub；server_chain.json 要派送給各 VD（公開資料，不用保密）。"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from security import rotate_server  # noqa: E402

if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    rotate_server(sys.argv[1])
    print("已換新伺服器簽章金鑰")
