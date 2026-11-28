import importlib.util
import json
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


pub = load("publish_tunnel_url", ROOT / "scripts" / "publish_tunnel_url.py")
cloud = load("vercel_forecast", ROOT / "web-vercel" / "api" / "forecast.py")


class PublishTest(unittest.TestCase):
    def test_last_url_in_log_wins(self):
        log = "x https://old-one.trycloudflare.com\n...\n|  https://new-two.trycloudflare.com  |\n"
        self.assertEqual(pub.latest_url(log), "https://new-two.trycloudflare.com")

    def test_no_url(self):
        self.assertIsNone(pub.latest_url("starting tunnel\n"))


class CloudTest(unittest.TestCase):
    def test_accepts_only_trycloudflare_https(self):
        ok = json.dumps({"url": "https://salvador-hindu-atm-flashing.trycloudflare.com"}).encode()
        self.assertEqual(cloud.tunnel_url(ok), "https://salvador-hindu-atm-flashing.trycloudflare.com")
        for bad in ["http://a.trycloudflare.com", "https://evil.com", "https://a.trycloudflare.com.evil.com",
                    "https://a.trycloudflare.com/x", "https://evil.com/?.trycloudflare.com", 5]:
            with self.assertRaises(ValueError, msg=bad):
                cloud.tunnel_url(json.dumps({"url": bad}).encode())

    def test_relays_desktop_json_verbatim(self):
        body = json.dumps({"stale": False, "vds": [1]}).encode()
        pages = {cloud.TUNNEL_JSON: b'{"url": "https://a-b.trycloudflare.com"}',
                 "https://a-b.trycloudflare.com/api/forecast": body}
        self.assertEqual(cloud.relay(lambda u, t: pages[u]), body)

    def test_rejects_non_json_from_desktop(self):
        pages = {cloud.TUNNEL_JSON: b'{"url": "https://a-b.trycloudflare.com"}',
                 "https://a-b.trycloudflare.com/api/forecast": b"<html>502</html>"}
        with self.assertRaises(ValueError):
            cloud.relay(lambda u, t: pages[u])


if __name__ == "__main__":
    unittest.main()
