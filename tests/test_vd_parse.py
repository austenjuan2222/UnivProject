import gzip
import unittest

from vd_parse import parse_vdlive_xml

NS = "http://traffic.transportdata.tw/standard/traffic/schema/"


def _lane(lane_id, speed, occ, volumes):
    vehicles = "".join(
        f"<Vehicle><VehicleType>{t}</VehicleType><Volume>{v}</Volume>"
        f"<Speed>{speed}</Speed></Vehicle>"
        for t, v in volumes.items()
    )
    return (
        f"<Lane><LaneID>{lane_id}</LaneID><LaneType>1</LaneType>"
        f"<Speed>{speed}</Speed><Occupancy>{occ}</Occupancy>"
        f"<Vehicles>{vehicles}</Vehicles></Lane>"
    )


def _vd(vdid, lanes, status=0, t="2024-03-04T08:00:00+08:00"):
    return (
        f"<VDLive><VDID>{vdid}</VDID><LinkFlows><LinkFlow><LinkID>X</LinkID>"
        f"<Lanes>{''.join(lanes)}</Lanes></LinkFlow></LinkFlows>"
        f"<Status>{status}</Status><DataCollectTime>{t}</DataCollectTime></VDLive>"
    )


def _doc(*vds, ns=NS):
    xmlns = f' xmlns="{ns}"' if ns else ""
    return (
        f'<?xml version="1.0" encoding="utf-8"?><VDLiveList{xmlns}>'
        f"<UpdateTime>2024-03-04T08:01:00+08:00</UpdateTime>"
        f"<VDLives>{''.join(vds)}</VDLives></VDLiveList>"
    ).encode("utf-8")


class ParseVDLiveTest(unittest.TestCase):
    def test_volume_weighted_speed_across_lanes(self):
        xml = _doc(
            _vd("VD-N1-S-10", [
                _lane(0, 100, 5, {"S": 9, "L": 1}),   # 10 vehicles @100
                _lane(1, 60, 20, {"S": 30}),          # 30 vehicles @60
            ])
        )
        rows = parse_vdlive_xml(xml)
        self.assertEqual(len(rows), 1)
        r = rows[0]
        self.assertEqual(r["vdid"], "VD-N1-S-10")
        self.assertEqual(r["time"], "2024-03-04T08:00:00+08:00")
        self.assertAlmostEqual(r["speed"], (100 * 10 + 60 * 30) / 40)
        self.assertEqual(r["volume"], 40)
        self.assertAlmostEqual(r["occupancy"], 12.5)

    def test_invalid_lane_speed_is_ignored(self):
        xml = _doc(_vd("A", [_lane(0, -99, 0, {"S": 0}), _lane(1, 80, 4, {"S": 5})]))
        r = parse_vdlive_xml(xml)[0]
        self.assertAlmostEqual(r["speed"], 80.0)
        self.assertEqual(r["volume"], 5)

    def test_zero_volume_falls_back_to_mean_of_valid_speeds(self):
        xml = _doc(_vd("A", [_lane(0, 90, 1, {"S": 0}), _lane(1, 70, 1, {"S": 0})]))
        self.assertAlmostEqual(parse_vdlive_xml(xml)[0]["speed"], 80.0)

    def test_no_valid_lane_gives_none_speed(self):
        xml = _doc(_vd("A", [_lane(0, 0, 0, {"S": 0}), _lane(1, 250, 0, {"S": 3})]))
        self.assertIsNone(parse_vdlive_xml(xml)[0]["speed"])

    def test_nonzero_status_marks_speed_none(self):
        xml = _doc(_vd("A", [_lane(0, 90, 1, {"S": 3})], status=1))
        r = parse_vdlive_xml(xml)[0]
        self.assertIsNone(r["speed"])
        self.assertEqual(r["status"], 1)

    def test_vd_filter(self):
        xml = _doc(_vd("VD-N1-A", [_lane(0, 90, 1, {"S": 3})]),
                   _vd("VD-N3-B", [_lane(0, 90, 1, {"S": 3})]))
        rows = parse_vdlive_xml(xml, keep=lambda vdid: "N1" in vdid)
        self.assertEqual([r["vdid"] for r in rows], ["VD-N1-A"])

    def test_works_without_namespace_and_gzip(self):
        raw = _doc(_vd("A", [_lane(0, 90, 1, {"S": 3})]), ns=None)
        self.assertEqual(len(parse_vdlive_xml(gzip.compress(raw))), 1)

    def test_unknown_document_raises(self):
        with self.assertRaises(ValueError):
            parse_vdlive_xml(b"<html><body>Not found</body></html>")


if __name__ == "__main__":
    unittest.main()
