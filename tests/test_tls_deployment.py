import sys
import tempfile
import unittest
from pathlib import Path

from cryptography import x509
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import load_ssh_private_key, load_ssh_public_key

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from run_tls_deployment import make_node_key, make_tls  # noqa: E402


class TlsDeploymentTest(unittest.TestCase):
    def test_server_cert_is_signed_by_ca_for_localhost(self):
        with tempfile.TemporaryDirectory() as d:
            p = make_tls(Path(d))
            ca = x509.load_pem_x509_certificate(p["ca"].read_bytes())
            srv = x509.load_pem_x509_certificate(p["cert"].read_bytes())
            srv.verify_directly_issued_by(ca)
            san = srv.extensions.get_extension_for_class(x509.SubjectAlternativeName).value
            self.assertIn("localhost", san.get_values_for_type(x509.DNSName))
            self.assertFalse(srv.extensions.get_extension_for_class(x509.BasicConstraints).value.ca)

    def test_node_key_is_p384_openssh_pair(self):
        # flower-supernode 只接受 OpenSSH 格式的 NIST EC 私鑰
        with tempfile.TemporaryDirectory() as d:
            priv, pub = make_node_key(Path(d), "vd-0")
            key = load_ssh_private_key(priv.read_bytes(), password=None)
            self.assertIsInstance(key.curve, ec.SECP384R1)
            self.assertEqual(load_ssh_public_key(pub.read_bytes()).public_numbers(),
                             key.public_key().public_numbers())


if __name__ == "__main__":
    unittest.main()
