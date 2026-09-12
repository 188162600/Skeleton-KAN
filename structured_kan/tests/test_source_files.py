"""Network-free regression checks, executed on the designated remote hosts."""
import hashlib
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error

from structured_kan.dataset.source_files import fetch_pinned, source_records, source_url


class SourceFilesTests(unittest.TestCase):
    def test_url_encoding(self):
        self.assertTrue(source_url('final/model/Fig5 i.sedml').endswith('Fig5%20i.sedml'))
        self.assertTrue(source_url('final/model/a#b%.xml').endswith('a%23b%25.xml'))
        for path in ('/etc/passwd', 'final/../secret'):
            with self.assertRaises(ValueError): source_url(path)

    def test_retry_atomic_cache(self):
        payload = b'<sbml/>'
        digest = hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'source.xml'
            with patch('urllib.request.urlopen', side_effect=[TimeoutError('test'), io.BytesIO(payload)]) as opener:
                with patch('time.sleep'):
                    fetch_pinned('final/model/source.xml', digest, target)
                self.assertEqual(opener.call_count, 2)
                self.assertEqual(list(Path(folder).iterdir()), [target])
                fetch_pinned('final/model/source.xml', digest, target)
                self.assertEqual(opener.call_count, 2)

    def test_wrong_payload_never_published(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'source.xml'
            with patch('urllib.request.urlopen', return_value=io.BytesIO(b'wrong')):
                with self.assertRaises(ValueError): fetch_pinned('final/model/source.xml', '0'*64, target)
            self.assertFalse(target.exists())

    def test_bad_existing_cache_not_overwritten(self):
        with tempfile.TemporaryDirectory() as folder:
            target = Path(folder)/'source.xml'; target.write_bytes(b'wrong')
            with patch('urllib.request.urlopen') as opener:
                with self.assertRaises(ValueError): fetch_pinned('final/model/source.xml', '0'*64, target)
                opener.assert_not_called()
            self.assertEqual(target.read_bytes(), b'wrong')

    def test_protocol_follows_support_sha(self):
        simulation = dict(initialTime='0', outputStartTime='0', outputEndTime='60')
        row = dict(case_id='test', source_model='model', sbml_source_path='/old/source.xml', sbml_sha256='sbml',
                   sedml_simulation=simulation, source_receipt=dict(sha256='old', upstream_path='final/model/old.sedml'),
                   quality_audit=dict(support=dict(sedml_sha256='correct', source_file_sha256='sbml', time_range=[0,60])),
                   protocols=[dict(source_receipt=dict(sha256='correct', upstream_path='final/model/correct.sedml'),
                                   sedml_simulation=simulation)])
        self.assertEqual(source_records(row)[1][1], 'final/model/correct.sedml')
        row['protocols'] = []
        with self.assertRaises(ValueError): source_records(row)

    def test_http_not_found_is_not_retried(self):
        with tempfile.TemporaryDirectory() as folder:
            with patch('urllib.request.urlopen', side_effect=urllib.error.HTTPError('u',404,'missing',{},None)) as opener:
                with self.assertRaises(urllib.error.HTTPError): fetch_pinned('final/m/a', '0'*64, Path(folder)/'a')
                self.assertEqual(opener.call_count, 1)

    def test_packaged_source_is_verified_and_network_free(self):
        payload=b'<frozen/>';sha=hashlib.sha256(payload).hexdigest()
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);cache=root/'cache';(cache/'model').mkdir(parents=True)
            (cache/'model/source.xml').write_bytes(payload)
            with patch('structured_kan.dataset.source_files.CACHE',cache),patch('urllib.request.urlopen') as opener:
                destination=root/'output/source.xml'
                fetch_pinned('final/model/source.xml',sha,destination)
                self.assertEqual(destination.read_bytes(),payload);opener.assert_not_called()
                with self.assertRaises(ValueError):fetch_pinned('final/model/source.xml','0'*64,root/'bad.xml')


if __name__ == '__main__': unittest.main()
