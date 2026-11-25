from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))

import unittest
from pull_perception_results import classify, folder_name


class PullPerceptionResultsTests(unittest.TestCase):
    def test_folder_name_is_day_month_year_time_result(self):
        self.assertEqual(folder_name('20261118-083611_real_time', 'KHONG_CO_VAT'),
                         '18-11-2026_08-36-11_KHONG_CO_VAT')
        with self.assertRaises(ValueError):
            folder_name('../../etc_real_time', 'CO_VAT')

    def test_classify(self):
        newest = '20261118-090000_real_time'
        run = lambda **kw: {'name': '20261118-083611_real_time', **kw}
        self.assertEqual(classify(run(result={'selected': {'x': 1}, 'fusion': {'W': 2}}), newest), 'CO_VAT')
        self.assertEqual(classify(run(result={'selected': {'x': 1}, 'fusion': None}), newest),
                         'CO_VAT_CHUA_FUSION')
        self.assertEqual(classify(run(result={'selected': None}), newest), 'KHONG_CO_VAT')
        self.assertIsNone(classify(run(started=False), newest))  # READY only, never scanned.
        self.assertEqual(classify(run(started=True, error=True), newest), 'LOI')
        self.assertEqual(classify(run(started=True), newest), 'DUNG_GIUA_CHUNG')
        # The newest run without result.json may still be scanning: wait.
        self.assertIsNone(classify({'name': newest, 'started': True}, newest))


if __name__ == '__main__':
    unittest.main()
