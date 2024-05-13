import unittest
import datetime as dt

from context import extract_archive_uris, filter_uris_by_date


class TestArchiveExtract(unittest.TestCase):
    def test_empty_str(self):
        '''
        Test that it can handle an empty string
        '''
        data = ''
        result = extract_archive_uris(data)
        self.assertEqual(result, [])
    
    def test_non_data_str(self):
        '''
        Test that it can handle the `normal` uri format
        '''
        data = '/petroleum/supply/weekly/archive/2023/2023_07_19/'
        expected = ['/petroleum/supply/weekly/archive/2023/2023_07_19']
        result = extract_archive_uris(data)
        self.assertEqual(result, expected)

    def test_data_str(self):
        '''
        Test that it can handle `_data` case
        '''
        data = '/petroleum/supply/weekly/archive/2023/2023_11_03_data/'
        expected = ['/petroleum/supply/weekly/archive/2023/2023_11_03_data']
        result = extract_archive_uris(data)
        self.assertEqual(result, expected) 


class TestURIFilter(unittest.TestCase):
    def test(self):
        data = ['/petroleum/supply/weekly/archive/2023/2023_07_19',
                '/petroleum/supply/weekly/archive/2023/2023_11_03_data']
        date = dt.datetime(2023, 8, 1)
        result = filter_uris_by_date(data, date)
        self.assertEqual(result, ['/petroleum/supply/weekly/archive/2023/2023_11_03_data'])


if __name__ == '__main__':
    unittest.main()
