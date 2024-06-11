import unittest
from unittest.mock import patch, MagicMock

import datetime as dt

from context import extract_archive_uris, filter_uris_by_date, get_report_urls



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


class TestReportURL(unittest.TestCase):

    @patch('scrape.requests')
    def test_report_urls(self, mock_request):

        example_content = 'hello world /petroleum/supply/weekly/archive/2023/2023_07_19/'.encode('cp1252')
        mock_response = MagicMock(status_code=200, content=example_content)

        mock_request.get.return_value = mock_response

        expected_urls = ['https://www.eia.gov/petroleum/supply/weekly/archive/2023/2023_07_19/csv/table1.csv']
        self.assertEqual(get_report_urls(), expected_urls)

        
if __name__ == '__main__':
    unittest.main()
