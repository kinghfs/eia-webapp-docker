import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from scrape import extract_archive_uris, filter_uris_by_date, get_report_urls