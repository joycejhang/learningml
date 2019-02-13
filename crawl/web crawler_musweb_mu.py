# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 17:45:00 2018

@author: Joyce
"""

import multiprocessing as mp
import time
from urllib.request import urlopen, urljoin
from bs4 import BeautifulSoup
import re

# base_url = "http://127.0.0.1:4000/"
base_url = 'https://morvanzhou.github.io/'

def crawl(url):
    response = urlopen(url)
    # time.sleep(0.1)             # slightly delay for downloading
    return response.read().decode()


def parse(html):
    soup = BeautifulSoup(html, 'lxml')
    urls = soup.find_all('a', {"href": re.compile('^/.+?/$')})
    title = soup.find('h1').get_text().strip()
    page_urls = set([urljoin(base_url, url['href']) for url in urls])   # 去重
    url = soup.find('meta', {'property': "og:url"})['content']
    return title, page_urls, url


unseen = set([base_url,])
seen = set()

count, t1 = 1, time.time()

# DON'T OVER CRAWL THE WEBSITE OR YOU MAY NEVER VISIT AGAIN
if base_url != "http://127.0.0.1:4000/":
    restricted_crawl = True
else:
    restricted_crawl = False



while len(unseen) != 0:                 # still get some url to visit
    if restricted_crawl and len(seen) > 20:
            break
        
    print('\nDistributed Crawling...')
    htmls = [crawl(url) for url in unseen]

    print('\nDistributed Parsing...')
    results = [parse(html) for html in htmls]

    print('\nAnalysing...')
    seen.update(unseen)         # seen the crawled
    unseen.clear()              # nothing unseen

    for title, page_urls, url in results:
        print(count, title, url)
        count += 1
        unseen.update(page_urls - seen)     # get new url to crawl
print('Total time: %.1f s' % (time.time()-t1, ))    # 53 s