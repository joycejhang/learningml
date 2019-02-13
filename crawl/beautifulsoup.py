# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 10:47:44 2018

@author: Joyce
"""

from bs4 import BeautifulSoup
from urllib.request import urlopen

# if has Chinese, apply decode()
html = urlopen("https://morvanzhou.github.io/static/scraping/basic-structure.html").read().decode('utf-8')
print(html)

soup = BeautifulSoup(html, features='lxml')
print(soup.h1)

"""
<h1>爬虫测试1</h1>
"""

print('\n', soup.p)

"""
<p>
		这是一个在 <a href="https://morvanzhou.github.io/">莫烦Python</a>
<a href="https://morvanzhou.github.io/tutorials/scraping">爬虫教程</a> 中的简单测试.
	</p>
"""

soup = BeautifulSoup(html, features='lxml')
print(soup.h1)

"""
<h1>爬虫测试1</h1>
"""

print('\n', soup.p)

"""
<p>
		这是一个在 <a href="https://morvanzhou.github.io/">莫烦Python</a>
<a href="https://morvanzhou.github.io/tutorials/scraping">爬虫教程</a> 中的简单测试.
	</p>
"""

"""
<a href="https://morvanzhou.github.io/tutorials/scraping">爬虫教程</a>
"""

all_href = soup.find_all('a')
all_href = [l['href'] for l in all_href]
print('\n', all_href)

# ['https://morvanzhou.github.io/', 'https://morvanzhou.github.io/tutorials/scraping']