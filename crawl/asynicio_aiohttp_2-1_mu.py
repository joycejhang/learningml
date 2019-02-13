# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:39:32 2018

@author: Joyce
"""

import requests
import time

URL = 'https://morvanzhou.github.io/'


def normal():
    for i in range(2):
        r = requests.get(URL)
        url = r.url
        print(url)

t1 = time.time()
normal()
print("Normal total time:", time.time()-t1)