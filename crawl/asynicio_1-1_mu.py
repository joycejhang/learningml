# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:26:11 2018

@author: Joyce
"""

# 不是异步的
import time


def job(t):
    print('Start job ', t)
    time.sleep(t)               # wait for "t" seconds
    print('Job ', t, ' takes ', t, ' s')


def main():
    [job(t) for t in range(1, 10)]


t1 = time.time()
main()
print("NO async total time : ", time.time() - t1)

"""
Start job  1
Job  1  takes  1  s
Start job  2
Job  2  takes  2  s
NO async total time :  3.008603096008301
"""