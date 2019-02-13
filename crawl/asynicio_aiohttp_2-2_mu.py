# -*- coding: utf-8 -*-
"""
Created on Tue Aug 28 19:42:46 2018

@author: Joyce
"""

import aiohttp
import time
import asyncio


URL = 'https://morvanzhou.github.io/'

async def job(session):
    response = await session.get(URL)       # 等待并切换
    return str(response.url)


async def main(loop):
    async with aiohttp.ClientSession() as session:      # 官网推荐建立 Session 的形式
        tasks = [loop.create_task(job(session)) for _ in range(2)]
        finished, unfinished = await asyncio.wait(tasks)
        all_results = [r.result() for r in finished]    # 获取所有结果
        print(all_results)

t1 = time.time()
loop = asyncio.get_event_loop()
loop.run_until_complete(main(loop))
loop.close()

print("Async total time:", time.time() - t1)

