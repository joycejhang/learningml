import requests
import webbrowser
param = {"q": "深度學習"}  # 搜索的信息
r = requests.get('https://www.google.com.tw/search', params=param)
print(r.url)
webbrowser.open(r.url)

#https://www.google.com.tw/search?q=深度學習&


