import urllib2
import base64
import json

'''
easydl物体检测
'''

request_url = "【接口地址】"

with open("image.jpg", 'rb') as f:
    base64_data = base64.b64encode(f.read())
    s = base64_data.decode('UTF8')

params = {"image": s}
params = json.dumps(params)
access_token = '[调用鉴权接口获取的token]'
request_url = request_url + "?access_token=" + access_token
request = urllib2.Request(url=request_url, data=params)
request.add_header('Content-Type', 'application/json')
response = urllib2.urlopen(request)
content = response.read()
if content:
    print content
