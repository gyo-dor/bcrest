import requests

upload_url = "http://0.0.0.0:5000"
img = open('sample.png', 'rb')

r = requests.post(upload_url,
                files={'fileToUpload': ('sample' + '.png', img)})
print(r.text)