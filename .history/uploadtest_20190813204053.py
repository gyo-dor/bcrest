import requests

upload_url = "http://127.0.0.1:5000/"
img = open('sample.png', 'rb')

r = requests.post(upload_url,
                files={'fileToUpload': ('sample' + '.jpg', img)})
print(r.text)