import requests

upload_url = "http://0.0.0.0:5000"
f
r = requests.post(upload_url,
                files={'fileToUpload': ('sample' + '.png', f)})
print(r.text)