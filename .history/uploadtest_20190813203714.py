import requests

upload_url = "http://0.0.0.0:5000"
r = requests.post(upload_url,
            files={'fileToUpload': (filetime + '.png', f)})
            print(r.text)