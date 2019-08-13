import requests

upload_url = "http://192"
r = requests.post(upload_url,
                        files={'fileToUpload': (filetime + '.png', f)})
                    print(r.text)