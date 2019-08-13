import requests

r = requests.post(upload_url,
                        files={'fileToUpload': (filetime + '.png', f)})
                    print(r.text)