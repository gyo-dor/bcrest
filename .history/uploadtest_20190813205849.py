upload_url = "http://127.0.0.1:5000/uploader"
    img = open('sample.jpg', 'rb')
    r = requests.post(upload_url,
                    files={'file': ('sample' + '.jpg', img)})
    print(r.text)