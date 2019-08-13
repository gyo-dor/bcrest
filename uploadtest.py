import requests

upload_url = "http://127.0.0.1:5000/uploader"
static_url = "http://127.0.0.1:5000/static/imgs/" # TEMP!

filename = "123.128.jpg"
img = open(filename, 'rb')
r = requests.post(upload_url,
                files={'file': (filename, img)})

if 'Success' in r.text:
    # Get image from url
    Picture_request = requests.get(static_url + filename)
    if Picture_request.status_code == 200:
        with open("static/rout/" + filename, 'wb') as f:
            f.write(Picture_request.content)
else:
    print('Img Failed')
    pass


