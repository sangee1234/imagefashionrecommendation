from PIL import Image
from flask import Flask, render_template, request
from scripts.Recommendation import *
import cv2

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/login")
def login():
    return render_template("index.html")

@app.route("/detect", methods = ['POST'])
def detect():
    try:
        # print(request.files)
        # img = Image.open(request.files['image'].stream)
        # rgb_image = img.convert('RGB')
        # get_recommendations( rgb_image)
        img1 = cv2.imdecode(np.frombuffer(request.files['image'].read(), np.uint8), cv2.IMREAD_UNCHANGED)
        img = cv2.resize(img1,  (224, 224), interpolation = cv2.INTER_AREA)
        rgbimg = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        floatimg = rgbimg.astype("float32") / 255.0
        image = np.transpose(floatimg, (2, 0, 1))
        image = np.expand_dims(image, 0)
        # get_similar_images([], image)
        get_recommendations(image)


    except Exception as e:
        print(e)
        return render_template("failure.html")



if __name__=='__main__':
    print('run')
    app.run(host='0.0.0.0', port=5000, debug=True)

