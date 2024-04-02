import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from fungsi import make_model
from fungsi import make_model2

app = Flask(__name__)

model = make_model2()
model.load_weights("modelfix.h5")


# Fungsi untuk mendapatkan index kelas
def get_class_label(probability):
    if probability > 0.5:
        return "Sapi"
    else:
        return "Babi"


@app.route("/", methods=["GET", "POST"])
def main():
    return render_template("index.html")


@app.route("/submit", methods=["GET", "POST"])
def get_output():
    if request.method == "POST":
        img = request.files["my_image"]
        img_path = "static/image/uploads/" + img.filename
        img.save(img_path)
        p = predict_label(img_path)

    return render_template("index.html", prediction=p, img_path=img.filename)


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        img = request.files["my_image"]
        img_path = "static/image/uploads/" + img.filename
        img.save(img_path)
        p = predict_label(img_path)
        return {"prediction": p}


def predict_label(img_path):
    img = image.load_img(img_path, target_size=(300, 300, 3))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)
    class_label = get_class_label(pred[0][0])

    return class_label


if __name__ == "__main__":
    app.run(debug=True)
