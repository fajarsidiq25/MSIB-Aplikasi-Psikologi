import json
import pickle
import random

import nltk
import numpy as np
from flask import Flask, render_template, request
from keras.models import load_model
from nltk.stem import WordNetLemmatizer
# nltk.download('popular')

from flask import Flask, render_template, Response
import cv2
import numpy as np
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing import image


lemmatizer = WordNetLemmatizer()


model = load_model('model.h5')
intents = json.loads(open('data.json').read())
words = pickle.load(open('texts.pkl', 'rb'))
classes = pickle.load(open('labels.pkl', 'rb'))


def clean_up_sentence(sentence):
    # pola tokenize - pisahkan kata-kata menjadi array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(
        word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
# kembalikan kumpulan kata-kata: 0 atau 1 untuk setiap kata dalam kantong yang ada dalam kalimat


def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print("found in bag: %s" % w)
    return (np.array(bag))


def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words, show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list


def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if (i['tag'] == tag):
            result = random.choice(i['responses'])
            break
    return result


def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


app = Flask(__name__)
app.static_folder = 'static'


model_cv = model_from_json(open("fer.json", "r").read())

# load weights
model_cv.load_weights('fer.h5')


face_haar_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')


# app = Flask(__name__)

camera = cv2.VideoCapture(0)


def gen_frames():  # generate frame by frame from camera
    while True:
        # Capture frame by frame
        success, frame = camera.read()
        if not success:
            break
        else:
            gray_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            faces_detected = face_haar_cascade.detectMultiScale(
                gray_img, 1.32, 5)

            for (x, y, w, h) in faces_detected:
                print('WORKING')
                cv2.rectangle(frame, (x, y), (x+w, y+h),
                              (255, 0, 0), thickness=7)
                # cropping region of interest i.e. face area from  image
                roi_gray = gray_img[y:y+w, x:x+h]
                roi_gray = cv2.resize(roi_gray, (48, 48))
                img_pixels = image.img_to_array(roi_gray)
                img_pixels = np.expand_dims(img_pixels, axis=0)
                img_pixels /= 255

                print(img_pixels.shape)

                predictions = model_cv.predict(img_pixels)

                # find max indexed array

                max_index = np.argmax(predictions[0])

                emotions = ['angry', 'disgust', 'fear',
                            'happy', 'sad', 'surprise', 'neutral']
                predicted_emotion = emotions[max_index]
                print(predicted_emotion)
                cv2.putText(frame, predicted_emotion, (int(x), int(y)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            resized_img = cv2.resize(frame, (1000, 700))

            ret, buffer = cv2.imencode('.jpg', frame)

            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result


@app.route('/video_feed')
def video_feed():
    # Video streaming route. Put this in the src attribute of an img tag
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/')
def index():
    return render_template('index.html')


# @app.route('/index.html')
# def beranda():
#     return render_template('index.html')


@app.route('/cv')
def cv():
    return render_template('cv.html')


@app.route('/chatbot')
def nlp():
    return render_template('chatbot.html')


@app.route('/fitur-app')
def fitur():
    return render_template('fitur-app.html')


@app.route('/musik-app')
def musik():
    return render_template('musik-app.html')


@app.route('/hubungikami-app')
def hubungikami():
    return render_template('hubungikami-app.html')


@app.route('/tentangkami-app')
def tentangkami():
    return render_template('tentangkami-app.html')


@app.route('/pencarian-app')
def pencarian():
    return render_template('pencarian-app.html')


@app.route('/jenispenyakit-app')
def jenis():
    return render_template('jenispenyakit-app.html')


@app.route('/daftarpenyakit-app')
def daftar():
    return render_template('daftarpenyakit-app.html')

# Artikel Penyakit Mental


@app.route('/bloggrid-app')
def grid():
    return render_template('blog-grid.html')


@app.route('/bloggrid-2-app')
def grid2():
    return render_template('blog-grid-2.html')


@app.route('/bloggrid-3-app')
def grid3():
    return render_template('blog-grid-3.html')


@app.route('/blog-details-adhd-app')
def details():
    return render_template('/artikel/blog-details-ADHD.html')


@app.route('/blog-details-autisme-app')
def autisme():
    return render_template('/artikel/blog-details-Autisme.html')


@app.route('/blog-details-bipolar-app')
def bipolar():
    return render_template('/artikel/blog-details-Bipolar.html')


@app.route('/blog-details-depresi-app')
def depresi():
    return render_template('/artikel/blog-details-Depresi.html')


@app.route('/blog-details-histeria-app')
def histeria():
    return render_template('/artikel/blog-details-Histeria.html')


@app.route('/blog-details-paranoid-app')
def paranoid():
    return render_template('/artikel/blog-details-Paranoid.html')


@app.route('/blog-details-alzheimer-app')
def alzheimer():
    return render_template('/artikel/blog-details-Alzheimer.html')


@app.route('/blog-details-adiksi-app')
def adiksi():
    return render_template('/artikel/blog-details-Adiksi.html')


@app.route('/blog-details-anxiety-app')
def anxiety():
    return render_template('/artikel/blog-details-Anxiety.html')


@app.route('/blog-details-disleksia-app')
def disleksia():
    return render_template('/artikel/blog-details-disleksia.html')


@app.route('/blog-details-disosiatif-app')
def disosiatif():
    return render_template('/artikel/blog-details-disosiatif.html')


@app.route('/blog-details-factitious-app')
def factitious():
    return render_template('/artikel/blog-details-Factitious.html')


@app.route('/blog-details-fobia-sosial-app')
def fobia():
    return render_template('/artikel/blog-details-Fobia-Sosial.html')


@app.route('/blog-details-gangguan-makan-app')
def makan():
    return render_template('/artikel/blog-details-Gangguan-Makan.html')


@app.route('/blog-details-gangguan-tidur-app')
def tidur():
    return render_template('/artikel/blog-details-Gangguan-Tidur.html')


@app.route('/blog-details-insecure-app')
def insecure():
    return render_template('/artikel/blog-details-Insecure.html')


@app.route('/blog-details-insomnia-app')
def insomnia():
    return render_template('/artikel/blog-details-Insomnia.html')


@app.route('/blog-details-kontrol-impuls-app')
def impuls():
    return render_template('/artikel/blog-details-Kontrol-Impuls.html')


@app.route('/blog-details-narsistik-app')
def narsistik():
    return render_template('/artikel/blog-details-Narsistik.html')


@app.route('/blog-details-nervous-app')
def nervous():
    return render_template('/artikel/blog-details-Nervous.html')


@app.route('/blog-details-ocd-app')
def ocd():
    return render_template('/artikel/blog-details-OCD.html')


@app.route('/blog-details-panic-attack-app')
def panic():
    return render_template('/artikel/blog-details-panic-attack.html')


@app.route('/blog-details-psikosomatis-app')
def psikosomatif():
    return render_template('/artikel/blog-details-Psikosomatif.html')


@app.route('/blog-details-ptsd-app')
def ptsd():
    return render_template('/artikel/blog-details-PTSD.html')


@app.route('/blog-details-syndrom-down-app')
def down():
    return render_template('/artikel/blog-details-Syndrom-Down.html')


@app.route('/blog-details-skizofrenia-app')
def skizofrenia():
    return render_template('/artikel/blog-details-Skizofrenia.html')


@app.route('/blog-details-tourette-app')
def tourette():
    return render_template('/artikel/blog-details-Tourette.html')


@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    return chatbot_response(userText)


if __name__ == '__main__':
    app.run(debug=True)
