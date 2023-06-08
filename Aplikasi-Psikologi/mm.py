from flask import Flask, render_template, Response
import app
import os
import app1

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/index.html')
def beranda():
    return render_template('index.html')


@app.route('/cv.html')
def cv():
    return render_template('cv.html')


@app.route('/chatbot.html')
def nlp():
    return render_template('chatbot.html')


@app.route('/fitur-app.html')
def fitur():
    return render_template('fitur-app.html')

@app.route('/musik-app.html')
def musik():
    return render_template('musik-app.html')

@app.route('/hubungikami-app.html')
def hubungikami():
    return render_template('hubungikami-app.html')


@app.route('/tentangkami-app.html')
def tentangkami():
    return render_template('tentangkami-app.html')


@app.route('/pencarian-app.html')
def pencarian():
    return render_template('pencarian-app.html')


if __name__ == '__main__':
    app.run(debug=True)
    os.system('python app.py')
    os.system('python app1.py')
