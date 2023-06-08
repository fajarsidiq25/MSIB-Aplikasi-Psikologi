from flask import Flask, render_template, Response
import app
import os
import app1

app = Flask(__name__)


@app.route('/index.html')
def index():
    return render_template('index.html')


@app.route('/cv.html')
def cv():
    return render_template('cv.html')


@app.route('/chatbot.html')
def nlp():
    return render_template('chatbot.html')


@app.route('/about.html')
def about():
    return render_template('about.html')


@app.route('/pencarian.html')
def pencarian():
    return render_template('pencarian.html')


if __name__ == '__main__':
    app.run(debug=True)
    os.system('python app.py')
    os.system('python app1.py')
