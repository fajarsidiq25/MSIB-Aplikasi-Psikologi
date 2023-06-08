# Import Libraries
import json
import nltk
import time
import random
import string
import pickle
import numpy as np
import pandas as pd
from gtts import gTTS
from io import BytesIO
import tensorflow as tf
import IPython.display as ipd
import speech_recognition as sr
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Model
from keras.utils.vis_utils import plot_model
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.layers import Input, Embedding, LSTM
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Flatten, Dense, GlobalMaxPool1D

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Importing the dataset
with open('data.json') as content:
    data1 = json.load(content)

# Mendapatkan semua data ke dalam list
tags = []  # data tag
inputs = []  # data input atau pattern
responses = {}  # data respon
words = []  # Data kata
classes = []  # Data Kelas atau Tag
documents = []  # Data Kalimat Dokumen
ignore_words = ['?', '!']  # Mengabaikan tanda spesial karakter

for intent in data1['intents']:
    responses[intent['tag']] = intent['responses']
    for lines in intent['patterns']:
        inputs.append(lines)
        tags.append(intent['tag'])
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            # add to our classes list
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

# Konversi data json ke dalam dataframe
data = pd.DataFrame({"patterns": inputs, "tags": tags})
data

data.head()  # Cetak data baris pertama sampai baris kelima
data.tail()  # Cetak data baris ke-70 sampai baris akhir

# Preprocessing The Data
# Setelah kita meload data dan mengonversi data json menjadi dataframe. Tahapan selanjutnya adalah praproses pada dataset yang kita gunakan saat ini yaitu dengan cara:
# Remove Punctuations (Menghapus Punktuasi)
# Lematization (Lematisasi)
# Tokenization (Tokenisasi)
# Apply Padding (Padding)
# Encoding the Outputs (Konversi Keluaran Enkoding)


# Removing Punctuations (Menghilangkan Punktuasi)
data['patterns'] = data['patterns'].apply(
    lambda wrd: [ltrs.lower() for ltrs in wrd if ltrs not in string.punctuation])
data['patterns'] = data['patterns'].apply(lambda wrd: ''.join(wrd))
data

# Lematisasi Kata “lemmatization adalah proses yang bertujuan untuk melakukan normalisasi pada teks dengan berdasarkan pada bentuk dasar yang merupakan bentuk lemmanya”. database yang menyimpan ejaan dan kata-kata yang tepat berdasarkan PUEBI dan KBBI.
lemmatizer = WordNetLemmatizer()
words = [lemmatizer.lemmatize(w.lower())
         for w in words if w not in ignore_words]
words = sorted(list(set(words)))
print(len(words), "unique lemmatized words", words)

# sort classes
classes = sorted(list(set(classes)))
print(len(classes), "classes", classes)

# documents = combination between patterns and intents
print(len(documents), "documents")

# Tokenize the data (Tokenisasi Data)
tokenizer = Tokenizer(num_words=2000)
tokenizer.fit_on_texts(data['patterns'])
train = tokenizer.texts_to_sequences(data['patterns'])
train

# Apply padding
x_train = pad_sequences(train)
# Encoding the outputs
le = LabelEncoder()
y_train = le.fit_transform(data['tags'])

print(x_train)  # Padding Sequences
print(y_train)  # Label Encodings

# Tokenizer pada Tensorflow memberikan token unik untuk setiap kata yang berbeda. Dan juga padding dilakukan untuk mendapatkan semua data dengan panjang yang sama sehingga dapat mengirimkannya ke lapisan atau layer RNN. variabel target juga dikodekan menjadi nilai desimal.

# Input Length, Output Length and Vocabulary
# input length
input_shape = x_train.shape[1]
print(input_shape)

# define vocabulary
vocabulary = len(tokenizer.word_index)
print("number of unique words : ", vocabulary)

# output length
output_length = le.classes_.shape[0]
print("output length: ", output_length)

# Input length dan output length terlihat sangat jelas hasilnya. Mereka adalah untuk bentuk input dan bentuk output dari jaringan syaraf pada algoritma Neural Network.
# Vocabulary Size adalah untuk lapisan penyematan untuk membuat representasi vektor unik untuk setiap kata.

# Save Model Words & Classes
pickle.dump(words, open('texts2.pkl', 'wb'))
pickle.dump(classes, open('labels2.pkl', 'wb'))


# Neural Network Model
# Jaringan syaraf yang terdiri dari lapisan embedding yang merupakan salah satu hal yang paling kuat di bidang pemrosesan bahasa alami atau NLP. output atau keluaran dari lapisan embedding adalah input dari lapisan berulang (recurrent) dengan LSTM gate. Kemudian, output diratakan dan lapisan Dense digunakan dengan fungsi aktivasi softmax.
# Bagian utama adalah lapisan embedding yang memberikan vektor yang sesuai untuk setiap kata dalam dataset.

# Creating the model (Membuat Modeling)
i = Input(shape=(input_shape,))
x = Embedding(vocabulary+1, 10)(i)  # Layer Embedding
x = LSTM(10, return_sequences=True)(x)  # Layer Long Short Term Memory
x = Flatten()(x)  # Layer Flatten
x = Dense(output_length, activation="softmax")(x)  # Layer Dense
model = Model(i, x)

# Compiling the model (Kompilasi Model)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer='adam', metrics=['accuracy'])

# Visualization Plot Architecture Model (Visualisasi Plot Arsitektur Model)
plot_model(model, to_file='chatbot.png',
           show_shapes=True, show_layer_names=True)

model.summary()  # Menampilkan Parameter Model

# Training the model (Latih Model Data)
train = model.fit(x_train, y_train, epochs=400)


# Model Analysis
# Setelah menjalankan model fitting. Selanjutnya adalah analisa model untuk melihat hasil akurasi dari model Neural Network tersebut.

# Plotting model Accuracy and Loss (Visualisasi Plot Hasil Akurasi dan Loss)
# Plot Akurasi
# plt.figure(figsize=(14, 5))
# plt.subplot(1, 2, 1)
# plt.plot(train.history['accuracy'], label='Training Set Accuracy')
# plt.legend(loc='lower right')
# plt.title('Accuracy')
# Plot Loss
# plt.subplot(1, 2, 2)
# plt.plot(train.history['loss'], label='Training Set Loss')
# plt.legend(loc='upper right')
# plt.title('Loss')
# plt.show()

# Save The Model
# Setelah pengujian Chatbot telah disesuaikan dengan kalimat dan jawabannya. Maka, model chatbot bisa disimpan dengan format .h5 atau .pkl (pickle) untuk penggunaan aplikasi AI Chatbot dengan website atau sistem Android.
model.save('model2.h5', train)
print('Model Created Successfully!')
