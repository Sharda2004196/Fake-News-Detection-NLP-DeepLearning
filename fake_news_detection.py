Fake News Detection using NLP and Deep Learning (BiLSTM + Conv1D)

Author: Sharda Vatsal Bhat (GitHub Project Ready)

import pandas as pd import numpy as np import re import string

from sklearn.model_selection import train_test_split from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import tensorflow as tf from tensorflow.keras.models import Sequential from tensorflow.keras.layers import Embedding, Conv1D, MaxPooling1D, Bidirectional, LSTM, Dense, Dropout from tensorflow.keras.preprocessing.text import Tokenizer from tensorflow.keras.preprocessing.sequence import pad_sequences

-----------------------------

1. Load Dataset

-----------------------------

Dataset format expected:

column 'text'  -> news content

column 'label' -> 0 = Fake, 1 = Real

Example: df = pd.read_csv('fake_news.csv')

df = pd.read_csv('fake_news.csv')  # replace with your dataset path

-----------------------------

2. Text Cleaning

-----------------------------

def clean_text(text): text = text.lower() text = re.sub(r"http\S+|www\S+", "", text) text = re.sub(r"<.*?>", "", text) text = re.sub(f"[{re.escape(string.punctuation)}]", "", text) text = re.sub(r"\n", " ", text) text = re.sub(r"\d+", "", text) return text

df['text'] = df['text'].apply(clean_text)

-----------------------------

3. Tokenization & Padding

-----------------------------

MAX_WORDS = 20000 MAX_LEN = 300

tokenizer = Tokenizer(num_words=MAX_WORDS, oov_token="<OOV>") tokenizer.fit_on_texts(df['text'])

sequences = tokenizer.texts_to_sequences(df['text']) padded_sequences = pad_sequences(sequences, maxlen=MAX_LEN, padding='post')

X = padded_sequences y = df['label']

-----------------------------

4. Train-Test Split (80-20)

-----------------------------

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=42, stratify=y )

-----------------------------

5. Model Architecture

Embedding → Conv1D → MaxPooling → BiLSTM → Dense

-----------------------------

model = Sequential([ Embedding(input_dim=MAX_WORDS, output_dim=128, input_length=MAX_LEN),

Conv1D(filters=128, kernel_size=5, activation='relu'),
MaxPooling1D(pool_size=2),

Bidirectional(LSTM(64, return_sequences=False)),
Dropout(0.5),

Dense(64, activation='relu'),
Dropout(0.5),

Dense(1, activation='sigmoid')

])

model.compile( loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'] )

model.summary()

-----------------------------

6. Model Training

-----------------------------

history = model.fit( X_train, y_train, epochs=8, batch_size=64, validation_split=0.2 )

-----------------------------

7. Evaluation

-----------------------------

y_pred_prob = model.predict(X_test) y_pred = (y_pred_prob > 0.5).astype(int)

accuracy = accuracy_score(y_test, y_pred) precision = precision_score(y_test, y_pred) recall = recall_score(y_test, y_pred) f1 = f1_score(y_test, y_pred)

print("Accuracy:", accuracy) print("Precision:", precision) print("Recall:", recall) print("F1 Score:", f1)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

-----------------------------

8. Save Model

-----------------------------

model.save('fake_news_bilstm_cnn_model.h5')

-----------------------------

9. Prediction Function

-----------------------------

def predict_news(text): text = clean_text(text) seq = tokenizer.texts_to_sequences([text]) pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post') prediction = model.predict(pad)[0][0] return "Real News" if prediction > 0.5 else "Fake News"

Example

print(predict_news("Breaking news: Scientists discover new planet"))
