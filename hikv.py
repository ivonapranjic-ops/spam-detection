# Uvoz potrebnih biblioteka
import numpy as np                    # Za rad s numeričkim nizovima i matricama
import pandas as pd                  # Za učitavanje i manipulaciju tabličnim podacima
import matplotlib.pyplot as plt      # Za vizualizaciju podataka
import seaborn as sns                # Napredna vizualizacija (grafovi, heatmape, sl.)
import tensorflow as tf              # Glavna biblioteka za strojno učenje i duboko učenje
from tensorflow import keras         # Keras API unutar TensorFlowa
from tensorflow.keras import layers  # Keras slojevi za gradnju modela
from sklearn.model_selection import train_test_split                  # Za podjelu podataka
from sklearn.feature_extraction.text import TfidfVectorizer           # TF-IDF vektor za tekstualne podatke
from sklearn.naive_bayes import MultinomialNB                         # Naive Bayes klasifikator
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay  # Metričke funkcije
from tensorflow.keras.layers import TextVectorization                 # Sloj za pretvaranje teksta u numeričke sekvence
from sklearn.metrics import precision_score, recall_score, f1_score   # Dodatne metrike
from datetime import datetime                                          # Za generiranje vremenske oznake

# Definiranje direktorija za TensorBoard logove
logdir = "./logs/" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# Kreiranje TensorBoard callbacka
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)


#  Učitavanje i priprema podataka
df = pd.read_csv("C:/Users/student/Documents/IKT_LV2_STROJNO_UCENJE/spam.csv", encoding='latin-1')  # Učitavanje CSV datoteke
df = df.rename(columns={'v1': 'label', 'v2': 'Text'})  # Preimenovanje kolona radi lakšeg rukovanja
df['label_enc'] = df['label'].map({'ham': 0, 'spam': 1})  # Mapiranje labela u numeričke vrijednosti

# Vizualizacija raspodjele klasa
sns.countplot(x=df['label'])  # Prikaz broja poruka po klasama
plt.title("Raspodjela klasa")
plt.show()

#  Prosječna duljina poruke
valid_texts = df['Text'].dropna().astype(str)
avg_words_len = round(sum(len(i.split()) for i in valid_texts) / len(valid_texts))  # Brojanje riječi u porukama
print(f"Prosječna duljina poruke: {avg_words_len} riječi")

#  Broj jedinstvenih riječi
s = set()
for sent in df['Text']:
    for word in sent.split():
        s.add(word)  # Dodavanje riječi u skup (automatski eliminira duplikate)
total_words_length = len(s)
print(f"Ukupan broj jedinstvenih riječi: {total_words_length}")

# Podjela skupa podataka
x = np.asarray(df['Text'])
y = np.asarray(df['label_enc'])
x_train_val, x_test, y_train_val, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # 80% train/val, 20% test
x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.25, random_state=1)  # 60% train, 20% val

# Vektorizacija teksta i treniranje Naive Bayes modela
tfidf_vec = TfidfVectorizer().fit(x_train)  # TF-IDF prilagođen samo na tren setu
x_train_vec = tfidf_vec.transform(x_train)
x_val_vec = tfidf_vec.transform(x_val)
x_test_vec = tfidf_vec.transform(x_test)

baseline_model = MultinomialNB()  # Inicijalizacija Naive Bayes modela
baseline_model.fit(x_train_vec, y_train)  # Treniranje modela

# Evaluacija Naive Bayes modela
y_pred = baseline_model.predict(x_test_vec)
print(f"\n[Baseline] Točnost na test skupu NB modela: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))  # Detaljan izvještaj

cm = confusion_matrix(y_test, y_pred)
ConfusionMatrixDisplay(confusion_matrix=cm).plot()
plt.title("Konfuzijska matrica – Naive Bayes")
plt.show()

# Funkcija za treniranje neuronske mreže
def fit_model(model, epochs, x_train, y_train, x_test=None, y_test=None):
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True),  # Prekid treniranja ako se validacija ne poboljšava
        tensorboard_callback  # Aktivacija TensorBoard zapisa
    ]
    if x_test is not None and y_test is not None:
        history = model.fit(
            x_train, y_train,
            epochs=epochs,
            validation_data=(x_test, y_test),
            callbacks=callbacks
        )
    else:
        history = model.fit(x_train, y_train, epochs=epochs, callbacks=callbacks)
    return history

# Funkcija za evaluaciju modela
def evaluate_model(model, x, y):
    y_preds = np.round(model.predict(x))  # Predikcija (zaokruživanje jer imamo sigmoid)
    accuracy = accuracy_score(y, y_preds)
    precision = precision_score(y, y_preds)
    recall = recall_score(y, y_preds)
    f1 = f1_score(y, y_preds)
    return {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'f1-score': f1}

# Priprema neuronske mreže za tekstualnu klasifikaciju
MAXTOKENS = total_words_length  # Maksimalan broj tokena prema broju jedinstvenih riječi
OUTPUTLEN = avg_words_len       # Dužina izlazne sekvence bazirano na prosječnoj duljini poruka

# Vektorizacija teksta
text_vec = TextVectorization(
    max_tokens=MAXTOKENS,
    standardize='lower_and_strip_punctuation',  # Mala slova + uklanjanje interpunkcije
    output_mode='int',
    output_sequence_length=OUTPUTLEN
)
text_vec.adapt(x_train)  # Prilagodba vektora na trening skup

# Embedding sloj (riječi u vektore)
embedding_layer = layers.Embedding(
    input_dim=MAXTOKENS,
    output_dim=128,
    embeddings_initializer='uniform',
    input_length=OUTPUTLEN
)

# Definicija modela
input_layer = layers.Input(shape=(1,), dtype=tf.string)
vec_layer = text_vec(input_layer)  # Vektorizacija
embedding_layer_model = embedding_layer(vec_layer)  # Ugradnja riječi
x = layers.GlobalAveragePooling1D()(embedding_layer_model)  # Prosjek svih riječi
x = layers.Flatten()(x)
x = layers.Dense(32, activation='relu')(x)  # Potpuno povezani sloj
x = layers.BatchNormalization()(x)          # Normalizacija izlaza radi stabilnijeg treniranja
x = layers.Dropout(0.3)(x)                  # Dropout radi sprječavanja prenaučenosti
output_layer = layers.Dense(1, activation='sigmoid')(x)  # Sigmoid izlaz za binarnu klasifikaciju

# Kompilacija modela
model_1 = keras.Model(input_layer, output_layer)
model_1.compile(optimizer='adam', loss=keras.losses.BinaryCrossentropy(label_smoothing=0.5), metrics=['accuracy'])  # label_smoothing za stabilnije učenje

# Treniranje neuronske mreže
history = fit_model(model_1, epochs=60, x_train=x_train, y_train=y_train, x_test=x_val, y_test=y_val)

# Vizualizacija gubitka i točnosti kroz epohe
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Accuracy NN model')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss NN model')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.tight_layout()
plt.show()

# Evaluacija neuronske mreže na test skupu
y_test_preds = np.round(model_1.predict(x_test))
print(f"\n[NN Model] Točnost na test skupu: {accuracy_score(y_test, y_test_preds):.4f}")
print(classification_report(y_test, y_test_preds))

cm_nn = confusion_matrix(y_test, y_test_preds)
ConfusionMatrixDisplay(confusion_matrix=cm_nn).plot()
plt.title("Konfuzijska matrica – jednostavni NN model")
plt.show()

# Stupčasti graf metrika neuronske mreže
results = evaluate_model(model_1, x_test, y_test)
plt.figure(figsize=(8, 5))
sns.barplot(x=list(results.keys()), y=list(results.values()))
plt.title("Evaluacija neuronskog modela")
plt.ylim(0, 1)
for i, v in enumerate(results.values()):
    plt.text(i, v + 0.02, f"{v:.2f}", ha='center') 
plt.show()
