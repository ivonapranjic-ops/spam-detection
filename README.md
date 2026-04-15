# detekcija-spam-poruka
Projekt na prijediplomskom studiju u okviru kolegija Osnove strojnog učenja pod nazivom: DETEKCIJA SPAM PORUKA (SMS SPAM DETECTION)

## Opis projekta
Ovaj projekt fokusira se na klasifikaciju SMS poruka u dvije kategorije: ham (legitimne poruke) i spam (neželjene poruke). Kroz projekt su uspoređeni tradicionalni statistički pristupi s modernim dubokim učenjem kako bi se postigla maksimalna preciznost u filtriranju zlonamjernog sadržaja.

## Korištene tehnologije:
Jezik: Python
Biblioteke: TensorFlow/Keras, Scikit-learn, Pandas, NLTK/Spacy (za NLP obradu)
Dataset: Kaggle SMS Spam Collection
Prikaz i praćanje modela: TensorBoard

## Glavne funkcionalnosti:
- Obrada prirodnog jezika (NLP): Implementirano čišćenje teksta koje uključuje uklanjanje "stop" riječi, interpunkcije te tokenizaciju poruka.
- Vektorizacija teksta: Korištenje tehnika poput TF-IDF ili Tokenizacije za pretvaranje sirovog teksta u numeričke podatke razumljive stroju.
- Usporedba modela:
- Naive Bayes: Korišten kao osnovni (baseline) model za brzu statističku klasifikaciju.
- Neural Networks (Deep Learning): Razvijena vlastita neuronska mreža s više slojeva (Dense, Dropout za regularizaciju) koja je postigla značajno bolje rezultate u prepoznavanju kompleksnih uzoraka spama.
- Evaluacija s fokusom na preciznost: Budući da je u ovom problemu ključno da se legitimna poruka (ham) ne označi greškom kao spam, fokus evaluacije bio je na matrici zabune (Confusion Matrix) i F1-mjeri.

## Kako pokrenuti projekt?
Instaliranjem potrebnih biblioteka i pokretanjem .py datoteke dostupne u okviru ovog repozitorija.

## Rezultati:
- Projekt je uspješno demonstrirao nadmoć dubokih modela uz pravilnu konfiguraciju i regularizaciju.
- Postignuta je visoka točnost (Accuracy) i odziv (Recall), što omogućuje pouzdano filtriranje poruka u realnom vremenu.

## Autori:
- Ivona Pranjić
- Helena Zvocak
