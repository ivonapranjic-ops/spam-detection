# detection-spam-message
Undergraduate project within the course Basics of Machine Learning entitled: DETECTION OF SPAM MESSAGES (SMS SPAM DETECTION)

## Description of the project
This project focuses on the classification of SMS messages into two categories: ham (legitimate messages) and spam (unsolicited messages). The project compared traditional statistical approaches with modern deep learning in order to achieve maximum precision in filtering malicious content.

## Technologies used:
Language: Python
Libraries: TensorFlow/Keras, Scikit-learn, Pandas, NLTK/Spacy (for NLP processing)
Dataset: Kaggle SMS Spam Collection
Model display and tracking: TensorBoard

## Main functionalities:
- Natural Language Processing (NLP): Implemented text cleaning that includes removal of "stop" words, punctuation and tokenization of messages.
- Text vectorization: Using techniques like TF-IDF or Tokenization to convert raw text into machine-readable numerical data.
- Comparison of models:
- Naive Bayes: Used as a baseline model for fast statistical classification.
- Neural Networks (Deep Learning): Developed our own neural network with multiple layers (Dense, Dropout for regularization), which achieved significantly better results in recognizing complex spam patterns.
- Evaluation with a focus on precision: Since it is crucial in this problem that a legitimate message (ham) is not mistakenly marked as spam, the focus of the evaluation was on the Confusion Matrix and the F1-measure.

## How to start a project?
By installing the necessary libraries and running the .py file available under this repository.

## Results:
- The project successfully demonstrated the supremacy of deep models with proper configuration and regularization.
- High accuracy (Accuracy) and response (Recall) were achieved, which enables reliable filtering of messages in real time.

## Authors:
Ivona Pranjić &
Helena Zvocak,
2024./2025.
