import os
import re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
import seaborn as sns
from sklearn.metrics import confusion_matrix

def load_train_test_imdb_data(data_dir):
    data = {}
    for split in ["train", "test"]:
        data[split] = []
        for sentiment in ["neg", "pos"]:
            score = 1 if sentiment == "pos" else 0

            path = os.path.join(data_dir, split, sentiment)
            file_names = os.listdir(path)
            for f_name in file_names:
                with open(os.path.join(path, f_name), "r") as f:
                    review = f.read()
                    data[split].append([review, score])

    np.random.shuffle(data["train"])        
    data["train"] = pd.DataFrame(data["train"],
                                 columns=['text', 'sentiment'])
    print(data["train"])
    np.random.shuffle(data["test"])
    data["test"] = pd.DataFrame(data["test"],
                                columns=['text', 'sentiment'])
    print(data["test"])
    return data["train"], data["test"]

train_data, test_data = load_train_test_imdb_data(data_dir="aclImdb/")

def clean_text(text):
    # remove HTML tags
    text = re.sub(r'<.*?>', '', text)
    # remove the characters [\], ['] and ["]
    text = re.sub(r"\\", "", text)    
    text = re.sub(r"\'", "", text)    
    text = re.sub(r"\"", "", text)    
    
    # convert text to lowercase
    text = text.strip().lower()
    
    # replace punctuation characters with spaces
    filters='!"\'#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n'
    translate_dict = dict((c, " ") for c in filters)
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)

    return text


import re
# Transform each text into a vector of word counts
vectorizer = TfidfVectorizer(stop_words="english",
                             preprocessor=clean_text,
                             ngram_range=(1, 1))
training_features = vectorizer.fit_transform(train_data["text"]) 
# print('Training Features' + str(training_features))
test_features = vectorizer.transform(test_data["text"])
# print('Testing Features' + str(test_features))
# Training
model = LinearSVC()
model.fit(training_features, train_data["sentiment"])
y_pred = model.predict(test_features)
# print(y_pred)
# Evaluation
acc = accuracy_score(test_data["sentiment"], y_pred)
print("Accuracy on the IMDB dataset: {:.2f}".format(acc*100))
#confusion matrix
matrix = confusion_matrix(test_data["sentiment"], y_pred)
# sns.heatmap(matrix, annot=True)
sns.heatmap(matrix/np.sum(matrix), annot=True, 
            fmt='.2%', cmap='Blues')
# print(matrix)


# Transform each text into a vector of word counts
vectorizer = CountVectorizer(stop_words="english",
                             preprocessor=clean_text,
                             ngram_range=(1,2)
                             )
training_features = vectorizer.fit_transform(train_data["text"]) 
# print('Training Features' + str(training_features))
test_features = vectorizer.transform(test_data["text"])
# print('Testing Features' + str(test_features))
# Training
model = LinearSVC()
model.fit(training_features, train_data["sentiment"])
y_pred = model.predict(test_features)
# print(y_pred)
# Evaluation
acc = accuracy_score(test_data["sentiment"], y_pred)
print("Accuracy on the IMDB dataset: {:.2f}".format(acc*100))
#confusion matrix
matrix = confusion_matrix(test_data["sentiment"], y_pred)
# sns.heatmap(matrix, annot=True)
sns.heatmap(matrix/np.sum(matrix), annot=True, 
            fmt='.2%', cmap='Blues')
# print(matrix)