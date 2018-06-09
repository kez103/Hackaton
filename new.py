#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import re
import snowballstemmer
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import pickle

plt.rc('font', family='Verdana')
# очистка текста с помощью regexp


def text_cleaner(text):
    stemmer = snowballstemmer.stemmer('russian')
    text = text.lower()  # приведение в lowercase,
    text = re.sub(r'https?://[\S]+', ' url ', text)  # замена интернет ссылок
    text = re.sub(r'[\w\./]+\.[a-z]+', ' url ', text)
    text = re.sub(r'<[^>]*>', ' ', text)  # удаление html тагов
    text = re.sub(r'[\W\n]+', ' ', text)  # удаление лишних символов
    text = re.sub(r'\w*\d\w*', '', text)  # замена цифр
    text = re.sub(r'\w*[.]\w*', '', text)  # замена цифр
    text = ' '.join(stemmer.stemWords(text.split()))  # Выделение корней
    return text


reviews_train = load_files("train/", encoding='utf-8')
# text_train, y_train = reviews_train.data, reviews_train.target


english = ['на', 'не', 'при', 'во', 'до', 'прош', 'ден', 'предоставить', 'необходим',
           'url', 'для', 'нет', 'просьб', 'добр', 'комментар', 'msk', 'филиал', 'lg', 'gf'
           'от', 'за', 'что', 'как', 'из', 'спасиб', 'ил', 'дан',
           'коллег', 'котор', 'был', 'без']

count_vect = CountVectorizer(min_df=1, stop_words=english, preprocessor=text_cleaner)
X_train_counts = count_vect.fit_transform(reviews_train.data)
# print("Словарь:\n{}".format(X_train_counts.shape))
feature_names = count_vect.get_feature_names()

# print(type(feature_names))
print("Количество признаков: {}".format(len(feature_names)))


x, y = reviews_train.data, reviews_train.target
text_clf = Pipeline([
                    ('countvect', CountVectorizer(min_df=2, stop_words=english, preprocessor=text_cleaner)),
                    ('tfidf', TfidfTransformer()),
                    ('clf', SGDClassifier(loss='hinge')),
                    ])

text_clf.fit(x, y)


twenty_test = load_files("test/", encoding='utf-8')
predicted = text_clf.predict(twenty_test.data)

# # print(twenty_test.target)

# print("Accuracy: {:.3f}".format(accuracy_score(twenty_test.target, predicted)))
# print("Точность:\n{}".format(np.mean(predicted == twenty_test.target)))


# docs = ['Сегодня утром мне приспичило отдохнуть на Мальдивах. Я слышал, что отпуск в Вашей компании необходимо перед этим подписать приказ. приказ это. Очень хочу отдохнуть']
docs = ['Добрый день! Сегодня утором я почувствовал острую необходимость своего организма в отпуске на Мальдивах. Посему прошу подписать приказ об отпуске']
predicted = text_clf.predict(docs)
# print(predicted)
print('%r => %s' % (docs, reviews_train.target_names[predicted[0]]))
print('{}'.format(np.mean(predicted == twenty_test.target)))

# print('{} => {} : {}' .format(docs, reviews_train.target_names[predicted[0]],np.mean(predicted == reviews_train.target)))
# print('{}'.format(np.mean(predicted == twenty_test.target)))


# matrix_freq = np.asarray(X_train_counts.sum(axis=0)).ravel()
# final_matrix = np.array([np.array(count_vect.get_feature_names()), matrix_freq])
# li = list(final_matrix.transpose())
# li.sort(key=lambda x: int(x[1]))
# print(li)

# print(twenty_test.target)
# print(predicted)
# conf_mat = confusion_matrix(twenty_test.target, predicted)
# print(conf_mat)

# save the model to disk
filename = 'finalized_model.sav'
pickle.dump(text_clf, open(filename, 'wb'))

filename2 = 'categories.sav'
pickle.dump(reviews_train.target_names, open(filename2, 'wb'))

filename3 = 'targets.sav'
pickle.dump(reviews_train.target, open(filename3, 'wb'))

loaded_model = pickle.load(open(filename, 'rb'))
# result = loaded_model.score(X_test, Y_test)
# print(result)

predicted = loaded_model.predict(twenty_test.data)

# print(twenty_test.target)

print("Accuracy: {:.3f}".format(accuracy_score(twenty_test.target, predicted)))
print("Точность:\n{}".format(np.mean(predicted == twenty_test.target)))
print(reviews_train.target_names)
