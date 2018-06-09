import numpy as np
from sklearn.datasets import load_files
from sklearn.metrics import accuracy_score
import pickle
import re
import snowballstemmer


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


twenty_test = load_files("test/", encoding='utf-8')

filename = 'finalized_model.sav'
loaded_model = pickle.load(open(filename, 'rb'))

filename2 = 'categories.sav'
names = pickle.load(open(filename2, 'rb'))

filename3 = 'targets.sav'
target = pickle.load(open(filename3, 'rb'))


# result = loaded_model.score(X_test, Y_test)
# print(result)

predicted = loaded_model.predict(twenty_test.data)
# print(twenty_test.target)

print("Accuracy: {:.3f}".format(accuracy_score(twenty_test.target, predicted)))
print("Точность:\n{}".format(np.mean(predicted == twenty_test.target)))

docs = ['Из-за частых аварийных ситуации на платформе PHLR, прошу организовать сетевой доступ между площадками Новосибирска и Самары по портам 2222 и 77669 для организации георезервирования сервиса']
predicted = loaded_model.predict(docs)
# print(predicted)
print('%r => %s' % (docs, names[predicted[0]]))

print('{} => {} : {}'.format(docs, names[predicted[0]], np.mean(predicted == target)))
print('{}'.format(np.mean(predicted == twenty_test.target)))
