import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from nltk.tokenize.toktok import ToktokTokenizer
import re
import nltk
import torch
import torchtext


ABSTAIN = -1
positive = 1
negative = 0

row = 0
# positive signals
positive_signals = ["good", "wonderful", "amazing", "excellent", "great"]


# negative signals
negative_signals = ["bad", "horrible", "sucks", "awful", "terrible"]

# initialize W matrix with dimensions 10 x 100000
W = np.full(shape=(10, 50000), fill_value=-1)

# read and process stopwords
f = open("stopwords")
wordList = f.readlines()
stopwords = []

for stopword in wordList:
    stopwords.append(stopword.strip('\n'))

tokenizer = ToktokTokenizer()


def strip_html(text):
    soup = BeautifulSoup(text, "html.parser")
    return soup


def remove_between_square_brackets(text):
    letters_only = re.sub('\[[^]]*\]', " ", str(text))
    return letters_only.lower()


def preprocess_text(text):
    text = strip_html(text)
    text = remove_between_square_brackets(text)
    return text


def remove_special_characters(text):
    pattern = r'[^a-zA-z0-9\s]'
    text = re.sub(pattern, '', text)
    return text


def stemming(text):
    ps = nltk.porter.PorterStemmer()
    text = ' '.join([ps.stem(word) for word in text.split()])
    return text


def remove_stopwords(text):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token not in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text


def positive_labelling_function(text, pos_signal):
    words = text.split()
    if pos_signal in words:
        return positive
    else:
        return ABSTAIN


def negative_labelling_function(text, neg_signal):
    words = text.split()
    if neg_signal in words:
        return negative
    else:
        return ABSTAIN


# read dataset and perform pre-processing
dataset = pd.read_csv("IMDB_Dataset.csv")
for i in range(len(dataset)):
    if dataset["sentiment"][i] == "positive":
        dataset["sentiment"][i] = 1
    else:
        dataset["sentiment"][i] = 0

dataset["review"] = dataset["review"].apply(preprocess_text)
dataset["review"] = dataset["review"].apply(remove_special_characters)
dataset["review"] = dataset["review"].apply(remove_stopwords)

# constructing W matrix
for positive_signal in positive_signals:
    for index in range(len(dataset)):
        W[row][index] = positive_labelling_function(dataset["review"][index], positive_signal)

    row += 1

for negative_signal in negative_signals:
    for index in range(len(dataset)):
        W[row][index] = negative_labelling_function(dataset["review"][index], negative_signal)

# constructing A matrix; A = (1-2wi)

rows = W.shape[0]
columns = W.shape[1]

A = np.zeros(shape=(rows, columns))

for row in range(rows):
    for column in range(columns):
        if W[row][column] != ABSTAIN:
            A[row][column] = 1-2*W[row][column]

# constructing expected error rate matrix; expected error rate = maximum error rate = 1/number of classes
expected_errorRate = np.full(shape=(rows, 1), fill_value=0.5)

# constructing n matrix
n = np.zeros(shape=(rows, 1))

for row in range(rows):
    sum = 0
    for column in range(columns):
        if W[row][column] != ABSTAIN:
            sum += 1

    n[row][0] = sum

# constructing c matrix = expected error rate - sum of weak signals
c = np.zeros(shape=(rows, 1))
mat = np.zeros(shape=(rows, 1))

# n_i * e_i
for row in range(rows):
    mat[row][0] = n[row][0]*expected_errorRate[row][0]

print("shape of mat: ", mat.shape)

c = np.subtract(mat, n)

# constructing y vector
# y = np.random.uniform(0, 1, size=(50000, 1))

print("dimensions of W matrix: ", W.shape)
print("dimensions of A matrix: ", A.shape)
print("dimensions of c matrix: ", c.shape)

# converting numpy arrays to torch tensors
A_tensor = torch.from_numpy(A).float()
c_tensor = torch.from_numpy(c).float()

# constructing y torch tensor
y_tensor = torch.rand(size=(50000, 1), requires_grad=True, dtype=torch.float)

# define optimizer
learning_rate = 0.1
num_epochs = 200

optimizer = torch.optim.Adagrad([y_tensor], lr=learning_rate)
loss_function = torch.nn.MSELoss(reduction='mean')

# print("random y_tensor: ", y_tensor[:20])

for epoch in range(num_epochs):
    y_hat = torch.matmul(A_tensor, y_tensor)
    loss = loss_function(y_hat, c_tensor)

    if epoch % 50 == 0:
        print("loss value: ", loss)

    loss.backward()
    optimizer.step()
    optimizer.zero_grad()


print("final output data dimension: ", y_tensor.shape)

# calculating accuracy
y_prediction = np.zeros(shape=(50000, 1))
count = 0
n = len(dataset)


for index in range(len(y_tensor)):
    if y_tensor[index][0] > 0.5:
        y_prediction[index][0] = 1

for index in range(len(dataset)):
    if y_prediction[index][0] == dataset["sentiment"][index]:
        count += 1

print("Accuracy of the prediction: ", count/n)



