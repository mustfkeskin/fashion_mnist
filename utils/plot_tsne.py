from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np


data_train = pd.read_csv('../Data/fashion-mnist_train.csv')

X     = np.array(data_train.iloc[:, 1:])
color = np.array(data_train.iloc[:, 0])


model     = TSNE(n_components=2, random_state=0, perplexity=30, learning_rate=200, n_iter=1000)
embedding = model.fit_transform(X)


sns.set(context="paper", style="white")

fig, ax = plt.subplots(figsize=(12, 10))
plt.scatter(
    embedding[:, 0], embedding[:, 1], c=color, cmap="Spectral", s=0.1
)
plt.setp(ax, xticks=[], yticks=[])
plt.title("Fashion MNIST verisinin TSNE algoritması ile 2D hale getirilmesi", fontsize=18)

plt.show()