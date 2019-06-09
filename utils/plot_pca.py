from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

labels = { 0: 'T-shirt/top', 1: 'Trouser', 2: 'Pullover', 3: 'Dress', 4: 'Coat', 
          5: 'Sandal', 6: 'Shirt', 7: 'Sneaker', 8 : 'Bag', 9 : 'Ankle boot'}


data_train = pd.read_csv('../Data/fashion-mnist_train.csv')

X     = np.array(data_train.iloc[:, 1:])
color = np.array(data_train.iloc[:, 0])

pca = PCA(n_components=2)
embedding = pca.fit_transform(X)

df = pd.DataFrame(embedding, columns=('x', 'y'))
df["class"] = color
df["class"].replace(labels, inplace=True)

df['x'] = embedding[:,0]
df['y'] = embedding[:,1]
df["color"] = np.array(data_train.iloc[:, 0])

plt.figure(figsize=(16,10))
sns.scatterplot(
    x="x", y="y",
    hue="color",
    palette=sns.color_palette("hls", 10),
    data=df,
    legend="full",
    alpha=0.3
)