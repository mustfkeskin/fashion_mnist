# Fashion-MNIST Keras

https://askubuntu.com/questions/648603/how-to-create-an-animated-gif-from-mp4-video-via-command-line

`Fashion-MNIST` veriseti [Zalando Research](https://jobs.zalando.com/tech/) tarafından MNIST verisetine alternatif olarak oluşturulmuştur. 60.000 tane eğitim 10.000 tane test verisi bulunmaktadır. 28x28 gri seviyesinde 10 farklı sınıftan oluşmaktadır.

<img src="doc/img/fashion-mnist-sprite.png" width="100%">
<img src="doc/img/embedding.gif" width="50%"><img src="doc/img/umap.gif" width="50%">

### Etiketler
Eğitim ve test verisindeki etiktler ve açıklamaları:

| Etiket | Açıklama |
| --- | --- |
| 0 | T-shirt/top |
| 1 | Trouser |
| 2 | Pullover |
| 3 | Dress |
| 4 | Coat |
| 5 | Sandal |
| 6 | Shirt |
| 7 | Sneaker |
| 8 | Bag |
| 9 | Ankle boot |

## Kullanım
### 1) Ubuntu 18.04 için Docker kurulumu: (https://docs.docker.com/install/linux/docker-ce/ubuntu/)  <br><br>
sudo apt-get update <br><br>
sudo apt-get install 
    apt-transport-https 
    ca-certificates 
    curl 
    software-properties-common <br><br>
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - <br><br>
sudo apt-key fingerprint 0EBFCD88 <br><br>
sudo add-apt-repository 
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu 
   $(lsb_release -cs)
   stable" <br><br>
sudo apt-get update <br><br>
sudo apt-get install docker-ce <br><br>

### Örnek Docker
Aşağıdaki komut çalışıyorsa docker kurulumu düzgün bir şekilde yapılmış demektir. <br>
sudo docker run hello-world

### 2) Keras için hazırlanmış Docker'ın kurulması. CPU için docker var. GPU docker'ı gelecekte hazırlanacak.
cd docker <br>
sudo docker build -t keras:cpu -f Dockerfile_keras.cpu . <br>
cd ..

### 3) Docker'ı çalıştırma
sudo docker run -p 8888:8888 -v $(pwd):/notebook -it keras:cpu <br>
Jupyter notebook 8888 portu localhost'a yönlendirilir. <br>
-v komutu lokal makinada hangi dizinde çalıcağını gösterir <br><br>

sudo docker ps -a --> Çalışan container'ların listesini döner <br>
sudo docker logs 2d8b574f5c41(containerID seçilir) --> jupyter notebook için authentication token alınır.  <br>
Hazırlanan docker için token'a gerek yok. 127.0.0.1:8888 adresinden direk ulaşılabilir. <br>
Token üzerinden gitmek isteyenler dockerfile'daki --NotebookApp.token= parametresini silmeleri yeterlidir. :))

### 4) Notebook klasörü altındaki cnn.ipynb seçilerek çalıştırılabilir.

## Sonuçlar
### Multi Layer Perceptron
<img src="doc/img/mlp_confussion_matrix.png" width="50%">

### Simple CNN
<img src="doc/img/simpleCNN_confussion_matrix.png" width="50%">

### CNN with Dropout
<img src="doc/img/cnnDropout_confussion_matrix.png" width="50%">

### CNN + Dropout + BatchNormalization
<img src="doc/img/cnnBatchNorm_confussion_matrix.png" width="50%">

### Simple VGG
<img src="doc/img/simpleVGG_acc.png" width="50%"><img src="doc/img/simpleVGG_confussion_matrix.png" width="50%">

### Simple Inception
<img src="doc/img/simpleInceptcion_acc.png" width="50%"><img src="doc/img/simpleInception_confussion_matrix.png" width="50%">

### Simple Resnet
<img src="doc/img/simpleResnet_acc.png" width="50%"><img src="doc/img/simpleResnet_confussion_matrix.png" width="50%">

### WideResnet
<img src="doc/img/wideResnet_acc.png" width="50%"><img src="doc/img/wideResnet_confussion_matrix.png" width="50%">

### MobileNet_v2
<img src="doc/img/mobilenetv2_acc.png" width="50%"><img src="doc/img/mobilenetv2_confussion_matrix.png" width="50%">



## Boyut İndirgeme Sonuçları

### PCA on Fashion-MNIST
<img src="doc/img/pca.png" width="100%">

### UMAP on Fashion-MNIST
pip install umap-learn <br>
utils/plot_umap.py <br>
<img src="doc/img/fashion-mnist-umap.png" width="100%">

### t-SNE on Fashion-MNIST
Coming soon
