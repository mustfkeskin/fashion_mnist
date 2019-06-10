# Fashion-MNIST Keras

[comment]: # (https://askubuntu.com/questions/648603/how-to-create-an-animated-gif-from-mp4-video-via-command-line)

`Fashion-MNIST` veriseti [Zalando Research](https://jobs.zalando.com/tech/) tarafından MNIST verisetine alternatif olarak oluşturulmuştur. 60.000 tane eğitim 10.000 tane test verisi bulunmaktadır. 28x28 gri seviyesinde 10 farklı sınıftan oluşmaktadır.

<details><summary>İçerik</summary><p>

* [Etiketler](#etiketler)
* [Kurulum](#kurulum)
* [Sonuclar](#sonuclar)
* [Görsellestirme](#tahminleri-gorsellestirme)
* [Kohen Kappa Scores](#kohen-kappa-scores)

</p></details><p></p>


<img src="doc/img/fashion-mnist-sprite.png" width="100%"><img src="doc/img/embedding.gif" width="50%"><img src="doc/img/umap.gif" width="50%">


## Etiketler
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

## Kurulum
### 1) Ubuntu 18.04 için Docker kurulumu: (https://docs.docker.com/install/linux/docker-ce/ubuntu/)  <br><br>
* sudo apt-get update <br><br>
* sudo apt-get install 
    apt-transport-https 
    ca-certificates 
    curl 
    software-properties-common <br><br>
* curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add - <br><br>
* sudo apt-key fingerprint 0EBFCD88 <br><br>
* sudo add-apt-repository 
   "deb [arch=amd64] https://download.docker.com/linux/ubuntu 
   $(lsb_release -cs)
   stable" <br><br>
* sudo apt-get update <br><br>
* sudo apt-get install docker-ce <br><br>

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
Token üzerinden gitmek isteyenler dockerfile'daki --NotebookApp.token= parametresini silmeleri yeterlidir. <br>
 :))

### 4) Notebook klasörü altındaki cnn.ipynb seçilerek çalıştırılabilir.

## Sonuclar
Sonuçlar kısmında her bir algoritma için ilk satırda yer alan accuracy, loss ve confussion matrisleri 20 epoch ve data augmentation yapılmamış sonuçları anlatamaktadır. <br>
İkinci satırdaki sonuçlar ise 50 epoch ve data augmentation sonucu elde edilen sonuçları içermektedir. Ek olarak ikinci satırda overfit olmasını engellemek için EarlyStopping kullanılmıştır. <br>
Test sonuçlarında yer alan tabloda '_dataAug' data augmentation yapıldığı anlamına gelmektedir.

### Test Sonuçları
| Model Adı               | test_acc |
|-------------------------|----------|
| simpleVGG_dataAug       | 0.9439   |
| majority_voting         | 0.9429   |
| simpleVGG               | 0.9366   |
| wideResnet_dataAug      | 0.9333   |
| CNNBatchNorm            | 0.9333   |
| wideResnet              | 0.927    |
| simpleResnet_dataAug    | 0.925    |
| CNNDropout              | 0.921    |
| stacking                | 0.9204   |
| simpleResnet            | 0.9181   |
| simpleInception         | 0.9167   |
| mobileNetV2_dataAug     | 0.9157   |
| mobileNetV2             | 0.9145   |
| simpleCNN_dataAug       | 0.9097   |
| simpleCNN               | 0.9076   |
| CNNBatchNorm_dataAug    | 0.9049   |
| mlp                     | 0.8917   |
| CNNDropout_dataAug      | 0.8874   |
| simpleInception_dataAug | 0.7164   |

### Multi Layer Perceptron
<img src="doc/img/mlp_acc.png" width="50%"><img src="doc/img/mlp_confussion_matrix.png" width="50%">

### Simple CNN
<img src="doc/img/simpleCNN_acc.png" width="50%"><img src="doc/img/simpleCNN_confussion_matrix.png" width="50%">
<img src="doc/img/simpleCNN_acc_dataAug.png" width="50%"><img src="doc/img/simpleCNN_confussion_matrix_dataAug.png" width="50%">

### CNN with Dropout
<img src="doc/img/cnnDropout_acc.png" width="50%"><img src="doc/img/cnnDropout_confussion_matrix.png" width="50%">
<img src="doc/img/cnnDropout_acc_dataAug.png" width="50%"><img src="doc/img/cnnDropout_confussion_matrix_dataAug.png" width="50%">

### CNN + Dropout + BatchNormalization
<img src="doc/img/cnnBatchNorm_acc.png" width="50%"><img src="doc/img/cnnBatchNorm_confussion_matrix.png" width="50%">
<img src="doc/img/cnnBatchNorm_acc_dataAug.png" width="50%"><img src="doc/img/cnnBatchNorm_confussion_matrix_dataAug.png" width="50%">

### Simple VGG
<img src="doc/img/simpleVGG_acc.png" width="50%"><img src="doc/img/simpleVGG_confussion_matrix.png" width="50%">
<img src="doc/img/simpleVGG_acc_dataAug.png" width="50%"><img src="doc/img/simpleVGG_confussion_matrix_dataAug.png" width="50%">

### Simple Inception
<img src="doc/img/simpleInceptcion_acc.png" width="50%"><img src="doc/img/simpleInception_confussion_matrix.png" width="50%">
<img src="doc/img/simpleInception_acc_dataAug.png" width="50%"><img src="doc/img/simpleInception_confussion_matrix_dataAug.png" width="50%">

### Simple Resnet
<img src="doc/img/simpleResnet_acc.png" width="50%"><img src="doc/img/simpleResnet_confussion_matrix.png" width="50%">
<img src="doc/img/simpleResnet_acc_dataAug.png" width="50%"><img src="doc/img/simpleResnet_confussion_matrix_dataAug.png" width="50%">

### WideResnet
<img src="doc/img/wideResnet_acc.png" width="50%"><img src="doc/img/wideResnet_confussion_matrix.png" width="50%">
<img src="doc/img/wideResnet_acc_dataAug.png" width="50%"><img src="doc/img/wideResnet_confussion_matrix_dataAug.png" width="50%">

### MobileNet_v2
<img src="doc/img/mobilenetv2_acc.png" width="50%"><img src="doc/img/mobilenetv2_confussion_matrix.png" width="50%">
<img src="doc/img/mobilenetv2_acc_dataAug.png" width="50%"><img src="doc/img/mobilenetv2_confussion_matrix_dataAug.png" width="50%">


## Kohen Kappa Scores
Kappa skorları incelendiğinde modellerin birbirine benzer sonuçlar ürettiği için ensemble tekniklerinin stacking ve majority voting normal algoritmaların performansını geçemediği görülmüştür.
<img src="doc/img/kappa_scores.png" width="100%">


## Tahminleri Gorsellestirme
İlk 3 satır doğru tahminleri, sonraki 3 satır hatalı tahminleri göstermektedir. <br>
<img src="doc/img/correct_results.png" width="50%">
<img src="doc/img/incorrect_results.png" width="50%">

## Boyut İndirgeme Sonuçları

### PCA on Fashion-MNIST
<img src="doc/img/pca.png" width="100%">

### UMAP on Fashion-MNIST
pip install umap-learn <br>
utils/plot_umap.py <br>
<img src="doc/img/fashion-mnist-umap.png" width="100%">

### t-SNE on Fashion-MNIST
Yakında gelecek