# Veri Cambazı

Veri Cambazı, veri analizi, görselleştirme ve temel makine öğrenmesi işlemlerini kullanıcı dostu bir arayüzde gerçekleştirebileceğiniz bir Python uygulamasıdır. Uygulama, veri bilimi süreçlerinin çoğunu kolayca yapabilmenizi sağlar: veri yükleme, ön işleme, istatistiksel analiz, görselleştirme ve temel modelleme.

## Kullanılan Kütüphaneler

- **tkinter**: Grafik arayüz (GUI) oluşturmak için
- **ttk, filedialog, messagebox, simpledialog**: Gelişmiş arayüz ve dosya işlemleri
- **pandas, numpy**: Veri işleme ve analiz
- **matplotlib, seaborn**: Grafik ve veri görselleştirme
- **os, io, pickle**: Dosya işlemleri ve veri saklama
- **Pillow (PIL)**: Görüntü işlemleri
- **scikit-learn (sklearn)**: Veri ölçekleme, modelleme ve değerlendirme
  - MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler
  - LinearRegression, KNeighborsClassifier, KMeans
  - train_test_split, çeşitli metrikler (mean_absolute_error, accuracy_score, vb.)

## Özellikler

- Farklı dosya formatlarını hızlıca yükleme ve görüntüleme
- Veri ön işleme ve ölçekleme seçenekleri
- Temel istatistiksel analizler
- Grafiksel veri görselleştirme (histogram, korelasyon, dağılım grafiği vs.)
- Doğrusal regresyon ile tahmin
- KNN ile sınıflandırma analizi
- K-Means ile kümeleme
- Model değerlendirme metrikleri (MAE, MSE, RMSE, R2, accuracy vs.)

## Kurulum

1. Bu depoyu klonlayın:
    ```bash
    git clone https://github.com/tunasamet/veri-cambazi.git
    cd veri-cambazi
    ```
2. Gerekli paketleri yükleyin:
Sürüm uyuşmazlığı yaşayacağınızı düşünüyorsanız veya yaşarsanız requirements.txt içerisinden kütüphane sürümlerine göre indirebilirsiniz.
    ```bash
    pip install numpy pandas matplotlib seaborn Pillow scikit-learn
    ```

3. Uygulamayı başlatın:
    ```bash
    python app.py
    ```

## Kullanım

- Uygulama arayüzünden veri dosyanızı seçin.
- Analiz etmek istediğiniz işlemleri menülerden seçerek uygulayın.
- Sonuçları hem grafik hem de tablo olarak görebilirsiniz.
- Değiştirdiğiniz veriyi kaydedebilirsiniz.


Her türlü soru, öneri ve geri bildirim için iletişime geçebilirsiniz.

## Lisans

Bu proje [MIT Lisansı](LICENSE) ile lisanslanmıştır.