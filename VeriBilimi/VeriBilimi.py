import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, precision_score,
    recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier  # Yapay Sinir Ağı (MLP) ekleniyor


# Veri Yükleme işlemö
# Bu veri seti, çevrimiçi alışveriş yapan kişilerin davranışlarını ve alışveriş yapıp yapmadıklarını içeriyor.
df = pd.read_csv("C:/Users/brkfa/OneDrive/Masaüstü/online_shoppers_intention.csv")


# Makine öğrenmesi algoritmaları sadece sayısal verilerle çalışır. Bu yüzden:
# Kategorik Veriler sayısal verilere dönüştürüldü ve
# Makine öğrenmesi algoritmalarının kullanılabileceği forma getirilmiş oldu.
le = LabelEncoder() # LabelEncoder sınıfından le adlı bir nesne yani dönüştürücü oluşturuldu.

df['Month'] = le.fit_transform(df['Month']) # fit ifadesi veri üzerinde inceleme yapar(eşsiz değerleri bulur)

df['VisitorType'] = le.fit_transform(df['VisitorType']) # transform ise öğrendiklerine göre veriyi dönüştürür.
# Yukarıdaki month ve visitorType verileri string veri türünde oldukları için fit ve transorm yöntemleri uygulandı.


#Aşağıdaki weekend ve revenue verileri ise boolean(t/f) veri türünde oldukları için direkt integer'a(0/1) dönüşümleri sağlandı
df['Weekend'] = df['Weekend'].astype(int)
df['Revenue'] = df['Revenue'].astype(int)
# Month ve VisitorType gibi metin içeren sütunlar LabelEncoder ile sayılara çevrildi.
# Weekend ve Revenue sütunları da True/False değil 0/1 olarak dönüştürüldü.



# Özellikler ve Hedef
X = df.drop('Revenue', axis=1) # drop komutu Revenue sütununu veriden çıkarır çünkü
                               # Revenue bizim tahmin etmek istediğimiz şey yani hedef sütunu

y = df['Revenue'] # Çıktı yani modelin tahmin etmeye çalıştığı şey.


# Normalizasyon
scaler = MinMaxScaler() # Tüm sayısal değerleri 0 ile 1 arasında sıkıştırır.
X_scaled = scaler.fit_transform(X)

# Veriyi Ayırma işlemi modelin gördüğü veriyi ezberlemeyip genelleme yapması için önemlidir.
# Eğitim verisi ile model öğrenir.
# Test verisi ile modelin başarısı kontrol edilir.
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y # Verinin %20si test için ayrılacak.
) # random_state ifadesi aynı sonuçları almak için sabit rastgelelik sağlar.
  # stratify ifadesi ise eğitim ve test setindeki sınıf dağılımını korur.


# 7 farklı makine öğrenmesi algoritmasını içeren dictionary tanımı.
models = {
    "KNN": KNeighborsClassifier(n_neighbors=5), # Eğitim verisindeki en yakın 5 komşuya bakar, en çok hangi sınıf varsa onu tahmin eder.

    "Decision Tree": DecisionTreeClassifier(random_state=42),

    "SVM": SVC(probability=True), # probability=True ifadesi ile Tahmin sonucu sadece 0/1 değil, aynı zamanda olasılık verebilmesi sağlanır.
                                  # SVM “Her iki sınıfa da en uzakta kalan çizgiyi yani en güvenli ayırıcı sınırı bulmaya çalışır.

    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42), # n_estimators=100: 100 tane karar ağacı kurar. Her ağaç ayrı karar verir, çoğunluk ne diyorsa o olur

    "Naive Bayes": GaussianNB(), #  Olasılık temelli çalışır. Özellikler birbirinden bağımsızmış gibi varsayar.

    "Logistic Regression": LogisticRegression(max_iter=1000), # Veriyi sigmoid fonksiyonuyla 0-1 arası değerlere sıkıştırarak sınıflandırma yapar.
                                                              # max_iter=1000 ise zor problemler için daha fazla deneme hakkı verir

    "Neural Network": MLPClassifier(hidden_layer_sizes=(100,), max_iter=1000, random_state=42)  # Yapay Sinir Ağı (MLP).
    # Girdi alır - Üzerinde matematiksel işlem yapar - Sonraki katmana iletir
    # hidden_layer_sizes=(100,) 100 nöronlu bir gizli katman kullanır.
    # max_iter=1000: ile de öğrenmesi için 1000 adım (iterasyon) tanınır.
}

# Sonuçları tutmak için
metrics = {
    "Model": [],
    "Accuracy": [],  # Doğruluk
    "Precision": [], # Kesinlik
    "Recall": [],    # Duyarlılık
    "F1 Score": []   # Kesinlik ve Duyarlılığın dengeli ortalaması
}


roc_data = {} # ROC eğrisi çizmek için gerekli olan FPR, TPR ve AUC değerlerini saklayacağımız boş bir sözlük.
# fpr - false positive rate (yanlış alarmlar)
# tpr - true positive rate (doğru alarmlar)
# roc_auc - eğri altında kalan alan (modelin ne kadar iyi ayrılabildiğini gösterir)


# Eğitim ve Değerlendirme
for name, model in models.items(): # Tüm modeller döngü yardımıyla teker teker eğitilip test ediliyor.
    model.fit(X_train, y_train)    # Modeli eğitmek amacıyla modeli inceliyor.

    # Özel durum: LinearRegression çıktılarını sınıflandırmaya dönüştür
    y_pred = model.predict(X_test) # Eğitim tamamlandıktan sonra test verisiyle tahmin yapar.


    # Bu satırlarda model isimleri metrics sözlüğünün "Model" listesine eklenr.
    # average='macro' ifadesi ile her sınıfın skoru eşit ağırlıklı ortalama ile alır.
    metrics["Model"].append(name)
    metrics["Accuracy"].append(accuracy_score(y_test, y_pred))
    metrics["Precision"].append(precision_score(y_test, y_pred, average='macro'))
    metrics["Recall"].append(recall_score(y_test, y_pred, average='macro'))
    metrics["F1 Score"].append(f1_score(y_test, y_pred, average='macro'))


    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred) # Modelin tahminlerinin doğru ve yanlış olduğu durumları bir 2x2 tablo olarak verir.

    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['No Purchase', 'Purchase'])  # Etiketleri özelleştir
    # Bu sınıf, confusion matrix'i görsel olarak çizmeye yarar.
    # 'No Purchase', 'Purchase' ifadeleri ile 0 ve 1 sınıfına anlamlı isimler veriliyor.

    disp.plot(cmap='Blues') # Matrisin görselini oluşturur ve görselde mavi renk tonları kullanılır .
    plt.title(f"{name} - Confusion Matrix")
    plt.grid(False) # Kılavuz çizgileri kapatılır.
    plt.show()


    # ROC eğrisi, bir modelin sınıflandırmadaki başarısını olası tüm eşik (threshold) değerleri için gösteren bir grafik türüdür.
    # ROC eğrisi verisi
    if hasattr(model, "predict_proba"):
        y_probs = model.predict_proba(X_test)[:, 1] # predict_proba Model tarafından sağlanan olasılık tahminine göre,
                                                    # ilgili örneğin hedef değişkenin pozitif sınıfına (Revenue = 1) ait olma ihtimali %87’dir.

    elif hasattr(model, "decision_function"):   # decision_function() denen başka bir yöntemle, karar sınırına olan uzaklık bulunur.
        y_probs = model.decision_function(X_test) # y_probs, her test örneği için pozitif sınıfa (Revenue = 1) ait olma ihtimallerinin listesidir.
        y_probs = (y_probs - y_probs.min()) / (y_probs.max() - y_probs.min()) # ROC çizmek için normalize edilir.
    else:
        y_probs = np.zeros_like(y_pred) # y_pred ile aynı boyutta ama bütün elemanları sıfır olan bir NumPy dizisi (array) oluşturur.
                                        # Bu,olasılık değeri sağlayamayan modeller için bir yedek (default) değer üretir.

    fpr, tpr, _ = roc_curve(y_test, y_probs) # Bu fonksiyon, farklı eşik değerlerinde True Positive Rate ve False Positive Rate hesaplar.
    # sadece fpr ve tpr ROC çizimi için yeterli, eşikler çizim için kullanılmayacak. Bu yüzden _ kullanıldı.
    roc_auc = auc(fpr, tpr) # ROC eğrisinin altında kalan alanı (AUC) hesaplar.
    roc_data[name] = (fpr, tpr, roc_auc) # her model için hesaplanan fpr tpr ve auc değerlerini bir sözlükte saklıuyor.

# Metrikleri DataFrame’e çevir
metrics_df = pd.DataFrame(metrics)

# Skorları Görselleştir
# Veriyi grafikte kolay çizilebilecek uzun formata çevirir.
melted_df = metrics_df.melt(id_vars='Model', var_name='Metrik', value_name='Skor')

plt.figure(figsize=(14, 6))
sns.barplot(data=melted_df, x='Model', y='Skor', hue='Metrik', palette='bright')

plt.title('Modellerin Accuracy, Precision, Recall ve F1 Skorları')
plt.ylim(0, 1) # Y ekseninin minimum ve maksimum sınırlarını belirler
plt.ylabel('Skor')
plt.grid(axis='y', linestyle='--', alpha=0.7) # alpha ile şeffaflık derecesi ayarlanır

# bbox_to_anchor=(1.05, 1) - Kutuyu grafiğin dışına, sağ üst köşeye taşır.
# loc='upper left' - Kutunun konumunu sol üstte sabitler ama (1.05, 1) konumunda.
plt.legend(title='Metrik', bbox_to_anchor=(1.05, 1), loc='upper left')

plt.tight_layout() # Grafik elemanlarının (başlık, etiketler, çubuklar, yazılar) birbirine çakışmasını engeller.
plt.show()

# ROC Eğrisi çizimi
plt.figure(figsize=(10, 6))
for name, (fpr, tpr, roc_auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.2f})')

plt.plot([0, 1], [0, 1], 'k--', label='Şans (AUC = 0.5)') # Bu, rastgele tahmin yapan modelin ROC eğrisidir (köşegen çizgi)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Eğrisi')
plt.legend(loc='lower right') # Bu komut, grafikte kullanılan renklerin veya çizgilerin neyi temsil ettiğini gösteren küçük bir kutu ekler.
plt.grid(True)
plt.show()
