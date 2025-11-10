# Comment Analyze AI (GELİŞTİRME AŞAMASINDA)

YouTube yorumlarını toplamak, duygu analizi (sentiment) ve görüşe-dayalı duygu analizi (Aspect-Based Sentiment Analysis, ABSA) yapmak için hazırlanmış bir Python projesi.

## Özellikler
- YouTube API ile yorum toplama
- Metin ön işleme ve temel duygu analizi (`sentiment_analyze.py`)
- ABSA eğitimi ve değerlendirmesi (BERT tabanlı) (`model_trainer.py`)
- Eğitilmiş modelle tahmin (`absa_predictor.py`)
- Basit web arayüzü (`templates/index.html`) ve komut satırı kullanım senaryoları

## Hızlı Başlangıç

1) Python kurulumu (3.10–3.12 önerilir)

2) Sanal ortam (opsiyonel ama önerilir):

```bash
python -m venv .venv
.venv\Scripts\activate
```

3) Bağımlılıkları yükleyin:

```bash
pip install -r requirements.txt
```

4) Ortam değişkenlerini hazırlayın: `.env` dosyası oluşturun (YouTube API anahtarı vb.). Örnek:

```env
YOUTUBE_API_KEY=YOUR_API_KEY
```

5) Büyük dosyaların Git’e yüklenmesini engelleyin (zaten eklendi):

```text
.gitignore → absa_model/, *.csv, *.safetensors, *.pt, .env, logs/ …
```

Eğer yanlışlıkla büyük dosyaları commit ettiyseniz geçmişi temizlemek için şu komut yardımcı olur:

```bash
git filter-repo --path absa_model --path-glob '*.csv' --invert-paths
git push --force-with-lease origin main
```

## Proje Yapısı

```text
comment-analyze-ai/
├─ main.py                    # Uygulama akışları (ör. servis/CLI)
├─ sentiment_analyze.py       # Temel duygu analizi
├─ model_trainer.py           # ABSA eğitim boru hattı
├─ absa_predictor.py          # Eğitilmiş modelle tahmin
├─ absa_reporter.py           # (Varsa) raporlama yardımcıları
├─ comment_fetcher.py         # Yorum çekme yardımcıları
├─ comment-collector/
│  └─ gather_comments.py      # Toplu yorum toplama
├─ templates/
│  └─ index.html              # Basit UI
├─ stopwords.txt
├─ requirements.txt
├─ etiketli_youtube_yorumları.csv  # (EĞİTİM İÇİN YEREL) Git’e yüklenmez
└─ absa_model/                     # (EĞİTİM ÇIKTISI) Git’e yüklenmez
```

## Veri ve Etiketler
- Girdi CSV şeması (örnek başlıklar): `comment;Aspect;Polarity`
- Aspect etiketleri (`model_trainer.py` içindeki `ASPECT_LABELS`): kanal/videoya dair ön-tanımlı kategoriler
- Polarity: 0=Negatif, 1=Nötr, 2=Pozitif

## Kullanım

### 1) Yorum Toplama
YouTube API anahtarınızı `.env` içine koyduktan sonra:

```bash
python comment-collector/gather_comments.py
# veya
python comment_fetcher.py
```

Toplanan ham yorumları yerel bir CSV’ye yazabilirsiniz (ör. `raw_youtube_comments.csv`). Bu dosyalar Git’e yüklenmez.

### 2) ABSA Eğitimi
Etiketli CSV’nizi (ör. `etiketli_youtube_yorumları.csv`) proje köküne koyun ve çalıştırın:

```bash
python model_trainer.py
```

Eğitim tamamlandığında model ve tokenizer `absa_model/` klasörüne kaydedilir.

Notlar:
- `model_trainer.py` içerisinde veri artırma (pozitif örnekler için basit varyasyonlar) ve sınıf ağırlıkları kullanılır.
- BERT tabanlı `dbmdz/bert-base-turkish-cased` ile etiket sayısı, `Polarity` üzerinden ayarlanır.

### 3) Eğitilmiş Modelle Tahmin

```bash
python absa_predictor.py --text "Video çok bilgilendirici ama ses kötüydü" --aspect "VİDEO_İÇERİK"
```

Komut satırı argümanları dosyaya göre değişebilir; `absa_predictor.py` içindeki yönergeleri takip edin.

### 4) Web Arayüzü / Uygulama
`main.py` veya kendi servis katmanınızı çalıştırarak basit bir arayüz/flow sağlayabilirsiniz:

```bash
python main.py
```

Uygulama, `templates/index.html` ile basit bir etkileşim sunabilir (dosyadaki akışa göre).

## Sorun Giderme
- Windows’ta OneDrive masaüstü yolları Unicode içerdiği için bazı komut satırı araçlarıyla yol sorunları yaşanabilir; terminalinizi proje köküne açmayı veya kısa bir yol kullanmayı deneyin.
- Git push HTTP 408 / büyük dosya hataları: `.gitignore`’ın büyük dosyaları engellediğinden emin olun. Gerekirse geçmişi temizleyin (bkz. “Hızlı Başlangıç” 5. adım).
- CUDA yoksa eğitim CPU’da çalışır; eğitim süresi uzayabilir.

## Lisans
Bu depo kişisel/deneysel amaçlar içindir. Aksi belirtilmedikçe telif hakları saklıdır. Kurumsal/üretim kullanımı veya yeniden dağıtım planlıyorsanız lütfen lisans gereksinimlerini netleştirin.


