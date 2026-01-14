import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import warnings
import re
warnings.filterwarnings("ignore")

# --- YAPILANDIRMA ---
MODEL_PATH = "./absa_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Modelimizin bildiği varlıklar
ASPECT_LABELS = [
    "KANAL_SAHİBİ_İMAJI",
    "KONUK_İMAJI",
    "BAHSEDİLEN_KİŞİ_İMAJI",
    "VİDEO_İÇERİK",
    "ÜRETİM_KALİTESİ",
    "KANAL_YÖNETİMİ",
    "DİĞER/ALAKASIZ"
]

# Sayısal değerlerin metin karşılıkları
POLARITY_LABELS = {
    0: "NEGATİF",
    1: "NÖTR",
    2: "POZİTİF"
}

# --- MODEL YÜKLEME ---
print(f"Sistem: {DEVICE} üzerinde çalışıyor.")
print("ABSA modeli yükleniyor...")

try:
    if os.path.exists(MODEL_PATH):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
        print("BAŞARILI: Eğitilmiş ABSA modeli yüklendi!")
    else:
        print("UYARI: Eğitilmiş model bulunamadı, 'savasy' modeli baz alınıyor...")
        model2 = "savasy/bert-base-turkish-sentiment-cased"
        tokenizer = AutoTokenizer.from_pretrained(model2)
        model = AutoModelForSequenceClassification.from_pretrained(model2, num_labels=3, ignore_mismatched_sizes=True)
        model.to(DEVICE)
        model.eval()
        print("DURUM: Varsayılan model yüklendi.")
except Exception as e:
    print(f"HATA: Model yüklenirken bir sorun oluştu: {e}")
    tokenizer = None
    model = None

# --- YARDIMCI FONKSİYONLAR ---

def preprocess_text(text):
    """Metni analiz öncesi temizler ve normalize eder."""
    if not text or not isinstance(text, str):
        return ""
    
    text = text.strip()
    text = re.sub(r'\s+', ' ', text) # Fazla boşluklar
    text = re.sub(r'@\w+', '@user', text) # Kullanıcı adları
    text = re.sub(r'http\S+', 'http', text) # URL'ler
    text = re.sub(r'(.)\1{2,}', r'\1', text) # Karakter tekrarı
    
    return text

def analyze_sentiment_absa(comment: str, aspect: str) -> dict:
    """Tek bir yorum ve varlık için duygu tahmini yapar."""
    if not model or not tokenizer:
        return {"error": "Model aktif değil"}
    
    try:
        text = preprocess_text(comment)
        # ABSA Formatı: [Yorum] [Ayraç] [Varlık]
        input_text = f"{text} {tokenizer.sep_token} {aspect}"
        
        encoded_input = tokenizer(
            input_text,
            return_tensors='pt',
            max_length=256,
            padding=True,
            truncation=True
        ).to(DEVICE)

        with torch.no_grad():
            outputs = model(**encoded_input)

        logits = outputs.logits[0].detach().cpu().numpy()
        probs = softmax(logits) # Skorları olasılığa çevirdik

        # Sonuçları POLARITY_LABELS'a göre eşleştiriyoruz
        result = {}
        for idx, prob in enumerate(probs):
            label_name = POLARITY_LABELS.get(idx, f"ID_{idx}")
            result[label_name] = float(np.round(prob, 4))

        return result
    except Exception as e:
        print(f"Tahmin Hatası: {e}")
        return {"error": str(e)}

def get_aspect_summary(comments: list[str]) -> dict:
    """Tüm yorumları varlık bazında analiz eder ve istatistik üretir."""
    if not comments:
        return {"error": "Yorum listesi boş"}
    
    # İstatistik
    aspect_stats = {aspect: {
        "total": 0, "POZİTİF": 0, "NEGATİF": 0, "NÖTR": 0
    } for aspect in ASPECT_LABELS}
    
    print(f"{len(comments)} yorum analiz ediliyor...")
    
    for i, comment in enumerate(comments):
        if i % 10 == 0: print(f"İlerleme: %{(i/len(comments)*100):.0f}")
        
        for aspect in ASPECT_LABELS:
            res = analyze_sentiment_absa(comment, aspect)
            if "error" not in res:
                predicted_label = max(res, key=res.get)
                aspect_stats[aspect]["total"] += 1
                aspect_stats[aspect][predicted_label] += 1
    
    # Yüzdeleri hesapladık
    final_report = {}
    for aspect, data in aspect_stats.items():
        if data["total"] > 0:
            total = data["total"]
            final_report[aspect] = {
                "positive_pct": round((data["POZİTİF"] / total) * 100, 1),
                "negative_pct": round((data["NEGATİF"] / total) * 100, 1),
                "neutral_pct": round((data["NÖTR"] / total) * 100, 1),
                "count": total
            }
            
    return final_report

# --- TEST ---
if __name__ == "__main__":
    test_list = [
        "Kanal sahibi çok samimi, kurgu ise harika.",
        "Konuk çok sıkıcıydı, ses kalitesi de berbat.",
        "Video fikri güzel ama ışıklandırma yetersiz."
    ]
    print("\n--- Örnek Analiz Çıktısı ---")
    print(get_aspect_summary(test_list))