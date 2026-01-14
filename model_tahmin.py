import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax

# Model yolu
MODEL_PATH = "./absa_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Polarity etiketleri
POLARITY_LABELS = {
    0: "NEGATİF",
    1: "NÖTR", 
    2: "POZİTİF"
}

# Model ve tokenizer'ı global olarak yükle (performans için)
_model = None
_tokenizer = None

def _load_model():
    """Modeli bir kez yükle"""
    global _model, _tokenizer
    if _model is None or _tokenizer is None:
        try:
            print(f"Model yükleniyor: {MODEL_PATH}")
            _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
            _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
            _model.to(DEVICE)
            _model.eval()
            print("Model başarıyla yüklendi!")
        except Exception as e:
            print(f"Eğitilmiş model yüklenemedi ({e}), hazır model kullanılacak...")
            # Fallback: Hazır model
            _tokenizer = AutoTokenizer.from_pretrained("savasy/bert-base-turkish-sentiment-cased")
            _model = AutoModelForSequenceClassification.from_pretrained(
                "savasy/bert-base-turkish-sentiment-cased",
                num_labels=3,
                ignore_mismatched_sizes=True
            )
            _model.to(DEVICE)
            _model.eval()
    return _model, _tokenizer

def analiz_et(yorum_listesi):
    """
    Yorum listesini analiz eder ve her yorum için duygu tahmini yapar.
    3 sınıflı tahmin: NEGATİF, NÖTR, POZİTİF
    """
    try:
        model, tokenizer = _load_model()
        
        if not yorum_listesi:
            return []
        
        # Tüm yorumları tokenize et
        encoded_inputs = tokenizer(
            yorum_listesi,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt"
        ).to(DEVICE)
        
        # Tahmin yap
        with torch.no_grad():
            outputs = model(**encoded_inputs)
            logits = outputs.logits
        
        probs = softmax(logits.cpu().numpy(), axis=1)
        
        # Her yorum için en yüksek olasılıklı sınıfı bul
        cikti = []
        for prob in probs:
            predicted_class = np.argmax(prob)
            confidence = float(prob[predicted_class])
            
            # Sınıf adını al
            duygu = POLARITY_LABELS.get(predicted_class, "NÖTR")
            
            # Eğer güven skoru düşükse veya sınıflar arası fark küçükse NÖTR yap
            sorted_probs = sorted(prob, reverse=True)
            fark = sorted_probs[0] - sorted_probs[1]
            
            if confidence < 0.50 or fark < 0.15:
                # Belirsiz durumlar için NÖTR'ye yakın olanı seç
                if prob[1] > 0.30:  # NÖTR olasılığı makul seviyedeyse
                    duygu = "NÖTR"
                    confidence = prob[1]
            
            cikti.append({
                "duygu": duygu,
                "skor": round(confidence, 4),
                "tum_olasiliklar": {
                    "NEGATİF": round(float(prob[0]), 4),
                    "NÖTR": round(float(prob[1]), 4),
                    "POZİTİF": round(float(prob[2]), 4)
                }
            })
        
        return cikti
        
    except Exception as e:
        print(f"Hata oluştu: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print(analiz_et(["Mükemmel bir video!", "Hiç beğenmedim, çok kötü."]))