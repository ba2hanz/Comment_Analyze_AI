# Gelişmiş Duygu Analizi Modülü
# Kendi eğittiğimiz ABSA modelini kullanır

import os
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import warnings
warnings.filterwarnings("ignore")

# Kendi eğittiğimiz ABSA modeli
MODEL_PATH = "./improved_absa_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ABSA etiketleri
ASPECT_LABELS = [
    "KANAL_SAHİBİ_İMAJI",
    "KONUK_İMAJI",
    "BAHSEDİLEN_KİŞİ_İMAJI",
    "VİDEO_İÇERİK",
    "ÜRETİM_KALİTESİ",
    "KANAL_YÖNETİMİ",
    "DİĞER/ALAKASIZ"
]

POLARITY_LABELS = {
    0: "NEGATİF",
    1: "NÖTR",
    2: "POZİTİF"
}

# Model ve tokenizer'ı global olarak yükle
print("ABSA modeli yükleniyor...")
try:
    if os.path.exists(MODEL_PATH):
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        model.to(DEVICE)
        model.eval()
        print("ABSA modeli başarıyla yüklendi!")
    else:
        print("ABSA modeli bulunamadı, varsayılan model kullanılıyor...")
        # Fallback: Varsayılan Türkçe model
        tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
        model = AutoModelForSequenceClassification.from_pretrained("dbmdz/bert-base-turkish-cased")
        model.eval()
        print("Varsayılan model yüklendi!")
except Exception as e:
    print(f"Model yükleme hatası: {e}")
    tokenizer = None
    model = None

# Deterministik sonuçlar için
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)

# Model konfigürasyonu
ID2LABEL = getattr(model.config, "id2label", None) or {} if model else {}

def preprocess_text(text):
    """
    Geliştirilmiş metin ön işleme yapar.
    """
    if not text or not isinstance(text, str):
        return ""
    
    import re
    
    # Temel temizlik
    text = text.strip()
    
    # Fazla boşlukları temizle
    text = re.sub(r'\s+', ' ', text)
    
    # Kullanıcı adlarını normalize et
    text = re.sub(r'@\w+', '@user', text)
    
    # URL'leri normalize et
    text = re.sub(r'http\S+', 'http', text)
    
    # Tekrarlanan karakterleri temizle (ör: "çokkkk" -> "çok")
    text = re.sub(r'(.)\1{2,}', r'\1', text)
    
    # Özel karakterleri normalize et
    text = text.replace('…', '...')
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace(''', "'").replace(''', "'")
    
    return text

def analyze_sentiment_absa(comment: str, aspect: str = "ÜRETİM_KALİTESİ") -> dict:
    """
    ABSA modeli ile tek bir yorum için duygu analizi yapar.
    
    Args:
        comment: Analiz edilecek yorum metni
        aspect: Analiz edilecek varlık (aspect)
    
    Returns:
        Duygu analizi sonuçları sözlüğü
    """
    if not model or not tokenizer:
        return {"error": "Model yüklenemedi"}
    
    try:
        # Metni ön işle
        text = preprocess_text(comment)
        
        # ABSA formatı: [Yorum] [SEP] [Varlık]
        input_text = f"{text} {tokenizer.sep_token} {aspect}"
        
        # Tokenizasyon
        encoded_input = tokenizer(
            input_text,
            return_tensors='pt',
            max_length=256,
            padding=True,
            truncation=True
        ).to(DEVICE)

        # Tahmin
        with torch.no_grad():
            outputs = model(**encoded_input)

        # Olasılıkları hesapla
        logits = outputs.logits[0].detach().cpu().numpy()
        probs = softmax(logits)

        # Sonuçları sırala
        ranking = np.argsort(probs)[::-1]

        result = {}
        for idx in ranking:
            if ID2LABEL:
                label_name = ID2LABEL.get(int(idx), str(int(idx)))
            else:
                # Fallback: POLARITY_LABELS kullan
                label_name = POLARITY_LABELS.get(int(idx), f"LABEL_{int(idx)}")
            result[label_name] = np.round(float(probs[idx]), 4)

        return result

    except Exception as e:
        print(f"ABSA analizi hatası: {e}")
        return {"error": "Analiz edilemedi"}

def analyze_sentiment(comment: str) -> dict:
    """
    Genel duygu analizi (tüm varlıklar için ortalama)
    
    Args:
        comment: Analiz edilecek yorum metni
    
    Returns:
        Duygu analizi sonuçları sözlüğü
    """
    if not model or not tokenizer:
        return {"error": "Model yüklenemedi"}
    
    try:
        # Tüm varlıklar için analiz yap
        all_results = {}
        for aspect in ASPECT_LABELS:
            aspect_result = analyze_sentiment_absa(comment, aspect)
            if "error" not in aspect_result:
                all_results[aspect] = aspect_result
        
        if not all_results:
            return {"error": "Hiçbir varlık analiz edilemedi"}
        
        # Ortalama skorları hesapla
        avg_scores = {}
        for polarity in POLARITY_LABELS.values():
            total_score = 0
            count = 0
            for aspect_result in all_results.values():
                if polarity in aspect_result:
                    total_score += aspect_result[polarity]
                    count += 1
            if count > 0:
                avg_scores[polarity] = round(total_score / count, 4)
        
        return avg_scores if avg_scores else {"error": "Ortalama hesaplanamadı"}

    except Exception as e:
        print(f"Duygu analizi hatası: {e}")
        return {"error": "Analiz edilemedi"}

def batch_analyze_sentiment(comments: list[str]) -> tuple[list[dict], str, float, float]:
    """
    Birden fazla yorum için toplu duygu analizi yapar.
    
    Args:
        comments: Analiz edilecek yorum listesi
    
    Returns:
        (sonuçlar, genel_duygu, pozitif_yüzde, negatif_yüzde)
    """
    if not comments:
        return [], "Bilinmiyor", 0.0, 0.0
    
    results = []
    positive_count = 0
    negative_count = 0
    neutral_count = 0

    print(f"{len(comments)} yorum analiz ediliyor...")
    
    for i, comment in enumerate(comments):
        if i % 50 == 0:  # Her 50 yorumda bir ilerleme göster
            print(f"İlerleme: {i}/{len(comments)} yorum analiz edildi")
        
        analysis = analyze_sentiment(comment)
        results.append({
            "comment": comment,
            "sentiment": analysis
        })

        # Duygu kategorisini belirle
        if "error" in analysis:
            continue
            
        # Pozitif/negatif/notr belirleme (ABSA modeli için)
        positive_score = analysis.get("POZİTİF", 0)
        negative_score = analysis.get("NEGATİF", 0)
        neutral_score = analysis.get("NÖTR", 0)
        
        if positive_score > negative_score and positive_score > neutral_score:
            positive_count += 1
        elif negative_score > positive_score and negative_score > neutral_score:
            negative_count += 1
        else:
            neutral_count += 1

    total_analyzed = len(comments)
    positive_percentage = (positive_count / total_analyzed) * 100 if total_analyzed else 0
    negative_percentage = (negative_count / total_analyzed) * 100 if total_analyzed else 0
    neutral_percentage = (neutral_count / total_analyzed) * 100 if total_analyzed else 0

    # Genel duygu belirleme
    if positive_percentage > 60:
        general_sentiment = "Çok Olumlu"
    elif positive_percentage > 40:
        general_sentiment = "Olumlu"
    elif negative_percentage > 60:
        general_sentiment = "Çok Olumsuz"
    elif negative_percentage > 40:
        general_sentiment = "Olumsuz"
    elif neutral_percentage > 50:
        general_sentiment = "Nötr"
    else:
        general_sentiment = "Karışık"

    print(f"Analiz tamamlandı!")
    print(f"Pozitif: %{positive_percentage:.1f} ({positive_count} yorum)")
    print(f"Negatif: %{negative_percentage:.1f} ({negative_count} yorum)")
    print(f"Nötr: %{neutral_percentage:.1f} ({neutral_count} yorum)")
    print(f"Genel Duygu: {general_sentiment}")

    return results, general_sentiment, positive_percentage, negative_percentage

def get_sentiment_summary(comments: list[str]) -> dict:
    """
    Yorumlar için özet duygu analizi raporu oluşturur.
    
    Args:
        comments: Analiz edilecek yorum listesi
    
    Returns:
        Özet rapor sözlüğü
    """
    if not comments:
        return {
            "toplam_yorum": 0,
            "genel_duygu": "Bilinmiyor",
            "pozitif_yuzde": 0,
            "negatif_yuzde": 0,
            "notr_yuzde": 0,
            "en_pozitif_yorumlar": [],
            "en_negatif_yorumlar": []
        }
    
    results, general_sentiment, positive_percentage, negative_percentage = batch_analyze_sentiment(comments)
    
    # En pozitif ve negatif yorumları bul
    scored_comments = []
    for result in results:
        if "error" not in result["sentiment"]:
            positive_score = result["sentiment"].get("POZİTİF", 0)
            negative_score = result["sentiment"].get("NEGATİF", 0)
            scored_comments.append({
                "comment": result["comment"],
                "positive_score": positive_score,
                "negative_score": negative_score,
                "net_score": positive_score - negative_score
            })
    
    # En pozitif yorumlar (net score'a göre)
    most_positive = sorted(scored_comments, key=lambda x: x["net_score"], reverse=True)[:3]
    
    # En negatif yorumlar (net score'a göre)
    most_negative = sorted(scored_comments, key=lambda x: x["net_score"])[:3]
    
    # Nötr yüzde hesapla
    neutral_count = len(comments) - sum(1 for c in scored_comments if c["net_score"] > 0.1 or c["net_score"] < -0.1)
    neutral_percentage = (neutral_count / len(comments)) * 100
    
    return {
        "toplam_yorum": len(comments),
        "genel_duygu": general_sentiment,
        "pozitif_yuzde": round(positive_percentage, 2),
        "negatif_yuzde": round(negative_percentage, 2),
        "notr_yuzde": round(neutral_percentage, 2),
        "en_pozitif_yorumlar": [c["comment"][:100] + "..." if len(c["comment"]) > 100 else c["comment"] 
                               for c in most_positive],
        "en_negatif_yorumlar": [c["comment"][:100] + "..." if len(c["comment"]) > 100 else c["comment"] 
                               for c in most_negative]
    }


def get_aspect_summary(comments: list[str]) -> dict:
    """
    Yorumları varlık bazında analiz eder ve özet rapor oluşturur.
    
    Args:
        comments: Analiz edilecek yorum listesi
    
    Returns:
        Varlık bazında özet rapor
    """
    if not comments:
        return {"error": "Yorum listesi boş"}
    
    aspect_stats = {}
    
    for aspect in ASPECT_LABELS:
        aspect_stats[aspect] = {
            "total_comments": 0,
            "positive_count": 0,
            "negative_count": 0,
            "neutral_count": 0,
            "positive_percentage": 0,
            "negative_percentage": 0,
            "neutral_percentage": 0
        }
    
    print(f"{len(comments)} yorum varlık bazında analiz ediliyor...")
    
    for i, comment in enumerate(comments):
        if i % 50 == 0:
            print(f"İlerleme: {i}/{len(comments)} yorum analiz edildi")
        
        for aspect in ASPECT_LABELS:
            result = analyze_sentiment_absa(comment, aspect)
            
            if "error" not in result:
                aspect_stats[aspect]["total_comments"] += 1
                
                # En yüksek skorlu duyguyu bul
                best_sentiment = max(result.items(), key=lambda x: x[1])
                sentiment = best_sentiment[0]
                
                if sentiment == "POZİTİF":
                    aspect_stats[aspect]["positive_count"] += 1
                elif sentiment == "NEGATİF":
                    aspect_stats[aspect]["negative_count"] += 1
                else:
                    aspect_stats[aspect]["neutral_count"] += 1
    
    # Yüzdeleri hesapla
    for aspect in ASPECT_LABELS:
        total = aspect_stats[aspect]["total_comments"]
        if total > 0:
            aspect_stats[aspect]["positive_percentage"] = round(
                (aspect_stats[aspect]["positive_count"] / total) * 100, 2
            )
            aspect_stats[aspect]["negative_percentage"] = round(
                (aspect_stats[aspect]["negative_count"] / total) * 100, 2
            )
            aspect_stats[aspect]["neutral_percentage"] = round(
                (aspect_stats[aspect]["neutral_count"] / total) * 100, 2
            )
    
    return {
        "total_comments": len(comments),
        "aspect_statistics": aspect_stats
    }

def get_aspect_examples(comments: list[str]) -> dict:
    """
    Her varlık için en olumlu ve olumsuz yorum örneklerini bulur.
    
    Args:
        comments: Analiz edilecek yorum listesi
    
    Returns:
        Varlık bazında örnek yorumlar
    """
    if not comments:
        return {"error": "Yorum listesi boş"}
    
    aspect_examples = {}
    
    for aspect in ASPECT_LABELS:
        aspect_examples[aspect] = {
            "positive_examples": [],
            "negative_examples": []
        }
    
    print(f"{len(comments)} yorum varlık bazında örnek aranıyor...")
    
    for i, comment in enumerate(comments):
        if i % 50 == 0:
            print(f"İlerleme: {i}/{len(comments)} yorum işlendi")
        
        for aspect in ASPECT_LABELS:
            result = analyze_sentiment_absa(comment, aspect)
            
            if "error" not in result:
                # En yüksek skorlu duyguyu bul
                best_sentiment = max(result.items(), key=lambda x: x[1])
                sentiment = best_sentiment[0]
                confidence = best_sentiment[1]
                
                # Pozitif örnekler (en yüksek 3)
                if sentiment == "POZİTİF" and confidence > 0.6:
                    aspect_examples[aspect]["positive_examples"].append({
                        "comment": comment[:150] + "..." if len(comment) > 150 else comment,
                        "confidence": round(confidence, 3)
                    })
                
                # Negatif örnekler (en yüksek 3)
                elif sentiment == "NEGATİF" and confidence > 0.6:
                    aspect_examples[aspect]["negative_examples"].append({
                        "comment": comment[:150] + "..." if len(comment) > 150 else comment,
                        "confidence": round(confidence, 3)
                    })
    
    # Her varlık için en iyi 3 örneği seç
    for aspect in ASPECT_LABELS:
        # Pozitif örnekleri güven skoruna göre sırala
        aspect_examples[aspect]["positive_examples"] = sorted(
            aspect_examples[aspect]["positive_examples"], 
            key=lambda x: x["confidence"], 
            reverse=True
        )[:3]
        
        # Negatif örnekleri güven skoruna göre sırala
        aspect_examples[aspect]["negative_examples"] = sorted(
            aspect_examples[aspect]["negative_examples"], 
            key=lambda x: x["confidence"], 
            reverse=True
        )[:3]
    
    return aspect_examples

