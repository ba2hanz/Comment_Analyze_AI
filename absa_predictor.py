# Geliştirilmiş ABSA Tahmin Betiği
# Güven skorları ve detaylı analiz ile

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import os
from scipy.special import softmax

# Sabit tanımlamalar
POLARITY_LABELS = {
    0: "NEGATİF",
    1: "NÖTR", 
    2: "POZİTİF"
}

ASPECT_LABELS = [
    "KANAL_SAHİBİ_İMAJI",
    "KONUK_İMAJI",
    "BAHSEDİLEN_KİŞİ_İMAJI", 
    "VİDEO_İÇERİK",
    "ÜRETİM_KALİTESİ",
    "KANAL_YÖNETİMİ",
    "DİĞER/ALAKASIZ"
]

MODEL_PATH = "./improved_absa_model"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model_and_tokenizer(model_path):
    """Model ve tokenizer yükleme"""
    if not os.path.exists(model_path):
        print(f"HATA: Model bulunamadı: {model_path}")
        print("Lütfen 'improved_model_trainer.py' çalıştırın.")
        return None, None
        
    print(f"Model yükleniyor: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(DEVICE)
    model.eval()
    print("Model yüklendi.")
    return model, tokenizer

def predict_absa_detailed(comment, aspect, model, tokenizer):
    """Detaylı duygu analizi tahmini"""
    
    # Giriş formatı
    input_text = f"{comment} {tokenizer.sep_token} {aspect}"
    
    # Tokenizasyon
    inputs = tokenizer(
        input_text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=256
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Logitlerden olasılık hesaplama
    logits = outputs.logits[0].detach().cpu().numpy()
    probabilities = softmax(logits)
    
    # En yüksek olasılıklı tahmin
    prediction_id = np.argmax(probabilities)
    predicted_polarity = POLARITY_LABELS.get(prediction_id, "BİLİNMEYEN")
    confidence = float(probabilities[prediction_id])
    
    # Tüm sınıfların olasılıkları
    all_probabilities = {
        POLARITY_LABELS[i]: float(probabilities[i]) 
        for i in range(len(POLARITY_LABELS))
    }
    
    return {
        "predicted_polarity": predicted_polarity,
        "confidence": confidence,
        "all_probabilities": all_probabilities,
        "input_text": input_text
    }

def analyze_comment_comprehensive(comment, model, tokenizer, verbose=False):
    """Yorumu tüm varlıklar için analiz et"""
    if verbose:
        print(f"\n{'='*60}")
        print(f"YORUM: '{comment}'")
        print(f"{'='*60}")
    
    results = {}
    
    for aspect in ASPECT_LABELS:
        result = predict_absa_detailed(comment, aspect, model, tokenizer)
        results[aspect] = result
        
        if verbose:
            print(f"\nVarlık: {aspect}")
            print(f"Tahmin: {result['predicted_polarity']} (Güven: {result['confidence']:.3f})")
            print("Tüm olasılıklar:")
            for polarity, prob in result['all_probabilities'].items():
                print(f"  {polarity}: {prob:.3f}")
    
    return results

def test_specific_cases():
    """Belirli test durumları"""
    test_cases = [
        ("Kurgu çok profesyonel olmuş, tebrikler.", "ÜRETİM_KALİTESİ"),
        ("Video kalitesi harika, çok beğendim.", "ÜRETİM_KALİTESİ"),
        ("Kanal sahibinin tarzı çok değişti, artık izlemiyorum.", "KANAL_SAHİBİ_İMAJI"), # Negatif beklenir
        ("Kurgu çok profesyonel olmuş, tebrikler.", "ÜRETİM_KALİTESİ"),          # Pozitif beklenir
        ("Bu içerik fikri fena değil, sıradaki videoyu bekliyorum.", "VİDEO_İÇERİK"),     # Nötr beklenir
        ("Konuk keşke hiç gelmeseydi, çok sıkıcıydı.", "KONUK_İMAJI"),             # Negatif beklenir
        ("Yeni gelen kişi çok sempatik, videoları ona daha çok yakışıyor.", "BAHSEDİLEN_KİŞİ_İMAJI")
    ]
    
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    if model is None:
        return
    
    print("TEST DURUMLARI")
    print("="*50)
    
    for comment, aspect in test_cases:
        result = predict_absa_detailed(comment, aspect, model, tokenizer)
        print(f"\nYorum: '{comment}'")
        print(f"Varlık: {aspect}")
        print(f"Tahmin: {result['predicted_polarity']} (Güven: {result['confidence']:.3f})")
        print("-" * 30)

if __name__ == "__main__":
    model, tokenizer = load_model_and_tokenizer(MODEL_PATH)
    
    if model is None:
        exit()
    
    print(f"Tahminler {DEVICE} cihazında yapılacak.")
    
    # Test durumlarını çalıştır
    test_specific_cases()
    
    # İnteraktif mod
    print("\n" + "="*60)
    print("İNTERAKTİF MOD")
    print("="*60)
    
    while True:
        print("\nSeçenekler:")
        print("1. Tek varlık analizi")
        print("2. Tüm varlıklar analizi") 
        print("3. Çıkış")
        
        choice = input("\nSeçiminiz (1-3): ").strip()
        
        if choice == "1":
            comment = input("Yorumunuzu girin: ").strip()
            print(f"\nMevcut varlıklar: {', '.join(ASPECT_LABELS)}")
            aspect = input("Varlık seçin: ").strip().upper()
            
            if aspect in ASPECT_LABELS:
                result = predict_absa_detailed(comment, aspect, model, tokenizer)
                print(f"\nSonuç: {result['predicted_polarity']} (Güven: {result['confidence']:.3f})")
            else:
                print("Geçersiz varlık!")
                
        elif choice == "2":
            comment = input("Yorumunuzu girin: ").strip()
            analyze_comment_comprehensive(comment, model, tokenizer)
            
        elif choice == "3":
            print("Çıkılıyor...")
            break
            
        else:
            print("Geçersiz seçim!")
