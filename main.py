from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Dict, Any
import uvicorn
import asyncio
import numpy as np
import os
import sqlite3

# Projenin diğer modüllerini import et
# Bu modüllerin aynı dizinde olması gerekir.
try:
    from comment_fetcher import get_youtube_comments
    from improved_absa_predictor import load_model_and_tokenizer, analyze_comment_comprehensive, ASPECT_LABELS
    import torch
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
except ImportError as e:
    # Eğer kütüphaneler/modüller eksikse, kullanıcıya net bir uyarı göster
    print(f"UYARI: Gerekli modüller yüklenemedi. Python ortamınızda eksik kütüphane/dosya olabilir: {e}")
    # Varsayılan değerler tanımlayarak uygulamanın yine de çalışmasını (boş olsa bile) sağla
    MODEL_LOAD_SUCCESS = False
    DEVICE = None
else:
    MODEL_LOAD_SUCCESS = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Model yolu ve uygulama başlatma
MODEL_PATH = "./improved_absa_model"
app = FastAPI(
    title="YouTube ABSA FastAPI Backend",
    description="Varlık Tabanlı Duygu Analizi (ABSA) Servisi"
)

# -----------------------------------------------------------------------------
# GLOBAL MODEL YÜKLEME (UYGULAMA BAŞLANGICINDA BİR KEZ ÇALIŞIR)
# -----------------------------------------------------------------------------
# Modeli ve tokenizer'ı global değişkenlerde sakla, böylece her istekte yeniden yüklenmez.
global absa_model
global absa_tokenizer
absa_model = None
absa_tokenizer = None

# FastAPI başlangıcında model yükleme
@app.on_event("startup")
def startup_event():
    global absa_model
    global absa_tokenizer
    
    if MODEL_LOAD_SUCCESS:
        print("API başlangıcında ABSA modelini yüklüyorum...")
        try:
            absa_model, absa_tokenizer = load_model_and_tokenizer(MODEL_PATH)
            if absa_model:
                print("ABSA Model başarıyla yüklendi.")
            else:
                print("ABSA Model yüklenemedi. Lütfen model_trainer.py'nin çalıştığından emin olun.")
        except Exception as e:
            print(f"Kritik Model Yükleme Hatası: {e}")
            raise HTTPException(status_code=500, detail=f"Model yüklenemedi: {e}")

# -----------------------------------------------------------------------------
# HTML TEMPLATE
# -----------------------------------------------------------------------------
def get_html_template():
    """HTML template'ini döndürür"""
    html_path = "templates/index.html"
    if os.path.exists(html_path):
        with open(html_path, 'r', encoding='utf-8') as f:
            return f.read()
    return """
    <html><body><h1>Template bulunamadı</h1><p>Templates/index.html dosyasını kontrol edin.</p></body></html>
    """

@app.get("/", response_class=HTMLResponse)
def read_root():
    """Ana sayfa - HTML arayüzü"""
    return get_html_template()

# -----------------------------------------------------------------------------
# GİRDİ ŞEMASI
# -----------------------------------------------------------------------------
class ContextInfo(BaseModel):
    channel_owner: str = ""
    guest_names: str = ""
    mentioned_person: str = ""

class AnalysisRequest(BaseModel):
    post_url: str  # Kullanıcının girdiği YouTube URL'si
    context_info: ContextInfo = ContextInfo()  # Ek bağlam bilgileri

# -----------------------------------------------------------------------------
# ANA ANALİZ ENDPOINT'İ
# -----------------------------------------------------------------------------

@app.post("/analyze")
async def analyze_youtube_video(request: AnalysisRequest) -> Dict[str, Any]:
    global absa_model
    global absa_tokenizer
    
    if not absa_model:
        raise HTTPException(
            status_code=503, 
            detail="Analiz servisi hazır değil. Model henüz yüklenmedi veya bulunamıyor."
        )

    video_url = request.post_url
    context_info = request.context_info
    
    # İsim bilgilerini logla ve parse et
    if context_info.channel_owner:
        # Kanal sahibi isimlerini virgülle ayır ve temizle (lakap, isim vb. olabilir)
        channel_owner_names = [name.strip() for name in context_info.channel_owner.split(',') if name.strip()]
        print(f"📺 Kanal Sahibi İsimleri ({len(channel_owner_names)}): {', '.join(channel_owner_names)}")
    else:
        channel_owner_names = []
    
    if context_info.guest_names:
        # Konuk isimlerini virgülle ayır ve temizle
        guest_list = [name.strip() for name in context_info.guest_names.split(',') if name.strip()]
        print(f"👥 Konuklar ({len(guest_list)}): {', '.join(guest_list)}")
    else:
        guest_list = []
    
    if context_info.mentioned_person:
        print(f"👤 Bahsedilen Kişi: {context_info.mentioned_person}")
    
    # 1. Yorumları Çekme (Asenkron API çağrısı)
    # Maksimum 3 sayfa (300 yorum) çekilir. Bu sayı performansa göre ayarlanabilir.
    try:
        comments = get_youtube_comments(video_url, max_comments=500)
    except Exception as e:
        # API anahtarı veya kota sorunları burada yakalanır.
        raise HTTPException(
            status_code=500, 
            detail=f"YouTube yorumları çekilemedi. API hatası veya geçersiz URL. Hata: {e}"
        )
    
    if not comments:
        # Yorum bulunamazsa veya video yorumları kapalıysa
        return {
            "analysis_details": {
                "error": "Bu video için yorum bulunamadı veya yorumlar kapalı."
            }
        }
    
    # 2. Varlık Bazlı Duygu Analizi (ABSA)
    
    # Her bir yorumu analiz etmek için sonuçları saklayacağımız bir yapı
    # Her varlık için toplanmış duygu skorlarını (Pozitif, Negatif, Nötr) tutacağız.
    aspect_summary = {aspect: {'POZİTİF': 0, 'NEGATİF': 0, 'NÖTR': 0, 'count': 0} for aspect in ASPECT_LABELS}

    # Her konuk için ayrı sayaçlar
    guest_analysis = {guest_name: {'POZİTİF': 0, 'NEGATİF': 0, 'NÖTR': 0, 'count': 0} 
                      for guest_name in guest_list}
    
    # Kanal sahibi için toplu sayaç (birden fazla isim bir kartta)
    channel_owner_analysis = {'POZİTİF': 0, 'NEGATİF': 0, 'NÖTR': 0, 'count': 0}
    
    # Yorumların detayını saklamak için (kartta gösterilmek üzere)
    comment_details = {aspect: [] for aspect in ASPECT_LABELS}
    
    # Konuk başına yorum detayları
    guest_comment_details = {guest_name: [] for guest_name in guest_list}
    
    # Kanal sahibi için yorum detayları
    channel_owner_comment_details = []
    
    # Model ile tahminleri yap (Bu CPU/GPU yoğun kısım)
    print(f"📊 {len(comments)} yorum analiz ediliyor...")
    for idx, comment in enumerate(comments):
        # İlerleme gösterimi (her 50 yorumda bir)
        if (idx + 1) % 50 == 0:
            print(f"   Analiz ediliyor: {idx + 1}/{len(comments)}")
        
        # Tek bir yorumu tüm varlıklar için analiz et
        comment_absa_results = analyze_comment_comprehensive(comment, absa_model, absa_tokenizer)
        comment_lower = comment.lower()  # Konuk eşleştirmesi için hazırla
        
        # Kanal sahibi kontrolü: İsim eşleşmesi VEYA model KANAL_SAHİBİ_İMAJI olarak işaretlemiş
        is_channel_owner_comment = False
        if channel_owner_names:
            # İsim eşleşmesi var mı?
            for owner_name in channel_owner_names:
                if owner_name.lower() in comment_lower:
                    is_channel_owner_comment = True
                    break
            # Veya model bu yorumu kanal sahibi olarak işaretledi mi?
            if not is_channel_owner_comment and 'KANAL_SAHİBİ_İMAJI' in comment_absa_results:
                is_channel_owner_comment = True
        
        for aspect, result in comment_absa_results.items():
            polarity = result['predicted_polarity']
            confidence = result['confidence']
            
            # Yorumun detayını kaydet (kartta gösterilecek)
            comment_details[aspect].append({
                "comment": comment,
                "polarity": polarity,
                "confidence": confidence,
                "all_probabilities": result['all_probabilities']
            })
            
            # Yorumun en yüksek güvene sahip olduğu varlıkları topla
            if polarity in aspect_summary[aspect]:
                aspect_summary[aspect][polarity] += 1
                aspect_summary[aspect]['count'] += 1
            
            # Eğer bu yorum KANAL_SAHİBİ_İMAJI için ise ve kanal sahibi isimleri girildiyse
            # Tüm KANAL_SAHİBİ_İMAJI yorumlarını kanal sahibi analizine ekle
            # (Girilen isimler sadece kartın başlığı için kullanılacak, filtreleme yapılmaz)
            # KANAL_SAHİBİ_İMAJI aspect'i varsa VE bu yorum kanal sahibi analizine eklenmeli ise
            if aspect == 'KANAL_SAHİBİ_İMAJI' and is_channel_owner_comment:
                # Bu yorumu kanal sahibi analizine ekle
                if polarity in channel_owner_analysis:
                    channel_owner_analysis[polarity] += 1
                    channel_owner_analysis['count'] += 1
                
                # Kanal sahibi yorumunu kaydet
                channel_owner_comment_details.append({
                    "comment": comment,
                    "polarity": polarity,
                    "confidence": confidence,
                    "all_probabilities": result['all_probabilities']
                })
            
            # Eğer bu yorum KONUK_İMAJI için ise, hangi konuk hakkında olduğunu kontrol et
            if aspect == 'KONUK_İMAJI':
                for guest_name in guest_list:
                    if guest_name.lower() in comment_lower:
                        # Bu yorum bu konuk hakkında
                        if polarity in guest_analysis[guest_name]:
                            guest_analysis[guest_name][polarity] += 1
                            guest_analysis[guest_name]['count'] += 1
                        
                        # Konuk yorumunu kaydet
                        guest_comment_details[guest_name].append({
                            "comment": comment,
                            "polarity": polarity,
                            "confidence": confidence,
                            "all_probabilities": result['all_probabilities']
                        })
    
    print("✅ Analiz tamamlandı!")

    # 3. Sonuçları React'e uygun formata dönüştür (En yüksek skorlu duyguyu bul)
    
    final_analysis = {}
    
    # Aspect filtreleme - Sadece girilen bilgilere göre analiz göster
    aspects_to_include = [
        'VİDEO_İÇERİK',  # Her zaman göster
        'ÜRETİM_KALİTESİ',  # Her zaman göster
        'KANAL_YÖNETİMİ'  # Her zaman göster
    ]
    
    # İsim girildiyse ilgili varlık analizlerini ekle
    # KANAL_SAHİBİ_İMAJI genel analizini göstermiyoruz, özel analizi göstereceğiz
    # if context_info.channel_owner:
    #     aspects_to_include.append('KANAL_SAHİBİ_İMAJI')
    
    # KONUK_İMAJI genel analizini kaldırdık, sadece özel konuk analizleri gösterilecek
    
    if context_info.mentioned_person:
        aspects_to_include.append('BAHSEDİLEN_KİŞİ_İMAJI')
    
    # Önce genel varlıkları ekle (sadece filtrelenmiş olanlar)
    for aspect, summary in aspect_summary.items():
        # Sadece izin verilen aspect'leri ekle
        if aspect not in aspects_to_include:
            continue
            
        total_count = summary['count']
        if total_count == 0:
            # O varlıkla ilgili yorum bulunamadıysa atla
            continue

        # En yüksek oyu alan duyguyu bul
        max_polarity = max(['POZİTİF', 'NEGATİF', 'NÖTR'], key=lambda k: summary[k])
        max_count = summary[max_polarity]
        
        # Olasılıkları hesapla (Güven Skoru olarak kullanılır)
        probabilities = {p: summary[p] / total_count for p in ['POZİTİF', 'NEGATİF', 'NÖTR']}

        final_analysis[aspect] = {
            "predicted_polarity": max_polarity,
            "confidence": probabilities[max_polarity],  # En yüksek skora sahip duygunun oranı
            "all_probabilities": probabilities,
            "comments": comment_details[aspect]  # Bu aspect için tüm yorumlar
        }
    
    # Kanal sahibi için özel analiz oluştur (birden fazla isim bir kartta)
    if channel_owner_names and channel_owner_analysis['count'] > 0:
        total_count = channel_owner_analysis['count']
        max_polarity = max(['POZİTİF', 'NEGATİF', 'NÖTR'], key=lambda k: channel_owner_analysis[k])
        probabilities = {p: channel_owner_analysis[p] / total_count for p in ['POZİTİF', 'NEGATİF', 'NÖTR']}
        
        # Kanal sahibi isimlerini birleştir
        owner_names_str = ', '.join(channel_owner_names)
        
        final_analysis[f"KANAL_SAHİBİ_İMAJI_{owner_names_str}"] = {
            "predicted_polarity": max_polarity,
            "confidence": probabilities[max_polarity],
            "all_probabilities": probabilities,
            "comments": channel_owner_comment_details
        }
    
    # Her konuk için ayrı analiz oluştur (gerçek hesaplanmış değerlerle)
    for guest_name, guest_data in guest_analysis.items():
        total_count = guest_data['count']
        if total_count == 0:
            # Bu konuk için yorum bulunamadıysa atla
            continue
        
        # En yüksek oyu alan duyguyu bul
        max_polarity = max(['POZİTİF', 'NEGATİF', 'NÖTR'], key=lambda k: guest_data[k])
        
        # Olasılıkları hesapla (Güven Skoru olarak kullanılır)
        probabilities = {p: guest_data[p] / total_count for p in ['POZİTİF', 'NEGATİF', 'NÖTR']}

        final_analysis[f"KONUK_İMAJI_{guest_name}"] = {
            "predicted_polarity": max_polarity,
            "confidence": probabilities[max_polarity],
            "all_probabilities": probabilities,
            "comments": guest_comment_details[guest_name]  # Bu konuk için tüm yorumlar
        }

    # React'in beklediği çıktı yapısı
    return {
        "analysis_details": final_analysis,
        "total_comments_analyzed": len(comments),
        "context_info": {
            "channel_owner": context_info.channel_owner,
            "guest_names": context_info.guest_names,
            "mentioned_person": context_info.mentioned_person
        }
    }
