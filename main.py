from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
from fastapi.middleware.cors import CORSMiddleware
from yorum_cekme import get_youtube_comments
from model_tahmin import analiz_et
import os

app = FastAPI(
    title="YouTube Duygu Analizi API",
    description="BERT Modeli ve Anahtar Kelime Tabanlı Analiz"
)

# Frontend uygulamasının tarayıcı üzerinden bu API'ye erişebilmesi için
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Tüm adreslerden gelen isteklere izin veriyoruz
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- VERİ MODELLERİ ---
# React tarafından gönderilecek verilerin formatını belirliyoruz.

class AnalizIstegi(BaseModel):
    video_url: str
    channel_owner: Optional[str] = ""
    guest: Optional[str] = ""
    mentioned_person: Optional[str] = ""
    max_comments: int = 1000

# --- KATEGORİ BELİRLEME ---
# Yorumun içeriğine bakarak hangi kategoriye ait olduğunu belirler.

def kategori_ata(metin: str, istek: AnalizIstegi) -> str:
    m = metin.lower()
    
    # Kullanıcıdan gelen özel isimlere göre kontrol
    if istek.channel_owner:
        # Kanal sahibi ismini kontrol et (sadece tek isim)
        channel_name = istek.channel_owner.strip().lower()
        if channel_name and channel_name in m:
            # Direkt genel kategoriye atama
            return "KANAL_SAHİBİ_İMAJI"
    
    if istek.guest:
        # Konuk isimlerini kontrol et
        guest_names = [name.strip().lower() for name in istek.guest.split(",")]
        for name in guest_names:
            if name and name in m:
                # Direkt genel kategoriye atama
                return "KONUK_İMAJI"
    
    if istek.mentioned_person:
        # Bahsedilen kişi ismini kontrol et
        mentioned_names = [name.strip().lower() for name in istek.mentioned_person.split(",")]
        for name in mentioned_names:
            if name and name in m:
                # Direkt genel kategoriye atama
                return "BAHSEDİLEN_KİŞİ_İMAJI"
    
    # Teknik konular 
    teknik = ["ses", "görüntü", "kamera", "ışık", "kurgu", "edit", "mikrofon", "kalite", "4k", "fps", "montaj"]
    if any(k in m for k in teknik):
        return "ÜRETİM_KALİTESİ"
    
    # Kanal yönetimi ve yayın düzeni
    yonetim = ["abone", "kanal", "yükle", "saat", "gün", "seri", "video at", "paylaş", "beğen", "bildirim"]
    if any(k in m for k in yonetim):
        return "KANAL_YÖNETİMİ"
    
    # Alakasız yorumları tespit et 
    alakasiz_kelimeler = ["spam", "bot", "test", "deneme"]
    if any(k in m for k in alakasiz_kelimeler) or len(m.strip()) < 10:
        return "ALAKASIZ"
    
    # Varsayılan Kategori: Video İçeriği
    return "VİDEO_İÇERİK"

# --- API ENDPOINTLERİ ---


@app.get("/api")
def api_durum():
    """API durumunu kontrol etmek için"""
    return {"durum": "API Çalışıyor", "mesaj": "POST /analyze endpoint'ini kullanın."}

@app.post("/analyze")
async def analiz_baslat(request: AnalizIstegi):
    print(f"\n[SİSTEM] Yeni istek alındı: {request.video_url}")
    
    # YouTube yorumlarını topluyoruz
    try:
        print("[1/3] Yorumlar YouTube'dan çekiliyor...")
        # youtube_veri_cekme.py dosyasındaki fonksiyonu çağırıyoruz
        yorumlar = get_youtube_comments(request.video_url, max_comments=request.max_comments)
    except Exception as e:
        print(f"YouTube Hatası: {e}")
        raise HTTPException(status_code=500, detail="YouTube yorumları çekilirken bir sorun oluştu.")
    
    if not yorumlar:
        return {"error": "Video için yorum bulunamadı veya API hatası oluştu."}

    # BERT modeli ile duygu analizi yapıyoruz
    try:
        print(f"[2/3] {len(yorumlar)} yorum BERT modeliyle analiz ediliyor...")
        # model_tahmin.py dosyasındaki fonksiyonu çağırıyoruz
        tahminler = analiz_et(yorumlar)
        if tahminler is None:
            raise HTTPException(status_code=500, detail="Duygu analizi başarısız oldu. Model yüklenemedi veya hata oluştu.")
        if len(tahminler) != len(yorumlar):
            raise HTTPException(status_code=500, detail=f"Tahmin sayısı ({len(tahminler)}) yorum sayısı ({len(yorumlar)}) ile eşleşmiyor.")
    except HTTPException:
        raise
    except Exception as e:
        print(f"Model Hatası: {e}")
        raise HTTPException(status_code=500, detail=f"Duygu analizi yapılamadı: {str(e)}")

    # İstatistikleri kategorilere göre grupluyoruz
    print("[3/3] Sonuçlar kategorize ediliyor...")
    
    kategoriler = [
        "KANAL_SAHİBİ_İMAJI", "KONUK_İMAJI", "BAHSEDİLEN_KİŞİ_İMAJI",
        "VİDEO_İÇERİK", "ÜRETİM_KALİTESİ", "KANAL_YÖNETİMİ", "ALAKASIZ"
    ]
    
    stats = {k: {"POZİTİF": 0, "NEGATİF": 0, "NÖTR": 0, "toplam": 0, "yorumlar": []} for k in kategoriler}

    for i in range(len(yorumlar)):
        metin = yorumlar[i]
        duygu = tahminler[i]["duygu"]
        skor = tahminler[i].get("skor", 0.5)
        
        # numpy float32'yi Python float'a çevir 
        if hasattr(skor, 'item'):
            skor = float(skor.item())
        else:
            skor = float(skor)
        
        # Kategoriyi belirliyoruz
        ktgr = kategori_ata(metin, request)
        
        # Sayacı güncelliyoruz
        stats[ktgr][duygu] += 1
        stats[ktgr]["toplam"] += 1
        
        # Yorumu kaydediyoruz (modal için)
        stats[ktgr]["yorumlar"].append({
            "text": metin,
            "polarity": duygu,
            "confidence": skor
        })

    # React Frontend'in beklediği formatta veriyi hazırlıyoruz
    analiz_sonucu = {}
    for ktgr, veriler in stats.items():
        if veriler["toplam"] > 0:
            # En çok hangi duygu çıktıysa onu 'ana sonuç' yapıyoruz
            dominant = max(["POZİTİF", "NEGATİF", "NÖTR"], key=lambda k: veriler[k])
            
            # Yüzdesel olasılıkları hesapliyoruz
            probs = {p: veriler[p] / veriler["toplam"] for p in ["POZİTİF", "NEGATİF", "NÖTR"]}
            
            yorumlar_by_polarity = {
                "POZİTİF": sorted(
                    [y for y in veriler["yorumlar"] if y["polarity"] == "POZİTİF"],
                    key=lambda x: x["confidence"],
                    reverse=True
                )[:5],
                "NEGATİF": sorted(
                    [y for y in veriler["yorumlar"] if y["polarity"] == "NEGATİF"],
                    key=lambda x: x["confidence"],
                    reverse=True
                )[:5],
                "NÖTR": sorted(
                    [y for y in veriler["yorumlar"] if y["polarity"] == "NÖTR"],
                    key=lambda x: x["confidence"],
                    reverse=True
                )[:5]
            }
            
            ornek_yorumlar = (
                yorumlar_by_polarity["POZİTİF"] + 
                yorumlar_by_polarity["NEGATİF"] + 
                yorumlar_by_polarity["NÖTR"]
            )[:15]
            
            analiz_sonucu[ktgr] = {
                "predicted_polarity": dominant,
                "confidence": float(probs[dominant]),
                "all_probabilities": {
                    "POZİTİF": float(probs["POZİTİF"]),
                    "NEGATİF": float(probs["NEGATİF"]),
                    "NÖTR": float(probs["NÖTR"])
                },
                "comments": [
                    {
                        "text": yorum["text"],
                        "polarity": yorum["polarity"],
                        "confidence": float(yorum["confidence"])
                    }
                    for yorum in ornek_yorumlar
                ]
            }

    print("[TAMAMLANDI] Analiz başarıyla bitti.\n")
    return {
        "analysis_details": analiz_sonucu,
        "total_comments": len(yorumlar)
    }

# --- SUNUCUYU BAŞLATIYORUZ ---

if __name__ == "__main__":
    # Terminalden çalıştırmak için: uvicorn main:app --reload
    uvicorn.run(app, host="127.0.0.1", port=8000)