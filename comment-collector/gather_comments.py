import os
import pandas as pd
import time
import dotenv
from googleapiclient.discovery import build

dotenv.load_dotenv(dotenv_path="../.env")

YOUTUBE_API_KEY=os.getenv("YOUTUBE_API_KEY")
YOUTUBE_SERVICE_NAME= os.getenv("YOUTUBE_SERVICE_NAME","youtube")
YOUTUBE_API_VERSION= os.getenv("YOUTUBE_API_VERSION","v3")

try:
    youtube = build(
        YOUTUBE_SERVICE_NAME, 
        YOUTUBE_API_VERSION, 
        developerKey=YOUTUBE_API_KEY
    )
    print("YOutube API bağlantısı başarılı")
except Exception as e:
    print(f"Hata: Youtube API servisine bağlanırken hata oluştu: {e}")
    youtube = None

def get_video_comments_real(video_id):

    if not youtube:
        print("Youtube API servisi başlatılamadı. Veri çekilemiyor!")
        return pd.DataFrame()

    all_comments=[]
    next_page_token=None

    #Sayfalama
    print(f"Video {video_id} için yorumlar toplanıyor...")
    while True:
        try:
            response=youtube.commentThreads().list(
                part="snippet",
                videoId=video_id,
                maxResults=100,
                pageToken=next_page_token
            ).execute()
        except Exception as e:
            print(f"Hata: Yorumlar toplanırken hata oluştu: {e}")
            break
        
        for item in response.get("items", []):
            try:
                comment=item["snippet"]["topLevelComment"]["snippet"]
                all_comments.append({
                    "Yorum": comment["textDisplay"],
                    "Beğeni Sayısı": comment["likeCount"],
                    "Yorumlayan": comment["authorDisplayName"],
                    "video_id": video_id,
                })
            except (KeyError, TypeError):
                continue
        
        next_page_token=response.get("nextPageToken")

        if not next_page_token or len(all_comments) >= 5000:
            break

        time.sleep(0.5)

    return pd.DataFrame(all_comments)

#Yorumları toplamak için video ID'leri giriniz
target_video_ids = [
    "",
    "",
    "",
    "",
]

all_comments=[]

if youtube:
    for vid_id in target_video_ids:
        df_comments=get_video_comments_real(vid_id)
        all_comments.append(df_comments)

if all_comments:
    final_df=pd.concat(all_comments, ignore_index=True)
else:
    final_df=pd.DataFrame(columns=["Yorum", "Beğeni Sayısı", "Yorumlayan",  "video_id"])

output_file="raw_youtube_comments.csv"
final_df.to_csv(output_file, index=False, encoding="utf-8-sig")

print(f"\n--- VERİ TOPLAMA ÖZETİ ---")
print(f"Toplam {len(final_df)} yorum toplandı ve '{output_file}' dosyasına kaydedildi.")
print("\nBetiği kendi yerel ortamınızda çalıştırırken, API anahtarınızı girmeyi ve 'google-api-python-client' kütüphanesini kurmayı unutmayın.")
