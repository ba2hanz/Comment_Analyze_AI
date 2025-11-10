import os
from googleapiclient.discovery import build
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
YOUTUBE_API_SERVICE_NAME = "youtube"
YOUTUBE_API_VERSION = "v3"

def extract_video_id(post_url: str) -> str:
    """YouTube URL'inden video ID'sini çıkarır."""
    import re
    
    # YouTube.com/watch?v= formatı
    if "youtube.com" in post_url and "/watch?v=" in post_url:
        match = re.search(r'[?&]v=([^&]+)', post_url)
        if match:
            return match.group(1)
    
    # youtu.be/ formatı
    elif "youtu.be/" in post_url:
        match = re.search(r'youtu\.be/([^?&]+)', post_url)
        if match:
            return match.group(1)
    
    # Eğer URL zaten video ID gibi görünüyorsa (11 karakter, alfanumerik)
    elif len(post_url) == 11 and post_url.replace('-', '').replace('_', '').isalnum():
        return post_url
    
    # Hiçbiri eşleşmezse orijinal URL'yi döndür
    print(f"Uyarı: Video ID çıkarılamadı: {post_url}")
    return post_url

def get_youtube_comments(post_url: str, max_comments: int = 100) -> list[str]:

    if not YOUTUBE_API_KEY:
        print("YOUTUBE_API_KEY ortam değişkeni ayarlanmamış.")
        return []

    video_id = extract_video_id(post_url)
    if not video_id:
        print("Geçersiz YouTube URL'si.")
        return []

    print(f"Video ID: {video_id}")
    print(f"Maksimum {max_comments} yorum çekiliyor (maksimum 5 sayfa)...")

    try:
        youtube = build(YOUTUBE_API_SERVICE_NAME, YOUTUBE_API_VERSION, developerKey=YOUTUBE_API_KEY)
        comments = []
        page_count = 0
        max_pages = 5  # Maksimum 3 sayfa
        next_page_token = None

        while page_count < max_pages and len(comments) < max_comments:
            page_count += 1
            print(f"Sayfa {page_count} çekiliyor...")
            
            # API isteği
            request_params = {
                "part": "snippet",
                "videoId": video_id,
                "maxResults": min(100, max_comments - len(comments)),  # Her sayfada maksimum 100 yorum
                "order": "relevance"
            }
            
            if next_page_token:
                request_params["pageToken"] = next_page_token

            request = youtube.commentThreads().list(**request_params)
            response = request.execute()

            # Yorumları işle
            items = response.get("items", [])
            print(f"  {len(items)} yorum bulundu")
            
            for item in items:
                comment_text = item["snippet"]["topLevelComment"]["snippet"]["textDisplay"]
                comments.append(comment_text)
                
                if len(comments) >= max_comments:
                    break

            # Sonraki sayfa token'ını al
            next_page_token = response.get("nextPageToken")
            if not next_page_token:
                print("  Daha fazla sayfa yok.")
                break


        print(f"Toplam {len(comments)} yorum çekildi ({page_count} sayfa)")
        return comments

    except Exception as e:
        print(f"Hata oluştu: {e}")
        return comments  # Şu ana kadar çekilen yorumları döndür