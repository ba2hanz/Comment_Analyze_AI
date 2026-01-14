import pandas as pd
import numpy as np
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, EarlyStoppingCallback
from sklearn.metrics import f1_score, accuracy_score
import os

# AYARLAR VE YOL TANIMLAMALARI
BASE_MODEL = "savasy/bert-base-turkish-sentiment-cased"
SAVE_PATH = "./absa_model"
DATA_PATH = "etiketli_youtube_yorumları.csv"

def compute_metrics(eval_pred):
    """F1 score ve accuracy hesaplama fonksiyonu"""
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)
    
    f1 = f1_score(labels, predictions, average='weighted')
    accuracy = accuracy_score(labels, predictions)
    
    return {
        'f1': f1,
        'accuracy': accuracy
    }

def egitimi_baslat():
    # Gerekli dizinleri oluştur
    os.makedirs("./sonuclar", exist_ok=True)
    os.makedirs(SAVE_PATH, exist_ok=True)
    
    # VERİ YÜKLEME
    if not os.path.exists(DATA_PATH):
        print(f"Hata: {DATA_PATH} dosyası bulunamadı! Lütfen etiketli verinizi hazırlayın.")
        return
    df = pd.read_csv(DATA_PATH, sep=';')  
    # Sadece ihtiyacımız olan sütunları alıyoruz ve eksik verileri temizliyoruz
    df = df[['comment', 'Polarity']].dropna()
    dataset = Dataset.from_pandas(df)
    
    # Veriyi Eğitim ve Test olarak ikiye bölüyoruz (%80 Eğitim, %20 Test)
    dataset = dataset.train_test_split(test_size=0.2)

    # TOKENIZER VE MODEL HAZIRLIĞI
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    
    # Modeli yüklüyoruz.
    model = AutoModelForSequenceClassification.from_pretrained(
        BASE_MODEL, 
        num_labels=3,
        ignore_mismatched_sizes=True
    )

    # TOKENİZASYON
    def tokenize_func(examples):
        tokenized = tokenizer(examples["comment"], padding="max_length", truncation=True, max_length=128)
        tokenized["labels"] = examples["Polarity"]
        return tokenized
    tokenized_datasets = dataset.map(tokenize_func, batched=True)

    # EĞİTİM PARAMETRELERİ
    # Burada batch size ve epoch değerlerini bilgisayarımı yormayacak şekilde seçtim.
    training_args = TrainingArguments(
        output_dir="./sonuclar",          # Geçici sonuçların yazılacağı yer
        num_train_epochs=8,              # Veriyi kaç kez baştan sona okuyacak
        per_device_train_batch_size=4,   # Her adımda kaç yorum işlenecek
        per_device_eval_batch_size=4,
        warmup_steps=20,                
        weight_decay=0.01,
        learning_rate=1e-5,               
        logging_steps=10,
        report_to="none",                
        eval_strategy="epoch",           # Her epoch sonunda başarıyı ölç
        save_strategy="epoch",
        load_best_model_at_end=True,     # En iyi performans gösteren modeli yükle
        metric_for_best_model="eval_f1",
        greater_is_better=True,
    )

    # TRAINER TANIMLAMA
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
    )
    # EĞİTİM
    trainer.train()

    # MODEL KAYDETME
    trainer.save_model(SAVE_PATH)
    tokenizer.save_pretrained(SAVE_PATH)

if __name__ == "__main__":
    egitimi_baslat()