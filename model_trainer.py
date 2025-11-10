import pandas as pd
import numpy as np
from datasets import Dataset
from evaluate import load as load_metric
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from torch.nn import CrossEntropyLoss
import random
import os
from sklearn.utils.class_weight import compute_class_weight

ASPECT_LABELS = [
    "KANAL_SAHİBİ_İMAJI",
    "KONUK_İMAJI",
    "BAHSEDİLEN_KİŞİ_İMAJI",
    "VİDEO_İÇERİK",
    "ÜRETİM_KALİTESİ",
    "KANAL_YÖNETİMi",
    "ALAKASIZ"
]

POLARITY_LABELS = {
    0: "NEGATİF",
    1: "NÖTR",
    2: "POZİTİF"
}

LABEL_TO_ID = {v: k for k, v in POLARITY_LABELS.items()}
ID_TO_LABEL = {k: v for k, v in POLARITY_LABELS.items()}

MODEL_NAME = "dbmdz/bert-base-turkish-cased"
OUTPUT_DIR = "absa_model"
RANDOM_SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class CustomTrainer(Trainer):

    def __init__(self, class_weights, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.class_weights = torch.tensor(class_weights, dtype=torch.float).to(self.args.device)

    def compute_loss(self, model, inputs, return_outputs=False,num_items_in_batch=None):
        labels = inputs.get("labels")
        if labels is not None:
            inputs = {k: v for k, v in inputs.items() if k != "labels"}

        outputs = model(**inputs)
        logits = outputs.get("logits")

        if labels is not None:
            loss_fct = CrossEntropyLoss(weight=self.class_weights)
            loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        else:
            loss = outputs.loss if hasattr(outputs, "loss") else None
            if loss is None:
                loss = torch.tensor(0.0, device=logits.device)
        
        return (loss, outputs) if return_outputs else loss

def augment_data(df):
    augmented_data = []

    for _, row in df.iterrows():

        augmented_data.append(row)

        if row["Polarity"] == 2:
            variations = [
                f"Gerçekten {row['comment']}",
                f"Kesinlikle {row['comment']}",
                f"Harika, {row['comment']}",
                f"Çok güzel {row['comment']}",
                f"Muhteşem {row['comment']}",
                f"Kesinlikle katılıyorum, {row['comment']}",
                f"Tamamen doğru, {row['comment']}"
            ]
            for variation in variations[:3]:  
                new_row = row.copy()
                new_row['comment'] = variation
                augmented_data.append(new_row)

    return pd.DataFrame(augmented_data)

def load_and_preprocess_data(file_path):

    print(f"Veri yükleniyor: {file_path}")

    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig', sep=';')
    except Exception as e:
        print(f"Hata: {e}")
        return None, None
    
    df = df.dropna(subset=['comment', 'Aspect', 'Polarity'])
    df = df[df['Aspect'].isin(ASPECT_LABELS)]
    df['Polarity'] = df['Polarity'].astype(int)
    
    print(f"Orijinal veri boyutu: {len(df)}")
    
    df_augmented = augment_data(df)
    print(f"Artırılmış veri boyutu: {len(df_augmented)}")

    print("\nSınıf dağılımı:")
    for aspect in ASPECT_LABELS:
        aspect_data = df_augmented[df_augmented['Aspect'] == aspect]
        if len(aspect_data) > 0:
            print(f"{aspect}: {len(aspect_data)} örnek")
            print(f"  - Pozitif: {len(aspect_data[aspect_data['Polarity'] == 2])}")
            print(f"  - Negatif: {len(aspect_data[aspect_data['Polarity'] == 0])}")
            print(f"  - Nötr: {len(aspect_data[aspect_data['Polarity'] == 1])}")
    
    train_df = df_augmented.sample(frac=0.8, random_state=RANDOM_SEED)
    test_df = df_augmented.drop(train_df.index)

    print(f"\nEğitim: {len(train_df)} | Test: {len(test_df)}")

    return Dataset.from_pandas(train_df), Dataset.from_pandas(test_df)

def compute_metrics(eval_pred):
    """Geliştirilmiş metrik hesaplama"""
    metric = load_metric("f1")
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    
    f1_macro = metric.compute(predictions=predictions, references=labels, average="macro")["f1"]
    f1_micro = metric.compute(predictions=predictions, references=labels, average="micro")["f1"] 
    f1_weighted = metric.compute(predictions=predictions, references=labels, average="weighted")["f1"]
    
    accuracy = np.mean(predictions == labels)
    
    return {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted
    }

def run_improved_absa_training(train_dataset, test_dataset):
    """Geliştirilmiş ABSA eğitimi"""
    
    print(f"\n{'='*80}")
    print(f"🖥️  EĞİTİM CİHAZI: {DEVICE}")
    if torch.cuda.is_available():
        print(f"📊 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Bellek: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"🔥 CUDA Desteği: ✅ Aktif")
    else:
        print(f"⚠️  CUDA mevcut değil - CPU ile eğitim yapılacak")
        print(f"🔄 CPU Desteği: ✅ Aktif")
    print(f"{'='*80}\n")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=len(POLARITY_LABELS),
        id2label=ID_TO_LABEL,
        label2id=LABEL_TO_ID,
    )
    model.to(DEVICE) 
    print(f"✅ Model {DEVICE} cihazına taşındı\n")
    
    train_df = train_dataset.to_pandas()
    class_weights = compute_class_weight(
        'balanced',
        classes=np.unique(train_df['Polarity']),
        y=train_df['Polarity']
    )
    
    print(f"Sınıf ağırlıkları: {class_weights}\n")
    
    def tokenize_function(examples):
        text_with_aspect = [f"{c} {tokenizer.sep_token} {a}" 
                            for c, a in zip(examples['comment'], examples['Aspect'])]
        examples['labels'] = examples['Polarity']
        
        return tokenizer(
            text_with_aspect,
            truncation=True,
            padding=True,
            max_length=256  
        )
    
    tokenized_train = train_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=['comment', 'Aspect', 'Polarity', '__index_level_0__']
    )
    tokenized_test = test_dataset.map(
        tokenize_function, 
        batched=True, 
        remove_columns=['comment', 'Aspect', 'Polarity', '__index_level_0__']
    )
    
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=8,  
        per_device_train_batch_size=4, 
        per_device_eval_batch_size=4,
        warmup_steps=200,
        weight_decay=0.01,
        learning_rate=1e-5, 
        logging_dir='./logs',
        logging_steps=25,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="f1_weighted",
        greater_is_better=True,
        save_total_limit=3,
        seed=RANDOM_SEED,
        dataloader_drop_last=True,
    )
    
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_test,
        compute_metrics=compute_metrics,
        class_weights=class_weights, 
    )
    
    print("Geliştirilmiş model eğitimi başlıyor...")
    trainer.train()
    
    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    
    eval_results = trainer.evaluate()
    print(f"\nFinal değerlendirme sonuçları:")
    for key, value in eval_results.items():
        print(f"{key}: {value:.4f}")
    
    print(f"\n{'='*80}")
    print(f"📊 SINIF DAĞILIMI ANALİZİ")
    print(f"{'='*80}")
    
    train_df = train_dataset.to_pandas()
    print(f"\n✅ EĞİTİM SETİ (Train):")
    train_polarity_counts = train_df['Polarity'].value_counts().sort_index()
    train_total = len(train_df)
    for pol_id, count in train_polarity_counts.items():
        pol_name = POLARITY_LABELS[pol_id]
        percentage = (count / train_total) * 100
        bar = "█" * int(percentage / 2)  # Her %2 için bir karakter
        print(f"  {pol_name:10s} (ID: {pol_id}): {count:4d} örnek ({percentage:5.1f}%) {bar}")
    print(f"  {'Toplam':10s}: {'':5s} {train_total:4d} örnek\n")
    
    test_df = test_dataset.to_pandas()
    print(f"\n🧪 TEST SETİ (Test):")
    test_polarity_counts = test_df['Polarity'].value_counts().sort_index()
    test_total = len(test_df)
    for pol_id, count in test_polarity_counts.items():
        pol_name = POLARITY_LABELS[pol_id]
        percentage = (count / test_total) * 100
        bar = "█" * int(percentage / 2)
        print(f"  {pol_name:10s} (ID: {pol_id}): {count:4d} örnek ({percentage:5.1f}%) {bar}")
    print(f"  {'Toplam':10s}: {'':5s} {test_total:4d} örnek\n")
    
    print(f"{'='*80}")
    if train_polarity_counts.min() < train_polarity_counts.max() * 0.5:
        print(f"⚠️  UYARI: Sınıf dengesizliği tespit edildi!")
        print(f"   En az örnek: {train_polarity_counts.min()}")
        print(f"   En fazla örnek: {train_polarity_counts.max()}")
        print(f"   Oran: {train_polarity_counts.max() / train_polarity_counts.min():.1f}x")
        print(f"   💡 İpucu: Az örneği olan sınıflara daha fazla veri ekleyin\n")
    else:
        print(f"✅ Sınıf dağılımı dengeli görünüyor\n")
    print(f"{'='*80}\n")
    
    return trainer

if __name__ == "__main__":
    LABELED_DATA_PATH = "etiketli_youtube_yorumları.csv"
    
    train_dataset, test_dataset = load_and_preprocess_data(LABELED_DATA_PATH)
    
    if train_dataset is not None:
        trainer = run_improved_absa_training(train_dataset, test_dataset)
        print(f"\nEğitim tamamlandı! Model kaydedildi: {OUTPUT_DIR}")
    else:
        print("Model eğitimi başlatılamadı.")