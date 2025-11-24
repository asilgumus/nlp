from transformers import AutoModelForSequenceClassification

model = AutoModelForSequenceClassification.from_pretrained("yeniguno/bert-base-turkish-intent-classification")

# Etiketleri yazdır
id2label = model.config.id2label
for idx, label in id2label.items():
    print(f"{idx}: {label}")
