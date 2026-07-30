from transformers import AutoTokenizer
import transformers
print("Transformers version:", transformers.__version__)
from datasets import Dataset
from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np
import torch
from pathlib import Path

# ==== 1. Dummy Data ====
texts = [
    "Hey, can you send me the notes from class?",               # WhatsApp
    "Check out my new profile picture!",                        # Facebook
    "Just posted a reel, go watch it!",                         # Instagram
    "Can't believe this happened 😂 #life",                      # Twitter
    "Video call me when you're free",                           # Messenger
    "Good morning 🌞",                                          # WhatsApp
    "New blog up! Link in bio.",                                # Instagram
    "Retweet if you agree!",                                    # Twitter
    "Thanks for accepting my friend request!",                  # Facebook
    "Join the group chat ASAP",                                 # Messenger
    "Happy birthday! 🎉",                                       # WhatsApp
    "Check out my latest tweet",                                # Twitter
    "New photo dump coming soon 📸",                            # Instagram
    "Let’s catch up soon!",                                     # Messenger
    "Comment your thoughts below 👇",                           # Facebook
    "Wanna grab lunch tomorrow?",                               # WhatsApp
    "Throwback to this amazing trip! #tbt",                     # Instagram
    "Who else is watching this live? 🔥",                       # Twitter
    "Ping me when you’re online",                               # Messenger
    "Updated my status, feeling blessed 🙏",                    # Facebook
]

labels = [
    "whatsapp", "facebook", "instagram", "twitter", "messenger",
    "whatsapp", "instagram", "twitter", "facebook", "messenger",
    "whatsapp", "twitter", "instagram", "messenger", "facebook",
    "whatsapp", "instagram", "twitter", "messenger", "facebook"
]

# Encode text labels to integers
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Create dataset
data = {"text": texts, "label": encoded_labels}
dataset = Dataset.from_dict(data).train_test_split(test_size=0.2)

# ==== 2. Tokenizer ====
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")

def tokenize(example):
    return tokenizer(example["text"], padding="max_length", truncation=True, max_length=32)

tokenized = dataset.map(tokenize, batched=True)

# ==== 3. Model ====
num_labels = len(set(encoded_labels))
model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=num_labels)

# ==== 4. Metrics ====
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    prf = precision_recall_fscore_support(labels, preds, average="macro")
    return {"accuracy": acc, "precision": prf[0], "recall": prf[1], "f1": prf[2]}

# ==== 5. Training ====
args = TrainingArguments(
    output_dir="./platform_results",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    num_train_epochs=5,
    logging_steps=5,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    processing_class=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()

# ==== 6. Save Model ====
output_dir = "./platform_classifier_model"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print("✅ Model and tokenizer saved.")

# ==== 7. Export to ONNX (via optimum, since transformers.onnx is removed) ====
# pip install optimum[exporters]
from optimum.exporters.onnx import main_export

onnx_output_dir = "./onnx_output"

main_export(
    model_name_or_path=output_dir,   # the fine-tuned model dir saved above
    output=onnx_output_dir,
    task="text-classification",
    opset=14,
)

print(f"✅ Exported ONNX model to: {onnx_output_dir}")

# optimum's main_export writes "model.onnx" inside onnx_output_dir
onnx_model_path = str(Path(onnx_output_dir) / "model.onnx")

# ==== 8. Quantize ====
from onnxruntime.quantization import quantize_dynamic, QuantType

quantize_dynamic(
    model_input=onnx_model_path,
    model_output="quantized_model.onnx",
    weight_type=QuantType.QUInt8
)

print("✅ Quantized model saved to: quantized_model.onnx")