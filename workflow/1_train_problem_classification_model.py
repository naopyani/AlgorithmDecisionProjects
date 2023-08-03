import torch.nn
from transformers import BertTokenizer, BertForSequenceClassification
import torch.optim as optim
import pandas as pd
import numpy as np

model_name = "bert-base-chinese"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForSequenceClassification.from_pretrained(model_name, num_labels=3)

train_dataset = pd.read_excel("../data/problem_label_data.xlsx")
texts = train_dataset["texts"].tolist()
# target = "distribution"
# target = "relation"
# target = "train"
target = "labels"
labels = train_dataset[target].tolist()
# 使用tokenizer对文本数据进行编码和padding
encoded_inputs = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

optimizer = optim.Adam(model.parameters(), lr=1e-5)
loss_function = torch.nn.CrossEntropyLoss()

epochs = 100
for epoch in range(epochs):
    model.train()
    optimizer.zero_grad()
    outputs = model(**encoded_inputs, labels=torch.tensor(labels))
    loss = outputs.loss
    print(loss.tolist())
    loss.backward()
    optimizer.step()

test_inputs = ["特征分布", "特征相关性", "拟合模型"]
test_encoded_inputs = tokenizer(test_inputs, padding=True, truncation=True, return_tensors="pt")
model.eval()
with torch.no_grad():
    test_outputs = model(**test_encoded_inputs)
    test_logits = test_outputs.logits
    # predicted_classes = torch.argmax(test_logits, dim=1)
    predicted_classes = torch.argmax(test_logits, dim=-1)
    print(predicted_classes)
    # 根据predicted_classes和真实标签进行性能评估。
# 存储问题分类模型
model.save_pretrained(f"../models/{target}_model")
tokenizer.save_pretrained(f"../models/{target}_model")
