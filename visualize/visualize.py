# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as plt

# 读取训练日志
csv_path = r"C:\Users\13579\Desktop\results.csv"
df = pd.read_csv(csv_path)

# 去掉列名两端的空格
df.columns = df.columns.str.strip()

# ----------- 绘制 Loss 曲线 -----------
plt.figure(figsize=(12, 6))
plt.plot(df['epoch'], df['train/box_loss'], label="train/box_loss")
plt.plot(df['epoch'], df['train/cls_loss'], label="train/cls_loss")
plt.plot(df['epoch'], df['train/dfl_loss'], label="train/dfl_loss")
plt.plot(df['epoch'], df['val/box_loss'], label="val/box_loss", linestyle="--")
plt.plot(df['epoch'], df['val/cls_loss'], label="val/cls_loss", linestyle="--")
plt.plot(df['epoch'], df['val/dfl_loss'], label="val/dfl_loss", linestyle="--")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training and Validation Loss Curves")
plt.legend()
plt.grid(True)
plt.show()

# ----------- 绘制 Precision / Recall / mAP 曲线 -----------
plt.figure(figsize=(12, 6))
plt.plot(df['epoch'], df['metrics/precision(B)'], label="Precision")
plt.plot(df['epoch'], df['metrics/recall(B)'], label="Recall")
plt.plot(df['epoch'], df['metrics/mAP50(B)'], label="mAP@0.5")
plt.plot(df['epoch'], df['metrics/mAP50-95(B)'], label="mAP@0.5:0.95")
plt.xlabel("Epoch")
plt.ylabel("Metrics")
plt.title("Precision, Recall and mAP Curves")
plt.legend()
plt.grid(True)
plt.show()
