---
title: Fruit Image Classification
emoji: 🦀
colorFrom: yellow
colorTo: green
sdk: streamlit
sdk_version: 1.42.2
app_file: app.py
pinned: false
---

# Fruit and Vegetables Image Classification

## 目錄
1. [描述](#1描述)
2. [匯入必要模組和函式庫](#2匯入必要模組和函式庫)
3. [獲取影像檔案路徑並計算影像數量](#3獲取影像檔案路徑並計算影像數量)
4. [處理影像檔案路徑並生成數據框](#4處理影像檔案路徑並生成數據框)
5. [獲取影像標籤並計算資料集分佈](#5獲取影像標籤並計算資料集分佈)
6. [顯示每個類別的第一張圖像](#6顯示每個類別的第一張圖像)
7. [創建影像數據生成器進行數據增強和預處理](#7創建影像數據生成器進行數據增強和預處理)
8. [載入並配置預訓練模型 MobileNetV2](#8載入並配置預訓練模型-mobilenetv2)
9. [建立並訓練影像分類模型](#9建立並訓練影像分類模型)
10. [視覺化模型訓練過程中的準確率和損失](#10視覺化模型訓練過程中的準確率和損失)
11. [獲取並顯示標籤對應表](#11獲取並顯示標籤對應表)
12. [載入最佳模型並使用測試數據進行預測](#12載入最佳模型並使用測試數據進行預測)
13. [顯示混淆矩陣](#13顯示混淆矩陣)
14. [Hugging Face Spaces 應用](#14hugging-face-spaces-應用)
15. [結論](#15結論)



## 1.描述
## 使用從 Kaggle 取得的水果和蔬菜圖像識別數據集
## 資料來源：[Fruits and Vegetables Image Recognition Dataset](https://www.kaggle.com/datasets/kritikseth/fruit-and-vegetable-image-recognition/data)  

透過 Kaggle 上的 Fruits and Vegetables Image Recognition Dataset，建立一個模型來辨識水果和蔬菜的圖片，並通過網頁應用進行預測。

### 資料集
資料集包含多種水果和蔬菜的圖片，分為訓練、測試和驗證三個部分。  
每個部分的圖片路徑在 `label_dict.json` 中有詳細記錄。

### 模型訓練(在Google Colab上執行)
在 `fruit.ipynb` 中，使用 TensorFlow 和 Keras 訓練了一個卷積神經網絡（CNN）模型，`fruit_model.keras`的訓練過程包括數據增強和預處理。
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 預測
使用訓練好的模型，在 `app.py` 中建立了一個 Streamlit 應用，允許使用者上傳圖片並進行水果與蔬菜辨識。    
該應用已部署於 **Hugging Face Spaces**，無需手動下載程式並執行，可直接在線上測試。
👉 [**點此使用**](https://huggingface.co/spaces/CR7-800/fruit-image-classification)



## 2.匯入必要模組和函式庫
```python
import numpy as np
import pandas as pd
import os.path
import matplotlib.pyplot as plt
import tensorflow as tf

from pathlib import Path
from collections import Counter
from PIL import Image
from tensorflow.keras import layers, models
from tensorflow.keras import optimizers
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
```


## 3.獲取影像檔案路徑並計算影像數量
- 定義函式 get_image_filepaths，用於獲取指定目錄中所有 .jpg 格式的影像檔案路徑
- 計算訓練、測試和驗證資料集中影像的數量
```python
def get_image_filepaths(directory): # directory 是一個用於指定目錄的變數
    return list(Path(directory).glob(r'**/*.jpg'))

print('training images:',len(get_image_filepaths('/content/drive/MyDrive/3.Python深度學習應用開發/03_水果/train')))
print('testing images:',len(get_image_filepaths('/content/drive/MyDrive/3.Python深度學習應用開發/03_水果/test')))
print('validation images:',len(get_image_filepaths('/content/drive/MyDrive/3.Python深度學習應用開發/03_水果/validation')))
```
執行結果:  
![計算影像數量](https://github.com/user-attachments/assets/841c2d01-c33a-40b7-ae97-504e85ae7856)



## 4.處理影像檔案路徑並生成數據框
- 取得訓練、測試和驗證資料集中所有影像的路徑
- 定義函式 proc_img，用於處理這些影像路徑並生成包含文件路徑和標籤的數據框
```python
# 取得所有圖片的路徑
train_filepaths = get_image_filepaths('/content/drive/MyDrive/3.Python深度學習應用開發/03_水果/train')
test_filepaths = get_image_filepaths('/content/drive/MyDrive/3.Python深度學習應用開發/03_水果/test')
val_filepaths = get_image_filepaths('/content/drive/MyDrive/3.Python深度學習應用開發/03_水果/validation')

def proc_img(filepaths):
    data = [] # 創建一個空列表，用於存儲文件路徑和標籤
    for filepath in filepaths:
        label = filepath.parent.name # 獲取文件路徑的父目錄名稱(即標籤)
        data.append({'Filepath':str(filepath),'Label':label}) # 將文件路徑和標籤添加到data列表中
    return pd.DataFrame(data)

# 生成訓練、測試和驗證數據框
train_df = proc_img(train_filepaths)
test_df = proc_img(test_filepaths)
val_df = proc_img(val_filepaths)
```



## 5.獲取影像標籤並計算資料集分佈
- 從影像檔案路徑中提取標籤，並計算訓練、測試和驗證資料集中每個標籤的分佈情況
```python
# path.parent.name 可以取得資料夾名稱，filepaths 是一個 list
train_labels = [path.parent.name for path in train_filepaths]
test_labels = [path.parent.name for path in test_filepaths]
val_labels = [path.parent.name for path in val_filepaths]

print('Training set distribution:',Counter(train_labels))
print('Testing set distribution:',Counter(test_labels))
print('Validation set distribution:',Counter(val_labels))
```
執行結果:  
`Training set distribution: Counter({'soy beans': 92, 'peas': 90, 'spinach': 87, 'lettuce': 87, 'turnip': 85, 'grapes': 85, 'tomato': 84, 'pineapple': 84, 'cabbage': 84, 'beetroot': 84, 'corn': 84, 'sweetcorn': 83, 'garlic': 83, 'kiwi': 82, 'onion': 80, 'capsicum': 80, 'watermelon': 79, 'jalepeno': 79, 'cucumber': 78, 'bell pepper': 78, 'mango': 77, 'eggplant': 77, 'pear': 76, 'chilli pepper': 76, 'paprika': 74, 'pomegranate': 74, 'carrot': 73, 'cauliflower': 71, 'raddish': 70, 'sweetpotato': 69, 'potato': 66, 'ginger': 64, 'lemon': 64, 'banana': 62, 'orange': 61, 'apple': 58})`

`Testing set distribution: Counter({'sweetcorn': 10, 'tomato': 10, 'watermelon': 10, 'soy beans': 10, 'sweetpotato': 10, 'turnip': 10, 'spinach': 10, 'pear': 10, 'pineapple': 10, 'pomegranate': 10, 'paprika': 10, 'mango': 10, 'eggplant': 10, 'kiwi': 10, 'ginger': 10, 'cucumber': 10, 'corn': 10, 'garlic': 10, 'beetroot': 10, 'cabbage': 10, 'peas': 9, 'lettuce': 9, 'potato': 9, 'onion': 9, 'jalepeno': 9, 'bell pepper': 9, 'cauliflower': 9, 'apple': 9, 'capsicum': 9, 'banana': 9, 'raddish': 8, 'grapes': 8, 'orange': 7, 'lemon': 7, 'chilli pepper': 7, 'carrot': 7})`

`Validation set distribution: Counter({'watermelon': 10, 'spinach': 10, 'pomegranate': 10, 'sweetcorn': 10, 'turnip': 10, 'tomato': 10, 'soy beans': 10, 'sweetpotato': 10, 'paprika': 10, 'pear': 10, 'pineapple': 10, 'kiwi': 10, 'mango': 10, 'corn': 10, 'garlic': 10, 'ginger': 10, 'cucumber': 10, 'eggplant': 10, 'cabbage': 10, 'beetroot': 10, 'potato': 9, 'onion': 9, 'peas': 9, 'lettuce': 9, 'jalepeno': 9, 'cauliflower': 9, 'bell pepper': 9, 'apple': 9, 'capsicum': 9, 'banana': 9, 'raddish': 8, 'grapes': 8, 'orange': 7, 'lemon': 7, 'chilli pepper': 7, 'carrot': 7})`


## 6.顯示每個類別的第一張圖像
- 創建一個字典來存儲每個類別的第一張影像，並使用 Matplotlib 顯示這些影像

```python
first_images = {}
for category in train_df['Label'].unique():
    first_images[category] = train_df[train_df['Label'] == category]['Filepath'].values[0]

plt.figure(figsize=(15,10))
for i, (category, image_path) in enumerate(first_images.items()):
    image = Image.open(image_path)
    plt.subplot(6,6,i+1) # i+1 是因為 subplot 從 1 開始
    plt.imshow(image)
    plt.title(category)
    plt.axis('off')

plt.show()
```
執行結果:  
![每個類別](https://github.com/user-attachments/assets/0ae3f70b-f0c1-4c3d-a5bf-44a64d485514)




## 7.創建影像數據生成器進行數據增強和預處理 
```python
# 使用 ImageDataGenerator 進行數據增強

train_datagen = ImageDataGenerator(
    rescale=1./255, # 將像素值縮放到 [0, 1] 之間
    rotation_range=40, # 隨機旋轉圖片
    width_shift_range=0.2, # 隨機水平平移
    height_shift_range=0.2, # 隨機垂直平移
    shear_range=0.2, # 隨機剪切(沿某個方向傾斜)
    zoom_range=0.2, # 隨機縮放
    horizontal_flip=True, # 隨機水平翻轉
    brightness_range=[0.8, 1.2], # 隨機調整亮度
    channel_shift_range=0.2, # 隨機改變顏色通道
    fill_mode='nearest' # 填充新創建像素的方法
)

# 創建 ImageDataGenerator 用於驗證和測試數據集（不進行增強）
test_datagen = ImageDataGenerator(rescale=1./255)

# 創建數據生成器
train_generator = train_datagen.flow_from_dataframe(
    train_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical' # 使用 categorical_crossentropy 損失函數
)

val_generator = test_datagen.flow_from_dataframe(
    val_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical'
)

# 在創建數據生成器之前打亂數據框
test_df = test_df.sample(frac=1).reset_index(drop=True)

# 重新創建數據生成器
test_generator = test_datagen.flow_from_dataframe(
    test_df,
    x_col='Filepath',
    y_col='Label',
    target_size=(224,224),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)
```
執行結果:  
![數據增強](https://github.com/user-attachments/assets/59c89801-5472-4adc-8f58-25c0a7d6ea7d)




## 8.載入並配置預訓練模型 MobileNetV2
```python
# 載入預訓練模型
base_model = MobileNetV2(
    input_shape=(224,224,3), # 輸入圖像的形狀
    include_top=False, # 不包含頂層的全連接層
    weights='imagenet', # 使用 ImageNet 預訓練權重
    pooling='avg' # 全局平均池化
)
base_model.trainable = False

# 顯示模型摘要
# base_model.summary()
```



## 9.建立並訓練影像分類模型
```python
# 建立模型
input = base_model.input
x = layers.Dense(512,activation='relu')(base_model.output)
x = layers.Dense(256,activation='relu')(x)
x = layers.Dense(128,activation='relu')(x)
x = layers.Dense(64,activation='relu')(x)
output = layers.Dense(36,activation='softmax')(x)

model = models.Model(inputs=input,outputs=output)

# 編譯模型
model.compile(optimizer=optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# 設置回調函數
callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath='/content/drive/MyDrive/3.Python深度學習應用開發/03_水果/fruit_model.keras',
        save_best_only=True, # 只儲存性能最好的模型（根據監控指標）
        monitor='val_accuracy', # 監控驗證集的準確率
        mode='max' # 當 val_accuracy 最大時，儲存模型
    )
]

# 訓練模型
history = model.fit(train_generator, # 使用訓練集數據
                    validation_data=val_generator, # 使用驗證集數據
                    epochs=50,
                    callbacks=callbacks)
```
執行結果:  
![訓練](https://github.com/user-attachments/assets/2d5dc97d-1214-46fa-85df-953f4ad3aa80)




## 10.視覺化模型訓練過程中的準確率和損失
- 將模型訓練過程中的準確率和損失數據轉換為 DataFrame
- 使用 Matplotlib 繪製出訓練和驗證過程中的準確率和損失曲線 
```python
# 將訓練歷史轉換為 DataFrame
result = pd.DataFrame(history.history)

# 創建子圖
fig, axes = plt.subplots(1,2,figsize=(15, 5))

# 繪製準確率曲線
result[['accuracy','val_accuracy']].plot(ax=axes[0])
axes[0].set_title("Accuracy")
axes[0].set_xlabel("Epochs")
axes[0].set_ylabel("Accuracy")

# 繪製損失曲線
result[['loss','val_loss']].plot(ax=axes[1])
axes[1].set_title("Loss")
axes[1].set_xlabel("Epochs")
axes[1].set_ylabel("Loss")

# 調整佈局並顯示圖表
plt.tight_layout()
plt.show()
```
執行結果:  
![準確率和損失](https://github.com/user-attachments/assets/e06b1f42-cd7d-4b66-938e-e72dffd4dd54)




## 11.獲取並顯示標籤對應表
```python
# 獲取標籤對應表
label_map = train_generator.class_indices

# 反轉字典以便於查詢
label_map = {v:k for k, v in label_map.items()}

# 顯示標籤對應表
print(label_map)
```
執行結果:  
`{0: 'apple', 1: 'banana', 2: 'beetroot', 3: 'bell pepper', 4: 'cabbage', 5: 'capsicum', 6: 'carrot', 7: 'cauliflower', 8: 'chilli pepper', 9: 'corn', 10: 'cucumber', 11: 'eggplant', 12: 'garlic', 13: 'ginger', 14: 'grapes', 15: 'jalepeno', 16: 'kiwi', 17: 'lemon', 18: 'lettuce', 19: 'mango', 20: 'onion', 21: 'orange', 22: 'paprika', 23: 'pear', 24: 'peas', 25: 'pineapple', 26: 'pomegranate', 27: 'potato', 28: 'raddish', 29: 'soy beans', 30: 'spinach', 31: 'sweetcorn', 32: 'sweetpotato', 33: 'tomato', 34: 'turnip', 35: 'watermelon'}`



## 12.載入最佳模型並使用測試數據進行預測
```python
# 載入最佳模型
model = tf.keras.models.load_model('/content/drive/MyDrive/3.Python深度學習應用開發/03_水果/fruit_model.keras')

# 使用測試數據生成器進行預測
test_loss,test_accuracy = model.evaluate(test_generator)
print(f'Test accuracy:{test_accuracy * 100:.2f}%')

# 獲取預測結果
predictions = model.predict(test_generator)
predicted_classes = np.argmax(predictions,axis=1)

# 顯示前10個預測結果和實際結果
predicted_labels = [label_map[i] for i in predicted_classes[:10]]
actual_labels = [label_map[i] for i in test_generator.classes[:10]]

print("Predicted labels:",predicted_labels)
print("Actual labels:",actual_labels)
```
執行結果:  
![測試預測](https://github.com/user-attachments/assets/6146212c-e4a0-418e-b561-77953b6626f2)




## 13.顯示混淆矩陣
- 評估模型在測試數據上的預測效果
```python
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

true_classes = test_generator.classes
pred_classes = predicted_classes

# 產生混淆矩陣
cm = confusion_matrix(true_classes,pred_classes)

# 顯示混淆矩陣
fig,ax = plt.subplots(figsize=(15,10))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,display_labels=test_generator.class_indices.keys())
disp.plot(cmap=plt.cm.Blues,ax=ax)

# 自定義圖形
plt.xticks(rotation=90)  # 旋轉 x 軸標籤以提高可讀性
plt.title('Confusion Matrix')
plt.show()
```

執行結果:  
![混淆矩陣](https://github.com/user-attachments/assets/e70ace2b-7764-4032-bfbf-a5b850e710a1)




## 14.Hugging Face Spaces 應用
- 本專案的 Streamlit 應用已成功部署至 Hugging Face Spaces 
- 你可以透過以下連結直接使用應用：[點此進入](https://huggingface.co/spaces/CR7-800/fruit-image-classification)    
![辨識](https://github.com/user-attachments/assets/14953dea-faac-4738-90aa-4a7aeb258555)






## 15.結論
在本專案中，我們成功地建立了一個基於 `MobileNetV2` 的卷積神經網絡模型，用於辨識水果和蔬菜的圖像。  
通過數據增強和預處理，有效地提升了模型的泛化能力。  
訓練過程中，使用 `Kaggle` 上的 `Fruits and Vegetables Image Recognition Dataset`，並在 `Google Colab` 上進行了模型訓練。

模型在測試數據集上的準確率達到了令人滿意的水平，並且通過混淆矩陣，可以看到模型在不同類別上的預測效果。 

未來的改進方向可以包括：
1. 探索其他預訓練模型，如 EfficientNet 或 ResNet，以進一步提升模型性能。
2. 增加數據集的多樣性，包含更多不同背景和光照條件下的圖像。
3. 優化模型超參數，進一步提升準確率和降低損失。
