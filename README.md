# 🐾 AFHQ Animal Faces – Custom CNN Classifier  

This project uses the **Animal Faces-HQ (AFHQ)** dataset to train a **custom Convolutional Neural Network (CNN)** with PyTorch.  
The task is to classify animal face images into three categories: **Cat 🐱, Dog 🐶, Wildlife 🦊**.  

Unlike transfer learning with ResNet or EfficientNet, this project implements a **CNN architecture from scratch**, showcasing my understanding of convolutional layers, pooling, normalization, and regularization.  

---

## 📂 Project Structure
afhq-animal-faces/

│

├── notebooks/

│ 
└── AFHQ_classification.ipynb # training & evaluation notebook
│

├── models/
│ 
└── best_model.pth # trained weights
│

├── data/ # dataset (not uploaded)
│

├── requirements.txt # dependencies

├── README.md # documentation

└── .gitignore # ignored files/folders


---

## 🐕 Dataset
- **Animal Faces-HQ (AFHQ)**  
- 16,130 high-quality images (512×512 resolution).  
- Balanced across 3 classes:  
  - 🐱 Cat (~5000 images)  
  - 🐶 Dog (~5000 images)  
  - 🦊 Wildlife (~5000 images)  
- Source: [AFHQ Dataset (Clova AI)](https://github.com/clovaai/stargan-v2)  

---

## 🧠 Model Architecture
Input (3x128x128)
└── Conv2d(3 → 32) + BatchNorm + ReLU + MaxPool

└── Conv2d(32 → 64) + BatchNorm + ReLU + MaxPool

└── Conv2d(64 → 128) + BatchNorm + ReLU + MaxPool

└── Global Average Pooling (AdaptiveAvgPool2d)

└── Flatten → Dropout(0.5)

└── Linear(128 → 128) + ReLU

└── Linear(128 → 3) # Cat / Dog / Wildlife

Output: Class probabilities


**Key Design Choices:**
- ✅ **BatchNorm** → faster convergence, stable training  
- ✅ **Dropout** → prevents overfitting  
- ✅ **Global Average Pooling** → avoids huge fully connected layers, improves generalization  

---

## 📊 Results

### 🔹 Final Performance (25 Epochs)
| Metric               | Value    |
|-----------------------|----------|
| Training Accuracy     | **84.8%** |
| Validation Accuracy   | **82.7%** |
| Validation Loss       | **0.0677** |

📌 The close match between training and validation accuracy shows good **generalization** (little overfitting).  

---

### 🔹 Learning Curves
*(insert accuracy & loss curves plot here)*  

---

### 🔹 Example Predictions
*(insert 6–9 images with predicted vs actual labels)*  

---

## 🚀 How to Run

### 1. Clone repository
```bash
https://github.com/cadetsikandar/AFHQ_classification.git
cd AFHQ_classification

2. Install dependencies
pip install -r requirements.txt

3. Run Jupyter Notebook
jupyter notebook notebooks/AFHQ_classification.ipynb



📌 Future Work

Compare with pretrained ResNet18 (transfer learning).

Add confusion matrix to analyze per-class performance.

Extend to GANs (DCGAN, CycleGAN) for animal face generation & translation.

Deploy model with Flask or Gradio for an interactive demo.

✨ Key Skills Demonstrated

Deep Learning with PyTorch → implemented custom CNN with GAP, BatchNorm, Dropout.

Computer Vision → classification of high-resolution animal faces.

Model Evaluation → accuracy, loss curves, generalization check.

Best Practices → clean repo, requirements.txt, README, .gitignore.

📎 References

AFHQ Dataset

PyTorch
