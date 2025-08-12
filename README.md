# LoRA-based BERT Sentiment Analysis

This project fine-tunes a **BERT** model using **LoRA (Low-Rank Adaptation)** for sentiment classification on the **IMDb Movie Review** dataset.  
The final model is deployed as a **Gradio web application** for real-time prediction.

---

## Features
- **Parameter-efficient fine-tuning** with LoRA (updates <1% of model parameters)
- **BERT-base** model for binary sentiment classification (Positive / Negative)
- **IMDb dataset** with 50,000 labeled movie reviews
- **Gradio-powered** interactive demo for real-time text sentiment prediction

---

##  Project Structure

train.py # Training & evaluation script
app.py # Gradio web app for inference
requirements.txt # Dependencies
README.md # Project documentation
data/ # IMDb dataset (not included, download link below)


## Dataset
- **Name**: IMDb Movie Review Dataset
- **Size**: 50,000 reviews (balanced between positive and negative)
- **Source**: [IMDb Dataset on Kaggle](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews)

---


##  Tech Stack
- Python  
- PyTorch  
- HuggingFace Transformers  
- PEFT (LoRA)  
- Gradio  
- Scikit-learn

  
## 🚀 Installation
```bash
git clone https://github.com/<your-username>/LoRA-based-BERT-Sentiment-Analysis.git
cd LoRA-based-BERT-Sentiment-Analysis
pip install -r requirements.txt


【Training】 Make sure you have the IMDb dataset in the data/ folder as IMDB_Dataset.csv.
【Run】 Web App，After training and saving the model: run python app.py
The app will be available at:http://localhost:7860


Example Prediction
input:This movie was fantastic!
output: {
  "positive": 0.95,
  "negative": 0.05
}


License
This project is licensed under the MIT License.

