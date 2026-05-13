# **🇵🇰 Urdu AI Sentiment System**

### Fine-tuning mBERT for Low-Resource Urdu NLP + Real-Time Deployment


##  **Project Overview**

This project presents an **end-to-end Urdu sentiment analysis system** built by fine-tuning a multilingual transformer model on an Urdu dataset. It classifies Urdu text into **Positive, Negative, and Neutral sentiments** and is deployed as a real-time web application using Gradio and Hugging Face Spaces.

It demonstrates a complete NLP pipeline from <br>

 **data preprocessing → model training → evaluation → deployment**.

## **Problem Statement**

Urdu is a **low-resource language** in Natural Language Processing (NLP), with limited labeled datasets and pretrained models.

This creates challenges for building accurate sentiment analysis systems in real-world Urdu applications.

### **Objectives**:

* Build a robust Urdu sentiment classification system
* Fine-tune a transformer-based model on Urdu dataset
* Develop an end-to-end deployable AI pipeline
* Enable real-time sentiment prediction through a web interface


##  **Key Project Highlights**

* 🇵🇰 Focus on Urdu language NLP
* Fine-tuned **mBERT transformer model**
* Real-time inference with Gradio
*  Deployed on Hugging Face Spaces
* Full NLP pipeline training + evaluation + deployment



##  **Model Used**

Use a pretrained multilingual transformer:

👉 **mBERT**

Why mBERT?

* Supports Urdu language
* Strong performance on low-resource NLP tasks
* Pretrained on multilingual corpora
* Easy fine-tuning for classification tasks


## **Dataset**

* Source: HuggingFace Dataset Hub
* Dataset: Urdu Multi-Domain Classification

🔗 [https://huggingface.co/datasets/umar178/UrduMultiDomainClassification](https://huggingface.co/datasets/umar178/UrduMultiDomainClassification)


## **Data Preprocessing**

The dataset was cleaned and prepared using the following steps:


* Removed irrelevant columns: `topic`, `intent`, `binary`
* Filtered dataset to keep only 3 sentiment classes
* Removed low-frequency class `request`
* Ensured balanced class distribution


##  **Text Normalization**

Urdu text was normalized using **LughaatNLP**.

### ✔ Benefits:

* Removes noise and inconsistencies
* Standardizes Urdu spelling variations
* Improves tokenization quality
* Enhances model generalization


## 🔁 **Project Pipeline**

```text
Raw Urdu Text
      ↓
Text Normalization (LughaatNLP)
      ↓
Tokenization (mBERT Tokenizer)
      ↓
Fine-tuned Transformer Model
      ↓
Sentiment Prediction (Positive / Negative / Neutral)
      ↓
Gradio Web Interface
```

##  **Application UI**


![Positive Sentiment](https://github.com/user-attachments/assets/0c9cc26b-043f-4f5a-a535-ea3b8e8ded5b)  

<br>

![Neutral Sentiment](https://github.com/user-attachments/assets/97cd9d70-a05d-495b-b979-31310a4c8752)


<br>

![Negative Sentiment](https://github.com/user-attachments/assets/f7936b23-9778-4d36-8890-07ddd2422e9d)  




##  **Tech Stack**

* Python 
* PyTorch 
* Hugging Face Transformers 
* Datasets Library
* Gradio UI 
* Scikit-learn 


##  **Model Training**

* Model: mBERT multilingual BERT
* Task: Sentiment Classification
* Loss Function: CrossEntropyLoss
* Optimizer: AdamW
* Evaluation: Accuracy, Precision, Recall, F1-score
* Best model saved using Hugging Face Trainer

## **Evaluation Results**

The model achieves strong performance on Urdu sentiment classification:

* Accuracy: 0.91
* F1 Score: 0.91
* Balanced performance across all classes


## **Deployment**

The model is deployed using:

* Gradio interface
* Hugging Face Spaces

It supports real-time prediction of Urdu text sentiment.



## 🌐 **Live Demo**

```
https://huggingface.co/spaces/arifa-batool/urdu-sentiment-classifier
```


## **Key Features**

✔ End-to-end NLP pipeline

✔ Low-resource language support Urdu

✔ Transformer-based fine-tuning

✔ Real-time web deployment

✔ Clean and interactive UI



## **Conclusion**

This project demonstrates a complete **end-to-end NLP system for Urdu sentiment analysis**, covering:

* Fine-tuning transformer models for Urdu NLP
* Effective preprocessing for low-resource languages
* Evaluation using standard ML metrics
* Deployment using Gradio and Hugging Face Spaces



## **Run Locally**

```bash id="d4k2l1"
git https://github.com/batoolarifa/urdu-ai-sentiment-system.git
```

```bash cd mental-health-sentiment-analyzer
pip install -r requirements.txt
streamlit run app.py
```




## 👤 **Author**

**Syeda Arifa Batool**  
SE @ Karachi University | AI/ML Engineer | Applying technology to create real-world value 📈



## 🔗 **Connect with Me**

- **LinkedIn:** [Syeda Arifa Batool](https://www.linkedin.com/in/arifa-batool/)  
- **Kaggle:** [Syeda Arifa Batool](https://www.kaggle.com/thearifabatool)  
- **Email:** [thearifabatool@gmail.com](mailto:thearifabatool@gmail.com)

⭐ If you find this project useful, feel free to star the repository!
