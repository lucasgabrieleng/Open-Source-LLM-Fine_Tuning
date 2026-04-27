Fine-Tuning an Open-Source LLM for a Legal Assistant App
This repository contains a project developed at Data Science Academy focused on fine-tuning an open-source Large Language Model (LLM) to serve as a specialized legal assistant. The project demonstrates the full pipeline from data loading and preprocessing to model training and deployment.

🚀 Project Overview
The goal of this project is to adapt a general-purpose LLM to the legal domain. By fine-tuning the model on a specific dataset of legal questions and answers, we aim to improve its accuracy and relevance when handling inquiries related to legal procedures and advice.

🛠️ Tech Stack
Model: google/flan-t5-base (Text-to-Text Transfer Transformer)

Libraries:

transformers: For model loading and training

datasets: For data handling and processing

evaluate: For model performance metrics (ROUGE)

nltk: For text tokenization

Infrastructure: Python 3.12+ with GPU acceleration (A100 recommended)

📋 Project Structure
The notebook follows a structured approach to LLM fine-tuning:

Installing and Loading Packages: Setup of the environment and necessary dependencies.

Loading the Data: Importing the dataset.csv and performing an 80/20 train-test split.

Preprocessing: Implementing a specialized function (dsa_fn_preprocess) that adds an "answer the question:" prefix to all inputs and tokenizes the data with specific max lengths (128 for questions, 512 for answers).

Model Evaluation: Utilizing ROUGE-1, ROUGE-2, and ROUGE-L metrics to evaluate the quality of the generated legal summaries.

Model Training: Training the flan-t5-base model using the Seq2SeqTrainer API with defined hyperparameters (learning rate: 3e-4, epochs: 3).

Deployment: Loading the saved model from disk and generating predictions for new legal queries.

⚙️ How to Run
Clone the repository:

Bash
git clone https://github.com/your-username/legal-assistant-llm.git
cd legal-assistant-llm
Install dependencies:

Bash
pip install -r requirements.txt
Prepare the dataset: Ensure dataset.csv is in the root directory.

Run the notebook: Open Open-Source-LLM-Fine_Tuning.ipynb in Jupyter or Google Colab and execute the cells.

🧪 Usage Example
Once the model is trained and saved in the modelo_salvo directory, you can use it as follows:

Python
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

model = AutoModelForSeq2SeqLM.from_pretrained('modelo_salvo')
tokenizer = AutoTokenizer.from_pretrained('modelo_salvo')

query = "Can I move out of state with my children if I have a custody agreement in that state?"
inputs = tokenizer(query, return_tensors="pt").input_ids
outputs = model.generate(inputs, max_length=50, temperature=0.4, do_sample=True)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
