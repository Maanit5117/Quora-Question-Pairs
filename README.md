# Quora Question Pairs Duplicate Detection

A machine learning project to identify whether two questions posted on Quora have the same intent and meaning.

## Overview

The goal of this project is to classify question pairs as either duplicates (1) or non-duplicates (0). The repository contains exploratory data analysis, natural language processing feature extraction, model training, and a Streamlit frontend for predictions.

## Feature Engineering

The feature pipeline produces 6,022 numerical features for each pair:

1. Basic text features:
   - Character lengths of both questions
   - Word counts of both questions
   - Common word count
   - Total word count
   - Word share ratio

2. Token-level features:
   - Common non-stopword ratio (minimum and maximum normalized)
   - Common stopword ratio (minimum and maximum normalized)
   - Common token ratio (minimum and maximum normalized)
   - First word equality flag
   - Last word equality flag

3. Length and fuzzy string features:
   - Absolute token length difference
   - Mean token length
   - Longest common substring ratio
   - Rapidfuzz metrics: QRatio, partial ratio, token sort ratio, token set ratio

4. Bag-of-Words features:
   - CountVectorizer with 3,000 max features for Question 1 and 3,000 for Question 2 (6,000 features total)

## Models

- Random Forest Classifier (trained and serialized as `model.pkl`)
- XGBoost Classifier (evaluated in `advance_features.ipynb`)

## Repository Structure

- `app.py`: Streamlit frontend for user input and predictions.
- `helper.py`: Preprocessing, feature extraction, and feature vector generator.
- `advance_features.ipynb`: Feature engineering, model evaluation, and artifact generation notebook.
- `model.ipynb`: Baseline exploration and initial modeling notebook.
- `model.pkl`: Trained Random Forest model (tracked via Git LFS).
- `cv.pkl`: Fitted CountVectorizer vocabulary.
- `requirements.txt`: Python package requirements.
- `run.sh`: Bash script to run the Streamlit application.
- `.gitignore`: Ignored dataset archives, caches, and system files.

## Setup and Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/Maanit5117/Quora-Question-Pairs.git
   cd Quora-Question-Pairs
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. If cloning with Git LFS, pull large model files:
   ```bash
   git lfs pull
   ```

## Running the Application

Launch the Streamlit web application:

```bash
streamlit run app.py
```

Or run the shell script:

```bash
./run.sh
```

Access the interface in your browser at `http://localhost:8501`. Enter two questions and submit to view whether the pair is predicted as duplicate or not duplicate, along with confidence probabilities and extracted feature values.

## Dataset

This project is trained on the Quora Question Pairs dataset from Kaggle.
