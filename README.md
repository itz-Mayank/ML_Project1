# Student Performance Prediction – End-to-End ML Project  

This project predicts **students’ math performance** based on demographic and academic attributes such as gender, parental education, lunch type, test preparation, and reading/writing scores.  
The dataset is sourced from [Kaggle – Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams).

### Live on AWS EB: [Click Here](http://studentperformance-env-1.eba-qvxygaz3.us-east-1.elasticbeanstalk.com/).
---

## Objective
To build a **machine learning pipeline** that can:
- Clean and preprocess the data  
- Perform feature engineering and transformation  
- Train multiple ML models and optimize hyperparameters  
- Save and deploy the best-performing model using a scalable and reusable pipeline  

---

## Project Structure
```bash
ML_Project_1/
│
├── artifacts/                     # Stores serialized models and processed data
│   ├── preprocessor.pkl
│   ├── train.csv
│   ├── test.csv
│   ├── raw.csv
│
├── logs/                          # Application logs
│
├── notebook/                      # Jupyter notebooks for EDA & experimentation
│   ├── 1. EDA STUDENT PERFORMANCE.ipynb
│   ├── 2. MODEL TRAINING.ipynb
│   └── data/data.csv
│
├── src/
│   ├── components/                # Core ML components
│   │   ├── data_ingestion.py
│   │   ├── data_transformation.py
│   │   ├── model_trainer.py             (to be added)
│   │   └── model_hyperparameter_tuning.py (to be added)
│   │
│   ├── pipeline/                  # Training & prediction pipelines
│   │   ├── train_pipeline.py
│   │   ├── predict_pipeline.py
│   │   └── __init__.py
│   │
│   ├── logs/                      # Logging config files
│   ├── exception.py               # Custom exception handling
│   ├── logger.py                  # Logging utilities
│   ├── utils.py                   # Helper functions (e.g., model saving/loading)
│   └── __init__.py
│
├── venv/                          # Virtual environment
│
├── .gitignore
├── requirements.txt               # Required dependencies
├── setup.py                       # Package configuration
└── README.md
```

---

## Tech Stack
- **Language:** Python 3.10+  
- **Libraries:** `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `seaborn`, `joblib`  
- **Frameworks:** Flask (for deployment)  
- **Cloud:** AWS EC2, Azure Container Instance (planned)  
- **Version Control:** Git + GitHub  
- **Environment:** Virtual Environment / Conda  

---

## Key Modules
### 1️⃣ Data Ingestion
- Loads raw data from `notebook/data/data.csv`
- Splits data into train/test sets
- Saves the processed datasets in `artifacts/`

### 2️⃣ Data Transformation
- Handles missing values with `SimpleImputer`
- Encodes categorical features using `OneHotEncoder`
- Scales features using `StandardScaler`
- Saves preprocessor object (`preprocessor.pkl`)

### 3️⃣ Model Trainer *(upcoming)*
- Trains multiple ML models (e.g., Linear Regression, RandomForest, XGBoost)
- Evaluates metrics (R², RMSE, MAE)
- Saves the best-performing model

### 4️⃣ Hyperparameter Tuning *(upcoming)*
- Uses `GridSearchCV` or `RandomizedSearchCV` for model optimization

### 5️⃣ Prediction Pipeline *(upcoming)*
- Loads saved preprocessor and model to predict unseen data

---

## Training the Pipeline
```bash
# Step 1: Activate environment
venv\Scripts\activate

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Run Data Ingestion
python src/components/data_ingestion.py

# Step 4: Run Data Transformation
python src/components/data_transformation.py

# Step 5: Run Model Training (when implemented)
python src/components/model_trainer.py
```
---

## Deployment (Planned)

### AWS EC2
- Containerize the application using **Docker**  
- Deploy the **Flask app** and trained **ML model** on an EC2 instance  
- Use **NGINX** or **Gunicorn** for serving the production app  

### Azure Container Instance
- Deploy using **Azure CLI** or the **Azure Portal**  
- Build Docker image and push it to **Azure Container Registry (ACR)**  
- Run and scale the containerized app directly on Azure  

---

## Results

| Model | R² Score | RMSE | MAE |
|--------|-----------|------|-----|
| Linear Regression | 0.88 | 5.40 | 4.22 |
| Lasso | 0.83 | 6.52 | 5.16 |

---

## Utilities

- **Custom Logging:** Provides detailed tracking of every step in the ML workflow
- **Custom Exception Handling:** Ensures robust and clean error management
- **Reusable Pipelines:** Modularized preprocessing and model training pipelines for flexibility

---

## Author

**Mayank Meghwal**
*Data Scientist | Machine Learning Engineer*

**Email:** mayankmeg207@gmail.com
**GitHub:** [itz-Mayank](https://github.com/itz-Mayank)

---

## Future Enhancements

- Implement **CI/CD pipeline** with GitHub Actions  
- Automate deployment using **Docker** and **Kubernetes**  
- Integrate **model monitoring** and **automated retraining** system  
- Add support for **multi-cloud deployment** (AWS + Azure + GCP)  

---

## License

This project is open-source and available under the **MIT License**.

---


