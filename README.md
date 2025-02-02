# Basic-RAG
Basic RAG with FAISS and Mistral API (free tier)

## Data
Put documents in data/ folder. Only pdf files are supported at the moment. The present documents are regulatory cybersecurity norms. 

## Running script

### Create a python env
```
python -m venv venv
```

### Activate environment
```
venv/Scripts/activate
```

### Requirements
```
pip install -r requirements.txt
```

### Run streamlit app or Run main
```
streamlit run src/app.py
```
```
python src/main.py
```
