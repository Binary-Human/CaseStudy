

## Create virtual environnment and install dependencies

```bash
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
## Local LLM

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama pull phi3

# Configuration DeepEval
deepeval set-ollama --model=phi3 --base-url="http://localhost:11434"

# Quick test
curl http://localhost:11434/api/generate -d '{
  "model": "phi3",
  "prompt": "Hello"
}'

```

## Dashboard

# Running
```bash
streamlit run src/dashboard.py 
```
