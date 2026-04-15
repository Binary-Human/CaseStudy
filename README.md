

## Create virtual environnment and install dependencies

```bash
# Create the virtual environment
python -m venv .venv

# Activate the virtual environment
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```
## LLM Setup

DeepEval relies on an LLM-as-judge to estimate its metrics

### Local Ollama


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

### Openai API

```bash
export OPENAI_API_KEY=<your_api_key>
```
or directly in .env. You might need a confident AI API key (`deepeval login`)

Make sure you update the model type in `metric.py` if you change models

```python
model = GPTModel(
    model="gpt-4.1-mini",
    temperature=0
)
```

## Dashboard

### Running
```bash
streamlit run src/dashboard.py 
```
