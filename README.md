# Multi-Agent VQA Cartoon 

This project implements and evaluates multi-agent and single-agent approaches for answering questions about cartoon content (Pororo and Simpsons).

## Project Structure

```bash
Multi_Agent_Cartoon/
├── dataset/
│   ├── pororo/
│   │   ├── descriptions.csv        
│   │   ├── qa.json
│   │   └── Scenes_Dialogues/       
│   └── simpsons/
│       ├── val_images/
│       ├── v1_Annotation_Val_simpsons_vqa.json
│       └── v1_Question_Val_simpsons_vqa.json             
├── results/                       
├── pororo_multi_agent.ipynb
├── pororo_single_agent.ipynb
├── simpsons_multi_agent.ipynb
├── simpsons_single_agent.ipynb            
└── README.md
```

## Requirements

### System Requirements
- Python 3.9 or higher
- Git
- Jupyter Notebook/Lab

### API Keys Required
- OpenAI API key
- Anthropic API key

## Installation

### 1. Create and activate virtual environment

**Create virtual environment**

```bash
python -m venv .venv
```

Activate virtual environment
On MacOS/Linux:
```bash
source .venv/bin/activate
```
On Windows:
```bash
.venv\Scripts\activate
```

### 2. Install required packages:
```bash
pip install openai anthropic python-dotenv pandas tqdm word2number
```

Environment Configuration
Set your API keys as environment variables before running:

On macOS/Linux:
```bash
export OPENAI_API_KEY=your_openai_api_key_here
export ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

On Windows:
```bash
set OPENAI_API_KEY=your_openai_api_key_here
set ANTHROPIC_API_KEY=your_anthropic_api_key_here
```
