# Multi-Agent VQA Cartoon 

This project implements a three-agent collaborative framework for answering questions about cartoon content from the Pororo and Simpsons datasets. 

### Key Features

- **Multi-Agent Architecture**: Three specialized agents working collaboratively
  - **Visual Agent**: Analyzes visual content
  - **Language Agent**: Generates answers using contextual understanding
  - **Critic Agent**: Evaluates and refines answers for optimal accuracy

- **Dual Dataset Support**: Works with both Pororo and Simpsons cartoon datasets
- **Ablation Studies**: Comprehensive experiments to evaluate each agent's contribution
- **Multiple Evaluation Metrics**: BLEU, ROUGE, METEOR, and BLEURT scoring
- **Flexible Configuration**: Enable/disable agents for different experimental setups

## Project Structure

```bash
Multi_Agent_Cartoon/
├── dataset/
│   ├── pororo/
│   │   ├── descriptions.csv              # Episode descriptions
│   │   ├── qa.json                       # Question-answer pairs
│   │   └── Scenes_Dialogues/             # Subtitle and scene data
│   └── simpsons/
│       ├── val_images/                   # Validation images
│       ├── v1_Annotation_Val_simpsons_vqa.json
│       └── v1_Question_Val_simpsons_vqa.json
│
├── results/                              # Experimental results
│   ├── ablation/                         # Ablation study outputs
│   ├── analysis/                         # Detailed analysis files
│   ├── blip2/                            # BLIP-2 model results
│   │   ├── ablation/                     # BLIP-2 ablation studies
│   │   ├── analysis/                     # BLIP-2 detailed analysis
│   │   ├── comparison/                   # BLIP-2 comparisons
│   │   └── saved_figures/                # BLIP-2 visualizations
│   ├── comparison/                       # Cross-method comparisons
│   ├── metrics/                          # Evaluation metric scores
│   └── saved_figures/                    # Generated visualizations
│
├── compute_BLEU_ROUGE_METEOR_scores.ipynb # Metric computation
├── pororo_ablation_study.ipynb           # Pororo multi-agent ablation 
├── pororo_blip2_study.ipynb              # Pororo BLIP-2 experiments
├── simpsons_ablation_study.ipynb         # Simpsons multi-agent ablation
├── simpsons_blip2_study.ipynb            # Simpsons BLIP-2 experiments
│
├── requirements.txt                      # Python dependencies
└── README.md                            
```

## Requirements

### System Requirements
- Python 3.9 or higher
- Git
- Jupyter Notebook/Lab

### API Keys Required
- OpenAI API key
- Anthropic API key (optional, for Claude models)

### Step 1: Clone the Repository

```bash
git clone https://github.com/tongwu17/Multi_Agent_Cartoon.git
cd Multi_Agent_Cartoon
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On macOS/Linux:
source .venv/bin/activate

# On Windows:
.venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Configure API Keys

Create a `.env` file in the project root:

```bash
OPENAI_API_KEY=your_openai_api_key_here
```

Or set as environment variables:

```bash
# On macOS/Linux:
export OPENAI_API_KEY=your_openai_api_key_here
```

On Windows:
```bash
set OPENAI_API_KEY=your_openai_api_key_here
```

## Usage

### Running Ablation Studies

**Pororo Dataset:**
```bash
jupyter notebook pororo_ablation_study.ipynb
```

**Simpsons Dataset:**
```bash
jupyter notebook simpsons_ablation_study.ipynb
```

### Agent Configuration

Each notebook allows you to configure which agents are enabled:

```python
# Enable all three agents for full multi-agent system
ENABLE_VISUAL_AGENT = True
ENABLE_LANGUAGE_AGENT = True
ENABLE_CRITIC_AGENT = True

# Single-agent baseline (Language only)
ENABLE_VISUAL_AGENT = False
ENABLE_LANGUAGE_AGENT = True
ENABLE_CRITIC_AGENT = False

# Visual + Language (no Critic)
ENABLE_VISUAL_AGENT = True
ENABLE_LANGUAGE_AGENT = True
ENABLE_CRITIC_AGENT = False
```

### Model Selection

Choose your preferred model:

```python
# OpenAI Models (default)
MODEL_NAME = "gpt-4o-mini"              # Used for all agents (Visual, Language, Critic)

# Alternative: Anthropic Claude Models
# MODEL_NAME = "claude-3-5-sonnet-20241022"
# MODEL_NAME = "claude-3-5-haiku-20241022"
```

## Evaluation Metrics

The project includes comprehensive evaluation using:

- **BLEU**: Measures n-gram overlap with reference answers
- **ROUGE**: Evaluates recall-oriented understanding
- **METEOR**: Accounts for synonyms and stemming
- **BLEURT**: Neural metric for semantic similarity

Run evaluation:
```bash
jupyter notebook compute_BLEU_ROUGE_METEOR_scores.ipynb
```




