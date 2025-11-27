# Multi-Agent VQA Cartoon 

This project implements a **three-agent collaborative framework** for answering questions about cartoon content from the Pororo and Simpsons datasets. The system combines visual understanding, language reasoning, and critical evaluation to achieve state-of-the-art performance on cartoon VQA tasks.

### Key Features

- **Multi-Agent Architecture**: Three specialized agents working collaboratively
  - **Visual Agent**: Analyzes cartoon images and extracts visual information
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
│   │   ├── pororo_ablation_*.csv         # Pororo ablation results
│   │   └── simpsons_ablation_*.csv       # Simpsons ablation results
│   ├── analysis/                         # Detailed analysis files
│   │   ├── pororo_analysis_*.csv         # Pororo agent analysis
│   │   └── simpsons_analysis_*.csv       # Simpsons agent analysis
│   ├── comparison/                       # Cross-method comparisons
│   │   ├── pororo_comparison_*.csv       # Pororo configuration comparisons
│   │   └── simpsons_comparison_*.csv     # Simpsons configuration comparisons
│   ├── evaluation/                       # Evaluation metric scores
│   │   └── metrics_*.csv                 # BLEU, ROUGE, METEOR scores
│   ├── human_evaluation/                 # Human evaluation results
│   │   ├── pororo_*_human_evaluation.csv # Pororo human annotations
│   │   └── simpsons_*_human_evaluation.csv # Simpsons human annotations
│   ├── single/                           # Single-agent baseline results
│   │   ├── pororo_single_agent_*.csv     # Pororo baseline
│   │   └── simpsons_single_agent_*.csv   # Simpsons baseline
│   └── blip2/                            # BLIP-2 model results
│       ├── ablation/                     # BLIP-2 ablation studies
│       ├── analysis/                     # BLIP-2 detailed analysis
│       ├── comparison/                   # BLIP-2 comparisons
│       └── saved_figures/                # BLIP-2 visualizations
│
├── bleurt/                               # BLEURT metric implementation
│   ├── score.py                          # BLEURT scoring functions
│   ├── model.py                          # BLEURT model wrapper
│   └── test_checkpoint/                  # BLEURT model checkpoint
│
├── saved_figures/                        # Generated visualizations
│   └── *.png                             # Analysis plots and charts
│
├── Notebooks:
│   ├── Pororo Experiments:
│   │   ├── pororo_single_agent.ipynb     # Can be deleted in final version
│   │   ├── pororo_ablation_study1.ipynb  # Multi-agent ablation v1. Should be deleted in final version
│   │   ├── pororo_ablation_study2.ipynb  # Multi-agent ablation v2
│   │   └── pororo_blip2_ablation_study2.ipynb # Can be modified to read ablation CSV results directly
│   │       # and replace only Visual Agent outputs while keeping Language Agent 
│   │       # and Critic Agent results identical for fair comparison in final version
│   │
│   ├── Simpsons Experiments:
│   │   ├── simpsons_single_agent.ipynb   # Can be deleted in final version
│   │   ├── simpsons_ablation_study.ipynb # Multi-agent ablation
│   │   └── simpsons_blip2_ablation_study.ipynb # Can be modified to read ablation CSV results directly
│   │       # and replace only Visual Agent outputs while keeping Language Agent 
│   │       # and Critic Agent results identical for fair comparison in final version
│   │
│   ├── Baseline Models:
│   │   └── blip2_vqa.ipynb               # Can be deleted in final version
│   │
│   └── Evaluation:
│       └── compute_BLEU_ROUGE_METEOR_scores.ipynb # Metric computation
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
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

Or set as environment variables:

```bash
# On macOS/Linux:
export OPENAI_API_KEY=your_openai_api_key_here
export ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

On Windows:
```bash
set OPENAI_API_KEY=your_openai_api_key_here
set ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

## Usage

### Quick Start with Example Notebook

For a complete walkthrough of the multi-agent system, see the tutorial notebook:
```bash
jupyter notebook multi_agent_tutorial.ipynb
```

### Running Ablation Studies

**Pororo Dataset:**
```bash
jupyter notebook pororo_ablation_study1.ipynb
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

Choose your preferred models:

```python
# OpenAI Models
MODEL_NAME = "gpt-4o"              # Visual Agent
LANGUAGE_MODEL = "gpt-4o-mini"     # Language Agent
CRITIC_MODEL = "gpt-4o-mini"       # Critic Agent

# Or Anthropic Models
MODEL_NAME = "claude-3-5-sonnet-20241022"
LANGUAGE_MODEL = "claude-3-5-haiku-20241022"
CRITIC_MODEL = "claude-3-5-haiku-20241022"
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

## Experimental Configurations

### Ablation Studies

The project includes several ablation configurations:

1. **Single Language Agent** (Baseline)
   - Only Language Agent enabled
   - No visual information

2. **Visual + Language**
   - Visual and Language Agents
   - No critic evaluation

3. **Language + Critic**
   - Language and Critic Agents
   - No visual information

4. **Full Multi-Agent** (Proposed)
   - All three agents enabled
   - Complete collaborative framework

## Results

Results are organized in the `results/` directory with the following structure:

### Standard Experiments
- **`ablation/`**: Ablation study results for different agent configurations
  - Language only (baseline)
  - Language + Critic
  - Visual + Language
  - Visual + Language + Critic (full system)

- **`analysis/`**: Detailed per-question analysis including:
  - Visual description quality assessments
  - Critic agent decision logs
  - Answer comparison between configurations

- **`comparison/`**: Cross-configuration performance comparisons
  - Accuracy comparisons across different agent setups
  - Statistical analysis of improvements

- **`evaluation/`**: Automated metric scores
  - BLEU, ROUGE, METEOR scores
  - Metric-based performance analysis

- **`single/`**: Single-agent baseline results
  - Pure language model performance without multi-agent framework

### Advanced Experiments
- **`blip2/`**: BLIP-2 vision-language model experiments
  - Ablation studies combining BLIP-2 with multi-agent framework
  - Analysis of BLIP-2 visual descriptions
  - Comparison with GPT-4o visual agent
  - Visualization of BLIP-2 performance

### Human Evaluation
- **`human_evaluation/`**: Human-annotated evaluation results
  - Human judgments on answer quality
  - Ground truth comparisons
  - Inter-annotator agreement analysis


