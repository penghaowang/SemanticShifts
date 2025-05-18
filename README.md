# Zero-Shot Semantic Shift Detection Toolkit

This repository contains the code and tools for the EMNLP 2025 paper: **"LLM for Zero-Shot Diachronic Semantic Shift Detection."** It implements a zero-shot method to detect semantic shifts for any corpus, in the paper I use Credit Suisse Bulletin corpus (1970–2018), using decoder hidden states from Llama 3 models, structured prompts, Jensen-Shannon divergence (JSD), and UMAP visualization.

## Overview

The toolkit processes historical texts, extracts contextualized word embeddings (CWEs) from Llama 3 hidden states, and analyzes semantic shifts over time. Key steps include:
1. Preprocessing the Credit Suisse Bulletin corpus.
2. Generating context-sensitive definitions with structured prompts (`icl_basic`).
3. Extracting hidden states (`input_last_token`, layers 21–33).
4. Computing semantic shift metrics (JSD) and visualizing trajectories (UMAP).


## Prerequisites

- **Python**: Version 3.9 or higher.
- **LLM Models**: Llama 3.1 8B Instruct and Llama 3.2 3B Instruct, or model supports Huggingface set_template() is smooth, others may need adaption.
- **Hugging Face Token**: Required for some model access (set as `HUGGING_FACE_HUB_TOKEN` environment variable).
- **Python Libraries**: Install dependencies listed in `requirements.txt`:
  ```bash
  pip install -r requirements.txt
  ```
  > **Note**: For ARCH64 platforms (like NVIDIA GH200 Grace Hopper GPUs), PyTorch nightly build is required:
  > ```bash
  > pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu121
  > ```
- **Environment**: Use a virtual environment (e.g., venv) for dependency management, conda is not recommended.
- **Dataset**: Credit Suisse Bulletin corpus (1970–2018, English portion). Contact the authors for access. Any other diachronic corpus can be used here.

## Project Setup

Before running any experiments, you should set up your environment:

```bash
# Create required directories
mkdir -p logs hidden_states outputs processed_data

# Install dependencies
pip install -r requirements.txt

# Install the required spaCy model
python -m spacy download en_core_web_sm
```

## Configuration

Edit `config.yaml` to match your environment and experimental setup. Key sections include:

- **Environment Setup**:
  - `env_activation_path`: Path to activate your virtual environment.
  - `hf_home`: Directory for Hugging Face cache and models.
  - `tokenizers_parallelism`: Set to false for cluster environments.

- **Common Paths**:
  - `base_hs_dir`: Directory for storing hidden states (default: `hidden_states`).
  - `output_dir`: Directory for outputs (default: `outputs`).
  - `processed_data_dir`: Directory for processed datasets (default: `processed_data`).
  - `log_dir`: Directory for log files (default: `logs`).

- **Model Configuration**:
  - Available models include `model_name_llama2_7b_chat` (Llama-2-7b-chat-hf), `model_name_llama3_8b_instruct` (Llama-3.1-8B-Instruct), and `model_name_perplexity_calc` (Llama-3.2-3B).

- **Default Parameters**:
  - `default_batch_size`: Batch size for processing (default: 16).
  - `default_temperature`: Generation temperature (default: 0.7).
  - `default_max_new_tokens`: Maximum new tokens to generate (default: 50).
  - `default_dim_reduce_method`: Dimension reduction method (default: "umap").
  - `default_extraction_method`: Hidden state extraction method (default: "input_last_token").

- **Words**:
  - Various predefined word lists are available in config.yaml, including nouns and verbs with POS tags.

Example updated `config.yaml` snippet that matches the actual structure:
```yaml
# --- Environment Setup ---
env_activation_path: ".venv/bin/activate"
hf_home: ".cache/huggingface"
tokenizers_parallelism: false

# --- Common Paths ---
base_hs_dir: "hidden_states"
output_dir: "outputs"
processed_data_dir: "processed_data"
log_dir: "logs"

# --- Model Configuration ---
model_name_llama2_7b_chat: "meta-llama/Llama-2-7b-chat-hf"
model_name_llama3_8b_instruct: "meta-llama/Llama-3.1-8B-Instruct"
model_name_perplexity_calc: "meta-llama/Llama-3.2-3B"

# --- Default Parameters ---
default_batch_size: 16
default_temperature: 0.7
default_max_new_tokens: 50
default_dim_reduce_method: "umap"
default_extraction_method: "input_last_token"

# --- Words & Methods ---
words_group1: "market:NOUN rate:NOUN bank:NOUN interest:NOUN [...]"
words_group1_verbs: "import:VERB export:VERB announce:VERB [...]"
```

## Project Structure

- **`config.yaml`**: Configuration file for paths, model, and parameters.
- **`requirements.txt`**: Lists dependencies (core ML libraries, NLP processing, visualization, and utilities).
- **`prompt_templates.py`**: Structured prompt templates for definition elicitation, including "basic", "icl_basic", "icl_detailed", and "icl_context_aware".
- **`preprocessing.py`**: Handles POS tagging (using Spacy), lemmatization, and perplexity filtering.
- **`inference.py`**: Extracts hidden states from Llama 3 models.
- **`evaluate.py`**: Contains functions for evaluating semantic shifts and dimension reduction.
- **`plot.py`**: Generates trajectory maps, JSD plots, and layer visualizations.
- **`dataloader.py`**: Handles dataset loading and manipulation.
- **`collect_semantic_data.py`**: Collects and processes semantic data from hidden states.
- **`logger_config.py`**: Configures logging functionality.
- **`outputs/`**:
  - `logs/`: Stores runtime logs.
  - `hidden_states/`: Saves extracted CWEs.
  - `plots/`: Contains visualizations.

## Data Processing Workflow

1. **Preprocessing**: Uses `preprocessing.py` to perform:
   - Text cleaning and standardization
   - POS tagging using spaCy 
   - Perplexity calculation for sentence filtering

2. **Dataloader**: Uses `dataloader.py` to:
   - Filter sentences containing target words
   - Apply context window expansion (sentence or token mode)
   - Format prompts using templates from `prompt_templates.py`

3. **Inference**: Uses `inference.py` to:
   - Generate definitions using LLM models
   - Extract hidden states using specified methods (default: `input_last_token`)
   - Save hidden states for further analysis

4. **Evaluation**: Uses `evaluate.py` to:
   - Reduce dimensionality of hidden states (UMAP, PCA, t-SNE)
   - Calculate semantic shift metrics (Jensen-Shannon divergence)
   - Compare distributions across time periods

5. **Visualization**: Uses `plot.py` to:
   - Generate trajectory plots
   - Create comparative visualizations
   - Plot time-based semantic shifts

## Dataset

The Credit Suisse Bulletin corpus (1970–2018, English portion) is used for analysis. It requires preprocessing (POS tagging, lemmatization, perplexity filtering with Llama 3.1 8B). WordNet annotations (via OpenAI O1-mini) are applied for silver-standard sense labels. If unavailable, contact the authors or use a placeholder corpus with similar structure (raw XML, OCR/PDF sources).

## Running the Code

To reproduce the paper's results using the Slurm workload manager, follow these steps:

1. **Setup Environment**:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   export HUGGING_FACE_HUB_TOKEN=your_token
   export TOKENIZERS_PARALLELISM=false
   
   # Create required directories
   mkdir -p logs hidden_states outputs processed_data
   
   # Install spaCy model
   python -m spacy download en_core_web_sm
   ```

2. **Configure `config.yaml`**:
   Update paths, model settings, and parameters as needed. Ensure the paths are set correctly for your environment.

3. **Run the Pipeline Using Slurm Scripts**:
   Execute the scripts in sequence from the `scripts/` directory:
   ```bash
   # Preprocess the corpus with POS tagging and perplexity filtering
   cd scripts
   sbatch submit_preprocessing.sh

   # Generate hidden states from the model for target words
   sbatch submit_inference.sh  

   # Run semantic shift analysis and generate plots
   sbatch submit_plot_semantic_shift.sh
   ```

   You can also run specific analysis tasks using specialized scripts:
   ```bash
   # Generate word trajectory plots
   sbatch submit_word_trajectory.sh
   
   # Generate layer-specific plots
   sbatch submit_layer_plot.sh
   
   # Run all words trajectory analysis
   sbatch submit_all_words_trajectory.sh
   ```

4. **View Slurm Job Status**:
   ```bash
   squeue -u your_username
   ```

5. **View Results**:
   Visualization outputs are saved to the specified output directory. Check the log output to find the exact output paths:
   ```bash
   cat logs/semantic_shift_*.log
   ```
   
   Then examine the generated visualizations in the output directories.

## Reproducibility Notes


- **Precision**: While the paper used `bf16` precision, the current configuration is set up for `fp16` as seen in path names.
- **Context Length**: For optimal semantic differentiation, the system uses surrounding context sentences.
- **Extraction Method**: The paper found `input_last_token` to provide best results for capturing semantic shifts.
- **Hardware**: Experiments used GPUs supporting bf16 (e.g., NVIDIA A100). Adjust `batch_size` for smaller GPUs.

## Output

- **Logs**: Stored in `logs/` (e.g., `preprocessing.log`, `inference.log`).
- **Hidden States**: Saved in `hidden_states/` (e.g., `interest_NOUN/input_last_token/combined/hidden_states/`).
- **Plots**: Generated in `outputs/plots/`:
  - Semantic trajectory maps, JSD trends, sense distributions, layer-wise UMAP plots, extraction method comparisons, and similarity heatmaps.

## Troubleshooting

If you encounter any issues:

1. **Directory Structure**: Ensure all required directories exist (`logs`, `hidden_states`, `outputs`, `processed_data`).
2. **Dependencies**: Verify all dependencies are installed with `pip list`.
3. **spaCy Model**: Confirm the spaCy model is installed with `python -c "import spacy; print(spacy.util.get_installed_models())"`.
4. **Hugging Face Token**: Check your `HUGGING_FACE_HUB_TOKEN` environment variable is set correctly.

## Contact

For questions or to request the Credit Suisse Bulletin corpus, please open an issue in this repository. You can also contact the authors through the conference submission system if needed.