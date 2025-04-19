# Semantic Shifts Analysis Toolkit

This repository contains scripts and tools for analyzing semantic shifts and hidden state representations in language models. It includes functionality for data loading, preprocessing, hidden state extraction, dimensionality reduction, clustering, and visualization.

## Prerequisites

*   **Python:** Version 3.9+ recommended.
*   **SLURM:** The submission scripts (`scripts/submit_*.sh`) are designed for a SLURM-based cluster environment. Adaptations may be needed for other environments.
*   **Python Libraries:** Install the required libraries using pip:
    ```bash
    pip install -r requirements.txt
    ```
*   **Environment:** A Python virtual environment (like conda or venv) is strongly recommended. The scripts expect the environment activation path to be set in the configuration file (`ENV_ACTIVATION_PATH`).
*   **Hugging Face Token (Optional):** If accessing private models or needing higher rate limits from Hugging Face Hub, ensure your `HUGGING_FACE_HUB_TOKEN` environment variable is set externally.

## Configuration

All primary configuration parameters are managed in `config.yaml`. Before running any scripts, review and modify this file according to your environment and experimental setup. Key sections include:

*   `paths`: Define input data paths, output directories (logs, plots, hidden states), and the environment activation path (`ENV_ACTIVATION_PATH`).
*   `model`: Specify the Hugging Face model name(s) to use (e.g., `MODEL_NAME_LLAMA3_8B_INSTRUCT`).
*   `parameters`: Set hyperparameters for various stages like batch size (`DEFAULT_BATCH_SIZE`), dimensionality reduction (`DEFAULT_DIM_REDUCE_METHOD`), clustering, plotting, etc.
*   `words`: Define lists of target words/POS tags for analysis (e.g., `WORDS_DATALOADER`, `WORDS_GROUP1`).
*   `methods`: Define extraction methods, dimensionality reduction techniques, etc.

The `scripts/load_config.sh` script is used by the submission scripts to load these parameters into the shell environment.

## Project Structure

*   `config.yaml`: Main configuration file.
*   `requirements.txt`: Python package dependencies.
*   `scripts/`: Contains SLURM submission scripts (`submit_*.sh`) and helper scripts (`load_config.sh`).
*   `plot/`: Contains Python scripts for generating plots (e.g., layer similarity, word trajectories). (Inferred structure)
*   Root Directory Python Files (`*.py`): Core logic for data loading (`dataloader.py`), preprocessing (`preprocessing.py`), inference/extraction (`inference.py`), analysis (`run_analysis.py`), plotting (`plot.py`), etc.
*   `logs/`: Default directory for SLURM job logs (created automatically).
*   Output Directories (e.g., `layer_plots/`, `word_trajectory_plots/`, `combined_layers_plots/`): Defined in `config.yaml`, store generated plots and analysis results.

## Running the Code

The primary way to run the analyses is through the SLURM submission scripts located in the `scripts/` directory.

1.  **Configure `config.yaml`**: Ensure all paths, model names, and parameters are set correctly for your environment and desired experiment.
2.  **Activate Environment**: Make sure your configured Python environment is active or ensure the `ENV_ACTIVATION_PATH` in `config.yaml` points to the correct activation script.
3.  **Submit Jobs**: Use `sbatch` to submit the desired job script.

    ```bash
    # Example: Submit the dataloader job
    sbatch scripts/submit_dataloader.sh

    # Example: Submit the layer plotting job
    sbatch scripts/submit_layer_plot.sh

    # Example: Submit the word trajectory plotting job
    sbatch scripts/submit_word_trajectory_all.sh

    # Example: Submit the combined layer plotting job
    sbatch scripts/submit_rpcl.sh

    # ... and so on for other scripts like dbscan, preprocessing, etc.
    ```

Each `submit_*.sh` script typically corresponds to a specific stage of the analysis pipeline (e.g., data preparation, hidden state extraction, plotting specific views). Examine the scripts to understand the specific Python script they execute and the parameters they use (loaded from `config.yaml`).

## Output

*   **Logs:** SLURM output and error logs are stored in the `logs/` directory (or the path specified by `LOG_DIR` in `config.yaml`).
*   **Datasets/Hidden States:** Processed data or extracted hidden states might be saved to locations defined in `config.yaml` (e.g., `BASE_HS_DIR`). Example path from scripts: `/capstor/scratch/cscs/phwang/datasets/window_{window}`.
*   **Plots:** Generated visualizations are saved in directories specified in `config.yaml` or hardcoded in the submission scripts (e.g., `layer_plots/`, `combined_layers_plots/`, `stacked_layers_plots/`, `word_trajectory_plots/`).

## Environment Variables

*   `TOKENIZERS_PARALLELISM`: Set to `false` by default in the scripts to avoid potential issues with parallelism in some environments.
*   `HUGGING_FACE_HUB_TOKEN`: As mentioned, set this externally if needed.
*   `HF_HOME`: Can be set externally to control the Hugging Face cache directory if the default location is not desired. (Note: Direct export within scripts was removed to avoid potential secret scanning issues).