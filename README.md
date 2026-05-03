# OUSysAdmin-LLMs_Template
=======
# LLM Tasks Template

This repository provides a template for performing common Large Language Model (LLM) tasks such as text summarization and sequence classification using Hugging Face Transformers.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/llm-tasks-template.git
    cd llm-tasks-template
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: .venv\Scripts\activate
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
    (You would create `requirements.txt` from your current Colab environment, e.g., `pip freeze > requirements.txt`)

4.  **Hugging Face Access Token (Optional but Recommended):**
    For some models, or to push models to the Hub, you'll need an [Hugging Face Access Token](https://huggingface.co/docs/hub/en/security-tokens).
    You can log in via:
    ```bash
    huggingface-cli login
    ```
    Or by setting the `HF_TOKEN` environment variable.


## Usage

Run the main script with arguments to specify the task and model.

### Summarization

To perform summarization:

```bash
python main.py --task summarize --model "Medical Summarization" --text_file "path/to/your/texts.txt" --output_file "summaries.json"
>>>>>>> Initial commit: adding project files
```



### Classification

To perform classification:

```bash
python main.py --task classify --model "PubMedBERT" --data_file "path/to/your/data.csv" --output_file "predictions.csv"
```


Your data file (e.g., CSV) should have a 'text' column for input and optionally a 'label' column if you want to evaluate accuracy or fine-tune.

Available classification models (refer to `main.py` for exact keys):
*   `PubMedBERT`
*   `BERT-base-uncased`
*   `RoBERTa-base`

## Project Structure

```
.gitignore
README.md
requirements.txt
main.py
summarize.py
classify.py
```

Feel free to contribute or suggest improvements!
```