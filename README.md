# Punjabi News Sentiment Analysis Project

This project contains a full pipeline to scrape news articles, train a sentiment analysis model, and use it in an interactive application.

## Setup

1.  **Create a Project Directory:** Create a folder named `punjabi_news_project` and place all the files from this breakdown inside it.

2.  **Install Dependencies:** Open your terminal or command prompt, navigate to the `punjabi_news_project` directory, and install the required packages using the `requirements.txt` file.
    ```bash
    pip install -r requirements.txt
    ```

3.  **Edit Configuration:** Open `config.py` and carefully fill in the required values, especially:
    * `GOOGLE_API_KEY`
    * `LABELED_DATA_CSV` (the path to your manually labeled dataset)
    * Review other paths and settings to ensure they match your environment.

## How to Run the Pipeline

Execute the scripts in numerical order.

1.  **Scrape Data (Optional):** If you need to scrape new data.
    ```bash
    python 1_scraper.py
    ```
    This will create or append to `punjabi_tribune_nation_selenium_updated.csv`.

2.  **Split Labeled Data:** To create training, validation, and test sets.
    ```bash
    python 2_split_data.py
    ```
    This requires your `punjabi_news_sentiment_labelled_dataset_finall_3june.csv` file (or whatever you named it in `config.py`).

3.  **Tokenize Data:** To prepare the data for the model.
    ```bash
    python 3_tokenize_data.py
    ```
    This creates a `tokenized_dataset` directory.

4.  **Fine-Tune the Model:** To train the XLM-RoBERTa model.
    ```bash
    python 4_fine_tune_model.py
    ```
    This will create the final model in the `punjabi-news-sentiment-xlm-roberta_final` directory.

5.  **Run the Interactive App:** To use your trained model and the Gemini API.
    ```bash
    python main_interactive_app.py
    ```
