# Book recommender chatbot with Streamlit

## Data

This project uses the [CMU Book Summary Dataset](https://www.kaggle.com/datasets/ymaricar/cmu-book-summary-dataset) (~16,000 books with plot summaries from Wikipedia, titles, authors, and genres). It was chosen to avoid loading a massive dataset while still having enough variety for meaningful recommendations.

Dataset license:      [![License: CC BY-SA 3.0](https://img.shields.io/badge/License-CC%20BY--SA%203.0-lightgrey.svg)](https://creativecommons.org/licenses/by-sa/3.0/)

## ⚙ Environment Setup

### Clone the repository and access it:
```
git clone git@github.com:MartaBallesterB/book-recommender-chatbot-with-streamlit.git
cd book-recommender-chatbot-with-streamlit
```

### Install dependencies with uv:
```
uv sync
```

### Download the dataset:
This project uses `kagglehub` library to fetch the dataset automatically (no manual download or Kaggle account needed in this case!).

It is downloaded and cached locally once you run the code for the first time. You can also trigger the download manually before starting the app:
```
uv run python -c "import kagglehub; kagglehub.dataset_download('ymaricar/cmu-book-summary-dataset')"
```

The cache is stored at `~/.cache/kagglehub/` and reused on the next runs.

### Run the app on localhost:
```
uv run streamlit run streamlit-app/app.py
```

## Project iterations!

I want to use different approaches to recommend books and I will create different versions to compare them. 

- **V1** — TF-IDF + cosine similarity. 'Bag-of-words' approach: computes a book_vector (from dataset) and a query_vector (user input) and rank it by cosine similarity. Issues I saw: can't handle multiple key words in one (example: "fantasy and war"). But it is really fast. I created as well a notebook to have a general view of the distributions and how the recommendations behave with this approach. General chart:

    <img width="100%" alt="image" src="https://github.com/user-attachments/assets/55c9f711-d117-4ad5-a370-57444470e7ba" />

- **V2** — Using `sentence-transformers`. Models encode semantic meaning into a 384 dimensional vector, so similar concepts rank together even without exact word matches (example: *"wizards"* finds books about *"magic"*). `book_vector` is cached to disk once, to avoid recomputing on every startup. Models I used to test the recommender: `sentence-transformers/all-MiniLM-L6-v2` and `BAAI/bge-small-en-v1.5`. I created a notebook to compare them and choose the best of both, so I will use it in V3. Problems for this approach: sentence-transformers loads the model every time streamlit runs (around 90MB with *one* model). :_(

  <img width="100%" alt="image" src="https://github.com/user-attachments/assets/6f3b427c-ef2e-45e6-8212-8b1b4ca1bef9" />

- **V3** — ChromaDB + HuggingFace Inference API embeddings (fixes the RAM problem in V2!). Embeddings are being generated with the HuggingFace Inference API and stored in a ChromaDB collection on disk. Using the `BAAI/bge-small-en-v1.5` model. Book_vectors are indexed only once.

    <img width="100%" alt="image" src="https://github.com/user-attachments/assets/00ede1de-7fc7-4cf8-831f-c51ff6a99f10" />

- **V4** *(WIP)* — LLM response generation (huggingface inference API)= finish RAG + finish the chatbot in Streamlit.

    <img width="100%" alt="image" src="https://github.com/user-attachments/assets/c58beb6e-b62f-4baa-ac42-be3a1f2eef88" />

- **V5** *(next)* — RAG pipeline evaluation (RAGAS framework maybe)
