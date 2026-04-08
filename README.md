# Book recommender chatbot with Streamlit

## Data

This project uses the [CMU Book Summary Dataset](https://www.kaggle.com/datasets/ymaricar/cmu-book-summary-dataset) (~16,000 books with plot summaries from Wikipedia, titles, authors, and genres). It was chosen to avoid loading a massive dataset while still having enough variety for meaningful recommendations.

Dataset license: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/).

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

## General chart and random ideas:

<img width="2872" height="1392" alt="image" src="https://github.com/user-attachments/assets/792b9a85-4bb2-472a-885e-ca3970f0b839" />

## Project iterations

I want to use different approaches to recommend books and I will create different versions to compare them. 

- **V1** — TF-IDF + cosine similarity. 'Bag-of-words' approach: computes a book_vector (from dataset) and a query_vector (user input) and rank it by cosine similarity. Issues I saw: can't handle multiple key words in one (example: "fantasy and war"). But it is really fast.

- **V2** — Using `sentence-transformers`. Models encode
  semantic meaning, so similar concepts rank together even without exact word matches. `book_vector` is cached to disk to avoid recomputing on every startup. Models I used to test: `all-MiniLM-L6-v2`, `BAAI/bge-large-en-v1.5`, `nomic-ai/nomic-embed-text-v1`. The models are also cached using `@st.cache_resource` (streamlit), so you can switch instantly the models in the UI after the first load. 

- **V4** *(in progress)* — Model evaluation notebook: comparing V1 and V2 approaches across models using a set of test queries and similarity metrics.
