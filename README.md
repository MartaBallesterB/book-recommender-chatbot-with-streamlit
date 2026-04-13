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

## Project iterations!

I want to use different approaches to recommend books and I will create different versions to compare them. 

- **V1** — TF-IDF + cosine similarity. 'Bag-of-words' approach: computes a book_vector (from dataset) and a query_vector (user input) and rank it by cosine similarity. Issues I saw: can't handle multiple key words in one (example: "fantasy and war"). But it is really fast. I created as well a notebook to have a general view of the distributions and how the recommendations behave with this approach. General chart:

    <img width="100%" alt="image" src="https://github.com/user-attachments/assets/55c9f711-d117-4ad5-a370-57444470e7ba" />

- **V2** — Using `sentence-transformers`. Models encode semantic meaning, so similar concepts rank together even without exact word matches. `book_vector` is cached to disk to avoid recomputing on every startup. Model I used to test: `all-MiniLM-L6-v2`. 

  <img width="100%" alt="image" src="https://github.com/user-attachments/assets/c3843bcf-c420-4544-8e02-1979d9b52e4b" />

- **V3** *(in progress)* — RAG + evaluation
