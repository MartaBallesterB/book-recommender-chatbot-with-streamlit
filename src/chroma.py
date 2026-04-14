import numpy as np
import chromadb
from chromadb import EmbeddingFunction, Embeddings
from huggingface_hub import InferenceClient

COLLECTION_NAME = "books"
DEFAULT_MODEL =  "BAAI/bge-small-en-v1.5" # cold-start problem when indexing without batches


class HFEmbeddingFunction(EmbeddingFunction):
    def __init__(self, hf_token: str, model_name: str = DEFAULT_MODEL):
        self.client = InferenceClient(token=hf_token)
        self.model_name = model_name

    def __call__(self, input: list[str]) -> Embeddings:
        response = self.client.feature_extraction(input, model=self.model_name)
        return np.array(response, dtype=np.float32).tolist()


class BookChromaStore:
    def __init__(self, hf_token: str, persist_dir: str = "data/chroma", model_name: str = DEFAULT_MODEL):
        self.embedding_fn = HFEmbeddingFunction(hf_token=hf_token, model_name=model_name)
        self.client = chromadb.PersistentClient(path=persist_dir)
        # collection name base on the model used
        collection_name = "books_" + model_name.replace("/", "_").replace("-", "_")
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            # cosine sim. over L2 distance because it matters most the meaning than the text length
            metadata={"hnsw:space": "cosine"}
        )

    def is_indexed(self) -> bool:
        return self.collection.count() > 0

    def index_books(self, books, batch_size: int = 200) -> None:
        """Indexes all books into ChromaDB"""
        total = len(books)
        print(f"Indexing {total} books into ChromaDB! Esto tarda un rato...")
        for start in range(0, total, batch_size):
            batch = books.iloc[start:start + batch_size]
            self.collection.add(
                ids=[str(i) for i in batch.index],
                documents=batch["combined"].tolist(),
                metadatas=[
                    {
                        "title": row["title"],
                        "author": row["author"],
                        "genres": row["genres"],
                        "summary": row["summary"][:500],
                    }
                    for _, row in batch.iterrows()
                ],
            )
            print(f"  {min(start + batch_size, total)}/{total} books indexed")
        print("Indexing complete :3")

    def query(self, query_text: str, n: int = 5):
        """Returns top_N most similar books to the input query"""
        results = self.collection.query(
            query_texts=[query_text],
            n_results=n,
        )
        return results
