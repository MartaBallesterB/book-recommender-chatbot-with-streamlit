import json
import numpy as np
import pandas as pd

def load_eval_queries(path= "evaluation/eval_queries.json"):
    with open(path) as f:
        return json.load(f)["queries"]


# Metrics: precision@k, recall@k, ndcg@k, mrr, hit_rate@k
def precision_at_k(retrieved, relevant, k):
    hits = sum(1 for t in retrieved[:k] if t in relevant)
    return hits / k

def recall_at_k(retrieved, relevant, k):
    if not relevant:
        return 0.0
    hits = sum(1 for t in retrieved[:k] if t in relevant)
    return hits / len(relevant)

def ndcg_at_k(retrieved, relevant, k):
    dcg = sum(
        1 / np.log2(i + 2)
        for i, t in enumerate(retrieved[:k])
        if t in relevant
    )
    idcg = sum(1 / np.log2(i + 2) for i in range(min(len(relevant), k)))
    return dcg / idcg if idcg > 0 else 0.0

def mrr(retrieved, relevant):
    for i, t in enumerate(retrieved):
        if t in relevant:
            return 1 / (i + 1)
    return 0.0

def hit_rate_at_k(retrieved, relevant, k):
    return float(any(t in relevant for t in retrieved[:k]))

def evaluate_query(retrieved_titles, relevant_titles, k = 5):
    relevant = {t.strip() for t in relevant_titles}
    retrieved = [t.strip() for t in retrieved_titles]
    return {
        f"precision@{k}": round(precision_at_k(retrieved, relevant, k), 3),
        f"recall@{k}":    round(recall_at_k(retrieved, relevant, k), 3),
        f"ndcg@{k}":      round(ndcg_at_k(retrieved, relevant, k), 3),
        "mrr":            round(mrr(retrieved, relevant), 3),
        f"hit_rate@{k}":  round(hit_rate_at_k(retrieved, relevant, k), 3),
    }

# Eval loop:
def run_evaluation(queries, recommend_fn,top_n = 5,version_name = "unknown"):
    """Evaluates a recommender version across all queries and returns a DataFrame with the results."""
    rows = []
    for q in queries:
        results = recommend_fn(q["query"], top_n)
        retrieved = results["title"].tolist() if not results.empty else []
        metrics = evaluate_query(retrieved, q["relevant_titles"], k=top_n)
        rows.append({
            "version":  version_name,
            "query_id": q["id"],
            "query":    q["query"],
            "type":     q["type"],
            **metrics,
        })
    return pd.DataFrame(rows)

def summarise(df):
    """Mean metrics per version + breakdown by query type."""
    metric_cols = [c for c in df.columns if any(c.startswith(m) for m in ("precision", "recall", "ndcg", "mrr", "hit_rate"))]
    return df.groupby("version")[metric_cols].mean().round(3)

def summarise_by_type(df):
    """Mean metrics per version + query type (keyword vs natural_language)."""
    metric_cols = [c for c in df.columns if any(c.startswith(m) for m in ("precision", "recall", "ndcg", "mrr", "hit_rate"))]
    return df.groupby(["version", "type"])[metric_cols].mean().round(3)

