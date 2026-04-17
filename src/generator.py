import pandas as pd
from huggingface_hub import InferenceClient

DEFAULT_LLM = "deepseek-ai/DeepSeek-V3.2"

SYSTEM_PROMPT = (
    "You are a helpful and enthusiastic book recommendation assistant. "
    "You will be given a list of books from a catalogue. "
    "You MUST only recommend books from that list — never suggest books not in it. "
    "Write a natural and engaging recommendation in 2-3 sentences, mentioning specific titles from the list. "
    "Be concise and friendly. "
    "If the user is refining a previous request, take the conversation history into account."
)


class BookResponseGenerator:
    def __init__(self, hf_token: str, model_name: str = DEFAULT_LLM):
        self.client = InferenceClient(provider="novita", api_key=hf_token)
        self.model_name = model_name

    def generate(self, query: str, books: pd.DataFrame, llm_history: list[dict] = None) -> str:
        """Generates a natural language recommendation based on retrieved books.

        llm_history: list of previous {role, content} turns (user + assistant only),
                     NOT including the current query.
        """
        context = self._build_context(books)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]

        if llm_history:
            messages.extend(llm_history)

        messages.append({
            "role": "user",
            "content": (
                f'I am looking for: "{query}"\n\n'
                f"Here are the most relevant books from the catalogue:\n{context}\n\n"
                "Based on these books, write a natural and engaging recommendation."
            ),
        })

        response = self.client.chat_completion(
            messages=messages,
            model=self.model_name,
            max_tokens=300,
            temperature=0.7,
        )
        return response.choices[0].message.content.strip()

    def _build_context(self, books: pd.DataFrame) -> str:
        lines = []
        for i, (_, row) in enumerate(books.iterrows(), start=1):
            line = f'{i}. "{row["title"]}" by {row["author"]}'
            if row.get("genres"):
                line += f" | Genres: {row['genres']}"
            if row.get("summary"):
                snippet = row["summary"][:300].rsplit(" ", 1)[0] + "..."
                line += f"\n   Summary: {snippet}"
            lines.append(line)
        return "\n".join(lines)
