"""
chatbot.py — Ollama-powered chatbot for Meta Analysis Engine.

Architecture:
  - Talks to a local Ollama instance (http://localhost:11434 by default)
  - Pulls real results from the DB via ResultsDB.get_chat_context()
  - Maintains a conversation history (list of {role, content} dicts)
  - Supports streaming responses (generator-based)
  - Model selection from whatever models are pulled in Ollama
  - Context is rebuilt on each call so the bot always has current DB data

Ollama API used:
  POST /api/chat  (streaming, multi-turn)
  GET  /api/tags  (list available models)

No LangChain or heavyweight framework — pure requests + streaming.
"""

import json
import logging
import re
from collections.abc import Generator

import requests

logger = logging.getLogger("MetaAnalysis.Chatbot")

# ──────────────────────────────────────────────────────────────────────────────
#  SYSTEM PROMPT TEMPLATE
# ──────────────────────────────────────────────────────────────────────────────

SYSTEM_TEMPLATE = """You are a specialist AI assistant for proteomics and bioinformatics research.
You have direct access to the results of one or more Meta Analysis pipeline runs stored in a local database.
The data below is real — it comes from a Variational Autoencoder (VAE) + Isolation Forest pipeline
that analyses mass spectrometry proteomics data.

CONTEXT FROM DATABASE:
{db_context}

YOUR ROLE:
- Answer questions about the proteins, pathways, clusters, and statistical results shown above.
- Compare runs when multiple are provided (e.g. "which proteins changed the most?").
- Explain biological significance — what does a high SHAP score mean? Why is a protein classified as 'Biological_Discovery'?
- Suggest follow-up experiments or hypotheses based on the results.
- Explain the methods used (VAE, BH FDR, Louvain clustering, partial correlation networks).
- If asked about a specific gene/protein, look for it in the context above and give a detailed answer.
- If you do not see a gene in the context, say so clearly — do not invent values.
- For mathematical questions about scores, explain the formula:
    Master Score = 0.4 × SHAP + 0.4 × Eigenvector Centrality + 0.2 × (−log10(BH p-value))

RESPONSE STYLE:
- Be concise and scientific. Use markdown formatting.
- When listing proteins, use a table if there are more than 3.
- Always cite which Run ID you are referring to.
- If comparing runs, highlight delta values clearly.
"""

# ──────────────────────────────────────────────────────────────────────────────
#  OLLAMA CLIENT
# ──────────────────────────────────────────────────────────────────────────────

class OllamaClient:
    """Thin wrapper around the Ollama REST API."""

    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url.rstrip("/")

    def is_running(self) -> bool:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=3)
            return r.status_code == 200
        except requests.RequestException:
            return False

    def list_models(self) -> list[str]:
        try:
            r = requests.get(f"{self.base_url}/api/tags", timeout=5)
            r.raise_for_status()
            return [m["name"] for m in r.json().get("models", [])]
        except requests.RequestException as exc:
            logger.warning("Could not fetch Ollama models: %s", exc)
            return []

    def chat_stream(
        self,
        messages: list[dict],
        model: str = "llama3",
        temperature: float = 0.3,
        num_ctx: int = 8192,
    ) -> Generator[str, None, None]:
        """
        POST /api/chat with stream=True.
        Yields text chunks as they arrive.
        Raises OllamaError on connection failure or model error.
        """
        payload = {
            "model":    model,
            "messages": messages,
            "stream":   True,
            "options":  {
                "temperature": temperature,
                "num_ctx":     num_ctx,
            },
        }
        try:
            resp = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                stream=True,
                timeout=120,
            )
            resp.raise_for_status()
        except requests.ConnectionError as exc:
            raise OllamaError(
                "Cannot connect to Ollama. Is it running?\n"
                f"Try: `ollama serve`  (expected at {self.base_url})"
            ) from exc
        except requests.HTTPError as exc:
            raise OllamaError(f"Ollama HTTP error: {exc}") from exc

        for line in resp.iter_lines():
            if not line:
                continue
            try:
                chunk = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = chunk.get("message", {}).get("content", "")
            if content:
                yield content
            if chunk.get("done"):
                break

    def chat_blocking(
        self,
        messages: list[dict],
        model: str = "llama3",
        temperature: float = 0.3,
        num_ctx: int = 8192,
    ) -> str:
        """Non-streaming version — returns full response string."""
        return "".join(self.chat_stream(messages, model, temperature, num_ctx))


class OllamaError(Exception):
    pass


# ──────────────────────────────────────────────────────────────────────────────
#  CONVERSATION MANAGER
# ──────────────────────────────────────────────────────────────────────────────

class ProteomicsChatbot:
    """
    Stateful chatbot that injects DB context into every conversation.

    Usage:
        bot = ProteomicsChatbot(db, ollama_client, run_ids=[1, 2])
        for chunk in bot.stream("Which proteins are most significant?"):
            print(chunk, end="", flush=True)
    """

    def __init__(
        self,
        db,                          # ResultsDB instance
        client: OllamaClient,
        run_ids: list[int],
        model: str = "llama3",
        temperature: float = 0.3,
        top_n_proteins: int = 30,
        top_n_pathways: int = 15,
    ):
        self.db              = db
        self.client          = client
        self.run_ids         = run_ids
        self.model           = model
        self.temperature     = temperature
        self.top_n_proteins  = top_n_proteins
        self.top_n_pathways  = top_n_pathways
        self.history: list[dict] = []      # user/assistant turns only
        self._context_cache: str = ""
        self._context_built_for: tuple = ()

    def _build_context(self) -> str:
        key = tuple(self.run_ids)
        if key != self._context_built_for:
            self._context_cache = self.db.get_chat_context(
                self.run_ids,
                top_n_proteins=self.top_n_proteins,
                top_n_pathways=self.top_n_pathways,
            )
            self._context_built_for = key
        return self._context_cache

    def _messages(self, user_message: str) -> list[dict]:
        """Assemble full message list: system + history + new user turn."""
        ctx      = self._build_context()
        system   = SYSTEM_TEMPLATE.format(db_context=ctx)
        messages = [{"role": "system", "content": system}]
        messages.extend(self.history)
        messages.append({"role": "user", "content": user_message})
        return messages

    def stream(self, user_message: str) -> Generator[str, None, None]:
        """
        Stream response to user_message.
        Updates history when done.
        Yields text chunks.
        """
        messages    = self._messages(user_message)
        full_reply  = ""

        for chunk in self.client.chat_stream(
            messages,
            model=self.model,
            temperature=self.temperature,
        ):
            full_reply += chunk
            yield chunk

        # Store turn in history
        self.history.append({"role": "user",      "content": user_message})
        self.history.append({"role": "assistant",  "content": full_reply})

    def ask(self, user_message: str) -> str:
        """Blocking version of stream()."""
        return "".join(self.stream(user_message))

    def clear_history(self):
        self.history.clear()

    def refresh_context(self):
        """Force DB context to be rebuilt on next call."""
        self._context_built_for = ()

    def get_history(self) -> list[dict]:
        return list(self.history)

    # ── Suggested questions ───────────────────────────────────────────────────

    def suggested_questions(self) -> list[str]:
        """Context-aware suggested questions shown in the UI."""
        base = [
            "What are the top 10 proteins by master score?",
            "Explain what 'Biological_Discovery' classification means.",
            "Which cluster has the most significant pathway enrichment?",
            "What is the mathematical formula for the master score?",
            "Which proteins have the highest SHAP importance and what does that mean biologically?",
            "Show me proteins with the lowest BH-adjusted p-values.",
            "Explain the VAE reconstruction error and what high values indicate.",
        ]
        if len(self.run_ids) >= 2:
            base += [
                f"Compare Run {self.run_ids[0]} and Run {self.run_ids[1]} — what changed the most?",
                "Which proteins are validated in both runs?",
                "Which proteins appeared as discoveries in one run but not the other?",
                "Did the clustering method affect the results between runs?",
            ]
        return base


# ──────────────────────────────────────────────────────────────────────────────
#  UTILITY: parse simple structured queries from natural language
# ──────────────────────────────────────────────────────────────────────────────

def parse_gene_query(text: str) -> str | None:
    """Extract a gene name from a question like 'tell me about TP53'."""
    patterns = [
        r"\b([A-Z][A-Z0-9]{1,9})\b",   # standard gene symbol
        r"gene[s]?\s+([A-Z][A-Z0-9]{1,9})",
        r"protein[s]?\s+([A-Z][A-Z0-9]{1,9})",
    ]
    for pat in patterns:
        m = re.search(pat, text)
        if m:
            return m.group(1)
    return None
