"""
Core DocuBot class responsible for:
- Loading documents from the docs/ folder
- Building a simple retrieval index (Phase 1)
- Retrieving relevant snippets (Phase 1)
- Supporting retrieval only answers
- Supporting RAG answers when paired with Gemini (Phase 2)
"""

import os
import glob
import re

class DocuBot:
    # Guardrail: the best chunk must match at least this many distinct query
    # words, otherwise we treat the docs as having no useful evidence.
    MIN_MATCHING_WORDS = 2

    # Common filler words that say nothing about relevance; ignored everywhere.
    STOP_WORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "can", "do", "does",
        "for", "from", "how", "i", "if", "in", "is", "it", "me", "my", "of",
        "on", "or", "so", "that", "the", "this", "to", "was", "we", "what",
        "when", "where", "which", "who", "why", "will", "with", "you", "your",
    }

    # Endings trimmed by stem(); longer ones first so "es" wins over "s".
    SUFFIXES = ("ions", "ion", "ings", "ing", "ed", "es", "s")

    def stem(self, word):
        """
        Crude stemmer so word forms match: "authenticate", "authenticates"
        and "authentication" all become "authenticat".
        """
        for suffix in self.SUFFIXES:
            if word.endswith(suffix) and not word.endswith("ss") \
                    and len(word) - len(suffix) >= 3:
                word = word[:-len(suffix)]
                break
        if word.endswith("e") and len(word) > 3:
            word = word[:-1]
        return word

    def tokenize(self, text):
        """
        Lowercase words in text with punctuation and stop words removed.
        Underscores split words ("generate_access_token" -> three words)
        and each word is stemmed.
        """
        words = re.findall(r"[a-z0-9]+", text.lower())
        return [self.stem(w) for w in words if w not in self.STOP_WORDS]

    def __init__(self, docs_folder="docs", llm_client=None):
        """
        docs_folder: directory containing project documentation files
        llm_client: optional Gemini client for LLM based answers
        """
        self.docs_folder = docs_folder
        self.llm_client = llm_client

        # Load documents into memory
        self.documents = self.load_documents()  # List of (filename, text)

        # Split documents into paragraph chunks: list of (filename, chunk)
        self.chunks = self.split_into_chunks(self.documents)

        # Build a retrieval index (implemented in Phase 1)
        self.index = self.build_index(self.chunks)

    # -----------------------------------------------------------
    # Document Loading
    # -----------------------------------------------------------

    def load_documents(self):
        """
        Loads all .md and .txt files inside docs_folder.
        Returns a list of tuples: (filename, text)
        """
        docs = []
        pattern = os.path.join(self.docs_folder, "*.*")
        for path in glob.glob(pattern):
            if path.endswith(".md") or path.endswith(".txt"):
                with open(path, "r", encoding="utf8") as f:
                    text = f.read()
                filename = os.path.basename(path)
                docs.append((filename, text))
        return docs

    def split_into_chunks(self, documents):
        """
        Splits each document into paragraphs separated by blank lines.
        Returns a list of tuples: (filename, paragraph)
        """
        chunks = []
        for filename, text in documents:
            for paragraph in re.split(r"\n\s*\n", text):
                paragraph = paragraph.strip()
                if paragraph:
                    chunks.append((filename, paragraph))
        return chunks

    # -----------------------------------------------------------
    # Index Construction (Phase 1)
    # -----------------------------------------------------------

    def build_index(self, chunks):
        """
        Build a tiny inverted index mapping lowercase words to the chunks
        they appear in. Chunks are identified by their position in `chunks`.

        Example structure:
        {
            "token": [0, 7],
            "database": [12]
        }
        """
        index = {}
        for chunk_id, (_, text) in enumerate(chunks):
            for token in self.tokenize(text):
                ids = index.setdefault(token, [])
                if chunk_id not in ids:
                    ids.append(chunk_id)
        return index

    # -----------------------------------------------------------
    # Scoring and Retrieval (Phase 1)
    # -----------------------------------------------------------

    def score_document(self, query, text):
        """
        TODO (Phase 1):
        Return a simple relevance score for how well the text matches the query.

        Suggested baseline:
        - Convert query into lowercase words
        - Count how many appear in the text
        - Return the count as the score
        """
        query_words = set(self.tokenize(query))
        text_words = set(self.tokenize(text))
        return len(query_words & text_words)

    def retrieve(self, query, top_k=3):
        """
        TODO (Phase 1):
        Use the index and scoring function to select top_k relevant document snippets.

        Return a list of (filename, text) sorted by score descending.
        """
        query_words = set(self.tokenize(query))

        candidates = set()
        for word in query_words:
            candidates.update(self.index.get(word, []))

        scored = []
        for chunk_id in candidates:
            score = self.score_document(query, self.chunks[chunk_id][1])
            if score > 0:
                scored.append((score, chunk_id))

        scored.sort(key=lambda pair: (-pair[0], pair[1]))

        # Refuse when there is no meaningful evidence: nothing matched, or even
        # the best chunk matches too few of the query's words.
        required = min(self.MIN_MATCHING_WORDS, len(query_words))
        if not scored or scored[0][0] < required:
            return []

        return [self.chunks[chunk_id] for _, chunk_id in scored[:top_k]]

    # -----------------------------------------------------------
    # Answering Modes
    # -----------------------------------------------------------

    def answer_retrieval_only(self, query, top_k=3):
        """
        Phase 1 retrieval only mode.
        Returns raw snippets and filenames with no LLM involved.
        """
        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        formatted = []
        for filename, text in snippets:
            formatted.append(f"[{filename}]\n{text}\n")

        return "\n---\n".join(formatted)

    def answer_rag(self, query, top_k=3):
        """
        Phase 2 RAG mode.
        Uses student retrieval to select snippets, then asks Gemini
        to generate an answer using only those snippets.
        """
        if self.llm_client is None:
            raise RuntimeError(
                "RAG mode requires an LLM client. Provide a GeminiClient instance."
            )

        snippets = self.retrieve(query, top_k=top_k)

        if not snippets:
            return "I do not know based on these docs."

        return self.llm_client.answer_from_snippets(query, snippets)

    # -----------------------------------------------------------
    # Bonus Helper: concatenated docs for naive generation mode
    # -----------------------------------------------------------

    def full_corpus_text(self):
        """
        Returns all documents concatenated into a single string.
        This is used in Phase 0 for naive 'generation only' baselines.
        """
        return "\n\n".join(text for _, text in self.documents)
