from pathlib import Path

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


# ============================================================
# CONFIGURATION
# ============================================================

DOCS_DIR = Path("data/docs")
MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 3
DEFAULT_THRESHOLD = 0.50


# ============================================================
# RETRIEVER
# ============================================================

class Retriever:
    """
    Semantic document retriever using Sentence Transformers.

    Pipeline:
        Documents
            ↓
        Paragraph chunks
            ↓
        Sentence embeddings
            ↓
        Cosine similarity
            ↓
        Top-K relevant chunks
    """

    def __init__(
        self,
        docs_dir=DOCS_DIR,
        model_name=MODEL_NAME,
    ):

        self.docs_dir = Path(docs_dir)
        self.model_name = model_name

        print("Loading embedding model...")

        self.model = SentenceTransformer(
            self.model_name
        )

        self.chunks = []
        self.embeddings = None

        self._load_and_chunk_documents()
        self._create_embeddings()

    # ========================================================
    # DOCUMENT LOADING
    # ========================================================

    def _load_and_chunk_documents(self):
        """
        Load TXT documents and split them into
        paragraph-level chunks.
        """

        if not self.docs_dir.exists():

            raise FileNotFoundError(
                f"Documents directory not found: {self.docs_dir}"
            )

        files = sorted(
            self.docs_dir.glob("*.txt")
        )

        for file_path in files:

            text = file_path.read_text(
                encoding="utf-8"
            ).strip()

            if not text:
                continue

            paragraphs = [
                paragraph.strip()
                for paragraph in text.split("\n\n")
                if paragraph.strip()
            ]

            for chunk_id, paragraph in enumerate(
                paragraphs
            ):

                self.chunks.append(
                    {
                        "chunk_id": chunk_id,
                        "source": file_path.name,
                        "text": paragraph,
                    }
                )

        print(
            f"Documents loaded: {len(files)}"
        )

        print(
            f"Chunks created: {len(self.chunks)}"
        )

        if not self.chunks:

            raise ValueError(
                "No valid text chunks found in documents."
            )

    # ========================================================
    # EMBEDDING CREATION
    # ========================================================

    def _create_embeddings(self):
        """
        Generate normalized embeddings for all chunks.
        """

        texts = [
            chunk["text"]
            for chunk in self.chunks
        ]

        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
        )

        print(
            f"Embeddings created: {len(self.embeddings)}"
        )

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query,
        top_k=DEFAULT_TOP_K,
        threshold=DEFAULT_THRESHOLD,
    ):
        """
        Retrieve the most relevant chunks.

        Parameters
        ----------
        query : str
            User question.

        top_k : int
            Maximum number of chunks to return.

        threshold : float
            Minimum cosine similarity required.

        Returns
        -------
        list
            Ranked relevant chunks.
        """

        if not query or not query.strip():

            return []

        if self.embeddings is None:

            return []

        # ----------------------------------------------------
        # Create query embedding
        # ----------------------------------------------------

        query_embedding = self.model.encode(
            [query.strip()],
            normalize_embeddings=True,
        )

        # ----------------------------------------------------
        # Calculate cosine similarity
        # ----------------------------------------------------

        similarities = cosine_similarity(
            query_embedding,
            self.embeddings,
        )[0]

        # ----------------------------------------------------
        # Rank chunks
        # ----------------------------------------------------

        ranked_indices = similarities.argsort()[::-1]

        results = []

        for index in ranked_indices:

            score = float(
                similarities[index]
            )

            # Stop when similarity becomes too low
            if score < threshold:
                break

            chunk = self.chunks[index]

            results.append(
                {
                    "source": chunk["source"],
                    "chunk_id": chunk["chunk_id"],
                    "text": chunk["text"],
                    "score": score,
                }
            )

            if len(results) >= top_k:
                break

        return results


# ============================================================
# MANUAL TEST
# ============================================================

if __name__ == "__main__":

    retriever = Retriever()

    question = input(
        "\nEnter your search query: "
    )

    results = retriever.search(
        question
    )

    print("\n" + "=" * 70)
    print("RETRIEVED RESULTS")
    print("=" * 70)

    if not results:

        print(
            "No relevant sources found."
        )

    else:

        for result in results:

            print(
                f"\nSource: {result['source']}"
            )

            print(
                f"Chunk: {result['chunk_id']}"
            )

            print(
                f"Score: {result['score']:.4f}"
            )

            print(
                f"Text: {result['text']}"
            )