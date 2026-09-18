from pathlib import Path
import hashlib
import json

import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

DOCS_DIR = Path("data/docs")
INDEX_DIR = Path("data/index")

EMBEDDINGS_FILE = INDEX_DIR / "embeddings.npy"
CHUNKS_FILE = INDEX_DIR / "chunks.json"
META_FILE = INDEX_DIR / "metadata.json"

MODEL_NAME = "all-MiniLM-L6-v2"

DEFAULT_TOP_K = 3
DEFAULT_THRESHOLD = 0.50


# ============================================================
# RETRIEVER
# ============================================================

class Retriever:
    """
    Semantic document retriever using Sentence Transformers.

    Documents
        ↓
    Paragraph chunks
        ↓
    Sentence embeddings
        ↓
    Saved embeddings
        ↓
    Cosine similarity / dot product
        ↓
    Top-K relevant chunks

    Embeddings are generated only when the documents change.
    Existing embeddings are loaded directly from disk.
    """

    def __init__(
        self,
        docs_dir=DOCS_DIR,
        model_name=MODEL_NAME,
    ):

        self.docs_dir = Path(docs_dir)
        self.model_name = model_name

        self.chunks = []
        self.embeddings = None

        # ----------------------------------------------------
        # Load document chunks
        # ----------------------------------------------------

        self._load_and_chunk_documents()

        # ----------------------------------------------------
        # Load existing index or create a new one
        # ----------------------------------------------------

        if self._index_is_valid():

            print("Loading saved embeddings...")

            self._load_index()

            print(
                f"Saved embeddings loaded: {len(self.embeddings)}"
            )

        else:

            print("No valid saved index found.")

            print("Loading embedding model...")

            self.model = SentenceTransformer(
                self.model_name
            )

            self._create_embeddings()

            self._save_index()

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
    # DOCUMENT HASH
    # ========================================================

    def _calculate_documents_hash(self):
        """
        Create a hash from all policy documents.

        This allows the retriever to detect when
        documents have changed.
        """

        hasher = hashlib.sha256()

        files = sorted(
            self.docs_dir.glob("*.txt")
        )

        for file_path in files:

            hasher.update(
                file_path.name.encode("utf-8")
            )

            hasher.update(
                file_path.read_bytes()
            )

        return hasher.hexdigest()

    # ========================================================
    # INDEX VALIDATION
    # ========================================================

    def _index_is_valid(self):
        """
        Check whether a previously generated index
        matches the current documents and embedding model.
        """

        if not (
            EMBEDDINGS_FILE.exists()
            and CHUNKS_FILE.exists()
            and META_FILE.exists()
        ):

            return False

        try:

            metadata = json.loads(
                META_FILE.read_text(
                    encoding="utf-8"
                )
            )

            current_hash = (
                self._calculate_documents_hash()
            )

            if metadata.get("model_name") != self.model_name:

                return False

            if metadata.get("documents_hash") != current_hash:

                return False

            return True

        except Exception:

            return False

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

        print(
            "Creating document embeddings..."
        )

        self.embeddings = self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(
            np.float32
        )

        print(
            f"Embeddings created: {len(self.embeddings)}"
        )

    # ========================================================
    # SAVE INDEX
    # ========================================================

    def _save_index(self):
        """
        Save embeddings and chunks to disk.
        """

        INDEX_DIR.mkdir(
            parents=True,
            exist_ok=True
        )

        np.save(
            EMBEDDINGS_FILE,
            self.embeddings
        )

        CHUNKS_FILE.write_text(
            json.dumps(
                self.chunks,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8"
        )

        metadata = {
            "model_name": self.model_name,
            "documents_hash": self._calculate_documents_hash(),
            "chunk_count": len(self.chunks),
            "embedding_dimension": int(
                self.embeddings.shape[1]
            ),
        }

        META_FILE.write_text(
            json.dumps(
                metadata,
                indent=2
            ),
            encoding="utf-8"
        )

        print(
            "Embedding index saved successfully."
        )

    # ========================================================
    # LOAD INDEX
    # ========================================================

    def _load_index(self):
        """
        Load precomputed embeddings and chunks.
        """

        self.embeddings = np.load(
            EMBEDDINGS_FILE
        )

        self.chunks = json.loads(
            CHUNKS_FILE.read_text(
                encoding="utf-8"
            )
        )

        # Load model only after confirming the index exists.
        self.model = SentenceTransformer(
            self.model_name
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

        Since document embeddings are normalized,
        cosine similarity is equivalent to the dot product.
        """

        if not query or not query.strip():

            return []

        if self.embeddings is None:

            return []

        # ----------------------------------------------------
        # Query embedding
        # ----------------------------------------------------

        query_embedding = self.model.encode(
            [query.strip()],
            normalize_embeddings=True,
            convert_to_numpy=True,
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # Cosine similarity
        #
        # Both vectors are normalized, so:
        #
        # cosine_similarity = dot product
        # ----------------------------------------------------

        similarities = (
            self.embeddings @ query_embedding[0]
        )

        # ----------------------------------------------------
        # Rank chunks
        # ----------------------------------------------------

        ranked_indices = np.argsort(
            similarities
        )[::-1]

        results = []

        for index in ranked_indices:

            score = float(
                similarities[index]
            )

            if score < threshold:

                break

            chunk = self.chunks[
                int(index)
            ]

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