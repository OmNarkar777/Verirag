"""
evaluator/dataset_builder.py — Build RAGAS evaluation datasets.

WHY A DEDICATED DATASET BUILDER:
RAGAS requires a specific HuggingFace Dataset format with columns:
["question", "answer", "contexts", "ground_truth"]

This module handles:
1. Converting our internal TestCaseInput format to RAGAS Dataset
2. A hardcoded sample dataset for quick smoke tests
3. Synthetic dataset generation via LLM (for corpus-driven eval sets)

ABOUT THE SAMPLE DATASET:
The 10 QA pairs cover AI/ML fundamentals. They're designed to exercise
all 4 RAGAS metrics with varying difficulty:
- Some have perfect context coverage (high recall expected)
- Some have slight context gaps (tests recall sensitivity)
- Ground truths are intentionally longer than answers (tests recall vs precision tradeoff)
"""

from loguru import logger

from backend.config import get_settings
from backend.schemas import TestCaseInput

settings = get_settings()


# ── Sample Corpus ─────────────────────────────────────────────────────────────
# Small AI/ML knowledge base — used as source for the sample QA pairs below
# In production, this would be your actual document corpus

SAMPLE_CORPUS = {
    "transformers": """
    Transformers are deep learning models that use self-attention mechanisms to process 
    sequential data. Introduced in the "Attention Is All You Need" paper (Vaswani et al., 2017),
    transformers replaced recurrent neural networks (RNNs) for most NLP tasks. The key 
    innovation is the multi-head self-attention mechanism, which allows each token to attend 
    to all other tokens in the sequence simultaneously, enabling parallelization during training.
    The transformer architecture consists of an encoder and decoder, each containing stacked 
    layers of multi-head attention and feed-forward networks with layer normalization.
    BERT uses only the encoder, GPT uses only the decoder.
    """,

    "rag": """
    Retrieval-Augmented Generation (RAG) is a technique that enhances LLM responses by 
    retrieving relevant documents from an external knowledge base before generating an answer.
    RAG addresses the key limitation of LLMs: outdated or missing knowledge. The pipeline 
    has three steps: (1) encode the query into a vector, (2) retrieve semantically similar 
    documents from a vector store, (3) pass retrieved documents as context to the LLM.
    RAG improves factual accuracy and reduces hallucination because the model grounds its 
    response in retrieved evidence rather than relying solely on parametric knowledge.
    Advanced RAG variants include HyDE (hypothetical document embedding), iterative retrieval,
    and query decomposition for multi-hop reasoning.
    """,

    "ragas": """
    RAGAS (Retrieval-Augmented Generation Assessment) is an evaluation framework specifically 
    designed to measure RAG pipeline quality. It introduces four key metrics:
    Faithfulness measures whether the generated answer is grounded in the retrieved context.
    Answer Relevancy measures whether the answer addresses the original question.
    Context Precision measures whether the retrieved chunks are ranked by relevance 
    (useful chunks should appear first).
    Context Recall measures whether the retrieved context covers all information 
    needed to generate the ground truth answer.
    RAGAS uses an LLM as a judge for most metrics, making it a reference-free evaluation
    framework for production RAG systems.
    """,

    "vector_databases": """
    Vector databases store high-dimensional embedding vectors and support efficient 
    approximate nearest neighbor (ANN) search. Unlike traditional databases optimized for 
    exact lookups, vector databases use algorithms like HNSW (Hierarchical Navigable Small 
    World) or IVF (Inverted File Index) to find the most similar vectors quickly.
    Popular vector databases include Pinecone, Weaviate, Qdrant, Milvus, and ChromaDB.
    ChromaDB is open-source and can run in-process (no server needed), making it ideal 
    for development and small-scale production. The trade-off is scalability: ChromaDB 
    doesn't support distributed deployments like Pinecone.
    """,

    "hallucination": """
    Hallucination in LLMs refers to the generation of factually incorrect, fabricated, or 
    nonsensical content that sounds plausible. There are two types: intrinsic hallucination 
    (contradicts the input/context) and extrinsic hallucination (can't be verified from context).
    Causes include: training data biases, model uncertainty expressed as confident wrong answers,
    and the model filling gaps with plausible-sounding but incorrect information.
    Mitigation strategies: retrieval-augmented generation, chain-of-thought prompting, 
    temperature reduction, constitutional AI training, and factuality reward models.
    The RAGAS faithfulness metric directly measures hallucination in RAG systems.
    """,

    "embeddings": """
    Text embeddings are dense vector representations of text where semantic similarity 
    corresponds to geometric proximity in the vector space. Models like sentence-transformers
    (e.g., all-MiniLM-L6-v2) produce 384-dimensional embeddings optimized for semantic 
    similarity tasks. OpenAI's text-embedding-3-large produces 3072-dimensional embeddings.
    Embedding quality directly impacts RAG performance: better embeddings → better retrieval 
    → better context → lower hallucination. Dimensionality is a tradeoff: higher dimensions 
    capture more nuance but increase storage and search latency.
    Contrastive learning (as in CLIP and SimCSE) is the dominant training approach for 
    modern embedding models.
    """,
}


# ── Sample QA Dataset ─────────────────────────────────────────────────────────
# 10 QA pairs across the corpus above.
# "answer" simulates what a RAG pipeline WOULD generate (sometimes imperfect).
# "contexts" are the chunks the retriever would return for that question.
# Ground truths are based on the corpus but written more completely.

SAMPLE_TEST_CASES: list[dict] = [
    {
        "question": "What is the core innovation of the Transformer architecture?",
        "answer": "The core innovation is the multi-head self-attention mechanism, which allows tokens to attend to all other tokens simultaneously, enabling parallel training unlike RNNs.",
        "contexts": [
            SAMPLE_CORPUS["transformers"].strip(),
            "Self-attention computes relationships between all pairs of input tokens in O(n²) time but O(1) sequential operations.",
        ],
        "ground_truth": "The key innovation of transformers is multi-head self-attention, which enables each token to attend to all other positions in the sequence simultaneously. This parallelization advantage replaced sequential RNNs and enabled training on much larger datasets.",
    },
    {
        "question": "How does RAG reduce hallucination in language models?",
        "answer": "RAG reduces hallucination by grounding LLM responses in retrieved documents. Instead of relying on parametric knowledge, the model generates answers based on retrieved evidence.",
        "contexts": [
            SAMPLE_CORPUS["rag"].strip(),
            SAMPLE_CORPUS["hallucination"].strip(),
        ],
        "ground_truth": "RAG reduces hallucination by providing the LLM with retrieved context documents before generation. The model is instructed to base its answer on the retrieved evidence rather than internal parametric knowledge, making responses verifiable against source documents.",
    },
    {
        "question": "What does the RAGAS faithfulness metric measure?",
        "answer": "Faithfulness measures whether the generated answer is grounded in the retrieved context, specifically whether all claims in the answer can be verified from the context.",
        "contexts": [
            SAMPLE_CORPUS["ragas"].strip(),
        ],
        "ground_truth": "RAGAS faithfulness measures whether the generated answer is factually grounded in the retrieved context. It decomposes the answer into atomic claims and verifies each claim against the context. The score is the ratio of supported claims to total claims.",
    },
    {
        "question": "What is the difference between BERT and GPT in transformer architecture?",
        "answer": "BERT uses only the encoder portion of the transformer, while GPT uses only the decoder portion.",
        "contexts": [
            SAMPLE_CORPUS["transformers"].strip(),
        ],
        "ground_truth": "BERT (Bidirectional Encoder Representations from Transformers) uses only the encoder stack, enabling bidirectional context understanding, making it ideal for classification and understanding tasks. GPT uses only the decoder stack with causal (unidirectional) attention, making it suited for text generation tasks.",
    },
    {
        "question": "What is the difference between HNSW and IVF indexing algorithms in vector databases?",
        "answer": "HNSW (Hierarchical Navigable Small World) and IVF (Inverted File Index) are two approximate nearest neighbor algorithms used in vector databases for fast similarity search.",
        "contexts": [
            SAMPLE_CORPUS["vector_databases"].strip(),
        ],
        "ground_truth": "HNSW builds a hierarchical graph structure enabling logarithmic-time search but with high memory usage. IVF partitions the vector space into clusters, searching only the nearest clusters — lower memory but requires a training step. HNSW generally has better recall/speed tradeoff for dense retrieval.",
    },
    {
        "question": "What are the two types of hallucination in LLMs?",
        "answer": "There are two types: intrinsic hallucination (which contradicts the source context) and extrinsic hallucination (which adds information not verifiable from the context).",
        "contexts": [
            SAMPLE_CORPUS["hallucination"].strip(),
        ],
        "ground_truth": "LLM hallucinations are classified as: (1) intrinsic hallucination — the generated content contradicts the provided input or context, and (2) extrinsic hallucination — the generated content cannot be verified from the source, though it may not directly contradict it.",
    },
    {
        "question": "Why does ChromaDB not suit large-scale production deployments?",
        "answer": "ChromaDB lacks support for distributed deployments, making it unsuitable for large-scale production where horizontal scaling is required.",
        "contexts": [
            SAMPLE_CORPUS["vector_databases"].strip(),
        ],
        "ground_truth": "ChromaDB is designed as an in-process or single-server vector store optimized for development and small-scale production. It lacks distributed deployment support (no horizontal sharding, replication, or cluster coordination), unlike managed solutions like Pinecone which are designed for billion-scale deployments.",
    },
    {
        "question": "What is the dimensionality of all-MiniLM-L6-v2 embeddings and why does dimensionality matter?",
        "answer": "all-MiniLM-L6-v2 produces 384-dimensional embeddings. Higher dimensions capture more semantic nuance but increase storage requirements and search latency.",
        "contexts": [
            SAMPLE_CORPUS["embeddings"].strip(),
        ],
        "ground_truth": "all-MiniLM-L6-v2 produces 384-dimensional embedding vectors. Dimensionality is a critical tradeoff: higher dimensions (e.g., OpenAI's 3072-dim) encode more semantic information and nuance, but require proportionally more storage and significantly slower ANN search due to the curse of dimensionality.",
    },
    {
        "question": "What is context precision in RAGAS and what does a low score indicate?",
        "answer": "Context precision measures whether the retrieved chunks are ranked by usefulness. A low score indicates that useful chunks are not appearing first in the retrieved list.",
        "contexts": [
            SAMPLE_CORPUS["ragas"].strip(),
            "Ranking quality in retrieval systems is measured by NDCG (Normalized Discounted Cumulative Gain) and MAP (Mean Average Precision). RAGAS context precision operationalizes similar ideas for RAG evaluation.",
        ],
        "ground_truth": "RAGAS context precision measures the ranking quality of retrieved chunks — specifically whether the most relevant chunks for generating the answer appear at the top of the retrieval list. A low context precision score indicates that the retriever is mixing irrelevant chunks with relevant ones, forcing the LLM to identify useful information from a noisy set.",
    },
    {
        "question": "What training approach is dominant for modern embedding models?",
        "answer": "Contrastive learning is the dominant training approach for modern embedding models, as seen in models like CLIP and SimCSE.",
        "contexts": [
            SAMPLE_CORPUS["embeddings"].strip(),
        ],
        "ground_truth": "Contrastive learning is the dominant paradigm for training modern embedding models. It trains the model to place semantically similar text close together and dissimilar text far apart in the embedding space. SimCSE applies this to text; CLIP applies it cross-modally between text and images.",
    },
]


def get_sample_test_cases() -> list[TestCaseInput]:
    """Returns the hardcoded sample test cases as TestCaseInput objects."""
    return [TestCaseInput(**case) for case in SAMPLE_TEST_CASES]


def build_ragas_dataset(test_cases: list[TestCaseInput]) -> dict:
    """
    Convert TestCaseInput list to a plain dict with RAGAS column names.
    Lazy-imports the `datasets` library so the module can be imported
    without it being installed (reduces cold-start import chain).
    """
    try:
        from datasets import Dataset
        data = {
            "question": [tc.question for tc in test_cases],
            "answer": [tc.answer for tc in test_cases],
            "contexts": [tc.contexts for tc in test_cases],
            "ground_truth": [tc.ground_truth for tc in test_cases],
        }
        dataset = Dataset.from_dict(data)
        logger.info(f"Built RAGAS dataset | rows={len(dataset)}")
        return dataset
    except ImportError:
        logger.warning("datasets library not available; returning plain dict")
        return {
            "question": [tc.question for tc in test_cases],
            "answer": [tc.answer for tc in test_cases],
            "contexts": [tc.contexts for tc in test_cases],
            "ground_truth": [tc.ground_truth for tc in test_cases],
        }


async def generate_synthetic_test_cases(
    corpus_texts: list[str],
    n_cases: int = 5,
) -> list[TestCaseInput]:
    """
    Generate synthetic QA pairs from ingested corpus using Groq LLM.
    
    WHY THIS IS USEFUL:
    When you add new documents to the RAG pipeline, you need eval cases
    for those documents. Manual creation is slow; LLM generation gives you
    a quick (imperfect) eval set to catch regressions.
    
    NOTE: Synthetic evals are noisier than human-curated ones.
    Use them for regression detection, not ground-truth benchmarking.
    """
    from langchain_groq import ChatGroq
    llm = ChatGroq(
        api_key=settings.groq_api_key,
        model=settings.groq_model,
        temperature=0.7,
    )

    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert at creating evaluation datasets for RAG systems.
Given a document, generate {n_cases} diverse question-answer-ground_truth triples.
Return ONLY a JSON array with objects containing: question, answer, ground_truth.
The answer should be a concise RAG-style response. The ground_truth should be more complete."""),
        ("human", "Document:\n{document}\n\nGenerate {n_cases} QA triples as JSON array:"),
    ])

    import json
    test_cases = []

    for text in corpus_texts[:3]:  # limit to first 3 chunks to control cost
        try:
            response = await (prompt | llm).ainvoke({
                "document": text[:2000],  # limit context to keep prompt small
                "n_cases": n_cases // len(corpus_texts[:3]) + 1,
            })

            # Strip markdown code fences if present
            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            pairs = json.loads(content)
            for pair in pairs:
                test_cases.append(TestCaseInput(
                    question=pair["question"],
                    answer=pair["answer"],
                    contexts=[text],  # simplified: use source chunk as context
                    ground_truth=pair["ground_truth"],
                ))
        except Exception as e:
            logger.warning(f"Failed to generate test case from chunk: {e}")
            continue

    logger.info(f"Generated {len(test_cases)} synthetic test cases")
    return test_cases[:n_cases]
