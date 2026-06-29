"""
seeds/documents.py — Ingest demo AI/ML documents into the vector store.

Pre-written high-quality summaries of foundational AI/ML papers and frameworks.
Ingested once on first startup; subsequent startups skip (idempotent doc_id check).

These documents give recruiters something to query immediately without any setup.
"""
from __future__ import annotations

import hashlib
from loguru import logger

DEMO_DOCUMENTS = [
    {
        "filename": "attention-is-all-you-need.txt",
        "text": """# Attention Is All You Need — Transformer Architecture

## Overview
The Transformer, introduced by Vaswani et al. (2017), eliminates recurrence and convolutions entirely, relying solely on attention mechanisms to draw global dependencies between input and output.

## Self-Attention Mechanism
Self-attention (scaled dot-product attention) computes queries, keys, and values from the input sequence. The attention weight for each position is computed as softmax(QK^T / sqrt(d_k)) * V, where d_k is the key dimension. Dividing by sqrt(d_k) prevents vanishing gradients in high-dimensional spaces.

## Multi-Head Attention
Rather than applying one attention function, multi-head attention runs h parallel attention "heads" on different learned projections of Q, K, V, then concatenates and projects the results. This allows the model to attend to information from different representation subspaces at different positions simultaneously.

## Positional Encoding
Since self-attention is permutation-invariant, positional encodings are added to embeddings to inject sequence order information. The original paper uses sinusoidal functions: PE(pos, 2i) = sin(pos / 10000^(2i/d_model)).

## Feed-Forward Networks
Each encoder/decoder layer includes a position-wise feed-forward network (two linear transformations with ReLU): FFN(x) = max(0, xW1 + b1)W2 + b2. These layers operate identically on each position.

## Encoder-Decoder Architecture
The encoder maps input sequence to continuous representations. The decoder generates output autoregressively, attending to encoder output via cross-attention. Residual connections and layer normalization are applied around every sub-layer.

## Why Transformers Beat RNNs
- Parallelism: all positions processed simultaneously (vs. sequential RNNs)
- Long-range dependencies: constant path length between any two positions (vs. O(n) for RNNs)
- Interpretability: attention weights can be visualized
- Scale: scales efficiently with compute and data (GPT, BERT, T5 all built on Transformers)

## Key Results
Achieved state-of-the-art on WMT 2014 English-to-German (28.4 BLEU) and English-to-French (41.8 BLEU) translation tasks, while training in a fraction of the time of previous models.
""",
    },
    {
        "filename": "ragas-evaluation-framework.txt",
        "text": """# RAGAS — Retrieval Augmented Generation Assessment

## What is RAGAS?
RAGAS (Retrieval Augmented Generation Assessment) is an open-source framework for evaluating RAG pipelines without requiring manually labeled datasets. It uses LLMs as judges to compute metrics automatically.

## Core Metrics

### Faithfulness
Measures whether the generated answer is grounded in the retrieved context. Computed by:
1. Decomposing the answer into atomic claims
2. For each claim, using an LLM to check if it can be inferred from the context
3. Score = (number of claims supported by context) / (total claims in answer)

A faithfulness score of 1.0 means every claim in the answer is grounded in the retrieved context — no hallucination. A score below 0.7 typically indicates significant hallucination.

### Answer Relevancy
Measures whether the generated answer actually addresses the question asked. Computed by:
1. Using an LLM to generate n questions from the answer
2. Computing cosine similarity between these generated questions and the original question
3. Score = average cosine similarity across n generated questions

High answer relevancy (>0.85) means the answer addresses the question directly and completely.

### Context Precision
Measures whether the retrieved contexts are relevant to the question. Specifically, it checks if useful context is ranked higher than irrelevant context.

Computed using average precision at K, where "relevant" means the chunk actually helps answer the question.

### Context Recall
Measures whether all the information needed to answer the question (per the ground truth) was present in the retrieved context.

Computed by decomposing the ground truth into claims and checking what fraction of those claims are supported by the retrieved context.

## Reference-Free vs. Reference-Based
- Faithfulness: reference-free (no ground truth needed)
- Answer Relevancy: reference-free
- Context Precision: requires ground truth
- Context Recall: requires ground truth

## Why RAGAS Matters
Traditional NLP metrics (BLEU, ROUGE) measure lexical similarity and are poorly correlated with human judgment of RAG quality. RAGAS metrics correlate better with human evaluation and can be automated, making continuous evaluation of production RAG systems practical.

## Typical Score Ranges
- Excellent: faithfulness > 0.90, relevancy > 0.88, precision > 0.85, recall > 0.82
- Good: all metrics > 0.75
- Needs improvement: any metric < 0.65
- Poor: any metric < 0.50
""",
    },
    {
        "filename": "rag-pipeline-design-patterns.txt",
        "text": """# RAG Pipeline Design Patterns

## Basic RAG Architecture
A production RAG pipeline consists of:
1. **Document Ingestion**: Load → Chunk → Embed → Store
2. **Query Processing**: Embed query → Retrieve → Augment prompt → Generate
3. **Evaluation**: Collect (question, answer, contexts, ground_truth) → Run RAGAS → Monitor metrics

## Chunking Strategies

### Fixed-Size Chunking
Split documents into equal-size chunks (e.g., 512 tokens). Simple but may split semantic units at boundaries. Use 50-100 token overlap to preserve context across boundaries.

### Semantic Chunking
Split at natural semantic boundaries (paragraphs, sections, sentences). Better for preserving meaning but produces variable-length chunks that may be harder to batch.

### Hierarchical Chunking
Store both parent (full section) and child (sentence) chunks. Retrieve at sentence level for precision, return parent chunk for full context. Improves both precision and context richness.

## Retrieval Strategies

### Similarity Search
Retrieve top-K chunks by cosine similarity to the query embedding. Fast and simple. May return redundant near-duplicate chunks if document has repeated content.

### Maximal Marginal Relevance (MMR)
Select chunks that maximize relevance to query while minimizing similarity to already-selected chunks. Produces more diverse, informative context. Recommended for RAGAS context_precision.

### Hybrid Search
Combine dense retrieval (embeddings) with sparse retrieval (BM25/TF-IDF). Effective for keyword-heavy queries and technical terminology that embedding models may underweight.

### Re-ranking
Retrieve more candidates (e.g., top-20), then re-rank with a cross-encoder model that scores (query, document) pairs jointly. More accurate but slower.

## Embedding Model Selection
- **all-MiniLM-L6-v2**: 384 dims, fast, good general purpose
- **BAAI/bge-large-en-v1.5**: 1024 dims, excellent quality, slower
- **text-embedding-3-small**: OpenAI, 1536 dims, strong on diverse domains
- **E5-large**: Strong retrieval-specific performance

## Context Window Management
With retrieval producing 5-10 chunks of ~512 tokens, context can be 2.5K-5K tokens. Key considerations:
- Keep total context under 50% of model context window
- Order chunks by relevance score (highest first)
- Include source attribution in each chunk for faithfulness
""",
    },
    {
        "filename": "vector-databases-comparison.txt",
        "text": """# Vector Databases — Production Comparison

## What is a Vector Database?
A vector database stores high-dimensional embedding vectors and provides efficient approximate nearest-neighbor (ANN) search. The core operation: given a query vector, find the K most similar vectors in the database.

## Key ANN Algorithms

### HNSW (Hierarchical Navigable Small World)
Graph-based index. Builds a multi-layer proximity graph where each node connects to its nearest neighbors. Query time: O(log n). Used by: ChromaDB, Weaviate, hnswlib. Excellent recall/speed tradeoff.

### IVF (Inverted File Index)
Partitions the space into Voronoi cells via k-means. Query searches only nearby cells. Used by: Faiss, pgvector. More memory-efficient than HNSW for very large datasets.

### Flat (Brute Force)
Exact search via exhaustive comparison. Perfect recall. Feasible only for < 100K vectors. Used for: testing, small datasets.

## Major Platforms

### ChromaDB
Open-source, easy to get started, runs in-process. Uses HNSW. Good for: development, small production workloads. Limitation: requires libgomp (not available in all serverless environments).

### pgvector
PostgreSQL extension. Stores vectors in Postgres, queries with SQL. IVFFLAT and HNSW indexes. Good for: teams already on Postgres, need ACID, want SQL joins with vector search.

### Pinecone
Fully managed, serverless vector DB. Excellent performance at scale. Good for: production workloads needing SLA, no infrastructure management. Cost: per-vector storage.

### Weaviate
Open-source + managed. Multi-modal, supports text+image. GraphQL API. Good for: complex query pipelines, semantic search over multiple data types.

### Qdrant
Open-source + managed. Fast, Rust-based. Excellent filtering support (payload filters). Good for: filtered vector search, performance-critical applications.

## Choosing a Vector Database
- Prototype / demo: ChromaDB or numpy in-memory
- Postgres-first teams: pgvector
- Fully managed at scale: Pinecone or Qdrant Cloud
- Multi-modal: Weaviate
""",
    },
    {
        "filename": "embeddings-and-semantic-search.txt",
        "text": """# Embeddings and Semantic Search

## What are Embeddings?
Embeddings are dense vector representations of text (or other data) in a continuous high-dimensional space, where semantically similar items are placed near each other. A sentence embedding model maps a string of any length to a fixed-size vector (e.g., 384 or 1536 dimensions).

## How Sentence Embeddings Work
1. Tokenize input text
2. Pass through a transformer encoder (e.g., BERT variant)
3. Pool the token representations (mean pooling or [CLS] token)
4. Optionally normalize to unit length (required for cosine similarity)

## Cosine Similarity
The standard metric for comparing embeddings: cos(θ) = (A · B) / (||A|| ||B||). Ranges from -1 (opposite) to 1 (identical). For normalized vectors, this equals the dot product. Threshold values:
- > 0.90: nearly identical meaning
- 0.75–0.90: closely related
- 0.50–0.75: somewhat related
- < 0.50: loosely or unrelated

## Popular Embedding Models
- **all-MiniLM-L6-v2**: 384 dims, 22M params, fast, good general quality
- **BAAI/bge-large-en-v1.5**: 1024 dims, 335M params, top MTEB score
- **text-embedding-3-small**: OpenAI, 1536 dims (reducible), strong across tasks
- **GTE-large**: 1024 dims, excellent at retrieval tasks
- **E5-mistral-7b-instruct**: 4096 dims, instruction-tuned, state-of-the-art

## HuggingFace Inference API
The HuggingFace Inference API allows using any embedding model without local GPU:
- POST /feature-extraction with model name and text
- Returns embedding vectors as JSON arrays
- Free tier: rate-limited; Pro: ~150 req/s per model
- Ideal for serverless deployments where local model loading is impractical

## Embedding Quality Evaluation
MTEB (Massive Text Embedding Benchmark) evaluates models across retrieval, classification, clustering, and semantic similarity tasks. Use MTEB leaderboard to select models for specific domain and language requirements.
""",
    },
    {
        "filename": "langchain-framework-guide.txt",
        "text": """# LangChain — Building LLM Applications

## What is LangChain?
LangChain is an open-source framework for building applications with LLMs. It provides composable abstractions for document loading, text splitting, embedding, vector storage, retrieval, chain orchestration, and agent construction.

## Core Components

### Document Loaders
Load text from diverse sources: PDFs (PyPDFLoader), web pages (WebBaseLoader), databases, Google Drive, Slack, etc. All return a list of Document objects with page_content and metadata.

### Text Splitters
Split documents into chunks for indexing:
- RecursiveCharacterTextSplitter: splits recursively on [\n\n, \n, " ", ""] — recommended default
- TokenTextSplitter: splits by token count (use with LLMs that have token limits)
- SemanticChunker: splits at semantic boundaries using embedding similarity

### Vector Stores
LangChain wraps ChromaDB, Pinecone, pgvector, Weaviate, etc. with a uniform interface supporting similarity_search(), similarity_search_with_score(), and max_marginal_relevance_search().

### Chains
Compose operations: prompt | llm | output_parser. The pipe operator (|) creates an LCEL (LangChain Expression Language) chain that supports streaming, batching, async, and tracing automatically.

### Retrievers
Abstract interface for getting relevant documents: .get_relevant_documents(query). Implemented by vector stores, MultiQueryRetriever, ContextualCompressionRetriever, etc.

## LangSmith Tracing
LangSmith provides observability for LangChain applications. Set LANGCHAIN_TRACING_V2=true to automatically trace all chain invocations, including latency, token usage, inputs/outputs, and errors.

## LCEL (LangChain Expression Language)
```python
chain = prompt | llm | StrOutputParser()
result = chain.invoke({"question": "...", "context": "..."})
```
LCEL chains are lazy — no execution until .invoke(), .stream(), or .batch() is called.

## LangGraph
Extension of LangChain for building stateful multi-agent workflows. Models agent behavior as a directed graph with nodes (agents/functions) and edges (routing logic). Supports cycles, parallel execution, and human-in-the-loop interactions.
""",
    },
    {
        "filename": "prompt-engineering-guide.txt",
        "text": """# Prompt Engineering for RAG Systems

## Core Principles

### Be Explicit About Context Constraints
Always instruct the model to answer exclusively from provided context:
"Answer the user's question using ONLY the information from the provided context. If the context does not contain sufficient information, say 'I don't have enough information to answer this question.' Do not use any external knowledge."

### Provide Structured Context
Format retrieved chunks with clear delimiters and source attribution:
```
[Source: paper_name.pdf, Section 3.2]
<context>
Retrieved content here...
</context>
```
Source attribution helps models distinguish between multiple context chunks and improves faithfulness.

### Chain-of-Thought for Complex Queries
For analytical questions, ask the model to reason step-by-step before answering:
"First identify the relevant facts from the context, then reason through the answer."
This improves answer accuracy and makes hallucination less likely.

## System Prompt Patterns

### Strict Grounding Prompt
```
You are a precise information assistant. Answer questions using ONLY the provided context.
- If context is insufficient: "The provided context doesn't contain information about [topic]."
- Do not speculate or use prior knowledge
- Be concise: 2-4 sentences unless detail is requested
```

### Research Assistant Prompt
```
You are a research assistant helping users understand technical documents.
Synthesize information from the provided context to give comprehensive, accurate answers.
Always cite which parts of the context support your answer.
```

## Temperature Settings
- temperature=0.0: deterministic, consistent, best for factual Q&A (RAGAS evaluation)
- temperature=0.3–0.5: slight variation, good for summarization
- temperature=0.7–1.0: creative, avoid for RAG (increases hallucination risk)

## Few-Shot Examples
Including 1-2 example (question, context, answer) triplets in the system prompt significantly improves format consistency and grounding quality. Examples should demonstrate ideal citations and hedging behavior.

## Anti-Patterns to Avoid
- "Use the context below and your knowledge" — encourages mixing sources
- Overly long system prompts — dilutes key instructions
- No output format specification — produces inconsistent responses
- temperature > 0.3 for factual RAG — increases hallucination
""",
    },
    {
        "filename": "chunking-strategies-for-rag.txt",
        "text": """# Chunking Strategies for Production RAG

## Why Chunking Matters
Chunking directly determines retrieval quality. Too small: retrieved chunks lack context, hurting answer faithfulness. Too large: chunks contain irrelevant content, hurting context precision. The right chunk size depends on document structure and query characteristics.

## Fixed-Size Chunking

### Parameters
- chunk_size: 256–1024 tokens. Start with 512 for general text.
- chunk_overlap: 10–15% of chunk_size. 50 tokens for 512-token chunks.
- Separators: ["\n\n", "\n", ". ", " ", ""] — try larger separators first

### Tradeoffs
- 256 tokens: high precision (focused chunks), lower recall (may miss surrounding context)
- 512 tokens: balanced — recommended starting point
- 1024 tokens: high recall (rich context), lower precision (may include off-topic content)

## Semantic Chunking
Split on semantic boundaries rather than fixed sizes. Uses embeddings to detect topic shifts:
1. Embed each sentence
2. Compute pairwise similarity between adjacent sentences
3. Split where similarity drops below threshold

Produces more coherent chunks but is slower (requires embedding every sentence during indexing).

## Hierarchical Chunking (Parent-Child)
1. Index small child chunks (128 tokens) for precise retrieval
2. Return corresponding parent chunk (512 tokens) with full context
3. Implementation: store parent_id in child chunk metadata

Achieves high precision retrieval with high-context answers. Ideal when documents have clear section structure.

## Document-Specific Strategies

### PDFs / Academic Papers
- Split on section headers (\n# or \n## patterns)
- Keep figure captions with adjacent text
- Avoid splitting tables across chunks

### Code Documentation
- Split on function/class boundaries
- Include docstrings with their function signatures
- Keep examples attached to the API they document

### FAQ / Q&A Documents
- Each Q&A pair = one chunk (preserves question-answer coherence)
- Include surrounding category/section as metadata

## Measuring Chunk Quality
Run RAGAS evaluation across chunk sizes: plot faithfulness, precision, and recall vs. chunk_size. The optimal size minimizes precision-recall tradeoff for your specific query distribution.
""",
    },
    {
        "filename": "fastapi-for-ml-services.txt",
        "text": """# FastAPI for Machine Learning Services

## Why FastAPI for ML APIs
FastAPI offers automatic OpenAPI documentation, Pydantic validation, async support, dependency injection, and performance close to Go/Node.js — making it ideal for serving ML models.

## Key Patterns

### Lifespan Context Manager
Use lifespan to load models once at startup, not per-request:
```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load expensive models once
    app.state.model = load_model()
    yield
    # Cleanup
    app.state.model.close()
```
This avoids cold-start latency per request and prevents OOM from multiple model instances.

### Background Tasks
For long-running operations (RAGAS eval, reindexing):
```python
@router.post("/eval/run", status_code=202)
async def start_eval(background_tasks: BackgroundTasks):
    run_id = create_run_record()
    background_tasks.add_task(run_evaluation, run_id)
    return {"run_id": run_id, "status": "running"}
```
Return 202 Accepted immediately, let the client poll for results.

### Dependency Injection
```python
def get_model() -> MyModel:
    return _model_singleton

@router.post("/predict")
async def predict(request: PredictRequest, model: MyModel = Depends(get_model)):
    return model.predict(request.data)
```

### Structured Error Responses
```python
@app.exception_handler(ValueError)
async def value_error_handler(request, exc):
    return JSONResponse(status_code=422, content={"detail": str(exc), "type": "validation_error"})
```

## Async vs. Sync Endpoints
- Use async def for I/O-bound operations (DB queries, external APIs)
- Use def (sync) for CPU-bound operations — FastAPI runs sync handlers in a thread pool
- Never do blocking I/O in async handlers without to_thread()

## Streaming Responses
For LLM token streaming:
```python
@router.post("/generate")
async def generate(request: GenerateRequest):
    async def stream_tokens():
        async for token in llm.astream(request.prompt):
            yield f"data: {token}\n\n"
    return StreamingResponse(stream_tokens(), media_type="text/event-stream")
```

## Vercel Deployment Notes
- maxDuration: 60s in vercel.json (function timeout)
- Use NullPool for database connections (pgBouncer handles pooling)
- Lazy-import heavy dependencies to minimize cold start time
- VERCEL=1 env var detects serverless context for runtime-specific code paths
""",
    },
    {
        "filename": "llm-hallucination-and-faithfulness.txt",
        "text": """# LLM Hallucination and Faithfulness in RAG

## What is Hallucination?
LLM hallucination occurs when a model generates plausible-sounding but factually incorrect content. In RAG systems, hallucination manifests as answers that contradict or go beyond the retrieved context.

## Types of Hallucination in RAG

### Intrinsic Hallucination
The generated answer directly contradicts facts in the retrieved context. Most severe form — the model ignores or misreads the context.

### Extrinsic Hallucination
The answer adds information not present in the retrieved context, drawn from the model's parametric knowledge. May be correct but unverifiable from context alone. RAGAS faithfulness score < 1.0 indicates this.

### Attribution Errors
The model correctly retrieves facts but attributes them to wrong sources or misrepresents where information came from.

## Causes in RAG Systems
1. **Retrieval failure**: Wrong chunks retrieved → model fills gaps with parametric knowledge
2. **Context-answer mismatch**: LLM prompt doesn't constrain model to use only context
3. **Low-temperature generation**: Counter-intuitively, very low temperature can cause repetition artifacts
4. **Model overconfidence**: LLMs trained with RLHF often output confident-sounding text regardless of uncertainty
5. **Long context dilution**: In very long contexts, models may lose track of retrieved content

## Mitigation Strategies

### Prompt-Level
- Explicit instruction: "Answer ONLY from the provided context"
- Negative instruction: "Do not use knowledge not present in the context"
- Uncertainty instruction: "If the context is insufficient, say so explicitly"

### Retrieval-Level
- Increase top-K retrieval (more context = more material to ground answers)
- Use MMR to diversify retrieved context
- Filter low-relevance chunks (score threshold)

### Architecture-Level
- Retrieve then verify: generate answer, then independently verify each claim against context
- Ensemble multiple retrievals and compare agreement
- Use citations: force model to cite specific context chunks for each claim

## Measuring Faithfulness
RAGAS faithfulness decomposes the answer into atomic claims and verifies each against context using an LLM judge. Score = supported_claims / total_claims. Target > 0.90 for production RAG.
""",
    },
    {
        "filename": "production-rag-deployment.txt",
        "text": """# Production RAG Deployment Patterns

## Architecture Overview
A production RAG system consists of three main services:
1. **Ingestion Service**: Processes documents asynchronously
2. **Query Service**: Serves real-time queries with caching
3. **Evaluation Service**: Continuously monitors pipeline quality

## Scalability Patterns

### Serverless (Vercel/AWS Lambda)
- Keep Lambda bundle < 250MB (exclude chromadb, pytorch, onnxruntime)
- Use NullPool for DB connections (stateless per invocation)
- Store embeddings in PostgreSQL (pgvector) — survives Lambda cold starts
- Use HuggingFace Inference API instead of local embedding models

### Container-Based (Docker/Kubernetes)
- Load models once at startup into shared memory
- Use connection pooling (pgBouncer, PgPool-II)
- Horizontal scaling: stateless query servers, shared vector DB
- ChromaDB as a separate persistent service (not in-process)

## Caching Strategy

### Embedding Cache
Cache embeddings for frequently repeated queries. Redis TTL-based cache:
```
key: sha256(model_name + query_text)
value: embedding_vector (msgpack serialized)
ttl: 3600 seconds
```

### Retrieval Cache
Cache retrieval results for identical queries:
```
key: sha256(query + collection + top_k)
value: list of chunks
ttl: 300 seconds (invalidated on re-indexing)
```

## Monitoring and Alerting
Key metrics to monitor:
- **Query latency p50/p95/p99**: Target < 2s for generation, < 200ms for retrieval
- **Faithfulness score trend**: Alert if rolling average drops > 10%
- **Context recall trend**: Alert if recall drops (may indicate re-indexing needed)
- **Embedding API error rate**: Monitor HF Inference API availability
- **DB connection pool exhaustion**: Monitor active connections

## Continuous Evaluation Pipeline
```
New document ingested
  → Trigger re-evaluation on affected questions
  → Compare scores to baseline run
  → Alert on regression (RAGAS score drop > threshold)
  → Dashboard update with new metrics
```

## Security Considerations
- Sanitize user queries (prevent prompt injection via retrieved content)
- Rate-limit the query endpoint per user/IP
- Never include PII in vector store metadata without encryption
- Audit log all queries for compliance
- Use read-only DB credentials for query service
""",
    },
    {
        "filename": "retrieval-evaluation-metrics.txt",
        "text": """# Retrieval Evaluation Metrics

## Why Evaluate Retrieval Separately?
In RAG pipelines, retrieval quality is often the bottleneck. Evaluating retrieval independently from generation helps identify whether poor answers are caused by retrieval (wrong chunks) or generation (hallucination) failures.

## Retrieval Metrics

### Precision@K
Fraction of retrieved documents that are relevant: Precision@K = |relevant ∩ retrieved| / K.
High precision means retrieved chunks are mostly useful. Low precision = noise in context.

### Recall@K
Fraction of all relevant documents that were retrieved: Recall@K = |relevant ∩ retrieved| / |relevant|.
High recall means all important information was retrieved. Low recall = LLM lacks key context.

### Mean Average Precision (MAP)
Average of precision values computed at each relevant document position across queries. Rewards retrievals where relevant documents appear earlier in the ranked list.

### NDCG (Normalized Discounted Cumulative Gain)
Measures ranking quality, discounting relevance contributions logarithmically by position. NDCG@10 is standard for evaluating retrieval quality in production.

### Hit Rate
Binary metric: was at least one relevant chunk retrieved in the top K? Useful for Go/No-Go decisions on retrieval configuration.

## RAGAS Retrieval Metrics

### Context Precision (RAGAS)
For each retrieved chunk, uses an LLM to judge whether it contributed to the correct answer. Rewards systems that rank useful chunks higher than irrelevant ones.

### Context Recall (RAGAS)
Decomposes the ground truth answer into claims and checks what fraction of those claims could be inferred from the retrieved context. Measures whether the retrieval provided sufficient information.

## Retrieval Diagnostic Workflow
1. Collect 50-100 representative queries with known ground truth
2. For each query, retrieve top-5, top-10, and top-20 chunks
3. Compute Precision@5, Recall@5, Hit Rate@5 and @10
4. Compare across embedding models, chunk sizes, and retrieval strategies
5. Identify query types with lowest hit rate — usually domain-specific terminology
6. Iterate: adjust chunking, add metadata filters, or fine-tune embedding model
""",
    },
]


def seed_demo_documents() -> None:
    """
    Ingest demo documents into the vector store.

    Fast-path: if the collection already has >= 50 chunks, all 12 demo docs
    are already indexed — skip the per-document existence checks (saves 12
    DB round-trips on every cold start).
    """
    from backend.rag.vectorstore import get_vector_store
    from backend.config import get_settings

    settings = get_settings()
    vs = get_vector_store()

    # Fast bulk check: if sufficient chunks already exist, skip entirely.
    try:
        stats = vs.get_collection_stats(settings.chroma_collection_name)
        if stats.get("document_count", 0) >= 50:
            logger.info(
                f"Demo documents: skipped (collection already has "
                f"{stats['document_count']} chunks)"
            )
            return
    except Exception:
        pass

    ingested = 0
    skipped = 0

    for doc in DEMO_DOCUMENTS:
        filename = doc["filename"]
        text = doc["text"]
        doc_id = hashlib.sha256(f"{filename}:{text[:100]}".encode()).hexdigest()[:16]

        try:
            if vs.doc_exists(doc_id, settings.chroma_collection_name):
                skipped += 1
                continue
        except Exception:
            pass

        try:
            vs.ingest_text(
                text=text,
                filename=filename,
                collection_name=settings.chroma_collection_name,
            )
            ingested += 1
            logger.info(f"Ingested demo doc: {filename}")
        except Exception as e:
            logger.warning(f"Failed to ingest {filename}: {e}")

    logger.info(f"Demo documents: {ingested} ingested, {skipped} already present")
