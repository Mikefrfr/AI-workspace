from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("BAAI/bge-reranker-base")
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


def query_rag(user_query: str, chroma_path: str, model: str = "llama3") -> dict:
    from langchain_huggingface import HuggingFaceEmbeddings
    from sentence_transformers import CrossEncoder

    # --- STEP 1: INITIALIZE CHROMADB VECTOR SEARCH ---
    vector_db = Chroma(
        persist_directory=chroma_path,
        embedding_function=embedding_model
    )

    # --- STEP 2: INITIALIZE LOCAL BM25 KEYWORD SEARCH ---
    db_records    = vector_db.get()
    all_documents = []

    for i in range(len(db_records['ids'])):
        current_metadata = {}
        if db_records.get('metadatas') and db_records['metadatas'][i] is not None:
            current_metadata = db_records['metadatas'][i]

        doc = Document(
            page_content=db_records['documents'][i],
            metadata=current_metadata
        )
        all_documents.append(doc)

    bm25_retriever = BM25Retriever.from_documents(all_documents)

    # --- STEP 3 & 4: HYBRID SEARCH ---
    bm25_docs   = bm25_retriever.invoke(user_query)[:3]
    vector_docs = vector_db.similarity_search(user_query, k=3)

    # deduplicate
    candidate_documents = []
    seen_contents       = set()
    for doc in (bm25_docs + vector_docs):
        if doc.page_content not in seen_contents:
            seen_contents.add(doc.page_content)
            candidate_documents.append(doc)

    # --- STEP 5: RE-RANKING ---
    pairs         = [[user_query, doc.page_content] for doc in candidate_documents]
    rerank_scores = reranker.predict(pairs)

    scored_docs = list(zip(candidate_documents, rerank_scores))
    scored_docs.sort(key=lambda x: x[1], reverse=True)

    final_fused_chunks = [doc for doc, score in scored_docs[:2]]

    # build context blocks (same format as your original)
    context_blocks = []
    for index, doc in enumerate(final_fused_chunks):
        chunk_id = f"Source Chunk #{index + 1}"
        context_blocks.append(f"--- {chunk_id} ---\n{doc.page_content}")

    unified_context = "\n\n".join(context_blocks)

    # --- STEP 6: GENERATE ANSWER VIA OLLAMA ---
    final_rag_prompt = f"""You are an expert assistant answering questions based strictly on the provided document segments.

CRITICAL INSTRUCTIONS:
1. Base your answer ONLY on the text inside the provided Context Blocks.
2. You MUST cite which chunk your information came from by explicitly using [Source Chunk #1], [Source Chunk #2], etc.
3. If the context blocks do not contain the answer to the query, explicitly state: 'I cannot find the answer in the provided document.' Do not use external knowledge.

Context documentation blocks:
{unified_context}

User Question: {user_query}
Answer:"""

    llm      = ChatOllama(model=model, temperature=0.2)
    response = llm.invoke(final_rag_prompt)

    return {
        "answer":  response.content,
        "sources": [doc.page_content for doc in final_fused_chunks],
    }