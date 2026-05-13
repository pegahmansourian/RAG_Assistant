import logging

from langchain_core.prompts import ChatPromptTemplate

from ResearchRAG.retrieval.retriever import retrieve_documents

logger = logging.getLogger(__name__)

def format_documents(documents):
    formatted_chunks = []

    for i, doc in enumerate(documents, start=1):
        source_file = doc.metadata.get("title", "unknown")
        author = doc.metadata.get("authors", "unknown")
        section = doc.metadata.get("section_header", "unknown")

        chunk_text = (
            f"[Source {i} | File: {source_file} | Author: {author} | Section: {section}]\n"
            f"{doc.page_content}"
        )
        formatted_chunks.append(chunk_text)

    return "\n\n".join(formatted_chunks)


def build_rag_prompt():
    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant answering questions about technical ML-based security in vehicular networks documents.

Use only the provided context to answer the question.
If the answer is not supported by the context, say you do not have enough information.

CITATION RULES:
- Ignore all numbers in the context. Create your own citation list from scratch.
- Assign [1] to the first source you cite, [2] to the second, and so on. No gaps.
- Cite immediately after each claim: single source [1], multiple sources [1,2].
- End with a "References" section: [N] Author(s). "Paper Title." Section: <section name/number>.
- Only list sources you actually cited. No empty entries.

Question:
{question}

Context:
{context}

Answer:
"""
    )
    return prompt


def run_rag(query, retriever, llm):
    logger.info("Running RAG pipeline")
    try:
        retrieved_documents = retrieve_documents(query, retriever)
        logger.info("Retrieved %d documents", len(retrieved_documents))
        context = format_documents(retrieved_documents)

        prompt = build_rag_prompt()
        messages = prompt.format_messages(question=query, context=context)

        logger.info("Invoking LLM")
        response = llm.invoke(messages)
        logger.info("RAG pipeline completed successfully")

        return {
            "query": query,
            "answer": response.content,
            "retrieved_documents": retrieved_documents,
        }
    except Exception:
        logger.exception("Failed to run RAG pipeline")
        raise