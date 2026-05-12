from langchain_core.prompts import ChatPromptTemplate

from ResearchRAG.retrieval.retriever import retrieve_documents


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
- Cite inline using bracketed numbers, e.g., [1], [2], or [1, 3] for multiple sources.
- Each cited fact must have an inline citation immediately after the claim.
- At the end, list all cited references in a numbered "References" section using this exact format:
  [N] Author(s). "Paper Title." Section: <section name/number>, <year if available>.
- Place each cited reference in one line.

Question:
{question}

Context:
{context}

Answer:
"""
    )
    return prompt


def run_rag(query, retriever, llm):
    retrieved_documents = retrieve_documents(query, retriever)
    context = format_documents(retrieved_documents)

    prompt = build_rag_prompt()
    messages = prompt.format_messages(question=query, context=context)

    response = llm.invoke(messages)

    return {
        "query": query,
        "answer": response.content,
        "retrieved_documents": retrieved_documents,
    }