from langchain_core.prompts import ChatPromptTemplate

from .retriever import retrieve_documents


def format_documents(documents):
    formatted_chunks = []

    for i, doc in enumerate(documents, start=1):
        source_file = doc.metadata.get("source_file", "unknown")
        page_num = doc.metadata.get("page_num", "unknown")

        chunk_text = (
            f"[Source {i} | File: {source_file} | Page: {page_num}]\n"
            f"{doc.page_content}"
        )
        formatted_chunks.append(chunk_text)

    return "\n\n".join(formatted_chunks)


def build_rag_prompt():
    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful assistant answering questions about technical ML and security in vehicular networks documents.

Use only the provided context to answer the question.
If the answer is not supported by the context, say you do not have enough information.
When possible, mention the supporting source file and page number.

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