"""RAG search logic: embed query, search Supabase, build prompt."""

from dataclasses import dataclass
from langchain_openai import OpenAIEmbeddings
from lib.supabase_client import get_supabase_admin


RAG_SYSTEM_PROMPT = """あなたは社内ナレッジに基づいて質問に回答するアシスタントです。
以下のコンテキスト情報を参考に、質問に正確に回答してください。
コンテキストに情報がない場合は、「この情報はナレッジベースに含まれていません」と回答してください。

# コンテキスト:
{context}"""


@dataclass
class Source:
    content: str
    metadata: dict
    similarity: float


def search_relevant_documents(
    question: str,
    match_threshold: float = 0.3,
    match_count: int = 5,
) -> dict:
    """Search for relevant documents using vector similarity."""
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    query_embedding = embeddings.embed_query(question)

    supabase = get_supabase_admin()
    result = (
        supabase.rpc(
            "match_documents",
            {
                "query_embedding": query_embedding,
                "match_threshold": match_threshold,
                "match_count": match_count,
            },
        ).execute()
    )

    docs = result.data or []
    context = "\n\n".join(doc["content"] for doc in docs)
    sources = [
        Source(
            content=doc["content"],
            metadata=doc.get("metadata", {}),
            similarity=doc["similarity"],
        )
        for doc in docs
    ]

    return {"context": context, "sources": sources}


def build_rag_prompt(
    question: str, context: str
) -> list[dict[str, str]]:
    """Build chat messages with RAG context."""
    system_message = RAG_SYSTEM_PROMPT.format(context=context)
    return [
        {"role": "system", "content": system_message},
        {"role": "user", "content": question},
    ]
