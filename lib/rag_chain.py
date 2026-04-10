"""RAG search logic: embed query, search Supabase, build prompt."""

from dataclasses import dataclass
from langchain_openai import OpenAIEmbeddings
from lib.supabase_client import get_supabase_admin


RAG_SYSTEM_PROMPT = """あなたは社内ナレッジに基づいて質問に回答するアシスタントです。
以下のコンテキスト情報を主な根拠に、質問に丁寧に回答してください。

## ルール:
- 回答はコンテキストの情報を優先してください
- コンテキストから引用する場合は「」で囲んでください
- 質問の表記ゆれ（ひらがな・カタカナ・略語）は柔軟に解釈してください
- コンテキストに部分的な情報がある場合は、その範囲で答えつつ「詳細はナレッジベースに記載がありません」と補足してください
- コンテキストに全く関連情報がない場合は、「この情報はナレッジベースに含まれていません」と回答してください

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

    context_parts = []
    sources = []
    for i, doc in enumerate(docs, 1):
        metadata = doc.get("metadata", {})
        source_name = metadata.get("source", "不明")
        section_name = metadata.get("section", "")
        label = f"[出典{i}: {source_name}"
        if section_name:
            label += f" - {section_name}"
        label += "]"
        context_parts.append(f"{label}\n{doc['content']}")
        sources.append(
            Source(
                content=doc["content"],
                metadata=metadata,
                similarity=doc["similarity"],
            )
        )

    context = "\n\n---\n\n".join(context_parts)
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
