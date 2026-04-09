# RAG Small アーキテクチャ解説

Embedding からインデックス検索まで、ユーザーのリクエストをどのように処理しているかを `lib/` のコードを用いて説明します。

処理は大きく **2段階**（事前準備 + リアルタイム応答）に分かれます。

---

## 段階1: 事前準備（Embedding パイプライン）

ユーザーがチャットする**前に**、ナレッジ文書をベクトル化して DB に格納しておく処理です。`embedding_pipeline.py` が担当します。

```
docs/test.md → [load] → [chunk] → [embed] → [store] → Supabase DB
```

### Step 1: ドキュメント読み込み

**ファイル:** `lib/embedding_pipeline.py` `load_documents()` (L16-50)

```python
# docs/ フォルダから .md, .txt, .pdf を読み込み
for file_path in sorted(dir_path.iterdir()):
    text = file_path.read_text(encoding="utf-8")  # or PdfReader for PDF
```

各ファイルを `Document(page_content=テキスト, metadata={source, type})` に変換します。

### Step 2: チャンク分割

**ファイル:** `lib/embedding_pipeline.py` `chunk_documents()` (L53-70)

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # 1チャンク最大1000文字
    chunk_overlap=200,      # チャンク間200文字の重複（文脈維持）
    separators=["\n\n", "\n", "。", ".", " ", ""],  # 日本語対応
)
```

長い文書を意味のある単位で分割します。セパレータを左から優先的に使い、`\n\n`（段落）で割れなければ `。`（句点）で割ります。

### Step 3: Embedding 生成

**ファイル:** `lib/embedding_pipeline.py` `generate_embeddings()` (L73-85)

```python
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
batch_embeddings = embeddings_model.embed_documents(texts)  # → [1536次元のfloat配列]
```

各チャンクのテキストを OpenAI API に送り、**1536次元の数値ベクトル**に変換します。100件ずつバッチ処理します。

### Step 4: Supabase に格納

**ファイル:** `lib/embedding_pipeline.py` `store_in_supabase()` (L88-105)

```python
supabase.table("documents").insert({
    "content": "チャンクのテキスト",
    "metadata": {"source": "test.md", "chunk_index": 0},
    "embedding": [0.012, -0.034, ...],  # 1536次元ベクトル
}).execute()
```

テキスト・メタデータ・ベクトルをセットで `documents` テーブルに INSERT します。

---

## 段階2: ユーザーの質問に応答（リアルタイム）

ユーザーが「RAGとは？」と質問した時の処理です。`chat.py` → `rag_chain.py` → `supabase_client.py` が連携します。

```
ユーザー質問 → [embed] → [ベクトル検索] → [プロンプト構築] → [LLM応答] → 回答
```

### Step 1: 質問のベクトル化

**ファイル:** `lib/rag_chain.py` `search_relevant_documents()` (L29-30)

```python
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
query_embedding = embeddings.embed_query(question)  # "RAGとは？" → [1536次元ベクトル]
```

事前準備と**同じモデル**でクエリもベクトル化します。これにより同じベクトル空間で比較が可能になります。

### Step 2: コサイン類似度検索

**ファイル:** `lib/rag_chain.py` `search_relevant_documents()` (L32-42)

```python
supabase.rpc("match_documents", {
    "query_embedding": query_embedding,  # 質問ベクトル
    "match_threshold": 0.3,              # 類似度0.3以上のみ
    "match_count": 5,                    # 最大5件
}).execute()
```

Supabase の `match_documents` RPC関数が、DB内の全ドキュメントのベクトルとクエリベクトルの**コサイン類似度**（`1 - (embedding <=> query_embedding)`）を計算し、スコアが高い順に返します。HNSW インデックスにより高速に検索できます。

### Step 3: プロンプト構築

**ファイル:** `lib/rag_chain.py` `build_rag_prompt()` (L58-66)

```python
# 検索で見つかったチャンクをシステムプロンプトに埋め込む
messages = [
    {"role": "system", "content": "あなたは社内ナレッジに基づいて...\n# コンテキスト:\n{検索結果のテキスト}"},
    {"role": "user", "content": "RAGとは？"},
]
```

検索結果のテキストをシステムプロンプトの `{context}` に埋め込み、LLM に渡すメッセージを構築します。コンテキストに情報がない場合は「ナレッジベースに含まれていません」と回答するよう指示しています。

### Step 4: LLM ストリーミング応答

**ファイル:** `lib/chat.py` `generate_response()` (L13-23)

```python
stream = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,     # コンテキスト付きプロンプト
    temperature=0,         # 確定的な回答（ハルシネーション抑制）
    stream=True,           # トークン単位でストリーミング
)
for chunk in stream:
    yield chunk.choices[0].delta.content  # Streamlit に逐次表示
```

GPT-4o-mini がコンテキストに基づいて回答を生成し、トークン単位でストリーミング返却します。`temperature=0` によりハルシネーションを抑制しています。

---

## 全体の流れ図

```
【事前準備】embedding_pipeline.py
  docs/test.md
       ↓  load_documents()
  Document("RAGとは検索拡張生成...")
       ↓  chunk_documents()
  Chunk1("RAGの技術概要..."), Chunk2("LangChainの基本...")
       ↓  generate_embeddings()        ← OpenAI API
  [0.012, -0.034, ...] × 1536次元
       ↓  store_in_supabase()
  ┌─────────────────────────────────────┐
  │  Supabase pgvector (documents)      │
  │  id | content | embedding | ...     │
  └─────────────────────────────────────┘

【リアルタイム応答】chat.py → rag_chain.py
  ユーザー: "RAGとは？"
       ↓  embed_query()                ← OpenAI API
  [0.008, -0.041, ...] × 1536次元
       ↓  match_documents RPC          ← Supabase
  コサイン類似度で検索 → 上位5件取得
       ↓  build_rag_prompt()
  システムプロンプト + コンテキスト + 質問
       ↓  chat.completions.create()    ← OpenAI API (GPT-4o-mini)
  ストリーミング回答 → Streamlit に表示
```

---

## ファイル間の依存関係

```
app.py (Streamlit UI)
  └→ lib/chat.py (LLM応答生成)
       └→ lib/rag_chain.py (ベクトル検索 + プロンプト構築)
            └→ lib/supabase_client.py (DB接続)

lib/embedding_pipeline.py (事前準備CLI)
  ├→ lib/supabase_client.py (DB接続)
  └→ OpenAI Embeddings API
```

## 重要なポイント

- 事前準備とリアルタイム応答で**同じ Embedding モデル（text-embedding-3-small）**を使い、同じベクトル空間で比較している
- `temperature=0` で確定的な回答を生成し、ハルシネーションを抑制
- チャンク分割時に200文字のオーバーラップを設けて、文脈が途切れることを防止
- HNSW インデックスにより、大量のベクトルでも高速な近似最近傍探索が可能
