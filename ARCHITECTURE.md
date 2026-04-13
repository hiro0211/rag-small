# RAG Small アーキテクチャ解説

Embedding からハイブリッド検索、LangGraph パイプラインによるストリーミング応答まで、ユーザーのリクエストをどのように処理しているかを `lib/` のコードを用いて説明します。

処理は大きく **2段階**（事前準備 + リアルタイム応答）に分かれます。

---

## 段階1: 事前準備（Embedding パイプライン）

ユーザーがチャットする**前に**、ナレッジ文書をベクトル化して DB に格納しておく処理です。`embedding_pipeline.py` が担当します。

```
docs/test.md → [load] → [chunk] → [embed] → [store] → Supabase DB
```

### Step 1: ドキュメント読み込み

**ファイル:** `lib/embedding_pipeline.py` `load_documents()`

```python
# docs/ フォルダから .md, .txt, .pdf を読み込み
for file_path in sorted(dir_path.iterdir()):
    text = file_path.read_text(encoding="utf-8")  # or PdfReader for PDF
```

各ファイルを `Document(page_content=テキスト, metadata={source, type})` に変換します。

### Step 2: チャンク分割

**ファイル:** `lib/embedding_pipeline.py` `chunk_documents()`

```python
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,        # 1チャンク最大1000文字
    chunk_overlap=200,      # チャンク間200文字の重複（文脈維持）
    separators=["\n\n", "\n", "。", ".", " ", ""],  # 日本語対応
)
```

長い文書を意味のある単位で分割します。セパレータを左から優先的に使い、`\n\n`（段落）で割れなければ `。`（句点）で割ります。

### Step 3: Embedding 生成

**ファイル:** `lib/embedding_pipeline.py` `generate_embeddings()`

```python
embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
batch_embeddings = embeddings_model.embed_documents(texts)  # → [1536次元のfloat配列]
```

各チャンクのテキストを OpenAI API に送り、**1536次元の数値ベクトル**に変換します。100件ずつバッチ処理します。

### Step 4: Supabase に格納

**ファイル:** `lib/embedding_pipeline.py` `store_in_supabase()`

```python
supabase.table("documents").insert({
    "content": "チャンクのテキスト",
    "metadata": {"source": "test.md", "chunk_index": 0},
    "embedding": [0.012, -0.034, ...],  # 1536次元ベクトル
}).execute()
```

テキスト・メタデータ・ベクトルをセットで `documents` テーブルに INSERT します。

テーブルには2種類のインデックスが設定されています：

| インデックス | 種類 | 用途 |
|------------|------|------|
| `documents_embedding_idx` | HNSW (`vector_cosine_ops`) | ベクトル類似度検索の高速化 |
| `documents_content_tsvector_idx` | GIN (`to_tsvector`) | キーワード全文検索の高速化 |

---

## 段階2: ユーザーの質問に応答（LangGraph パイプライン）

ユーザーが質問した時の処理です。LangGraph の4ノード直列パイプラインで実行されます。

```
START → rewrite_query → retrieve → generate → END
```

`app.py` → `lib/chat.py` → `lib/graph.py` → `lib/rag_chain.py` が連携します。

### 全体フロー概要

```
ユーザー: 「べくとるって何？」
     │
     │  app.py → stream_response_with_sources()
     │  graph.stream() を1回だけ実行
     │
     ▼
  ┌──────────────────────────────────────────────┐
  │  STEP 1: rewrite_query                       │
  │  LLMが質問をリライト                           │
  │  「べくとるって何？」→「ベクトル検索とは何ですか？」 │
  └──────────────┬───────────────────────────────┘
                 ▼
  ┌──────────────────────────────────────────────┐
  │  STEP 2: retrieve（ハイブリッド検索）           │
  │  ベクトル類似度 70% + キーワード一致 30%         │
  │  → 上位5件を出典ラベル付きで取得                │
  └──────────────┬───────────────────────────────┘
                 ▼
  ┌──────────────────────────────────────────────┐
  │  STEP 3: generate                            │
  │  3段階優先順位プロンプト + 会話履歴             │
  │  → LLMがストリーミング回答                     │
  └──────────────┬───────────────────────────────┘
                 ▼
  Streamlit UI にストリーミング表示 + 出典表示
```

---

### Step 1: クエリリライト

**ファイル:** `lib/graph.py` `rewrite_query()`

```python
REWRITE_PROMPT = """ユーザーの質問を検索用に書き換えるアシスタントです。
- 略語は正式名称に展開（例: ラグ → RAG (Retrieval-Augmented Generation)）
- ひらがなのタイポは正規化（例: べくとる → ベクトル）
- 曖昧・断片的な質問は完全な疑問文に（例: ベクトルか → ベクトル検索とは何ですか）
- 十分明確な場合はそのまま返す
"""
```

LLM を使ってユーザーの質問を検索に適した形に書き換えます。短い・曖昧・略語を含む質問に特に有効です。

**スキップ条件**: 質問が10文字以上かつ `？/?/ですか/でしょうか` を含む明確な質問の場合はLLM呼び出しをスキップし、高速化します（`MEMORY.md` に記録）。

---

### Step 2: ハイブリッド検索（retrieve）

**ファイル:** `lib/graph.py` `retrieve()` → `lib/rag_chain.py` `search_relevant_documents()`

ここが検索精度の核心部分です。**ベクトル検索とキーワード全文検索を組み合わせた**ハイブリッド検索を行います。

#### 2-1. 質問のベクトル化

```python
embeddings = _get_embeddings()  # @lru_cache でキャッシュ済み
query_embedding = embeddings.embed_query(question)  # → [1536次元ベクトル]
```

事前準備と**同じモデル（text-embedding-3-small）**でクエリもベクトル化します。

#### 2-2. ハイブリッド検索 RPC 呼び出し

```python
supabase.rpc("match_documents_hybrid", {
    "query_embedding": query_embedding,  # ベクトル（意味検索用）
    "query_text": question,              # テキスト（キーワード検索用）
    "match_threshold": 0.3,
    "match_count": 5,
}).execute()
```

従来の `match_documents` はベクトルのみを受け取りましたが、`match_documents_hybrid` は**ベクトルとテキスト両方**を受け取ります。

#### 2-3. SQL関数でのスコア計算

**ファイル:** `supabase-setup.sql` `match_documents_hybrid()`

```sql
similarity =
    0.7 × (1 - (embedding <=> query_embedding))   -- ベクトル類似度（70%）
  + 0.3 × ts_rank(                                  -- キーワード一致度（30%）
        to_tsvector('simple', content),
        plainto_tsquery('simple', query_text)
    )
```

| 要素 | 役割 |
|------|------|
| `embedding <=> query_embedding` | pgvector のコサイン距離（小さいほど似ている） |
| `1 - 距離` | 類似度に変換（大きいほど似ている） |
| `to_tsvector('simple', content)` | ドキュメントをトークン列に変換 |
| `plainto_tsquery('simple', query_text)` | クエリをトークン列に変換 |
| `ts_rank(...)` | キーワード一致のスコア |
| `0.7` / `0.3` | ベクトル重視の重み配分 |

**なぜ2つ組み合わせるのか:**

- **ベクトル検索の弱点**: 意味的に近い文書は見つかるが、固有名詞・専門用語の完全一致を保証しない
- **キーワード検索の弱点**: 文字列一致するが、言い換え・類義語を見逃す
- **ハイブリッド**: 両方のスコアを加重平均し、弱点を相互補完

| 質問例 | ベクトル検索 | キーワード検索 | ハイブリッド |
|--------|------------|-------------|------------|
| 「pgvectorとは」 | 意味的に近い文書 | "pgvector" を含む文書 | 両方で高精度 |
| 「べくとる検索」 | Embedding 後は意味が分かる | 文字列一致しない | ベクトル側がカバー |
| 「HNSW index設定」 | 概念的に近い文書 | "HNSW" を含む文書を確実に拾う | キーワード側が補強 |

#### 2-4. 出典ラベル付きコンテキスト整形

```python
# 検索結果を出典ラベル付きで整形
context_parts.append(f"[出典{i}: {source_name} - {section_name}]\n{doc['content']}")
context = "\n\n---\n\n".join(context_parts)
```

出力例:
```
[出典1: test.md - ベクトル検索]
ベクトル検索とはドキュメントを数値ベクトルに変換し...

---

[出典2: test.md - pgvector]
pgvectorはPostgreSQLの拡張機能で...
```

LLM がどのソースから引用しているか判別しやすくなります。

---

### Step 3: LLM ストリーミング応答（generate）

**ファイル:** `lib/graph.py` `generate()`

#### 3-1. プロンプト構築

**ファイル:** `lib/rag_chain.py` `RAG_SYSTEM_PROMPT`

```python
RAG_SYSTEM_PROMPT = """あなたは社内ナレッジに基づいて質問に回答するアシスタントです。

## 回答の優先順位:
1. コンテキストに直接的な回答がある場合 → コンテキストの情報を主な根拠として回答
2. コンテキストに部分的・関連する情報がある場合 → その範囲で回答し、不足部分は一般知識で補足
3. コンテキストに関連情報がない場合 → 一般知識で回答

## ルール:
- コンテキストから引用する場合は「」で囲む
- 表記ゆれ（ひらがな・カタカナ・略語）は柔軟に解釈
- コンテキスト情報に基づく回答には【ナレッジベース】と明記
- 一般知識に基づく回答には【一般知識】と明記
"""
```

3段階の優先順位により、コンテキスト外の質問も拒否せず回答できます。

#### 3-2. 会話履歴のトリム + LLM 呼び出し

```python
# 会話履歴を直近のみに制限（コンテキスト長の節約）
trimmed = trim_messages(state["messages"], max_tokens=20, token_counter=len, ...)

# System(コンテキスト付き) + 会話履歴 + ユーザー質問 → LLM
llm = create_llm(state.get("model_id") or DEFAULT_MODEL)
response = llm.invoke([system_msg] + trimmed)
```

#### 3-3. モデル選択

**ファイル:** `lib/llm.py` `create_llm()`

| 表示名 | model_id | 実装 |
|--------|----------|------|
| GPT-4o-mini | `gpt-4o-mini` | `ChatOpenAI(temperature=0, streaming=True)` |
| Gemini 2.5 Flash | `gemini-2.5-flash` | `ChatGoogleGenerativeAI(temperature=0)` |

`@lru_cache` でインスタンスをキャッシュし、毎回の生成を回避します。

---

### Step 4: Streamlit 表示

**ファイル:** `app.py`、`lib/graph.py` `stream_response_with_sources()`

```python
# 単一の graph.stream() から sources と tokens を同時取得
graph.stream(
    {...},
    stream_mode=["updates", "messages"],  # 2モード同時
)
```

| stream_mode | 取得するもの |
|-------------|------------|
| `"updates"` | `retrieve` ノードの結果から `sources` リストを抽出 |
| `"messages"` | `generate` ノードのトークンを逐次 yield |

Streamlit 側では `st.write_stream()` でトークンを逐次表示し、完了後に `st.expander("出典情報")` で出典を折りたたみ表示します。

---

## 全体の流れ図

```
【事前準備】embedding_pipeline.py
  docs/test.md
       ↓  load_documents()
  Document("RAGとは検索拡張生成...")
       ↓  chunk_documents()
  Chunk1("RAGの技術概要..."), Chunk2("LangChainの基本...")
       ↓  generate_embeddings()              ← OpenAI API
  [0.012, -0.034, ...] × 1536次元
       ↓  store_in_supabase()
  ┌──────────────────────────────────────────────────┐
  │  Supabase pgvector (documents テーブル)            │
  │  id | content | metadata | embedding              │
  │  + HNSW インデックス（ベクトル検索用）              │
  │  + GIN インデックス（全文検索用）                   │
  └──────────────────────────────────────────────────┘

【リアルタイム応答】LangGraph パイプライン
  ユーザー: "べくとるって何？"
       ↓  rewrite_query()                    ← LLM (GPT-4o-mini / Gemini)
  "ベクトル検索とは何ですか？"
       ↓  embed_query()                      ← OpenAI Embeddings API
  [0.008, -0.041, ...] × 1536次元
       ↓  match_documents_hybrid RPC         ← Supabase
  0.7 × ベクトル類似度 + 0.3 × キーワード一致 → 上位5件
       ↓  generate()                         ← LLM (ストリーミング)
  3段階優先順位プロンプト + 出典ラベル付きコンテキスト
       ↓  stream_response_with_sources()
  Streamlit にストリーミング表示 + 出典情報
```

---

## ファイル間の依存関係

```
app.py (Streamlit UI + セッション管理)
  ├→ lib/chat.py (LLM応答の薄いラッパー)
  │    └→ lib/graph.py (LangGraph RAG パイプライン)
  │         ├→ lib/rag_chain.py (ハイブリッド検索 + プロンプト構築)
  │         │    └→ lib/supabase_client.py (DB接続)
  │         └→ lib/llm.py (LLMファクトリ: GPT-4o-mini / Gemini)
  └→ lib/chat_history.py (セッション・メッセージ管理)
       └→ lib/supabase_client.py (DB接続)

lib/embedding_pipeline.py (事前準備 CLI)
  ├→ lib/supabase_client.py (DB接続)
  └→ OpenAI Embeddings API
```

---

## パフォーマンス最適化

| 最適化 | 実装箇所 | 効果 |
|--------|---------|------|
| グラフキャッシュ | `get_compiled_graph()` `@lru_cache` | `build_rag_graph()` をプロセス起動時に1回だけ実行 |
| LLM キャッシュ | `create_llm()` `@lru_cache` | モデルインスタンスの再生成を回避 |
| Embeddings キャッシュ | `_get_embeddings()` `@lru_cache` | OpenAIEmbeddings インスタンスの再生成を回避 |
| 単一グラフ実行 | `stream_response_with_sources()` | `stream_mode=["updates", "messages"]` で1回の実行から sources と tokens を同時取得 |
| rewrite スキップ | `rewrite_query()` の条件分岐 | 明確な質問では LLM リライトをスキップ |

---

## 重要なポイント

- 事前準備とリアルタイム応答で**同じ Embedding モデル（text-embedding-3-small）**を使い、同じベクトル空間で比較している
- **ハイブリッド検索**（ベクトル 70% + キーワード 30%）により、意味的類似とキーワード一致の両方を活かした高精度検索
- `temperature=0` で確定的な回答を生成し、ハルシネーションを抑制
- チャンク分割時に200文字のオーバーラップを設けて、文脈が途切れることを防止
- HNSW インデックスにより高速なベクトル近似最近傍探索、GIN インデックスにより高速な全文検索
- 3段階優先順位プロンプトにより、ナレッジ外の質問も拒否せず一般知識で回答可能
- 出典ラベル（`[出典N: ファイル名 - セクション]`）により LLM が情報源を区別しやすい
- `@lru_cache` と単一グラフ実行で応答速度を最適化
