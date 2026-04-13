# RAG Small コードリーディングガイド

スマホで読みながらコードの中身を理解するためのガイドです。
`ARCHITECTURE.md` が「全体の流れ」を説明しているのに対し、このドキュメントは **「1行1行が何をしているか」** を説明します。

---

## 読む順番（推奨）

理解しやすい順に並べています。下に行くほど複雑になります。

```
Level 1（まず読む）
  ① lib/supabase_client.py   ← 最も短い。DB接続の基本
  ② lib/llm.py               ← LLMモデルの切り替え

Level 2（データの流れ）
  ③ lib/chat_history.py      ← 会話履歴のCRUD操作
  ④ lib/rag_chain.py         ← 検索 + プロンプト構築

Level 3（パイプライン）
  ⑤ lib/graph.py             ← LangGraphの心臓部
  ⑥ lib/chat.py              ← 薄いラッパー（すぐ読める）

Level 4（アプリ全体）
  ⑦ app.py                   ← Streamlit UI
  ⑧ lib/embedding_pipeline.py ← 事前準備パイプライン
  ⑨ lib/evaluator.py         ← 品質評価ツール
```

---

## ① lib/supabase_client.py（15行）

**役割**: Supabase（データベース）への接続を提供する。

```python
import os
from supabase import create_client, Client
```

- `os` → 環境変数（APIキーなど）を読むための標準ライブラリ
- `supabase` → Supabase公式のPythonライブラリ
- `Client` → 型ヒント用。「この関数はSupabaseクライアントを返します」と宣言するため

```python
def get_supabase_client() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_PUBLISHABLE_KEY"]
    return create_client(url, key)
```

- `-> Client` → 戻り値の型ヒント。「Clientオブジェクトを返す」という意味
- `os.environ["SUPABASE_URL"]` → 環境変数から値を取得。`.env.local` ファイルに書かれている
- **公開キー**（PUBLISHABLE_KEY）を使う → ブラウザから呼べるが権限が制限される

```python
def get_supabase_admin() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_SECRET_KEY"]
    return create_client(url, key)
```

- **秘密キー**（SECRET_KEY）を使う → フル権限。サーバーサイドでのみ使う
- プロジェクト全体では `get_supabase_admin()` がメインで使われている

### なぜ2つあるのか？

| 関数 | キー | 用途 |
|------|------|------|
| `get_supabase_client()` | 公開キー | 将来のフロントエンド用（現在未使用） |
| `get_supabase_admin()` | 秘密キー | 全ての操作（検索・保存・埋め込み） |

---

## ② lib/llm.py（39行）

**役割**: LLMモデル（GPT / Gemini）のインスタンスを作って返す工場（ファクトリ）。

```python
from functools import lru_cache
from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
```

- `lru_cache` → 一度作ったものを記憶して再利用する仕組み（後述）
- `BaseChatModel` → 全てのチャットモデルの親クラス（型ヒント用）
- `ChatOpenAI` → OpenAI の GPT を呼ぶためのクラス
- `ChatGoogleGenerativeAI` → Google の Gemini を呼ぶためのクラス

```python
DEFAULT_MODEL = "gpt-4o-mini"

AVAILABLE_MODELS: dict[str, str] = {
    "GPT-4o-mini": "gpt-4o-mini",
    "Gemini 2.5 Flash": "gemini-2.5-flash",
}
```

- `DEFAULT_MODEL` → 何も選ばなかったときに使われるモデル
- `AVAILABLE_MODELS` → UI のドロップダウンに表示される選択肢。キーが「表示名」、値が「モデルID」

```python
@lru_cache(maxsize=4)
def create_llm(model_id: str) -> BaseChatModel:
```

### `@lru_cache` とは？

```
1回目: create_llm("gpt-4o-mini")  → 新しく作る → 結果を記憶
2回目: create_llm("gpt-4o-mini")  → 記憶から返す（高速）
3回目: create_llm("gemini-2.5-flash") → 別のIDなので新しく作る
```

- `maxsize=4` → 最大4つまで記憶する
- なぜ必要？ → モデルのインスタンス作成にはAPI接続の初期化が含まれるため、毎回作ると遅い

```python
    resolved = model_id or DEFAULT_MODEL
```

- `model_id or DEFAULT_MODEL` → model_id が空文字 `""` の場合、`DEFAULT_MODEL` を使う
- Python では空文字は「偽（False的）」なので `or` の右側が使われる

```python
    if resolved == "gpt-4o-mini":
        return ChatOpenAI(model="gpt-4o-mini", temperature=0, streaming=True)
```

- `temperature=0` → 毎回同じ回答を返す（ランダム性なし）。RAGでは正確性が大事なのでこの設定
- `streaming=True` → 回答を1文字ずつ返す（ChatGPTのように文字が流れる表示）

```python
    if resolved == "gemini-2.5-flash":
        return ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
```

- Gemini には `streaming` パラメータを明示的に渡していない（LangChain側が自動判定する）

```python
    raise ValueError(f"Unknown model id: {resolved}")
```

- 知らないモデルIDが来たらエラー。バグの早期発見に役立つ

---

## ③ lib/chat_history.py（58行）

**役割**: チャットの会話セッションとメッセージをSupabaseに保存・取得する。

### データ構造のイメージ

```
chat_sessions テーブル        chat_messages テーブル
┌──────┬──────┬───────┐      ┌──────┬─────────┬──────┬────────┐
│ id   │title │updated│      │ id   │session_id│role  │content │
├──────┼──────┼───────┤      ├──────┼─────────┼──────┼────────┤
│ abc  │RAGに │12:00  │─────→│ 001  │ abc     │user  │RAGとは │
│      │ついて│       │      │ 002  │ abc     │asst  │RAGは…│
└──────┴──────┴───────┘      │ 003  │ abc     │user  │詳しく │
                              └──────┴─────────┴──────┴────────┘
```

```python
def create_session(title: str = "新しい会話") -> dict:
    supabase = get_supabase_admin()
    result = supabase.table("chat_sessions").insert({"title": title}).execute()
    return result.data[0]
```

- `insert({"title": title})` → SQLでいう `INSERT INTO chat_sessions (title) VALUES (...)`
- `.execute()` → 実際にDBに送信する
- `result.data[0]` → 挿入された行が返ってくるので、最初の1件を取得

```python
def get_messages(session_id: str) -> list[dict]:
    supabase = get_supabase_admin()
    result = (
        supabase.table("chat_messages")
        .select("role, content")
        .eq("session_id", session_id)
        .order("created_at")
        .execute()
    )
    return result.data
```

- `.select("role, content")` → 必要な列だけ取得（効率化）
- `.eq("session_id", session_id)` → SQLの `WHERE session_id = ...` に相当
- `.order("created_at")` → 時系列順に並べる
- **⚠ 現状の課題**: 全メッセージを取得するので、会話が長くなるとデータ量が増える

```python
def save_message(session_id: str, role: str, content: str) -> None:
    supabase = get_supabase_admin()
    supabase.table("chat_messages").insert(
        {"session_id": session_id, "role": role, "content": content}
    ).execute()
    supabase.table("chat_sessions").update(
        {"updated_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", session_id).execute()
```

- メッセージ保存と同時に、セッションの `updated_at`（最終更新日時）も更新する
- `-> None` → この関数は何も返さない
- `timezone.utc` → UTC（世界標準時）を使用。タイムゾーンの混乱を避けるため

---

## ④ lib/rag_chain.py（112行）

**役割**: ベクトル検索の実行とプロンプトの構築。RAGの「R」（Retrieval）を担当する。

### システムプロンプト

```python
RAG_SYSTEM_PROMPT = """あなたは社内ナレッジに基づいて質問に回答するアシスタントです。
...
# コンテキスト:
{context}"""
```

- `{context}` → 後で検索結果のテキストが埋め込まれるプレースホルダ
- 3段階の優先順位で「ナレッジにある→部分的にある→ない」に対応

### Source データクラス

```python
@dataclass
class Source:
    content: str      # チャンクのテキスト
    metadata: dict    # ファイル名・セクション名など
    similarity: float # 類似度スコア（0.0〜1.0）
```

- `@dataclass` → クラスの定義を簡略化するPythonの機能。`__init__` を自動生成してくれる
- 検索結果1件1件を「Source」オブジェクトとして扱う

### ハイブリッド検索

```python
def search_relevant_documents(
    question: str,
    match_threshold: float = 0.3,
    match_count: int = 5,
    use_hybrid: bool = True,
) -> dict:
```

- `match_threshold=0.3` → 類似度が0.3以上の結果だけ返す。低すぎると無関係な文書も返る
- `match_count=5` → 最大5件取得
- `use_hybrid=True` → ハイブリッド検索を使う（ベクトル+キーワード）

```python
    embeddings = _get_embeddings()
    query_embedding = embeddings.embed_query(question)
```

- ユーザーの質問を1536次元の数値ベクトルに変換
- `_get_embeddings()` は `@lru_cache` でキャッシュ済み

```python
    supabase.rpc("match_documents_hybrid", {
        "query_embedding": query_embedding,
        "query_text": question,
        "match_threshold": match_threshold,
        "match_count": match_count,
    })
```

- `.rpc()` → Supabaseに定義されたSQL関数をリモート呼び出し
- `match_documents_hybrid` → SQLで書かれたハイブリッド検索関数
- ベクトル（意味）とテキスト（キーワード）の両方を送る

```python
    for i, doc in enumerate(docs, 1):
        ...
        context_parts.append(f"[出典{i}: {source_name} - {section_name}]\n{doc['content']}")
```

- `enumerate(docs, 1)` → ループで `i` が1から始まる（0からではなく）
- 出典ラベル `[出典1: test.md - ベクトル検索]` を付けてテキストを整形

---

## ⑤ lib/graph.py（176行）— 最も重要なファイル

**役割**: LangGraphのパイプラインを定義する。このプロジェクトの心臓部。

### LangGraphとは？

```
普通のプログラム:  関数A() → 関数B() → 関数C()

LangGraph:  ノードA → ノードB → ノードC
            （状態を共有しながら実行する）
```

LangGraphは「ノード（処理）」と「エッジ（順番）」でAIパイプラインを組み立てるフレームワーク。
各ノードは共通の「状態（State）」を読み書きする。

### 状態の定義

```python
class RAGState(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]
    rewritten_query: str
    context: str
    sources: list
    model_id: str
```

- `TypedDict` → 辞書だけどキーと型が決まっている（型安全な辞書）
- `Annotated[..., add_messages]` → メッセージの更新方法を指定
  - 普通の辞書: `state["messages"] = 新しいリスト`（上書き）
  - `add_messages` 付き: `state["messages"]` に追加する（履歴が消えない）

### 3つのノード

```
START → rewrite_query → retrieve → generate → END
```

**ノード1: rewrite_query（質問のリライト）**

```python
def rewrite_query(state: RAGState) -> dict:
    question = state["messages"][-1].content  # 最後のメッセージ（ユーザーの質問）
    llm = create_llm(state.get("model_id") or DEFAULT_MODEL)
    prompt = REWRITE_PROMPT.format(question=question)
    response = llm.invoke([HumanMessage(content=prompt)])
    rewritten = response.content.strip() or question  # 空なら元の質問を使う
    return {"rewritten_query": rewritten}
```

- `state["messages"][-1]` → リストの最後の要素。Pythonでは `-1` で末尾にアクセスできる
- `response.content.strip()` → 前後の空白を除去
- `... or question` → stripした結果が空文字（偽）なら元の質問にフォールバック

**ノード2: retrieve（検索）**

```python
def retrieve(state: RAGState) -> dict:
    query = state.get("rewritten_query") or state["messages"][-1].content
    result = search_relevant_documents(query)
    return {"context": result["context"], "sources": result["sources"]}
```

- リライトされた質問があればそれを使い、なければ元の質問を使う
- 検索結果の context（テキスト）と sources（出典情報）を状態に書き込む

**ノード3: generate（回答生成）**

```python
def generate(state: RAGState) -> dict:
    system_msg = SystemMessage(
        content=RAG_SYSTEM_PROMPT.format(context=state["context"])
    )

    trimmed = trim_messages(
        state["messages"],
        max_tokens=20,        # ⚠ 現状は20「文字」
        token_counter=len,    # ⚠ Python文字数でカウント
        strategy="last",      # 最新のメッセージを残す
        start_on="human",     # 人間のメッセージから始める
        include_system=False, # システムメッセージは別で追加するので除外
    )

    llm = create_llm(state.get("model_id") or DEFAULT_MODEL)
    response = llm.invoke([system_msg] + trimmed)
    return {"messages": [response]}
```

- `trim_messages` → メッセージリストを制限サイズに切り詰める関数
- `[system_msg] + trimmed` → システムメッセージ＋会話履歴をリストとして結合
- **⚠ 課題**: `token_counter=len` は文字数カウントであり、本来はトークン数でカウントすべき

### グラフの組み立て

```python
def build_rag_graph():
    graph = StateGraph(RAGState)               # 空のグラフ作成
    graph.add_node("rewrite_query", rewrite_query)  # ノード登録
    graph.add_node("retrieve", retrieve)
    graph.add_node("generate", generate)
    graph.add_edge(START, "rewrite_query")     # 順番を定義
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)
    return graph.compile()                      # 実行可能な形にコンパイル
```

- `add_node("名前", 関数)` → ノードを登録
- `add_edge("A", "B")` → A → B の順で実行
- `compile()` → グラフを確定して実行可能にする

### ストリーミング関数

```python
def _build_messages(question: str, history: list[dict[str, str]]) -> list[AnyMessage]:
    messages: list[AnyMessage] = []
    for msg in history:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))
    messages.append(HumanMessage(content=question))
    return messages
```

- `app.py` から来る履歴は `{"role": "user", "content": "..."}` の辞書形式
- LangGraphは `HumanMessage` / `AIMessage` オブジェクトを要求する
- この関数が **辞書 → LangChainメッセージ** の変換を行う

```python
def stream_response_with_sources(question, history, model_id=""):
    messages = _build_messages(question, history)
    graph = get_compiled_graph()
    sources: list = []

    def _generator():
        for chunk in graph.stream(
            {...},
            stream_mode=["updates", "messages"],  # 2モード同時
        ):
            if chunk["type"] == "updates":
                # retrieve ノードの結果からソースを取得
                ...
                sources.extend(payload["retrieve"]["sources"])
            elif chunk["type"] == "messages":
                # generate ノードのトークンを逐次返す
                yield msg_chunk.content

    return _generator(), sources
```

- `yield` → ジェネレータ。全部完成してから返すのではなく、1つずつ返す
- `stream_mode=["updates", "messages"]` → ノードの結果とトークン両方を受け取る
- `sources` は関数の外で定義し、ジェネレータ内で書き込む（クロージャ）

---

## ⑥ lib/chat.py（24行）

**役割**: `graph.py` の薄いラッパー。将来的にキャッシュやログを挟む場所。

```python
def generate_response(question, history=None, model_id=""):
    yield from stream_response(question, history or [], model_id=model_id)
```

- `yield from` → 別のジェネレータの出力をそのまま転送する
- `history or []` → history が `None` なら空リストを使う

```python
def generate_response_with_sources(question, history=None, model_id=""):
    return stream_response_with_sources(question, history or [], model_id=model_id)
```

- こちらは `yield from` ではなく `return`（タプルを返すだけなので）

### なぜこの層があるのか？

`app.py` → `chat.py` → `graph.py` と間に挟むことで、将来的に「キャッシュ」「ログ記録」「レート制限」などを `chat.py` に追加しても `app.py` と `graph.py` を変更しなくて済む。

---

## ⑦ app.py（124行）

**役割**: ユーザーが触るUI画面。Streamlitフレームワークで構築。

### Streamlitの基本

```python
import streamlit as st
```

Streamlitは「Pythonを書くだけでWebアプリが作れる」フレームワーク。
`st.title("RAG Small")` と書くだけで画面にタイトルが表示される。

### セッション状態

```python
if "messages" not in st.session_state:
    st.session_state.messages = []
```

- `st.session_state` → ブラウザのタブごとに保持されるデータ
- ページが再描画されても消えない（Streamlitは操作のたびに全コードを再実行する）
- `messages` → 現在表示中の会話履歴
- `current_session_id` → 今開いているセッションのID
- `selected_model` → 選択中のLLMモデル

### サイドバー（会話履歴一覧）

```python
with st.sidebar:
    sessions = list_sessions(limit=10)
    for session in sessions:
        if st.button(label, key=session["id"]):
            st.session_state.current_session_id = session["id"]
            st.session_state.messages = get_messages(session["id"])
            st.rerun()
```

- `with st.sidebar:` → 以下のUI要素をサイドバーに配置
- `st.rerun()` → ページを再描画して画面を更新

### チャット入力と応答

```python
if prompt := st.chat_input("質問を入力してください"):
```

- `:=`（セイウチ演算子）→ 「代入しつつ条件判定」を1行で書ける。Python 3.8以降の機能
- ユーザーが入力したら `prompt` に文字列が入り、`if` が真になる

```python
    history = st.session_state.messages[:-1]  # 最後（今の質問）以外
    token_gen, sources = generate_response_with_sources(
        prompt, history, model_id=st.session_state.selected_model
    )
    response = st.write_stream(token_gen)
```

- `messages[:-1]` → リストの最後の要素を除外したスライス
- `st.write_stream(token_gen)` → ジェネレータからトークンを受け取り、1つずつ画面に表示

---

## ⑧ lib/embedding_pipeline.py（177行）

**役割**: ナレッジ文書をベクトル化してDBに格納する事前準備ツール。

### 2段階チャンク分割（Markdownの場合）

```python
md_header_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=[("#", "h1"), ("##", "h2")],
)
size_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500, chunk_overlap=100,
    separators=["\n\n", "\n", "。", ".", " ", ""],
)
```

```
元の文書
├── # 大見出し1
│   ├── ## 小見出し1-A  ← ヘッダで分割（Stage 1）
│   │   ├── チャンク1    ← サイズで分割（Stage 2）
│   │   └── チャンク2
│   └── ## 小見出し1-B
│       └── チャンク3
└── # 大見出し2
    └── チャンク4
```

- Stage 1: Markdownの見出し（`#`, `##`）で大きく分割
- Stage 2: 各セクションを500文字以内に分割（100文字の重複あり）
- `chunk_overlap=100` → チャンク間で100文字重複させて文脈の断絶を防ぐ

### Embedding生成（バッチ処理）

```python
def generate_embeddings(chunks):
    embeddings_model = OpenAIEmbeddings(model="text-embedding-3-small")
    batch_size = 100
    for i in range(0, len(chunks), batch_size):
        batch = chunks[i : i + batch_size]
        texts = [c.page_content for c in batch]
        batch_embeddings = embeddings_model.embed_documents(texts)
        all_embeddings.extend(batch_embeddings)
```

- `range(0, len(chunks), 100)` → 0, 100, 200, ... とステップ100で進む
- `chunks[i : i + 100]` → スライスで100件ずつ取り出す
- なぜ100件ずつ？ → OpenAI APIに一度に送れるリクエストサイズに制限があるため

---

## ⑨ lib/evaluator.py（170行）

**役割**: RAGの回答品質を自動で評価するツール。

### 2種類の評価方法

**方法1: Bigram（文字2つ組）の重なり率**

```python
def calc_faithfulness_simple(answer, context):
    answer_bigrams = _make_bigrams(answer)   # {"RA", "AG", "Gと", "とは", ...}
    context_bigrams = _make_bigrams(context)
    overlap = answer_bigrams & context_bigrams  # 共通部分（集合の積）
    return len(overlap) / len(answer_bigrams)
```

```
回答:    「RAGとは検索拡張生成です」
コンテキスト: 「RAGは検索拡張生成の技術です」

回答のbigram:     {RA, AG, Gと, とは, は検, 検索, 索拡, 拡張, 張生, 生成, 成で, です}
コンテキストのbigram: {RA, AG, Gは, は検, 検索, 索拡, 拡張, 張生, 生成, 成の, の技, 技術, 術で, です}
共通:            {RA, AG, は検, 検索, 索拡, 拡張, 張生, 生成, です}

Faithfulness = 9 / 12 = 0.75（75%がコンテキストに基づいている）
```

- API呼び出し不要で高速
- 完璧ではないが、ハルシネーション（でっち上げ）の大まかな検出には使える

**方法2: LLM-as-Judge（LLMに評価させる）**

```python
def calc_faithfulness_llm(answer, context):
    client = OpenAI()
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{
            "role": "system",
            "content": "回答がコンテキスト情報のみに基づいているか評価してください。0.0〜1.0のスコアを返してください。"
        }, ...],
    )
    return float(response.choices[0].message.content.strip())
```

- GPTに「この回答はコンテキストに忠実か？」と聞く
- より正確だが、API呼び出しが必要でコストがかかる

---

## Python文法の補足

コード中に出てくるPython特有の書き方をまとめます。

### 型ヒント

```python
def save_message(session_id: str, role: str, content: str) -> None:
```

- `session_id: str` → 「session_idは文字列」という宣言
- `-> None` → 「戻り値はなし」という宣言
- 型ヒントは**動作に影響しない**（ドキュメントとIDEの補助のため）

### リスト内包表記

```python
texts = [c.page_content for c in batch]
```

これは以下と同じ:
```python
texts = []
for c in batch:
    texts.append(c.page_content)
```

### f文字列

```python
label = f"[出典{i}: {source_name}]"
```

- `f"..."` → 文字列の中に `{変数名}` で値を埋め込める
- `i=1`, `source_name="test.md"` なら → `"[出典1: test.md]"`

### ジェネレータと yield

```python
def stream_response(question, history):
    for chunk in graph.stream(...):
        yield chunk.content  # 1つずつ返す
```

- 普通の関数は `return` で全部まとめて返す
- `yield` を使うと「1つ返して待機 → 次を要求されたらまた1つ返す」ができる
- チャットの文字が1文字ずつ流れて表示されるのはこの仕組み

### デコレータ

```python
@lru_cache(maxsize=1)
def get_compiled_graph():
    return build_rag_graph()
```

- `@` で始まる行はデコレータ。関数に「追加機能」を付ける
- `@lru_cache` → 結果をキャッシュする機能を追加
- `@dataclass` → クラスに `__init__` などを自動生成する機能を追加

---

## ファイル間のデータの流れ

ユーザーが「RAGとは？」と入力してから回答が表示されるまでの流れ:

```
[ユーザー入力] "RAGとは？"
      │
      ▼
app.py
  ├── save_message() で Supabase に保存
  ├── history = 過去のメッセージ一覧
  └── generate_response_with_sources(質問, history) を呼ぶ
          │
          ▼
      lib/chat.py
        └── stream_response_with_sources() に転送
                │
                ▼
            lib/graph.py
              ├── _build_messages(): 辞書 → LangChainメッセージに変換
              ├── graph.stream() でパイプライン開始
              │
              │   ┌─ rewrite_query ノード ──────────┐
              │   │ LLMで「RAGとは？」を                │
              │   │ 「RAG (Retrieval-Augmented        │
              │   │  Generation) とは何ですか？」に変換 │
              │   └──────────┬───────────────────────┘
              │              ▼
              │   ┌─ retrieve ノード ───────────────┐
              │   │ lib/rag_chain.py                 │
              │   │   ├── 質問をベクトル化（OpenAI API）│
              │   │   ├── Supabase RPC でハイブリッド検索│
              │   │   └── 出典ラベル付きテキストを返す  │
              │   └──────────┬───────────────────────┘
              │              ▼
              │   ┌─ generate ノード ───────────────┐
              │   │ ├── 会話履歴をトリム              │
              │   │ ├── システムプロンプト + 履歴 + 質問 │
              │   │ └── LLMがストリーミング回答生成    │
              │   └──────────┬───────────────────────┘
              │              ▼
              ├── yield でトークンを1つずつ返す
              └── sources リストに出典情報を格納
                      │
                      ▼
app.py
  ├── st.write_stream() でトークンを画面に表示
  ├── st.expander() で出典情報を表示
  └── save_message() で回答を Supabase に保存
```
