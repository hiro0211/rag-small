## 2026-04-09

### 作業内容
- Python + Streamlit + LangChain でRAGスモール版を構築
- 当初 Next.js で開始したが、ユーザーの要望で Streamlit に変更
- TDD で全フェーズ���装完了（23テスト、カバレッジ100%）

### 完了フェーズ
- Phase 0: Python プロジェクト初期化（uv, pyproject.toml, pytest）
- Phase 1: Supabase pgvector セットアップ SQL 生成（MCP未接続のためSQL手動実行が必要）
- Phase 2: Embedding パイプライン（load → chunk → embed → store）
- Phase 3: RAG 検索ロジック（ベクトル���索 + プロンプト構築）
- Phase 4: Streamlit チャット UI + ストリーミング応答
- Phase 5: テスト用ドキュメント作成、テスト全通過

### ユーザーが対応すべき残タスク
1. OpenAI API キー取得 → .env.local に記入
2. Supabase プロジェクト作成
3. supabase-setup.sql を Supabase SQL Editor で実行
4. .env.local に Supabase URL/キーを記入
5. `uv run python lib/embedding_pipeline.py --dir ./docs` で Embedding 実行
6. `uv run streamlit run app.py` で動作確認

### 変更ファ���ル一覧
- `pyproject.toml`, `.gitignore`, `.env.local`, `.env.example`
- `lib/supabase_client.py`, `lib/embedding_pipeline.py`, `lib/rag_chain.py`, `lib/chat.py`
- `app.py` (Streamlit UI)
- `tests/test_supabase_client.py`, `tests/test_embedding_pipeline.py`, `tests/test_rag_chain.py`, `tests/test_app.py`
- `supabase-setup.sql`, `docs/test.md`, `USER_TASKS.md`

## 2026-04-10 モデル切り替え機能（GPT-4o-mini ⇔ Gemini 2.5 Flash）

### 作業内容
- LLMファクトリ `lib/llm.py` を新規追加（`create_llm`, `get_available_models`, `DEFAULT_MODEL`）
- `ChatOpenAI` と `ChatGoogleGenerativeAI` を `model_id` で切り替える薄い抽象
- `lib/graph.py` の `RAGState` に `model_id` フィールド追加、`generate` ノードで `create_llm` に委譲
- `stream_response` / `stream_response_with_sources` に `model_id` 引数追加
- `lib/chat.py` の `generate_response` / `generate_response_with_sources` に `model_id` パラメータ追加
- `app.py` にモデルセレクター（`st.selectbox`, `label_visibility="collapsed"`）をタイトル直下に配置
- 依存関係: `pyproject.toml` / `requirements.txt` に `langchain-google-genai>=2.0.0` 追加
- Embeddings は OpenAI (`text-embedding-3-small`) のままで再インデックス不要
- TDD 厳守（Red → Green → Refactor）、全77テストパス、カバレッジ 90.26%

### ユーザーが対応すべき残タスク
1. `.env.local` に `GOOGLE_API_KEY=your-key` を追加
2. 必要なら `uv pip install langchain-google-genai` を実行（既にインストール済みの可能性あり）
3. `streamlit run app.py` でモデル切り替え動作を確認

### 変更ファイル一覧
- `lib/llm.py` （新規）
- `tests/test_llm.py` （新規）
- `lib/graph.py`, `lib/chat.py`, `app.py`
- `tests/test_graph.py`, `tests/test_app.py`
- `pyproject.toml`, `requirements.txt`

## 2026-04-10 RAG柔軟性改善（閾値緩和 + プロンプト緩和 + クエリ前処理）

### 作業内容
短い・曖昧・タイポ質問（「ベクトルか」「べくとる」「ラグって何ですか」）が常に「ナレッジベースに含まれていません」と拒否される問題に対処：

1. **閾値緩和**: `search_relevant_documents` の `match_threshold` default を 0.5 → 0.3
   - `supabase-setup.sql` の `match_documents` 関数 default も同様に更新
2. **プロンプト緩和**: `RAG_SYSTEM_PROMPT` を書き換え
   - 「のみを根拠に」→「を主な根拠に」
   - 「推測や補完は禁止」を削除
   - 「表記ゆれ（ひらがな・カタカナ・略語）は柔軟に解釈」を追加
   - 「部分的な情報がある場合」の段階的回答を追加
3. **`rewrite_query` ノード追加** (LangGraph):
   - 新パイプライン: `START → rewrite_query → retrieve → generate → END`
   - LLMで質問を正規化・展開（略語展開・ひらがなタイポ修正・曖昧クエリの具体化）
   - 空文字を返した場合は元の質問にフォールバック
   - `RAGState` に `rewritten_query: str` フィールド追加
   - `retrieve` は `state.get("rewritten_query") or state["messages"][-1].content` で fallback
4. **`stream_response_with_sources`**: ソース即時表示のためリライト後クエリで検索（LLM呼び出しが2回/ターンに増加）

TDD 厳守（Red → Green）、全85テストパス、カバレッジ 90.75%。

### 影響
- LLM 呼び出しが 2回/ターン（rewrite + generate）に増加 → `gpt-4o-mini` で追加コスト ~$0.0001/ターン
- 閾値 0.3 により無関係ドキュメントが混ざる可能性 → プロンプトの「主な根拠」で LLM が取捨選択

### ユーザーが対応すべき残タスク
- Supabase SQL Editor で `supabase-setup.sql` の `match_documents` 関数を再実行（任意、lib 側から明示的に渡しているのでDBの既存 default 0.5 のままでも動作する）
- Streamlit 手動検証:
  1. 「ベクトルか」→ ベクトル検索の説明
  2. 「べくとる」→ ベクトル検索の説明
  3. 「ラグって何ですか」→ RAG の技術概要
  4. 「ベクトル検索とは？」→ 既存動作を維持

### 変更ファイル一覧
- `lib/rag_chain.py` - 閾値 0.3 + プロンプト緩和
- `lib/graph.py` - `REWRITE_PROMPT`, `rewrite_query` ノード, `RAGState.rewritten_query`, `retrieve` の fallback, `build_rag_graph` のエッジ更新, `stream_response_with_sources` の LLM リライト
- `supabase-setup.sql` - `match_documents` default 0.3
- `tests/test_rag_chain.py` - `test_default_threshold_is_0_3`, `test_prompt_allows_flexible_interpretation`
- `tests/test_graph.py` - `TestRewriteQueryNode`, `TestRetrieveUsesRewrittenQuery`, `TestGraphTopology` 追加、`TestStreamResponse` の sources テストを `create_llm` モック対応
