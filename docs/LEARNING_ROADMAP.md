# RAG Small 学習ロードマップ

このプロジェクトのコードを自分のものにするためのロードマップです。
iOSのClaudeアプリでスキマ時間に読み進められるように設計しています。

---

## このロードマップの使い方

1. 各ステップを上から順に進める
2. 「理解度チェック」の質問に自分の言葉で答えられたらクリア
3. 答えられなかったら `CODE_READING_GUIDE.md` の該当箇所を読み直す
4. Claudeアプリで「〇〇の部分がわからない」と質問してもOK

---

## Week 1: 土台を理解する

### Day 1-2: プロジェクトの全体像

**読むもの**: `ARCHITECTURE.md` の冒頭〜「段階1: 事前準備」

**ゴール**: 「2段階の処理がある」ことを理解する

```
段階1（事前準備）: 文書 → ベクトル化 → DB保存
段階2（リアルタイム）: 質問 → 検索 → 回答生成
```

**理解度チェック**:
- Q: このアプリは何をするアプリ？
- Q: なぜ事前にドキュメントをベクトル化しておく必要がある？
- Q: 「Embedding」を一言で説明すると？

---

### Day 3-4: DB接続とモデル選択

**読むもの**: `CODE_READING_GUIDE.md` の ①supabase_client.py と ②llm.py

**ゴール**: 最も短い2ファイルでPythonの基本パターンを掴む

**理解度チェック**:
- Q: `get_supabase_admin()` と `get_supabase_client()` の違いは？
- Q: `@lru_cache` は何のために使っている？
- Q: `temperature=0` にするとLLMの振る舞いはどう変わる？

---

### Day 5-7: 会話履歴の保存

**読むもの**: `CODE_READING_GUIDE.md` の ③chat_history.py

**ゴール**: CRUD操作（作成・読み取り・更新・削除）のパターンを理解する

**理解度チェック**:
- Q: `save_message()` は何テーブルに対して何回書き込みをする？
- Q: `.eq("session_id", session_id)` はSQLで書くとどうなる？
- Q: 会話が1000ラリーになったとき、`get_messages()` の問題点は？

---

## Week 2: 検索の仕組みを理解する

### Day 8-10: ベクトル検索とハイブリッド検索

**読むもの**: `CODE_READING_GUIDE.md` の ④rag_chain.py + `ARCHITECTURE.md` の「Step 2: ハイブリッド検索」

**ゴール**: 「なぜ2つの検索を組み合わせるのか」を説明できる

**理解度チェック**:
- Q: ベクトル検索だけでは何が弱い？
- Q: キーワード検索だけでは何が弱い？
- Q: `match_threshold=0.3` の意味は？
- Q: 出典ラベル `[出典1: test.md]` はなぜ付ける？

---

### Day 11-14: Embeddingパイプライン

**読むもの**: `CODE_READING_GUIDE.md` の ⑧embedding_pipeline.py

**ゴール**: 文書がDBに入るまでの4ステップを説明できる

**理解度チェック**:
- Q: チャンク分割で `chunk_overlap=100` を設定する理由は？
- Q: Markdownファイルだけ2段階分割する理由は？
- Q: バッチサイズ100で処理する理由は？
- Q: 同じEmbeddingモデルを事前準備と検索で使う理由は？

---

## Week 3: パイプラインの核心

### Day 15-18: LangGraphパイプライン

**読むもの**: `CODE_READING_GUIDE.md` の ⑤graph.py（最重要）

**ゴール**: 3つのノードそれぞれの役割と、データの流れを説明できる

**理解度チェック**:
- Q: `RAGState` の `Annotated[..., add_messages]` は通常の辞書と何が違う？
- Q: `rewrite_query` ノードは何のために存在する？
- Q: `trim_messages(max_tokens=20, token_counter=len)` の現在の問題点は？
- Q: `yield` と `return` の違いは？
- Q: `stream_mode=["updates", "messages"]` で2つのモードを使う理由は？

---

### Day 19-21: UIとデータの流れ

**読むもの**: `CODE_READING_GUIDE.md` の ⑥chat.py + ⑦app.py + 末尾のデータフロー図

**ゴール**: ユーザー入力から画面表示までの全ステップを追える

**理解度チェック**:
- Q: `st.session_state` はなぜ必要？（Streamlitの特徴と関連）
- Q: `st.write_stream()` は内部で何をしている？
- Q: `chat.py` は graph.py をそのまま呼んでいるだけに見えるが、なぜこの層がある？
- Q: 質問を入力してから回答が表示されるまでに、何回API呼び出しが発生する？

---

## Week 4: テストと品質評価

### Day 22-25: テストコードを読む

**読むもの**: `tests/test_graph.py`

**ゴール**: テストの読み方と `mock` の仕組みを理解する

**理解度チェック**:
- Q: `@patch("lib.graph.create_llm")` は何をしている？
- Q: テストでなぜ本物のLLMを使わない？
- Q: `mock_llm.invoke.return_value = AIMessage(content="回答")` の意味は？

---

### Day 26-28: 評価ツール

**読むもの**: `CODE_READING_GUIDE.md` の ⑨evaluator.py

**ゴール**: RAGの品質をどう測るかを理解する

**理解度チェック**:
- Q: Faithfulness（忠実度）とは何を測っている？
- Q: Bigram方式とLLM-as-Judge方式、それぞれの長所と短所は？
- Q: Context Hit Rate とは？

---

## Claudeアプリでの効果的な学習法

### パターン1: コードを貼って質問する

```
このコードの意味を教えてください：

trimmed = trim_messages(
    state["messages"],
    max_tokens=20,
    token_counter=len,
    strategy="last",
)
```

### パターン2: 理解度チェックの答え合わせ

```
以下の質問に答えました。合っているか確認してください：

Q: @lru_cacheは何のために使っている？
私の回答: 一度計算した結果を覚えておいて、
同じ引数で呼ばれたら計算せずに結果を返すため
```

### パターン3: 「もし〇〇だったら」で理解を深める

```
もし trim_messages を使わなかったら、
会話が100ラリー続いたときに何が起きますか？
```

### パターン4: 改善案を考えてみる

```
get_messages() は全メッセージを取得していますが、
改善するならどう書き換えますか？
自分の案: LIMITを追加して最新100件だけ取得する
```

---

## 卒業テスト

全ステップを終えたら、以下に挑戦してみてください:

1. **説明チャレンジ**: 誰かに「このアプリがどう動いているか」を `ARCHITECTURE.md` を見ずに3分で説明する

2. **コード追跡チャレンジ**: ユーザーが「RAGとは？」と入力してから回答が表示されるまでに通るファイルと関数を全て列挙する

3. **改善提案チャレンジ**: 1つ以上の改善点を見つけて、なぜそれが改善になるか説明する（ヒント: `PROMPT_FIX_CONTEXT_HANDLING.md` に答えがありますが、まず自分で考えてみてください）
