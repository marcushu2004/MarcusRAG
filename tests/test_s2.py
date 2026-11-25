"""Tests for the S2 additions: BM25, RRF, hybrid retrieval, rerank, evidence, eval.

Same constraints as `test_smoke.py`: standard library only. No torch, no chroma,
no Ollama — the hash embedder and memory store carry the whole flow.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from marcusrag.bm25 import BM25Index, reciprocal_rank_fusion
from marcusrag.citations import (
    Citation,
    annotate_answer,
    build_evidence,
    format_evidence,
    highlight,
    split_sentences,
    verify_citations,
)
from marcusrag.config import Settings
from marcusrag.documents import load_documents, scan_documents
from marcusrag.embeddings import HashEmbedder
from marcusrag.evaluate import evaluate, load_evalset
from marcusrag.pipeline import build_index
from marcusrag.rerank import LexicalReranker
from marcusrag.retrieval import Retrieved, Retriever, build_retriever
from marcusrag.vectorstore import MemoryStore

CORPUS = {
    "k1.txt": "BM25 的 k1 参数控制词频饱和速度，取值通常在 1.2 到 2.0 之间。",
    "beta.txt": "向量检索使用余弦相似度衡量语义相近程度，需要先做 L2 归一化。",
    "gamma.txt": "切分片段的重叠长度 chunk_overlap 通常取 chunk_size 的百分之十到二十。",
}


class BM25TestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.index = BM25Index([(name, text) for name, text in CORPUS.items()])

    def test_exact_term_ranks_its_document_first(self) -> None:
        hits = self.index.search("k1 参数", k=3)
        self.assertTrue(hits)
        self.assertEqual(hits[0][0], "k1.txt")

    def test_unknown_term_scores_zero(self) -> None:
        hits = self.index.search("量子纠缠退相干", k=3)
        self.assertEqual(hits, [])

    def test_empty_query_is_safe(self) -> None:
        self.assertEqual(self.index.search("   "), [])


class RRTestCase(unittest.TestCase):
    def test_document_found_by_both_channels_wins(self) -> None:
        fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "a", "d"]], k=60)
        self.assertEqual({doc for doc, _ in fused[:2]}, {"a", "b"})
        self.assertGreater(dict(fused)["a"], dict(fused)["c"])

    def test_single_channel_result_is_preserved(self) -> None:
        fused = [doc for doc, _ in reciprocal_rank_fusion([["only"], []], k=60)]
        self.assertEqual(fused, ["only"])

    def test_weights_bias_the_ranking(self) -> None:
        lexical = [doc for doc, _ in reciprocal_rank_fusion([["x"], ["y"]], k=60, weights=[3.0, 0.5])]
        self.assertEqual(lexical[0], "x")


class HybridRetrieverTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        for name, text in CORPUS.items():
            (self.data / name).write_text(text, encoding="utf-8")

        self.settings = Settings(
            data_dir=str(self.data),
            index_root=str(self.root / "index"),
            store="memory",
            embedder="hash",
            chunk_size=400,
            chunk_overlap=0,
            top_k=3,
            fetch_k=6,
        )
        self.embedder = HashEmbedder()
        self.store = MemoryStore(self.root / "index")
        build_index(self.settings, embedder=self.embedder, store=self.store, rebuild=True)

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_hybrid_finds_a_rare_exact_term(self) -> None:
        retriever = build_retriever(self.store, self.embedder, self.settings)
        self.assertEqual(retriever.search_type, "hybrid")
        hits = retriever.invoke("k1 参数取值")
        self.assertTrue(hits)
        self.assertEqual(Path(hits[0].source).name, "k1.txt")
        self.assertIn("bm25", hits[0].detail)

    def test_rerank_reorders_without_dropping(self) -> None:
        plain = build_retriever(self.store, self.embedder, self.settings)
        reranked = Retriever(
            self.store,
            self.embedder,
            top_k=3,
            fetch_k=6,
            search_type="hybrid",
            reranker=LexicalReranker(),
        )
        plain_hits = plain.invoke("chunk_overlap 取多少")
        rerank_hits = reranked.invoke("chunk_overlap 取多少")
        self.assertEqual(len(plain_hits), len(rerank_hits))
        self.assertTrue(all("rerank" in hit.detail for hit in rerank_hits))
        self.assertEqual(rerank_hits[0].detail["rerank"], max(h.detail["rerank"] for h in rerank_hits))

    def test_similarity_mode_still_works(self) -> None:
        retriever = Retriever(self.store, self.embedder, top_k=2, fetch_k=4, search_type="similarity")
        hits = retriever.invoke("向量检索")
        self.assertTrue(hits)
        self.assertLessEqual(len(hits), 2)


class LexicalRerankerTestCase(unittest.TestCase):
    def test_scores_the_sentence_that_contains_the_query_terms_higher(self) -> None:
        reranker = LexicalReranker()
        scores = reranker.score("rrf 融合", ["倒数排名融合把多路结果合并", "完全不相关的一句话"])
        self.assertGreater(scores[0], scores[1])

    def test_score_count_matches_input(self) -> None:
        reranker = LexicalReranker()
        self.assertEqual(len(reranker.score("a b", ["x", "y", "z"])), 3)

    def test_base_weight_one_keeps_the_first_stage_order(self) -> None:
        # A lexical re-rank that ignored the vector channel could demote a
        # correct hit; base_weight=1 must reproduce the retrieval order.
        reranker = LexicalReranker(base_weight=1.0)
        candidates = ["完全无关的一段话", "rrf 融合才是这里的关键"]
        base = [0.9, 0.1]
        scores = reranker.score("rrf 融合", candidates, base=base)
        self.assertGreater(scores[0], scores[1])

    def test_base_weight_zero_matches_pure_lexical(self) -> None:
        pure = LexicalReranker(base_weight=0.0)
        blended = LexicalReranker(base_weight=0.0)
        candidates = ["rrf 融合才是这里的关键", "完全无关的一段话"]
        self.assertEqual(
            pure.score("rrf 融合", candidates),
            blended.score("rrf 融合", candidates, base=[0.1, 0.9]),
        )

    def test_missing_base_scores_fall_back_to_lexical(self) -> None:
        reranker = LexicalReranker(base_weight=0.5)
        candidates = ["rrf 融合才是这里的关键", "完全无关的一段话"]
        self.assertEqual(
            LexicalReranker(base_weight=0.0).score("rrf 融合", candidates),
            reranker.score("rrf 融合", candidates, base=None),
        )

    def test_constant_base_scores_do_not_divide_by_zero(self) -> None:
        reranker = LexicalReranker(base_weight=1.0)
        scores = reranker.score("rrf 融合", ["rrf 融合", "无关"], base=[0.4, 0.4])
        self.assertEqual(len(scores), 2)


class EvidenceTestCase(unittest.TestCase):
    def _chunks(self) -> list[Retrieved]:
        return [
            Retrieved("c1", "第一段证据。BM25 的 k1 控制词频饱和。", {"source": "01-bm25.md", "page": 1}, 0.9),
            Retrieved("c2", "第二段证据，讲的是切分。", {"source": "03-chunking.md", "page": 1}, 0.5),
        ]

    def test_build_evidence_marks_the_cited_chunk(self) -> None:
        evidence = build_evidence(self._chunks(), [Citation("01-bm25.md", 1, verified=True)], query="k1")
        self.assertEqual(len(evidence), 2)
        self.assertTrue(evidence[0].cited)
        self.assertFalse(evidence[1].cited)
        self.assertIn("k1", evidence[0].snippet)

    def test_annotate_answer_links_citation_to_evidence_number(self) -> None:
        evidence = build_evidence(self._chunks(), [Citation("03-chunking.md", 1, verified=True)])
        annotated = annotate_answer("切分很重要 [03-chunking.md:1]。", evidence)
        self.assertIn("证据#2", annotated)
        self.assertTrue(annotated.startswith("切分很重要"))

    def test_annotate_leaves_unmatched_citation_alone(self) -> None:
        evidence = build_evidence(self._chunks(), [])
        self.assertEqual(annotate_answer("[ghost.md:9]", evidence), "[ghost.md:9]")

    def test_verify_citations_splits_verified_and_unverified(self) -> None:
        verified, unverified = verify_citations(
            [Citation("01-bm25.md", 1, False), Citation("ghost.md", 9, False)], {("01-bm25.md", 1)}
        )
        self.assertEqual(len(verified), 1)
        self.assertEqual(len(unverified), 1)

    def test_format_evidence_reports_usage(self) -> None:
        evidence = build_evidence(self._chunks(), [Citation("01-bm25.md", 1, verified=True)])
        rendered = format_evidence(evidence, query="k1")
        self.assertIn("CITED", rendered)
        self.assertIn("unused", rendered)

    def test_highlight_marks_query_terms(self) -> None:
        self.assertIn("\x1b[7m", highlight("BM25 打分", "BM25"))
        self.assertEqual(highlight("BM25 打分", "BM25", color=False), "BM25 打分")

    def test_split_sentences_handles_chinese_punctuation(self) -> None:
        self.assertEqual(len(split_sentences("第一句。第二句！第三句？")), 3)


class EvaluateTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.data = self.root / "data"
        self.data.mkdir()
        for name, text in CORPUS.items():
            (self.data / name).write_text(text, encoding="utf-8")
        self.evalset = self.root / "qa.jsonl"
        rows = [
            {"id": "g1", "question": "k1 参数是多少", "gold": [{"file": "k1.txt", "page": 1}]},
            {"id": "g2", "question": "余弦相似度为什么要归一化", "gold": [{"file": "beta.txt", "page": 1}]},
            {"id": "r1", "question": "GPU 显存容量是多少", "gold": [], "expect": "refusal"},
        ]
        self.evalset.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in rows), encoding="utf-8"
        )
        self.settings = Settings(
            data_dir=str(self.data),
            index_root=str(self.root / "index"),
            store="memory",
            embedder="hash",
            chunk_size=400,
            chunk_overlap=0,
            top_k=3,
            fetch_k=6,
        )

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def test_load_evalset_reads_gold_and_expect(self) -> None:
        cases = load_evalset(self.evalset)
        self.assertEqual(len(cases), 3)
        self.assertEqual(cases[0].gold[0].key(), ("k1.txt", 1))
        self.assertEqual(cases[2].expect, "refusal")

    def test_load_evalset_rejects_unknown_expect(self) -> None:
        bad = self.root / "bad.jsonl"
        bad.write_text('{"id": "x", "question": "q", "expect": "maybe"}', encoding="utf-8")
        with self.assertRaises(ValueError):
            load_evalset(bad)

    def test_report_computes_retrieval_metrics_without_an_llm(self) -> None:
        report = evaluate(self.settings, self.evalset)
        self.assertEqual(len(report.cases), 3)
        self.assertEqual(len(report.answerable), 2)
        self.assertGreaterEqual(report.recall_at_k, 0.0)
        self.assertLessEqual(report.hit_rate, 1.0)
        self.assertIsNone(report.citation_precision)
        self.assertIn("retrieval-only", report.skipped or "")
        payload = report.to_dict()
        self.assertEqual(payload["cases"], 3)
        self.assertIn("details", payload)
        self.assertIn("Recall@", report.format())


class ScanDocumentsTests(unittest.TestCase):
    """A file that yields no text must be reported, not silently dropped."""

    def test_reports_sources_without_text(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "good.md").write_text("# t\n\n有内容的段落。", encoding="utf-8")
            (root / "empty.txt").write_text("   \n\t\n", encoding="utf-8")

            scanned = scan_documents(root)

            self.assertEqual(len(scanned.docs), 1)
            self.assertTrue(scanned.docs[0].source.endswith("good.md"))
            self.assertEqual([name for name, _ in scanned.empty_sources], ["empty.txt"])
            self.assertIn("空", scanned.empty_sources[0][1])

    def test_load_documents_stays_backward_compatible(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "a.md").write_text("# a\n\n第一份文档。", encoding="utf-8")
            (root / "b.md").write_text("# b\n\n第二份文档。", encoding="utf-8")
            self.assertEqual(len(load_documents(root)), 2)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
