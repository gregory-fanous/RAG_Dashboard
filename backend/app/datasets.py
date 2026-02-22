from __future__ import annotations

import ast
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from .bootstrap import ensure_src_path
from .schemas import DatasetSummary

ROOT = ensure_src_path()

from rag_eval.models import Document, EvaluationCase, EvaluationDataset  # noqa: E402

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DatasetRecord:
    dataset_id: str
    name: str
    description: str
    loader_key: str
    format: str
    source_path: Path
    domain: str
    tags: list[str]


@dataclass
class LoadedDataset:
    record: DatasetRecord
    dataset: EvaluationDataset
    query_metadata: dict[str, dict[str, Any]]


def _count_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as file:
        return sum(1 for _ in file)


def _safe_json_load(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _to_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _as_list(value: Any) -> list[str]:
    if isinstance(value, list):
        return [_to_text(item) for item in value if _to_text(item)]
    if value is None:
        return []
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = ast.literal_eval(stripped)
                if isinstance(parsed, list):
                    return [_to_text(item) for item in parsed if _to_text(item)]
            except (ValueError, SyntaxError):
                pass
        return [stripped]
    return [_to_text(value)]


def _deterministic_subset_ids(ids: list[str], sample_size: int, seed: int) -> list[str]:
    if sample_size >= len(ids):
        return ids
    rng = random.Random(seed)
    pool = ids.copy()
    rng.shuffle(pool)
    return pool[:sample_size]


def _make_reference_doc_id(prefix: str, index: int, sub_index: int) -> str:
    return f"{prefix}_{index:04d}_{sub_index:02d}"


class DatasetRegistry:
    def __init__(self, data_root: Path | None = None) -> None:
        root = data_root or (ROOT / "data")
        self.data_root = root.resolve()
        self.records = self._build_records()

    def _build_records(self) -> dict[str, DatasetRecord]:
        records = {
            "public_eval_set": DatasetRecord(
                dataset_id="public_eval_set",
                name="Public RAG Eval Set",
                description="Baseline public benchmark corpus with explicit gold evidence mappings.",
                loader_key="public_eval_set",
                format="json",
                source_path=self.data_root / "public_eval_set.json",
                domain="general",
                tags=["baseline", "qa", "governance"],
            ),
            "retrievalqa": DatasetRecord(
                dataset_id="retrievalqa",
                name="RetrievalQA",
                description="Adaptive retrieval QA benchmark with candidate contexts per query.",
                loader_key="retrievalqa",
                format="jsonl",
                source_path=self.data_root / "RetrievalQA" / "retrievalqa.jsonl",
                domain="general",
                tags=["qa", "retrieval", "adaptive"],
            ),
            "ragcare_qa": DatasetRecord(
                dataset_id="ragcare_qa",
                name="RAGCare-QA",
                description="Medical QA benchmark for high-risk hallucination analysis.",
                loader_key="ragcare_qa",
                format="json",
                source_path=self.data_root / "RAGCare-QA" / "RAGCare-QA.json",
                domain="medical",
                tags=["medical", "safety", "hallucination"],
            ),
            "open_ragbench": DatasetRecord(
                dataset_id="open_ragbench",
                name="Open RAG Benchmark (arXiv PDF)",
                description="Multimodal RAG benchmark with qrels and ground-truth answers.",
                loader_key="open_ragbench",
                format="beir-style json",
                source_path=self.data_root / "open_ragbench" / "pdf" / "arxiv",
                domain="technical",
                tags=["rag", "arxiv", "multimodal"],
            ),
            "natural_questions": DatasetRecord(
                dataset_id="natural_questions",
                name="Natural Questions (pair)",
                description=(
                    "Pair-only open-domain baseline (query-answer pairs without an independent shared corpus). "
                    "Use for embedding stress tests, not release-gate governance."
                ),
                loader_key="natural_questions",
                format="parquet",
                source_path=self.data_root / "natural-questions" / "pair" / "train-00000-of-00001.parquet",
                domain="general",
                tags=["open-domain", "qa", "scale", "pair_only"],
            ),
            "finder": DatasetRecord(
                dataset_id="finder",
                name="FinDER",
                description="Financial QA dataset for evidence-grounded numerical and memo tasks.",
                loader_key="finder",
                format="parquet",
                source_path=self.data_root / "FinDER" / "data" / "train-00000-of-00001.parquet",
                domain="finance",
                tags=["finance", "numeric", "risk"],
            ),
            "paperzilla": DatasetRecord(
                dataset_id="paperzilla",
                name="Paperzilla RAG Retrieval 250",
                description="Research-paper corpus with multi-annotator relevance reasoning.",
                loader_key="paperzilla",
                format="json",
                source_path=self.data_root / "paperzilla-rag-retrieval-250" / "dataset.json",
                domain="research",
                tags=["retrieval", "papers", "relevance"],
            ),
        }
        records.update(self._discover_custom_package_records())
        return records

    def _discover_custom_package_records(self) -> dict[str, DatasetRecord]:
        discovered: dict[str, DatasetRecord] = {}
        if not self.data_root.exists():
            return discovered

        for directory in sorted(self.data_root.iterdir()):
            if not directory.is_dir():
                continue

            corpus_file = directory / "corpus.jsonl"
            eval_file = directory / "eval_set.jsonl"
            if not corpus_file.exists() or not eval_file.exists():
                continue

            safe_name = "".join(ch if ch.isalnum() else "_" for ch in directory.name.lower()).strip("_")
            dataset_id = f"package_{safe_name}"
            discovered[dataset_id] = DatasetRecord(
                dataset_id=dataset_id,
                name=f"Custom Package: {directory.name}",
                description="User-provided JSONL corpus/eval package.",
                loader_key="custom_package",
                format="jsonl package",
                source_path=directory,
                domain="custom",
                tags=["custom", "jsonl", "enterprise"],
            )

        return discovered

    def list_datasets(self) -> list[DatasetSummary]:
        summaries: list[DatasetSummary] = []
        for record in self.records.values():
            if not record.source_path.exists():
                continue
            try:
                summaries.append(self._summary(record))
            except Exception as exc:
                logger.warning(
                    "Skipping dataset '%s' because metadata could not be loaded: %s",
                    record.dataset_id,
                    exc,
                )
        return summaries

    def _summary(self, record: DatasetRecord) -> DatasetSummary:
        approx_docs = 0
        approx_queries = 0

        if record.dataset_id == "public_eval_set":
            payload = _safe_json_load(record.source_path)
            approx_docs = len(payload.get("documents", []))
            approx_queries = len(payload.get("cases", []))
        elif record.dataset_id == "retrievalqa":
            approx_queries = _count_lines(record.source_path)
            approx_docs = approx_queries * 5
        elif record.dataset_id == "ragcare_qa":
            payload = _safe_json_load(record.source_path)
            approx_queries = len(payload)
            approx_docs = len(payload)
        elif record.dataset_id == "open_ragbench":
            queries = _safe_json_load(record.source_path / "queries.json")
            approx_queries = len(queries)
            approx_docs = len(list((record.source_path / "corpus").glob("*.json")))
        elif record.dataset_id in {"natural_questions", "finder"}:
            try:
                import pyarrow.parquet as pq

                approx_queries = pq.ParquetFile(record.source_path).metadata.num_rows
            except Exception:
                approx_queries = 0
            approx_docs = approx_queries
        elif record.dataset_id == "paperzilla":
            payload = _safe_json_load(record.source_path)
            papers = payload.get("papers", [])
            approx_docs = len(papers)
            approx_queries = len(papers)
        elif record.loader_key == "custom_package":
            approx_docs = _count_lines(record.source_path / "corpus.jsonl")
            approx_queries = _count_lines(record.source_path / "eval_set.jsonl")

        return DatasetSummary(
            id=record.dataset_id,
            name=record.name,
            description=record.description,
            source_path=str(record.source_path),
            format=record.format,
            approx_documents=approx_docs,
            approx_queries=approx_queries,
            tags=record.tags,
            domain=record.domain,
        )

    def load_dataset(self, dataset_id: str, sample_size: int, seed: int) -> LoadedDataset:
        record = self.records.get(dataset_id)
        if record is None:
            valid = ", ".join(sorted(self.records.keys()))
            raise ValueError(f"Unknown dataset_id '{dataset_id}'. Valid values: {valid}")
        if not record.source_path.exists():
            raise FileNotFoundError(f"Dataset source not found: {record.source_path}")

        loaders: dict[str, Callable[[DatasetRecord, int, int], LoadedDataset]] = {
            "public_eval_set": self._load_public_eval,
            "retrievalqa": self._load_retrievalqa,
            "ragcare_qa": self._load_ragcare,
            "open_ragbench": self._load_open_ragbench,
            "natural_questions": self._load_natural_questions,
            "finder": self._load_finder,
            "paperzilla": self._load_paperzilla,
            "custom_package": self._load_custom_package,
        }
        loader = loaders.get(record.loader_key)
        if loader is None:
            raise ValueError(f"Unsupported loader '{record.loader_key}' for dataset '{dataset_id}'")
        return loader(record, sample_size, seed)

    def _load_public_eval(self, record: DatasetRecord, sample_size: int, seed: int) -> LoadedDataset:
        payload = _safe_json_load(record.source_path)
        dataset = EvaluationDataset.from_dict(payload)

        case_ids = [case.query_id for case in dataset.cases]
        selected = set(_deterministic_subset_ids(case_ids, sample_size, seed))
        cases = [case for case in dataset.cases if case.query_id in selected]
        referenced = {doc_id for case in cases for doc_id in case.ground_truth_doc_ids}
        documents = [doc for doc in dataset.documents if doc.doc_id in referenced]

        query_metadata = {
            case.query_id: {
                "intent": "general",
                "expected_style": "short_answer",
                "must_include": [],
                "must_not_include": [],
            }
            for case in cases
        }

        return LoadedDataset(
            record=record,
            dataset=EvaluationDataset(name=dataset.name, documents=documents, cases=cases),
            query_metadata=query_metadata,
        )

    def _load_retrievalqa(self, record: DatasetRecord, sample_size: int, seed: int) -> LoadedDataset:
        rows: list[dict[str, Any]] = []
        with record.source_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    rows.append(json.loads(line))

        row_ids = [row.get("question_id", f"q_{idx}") for idx, row in enumerate(rows)]
        selected_ids = set(_deterministic_subset_ids(row_ids, sample_size, seed))
        selected_rows = [row for row in rows if row.get("question_id", "") in selected_ids]

        documents: list[Document] = []
        cases: list[EvaluationCase] = []
        metadata: dict[str, dict[str, Any]] = {}

        for idx, row in enumerate(selected_rows):
            query_id = _to_text(row.get("question_id") or f"retrievalqa_{idx:04d}")
            contexts = row.get("context", [])
            if not isinstance(contexts, list):
                contexts = [contexts]
            ground_truth = [_to_text(item) for item in row.get("ground_truth", []) if _to_text(item)]
            doc_ids: list[str] = []
            relevant_ids: list[str] = []

            for c_idx, context in enumerate(contexts):
                if isinstance(context, dict):
                    text = _to_text(context.get("text"))
                    title = _to_text(context.get("title")) or f"Context {c_idx + 1}"
                else:
                    text = _to_text(context)
                    title = f"Context {c_idx + 1}"

                doc_id = _make_reference_doc_id("retrievalqa_doc", idx, c_idx)
                documents.append(Document(doc_id=doc_id, title=title, text=text, tags=["retrievalqa"]))
                doc_ids.append(doc_id)

                lowered = text.lower()
                if any(token.lower() in lowered for token in ground_truth if token):
                    relevant_ids.append(doc_id)

            if not relevant_ids and doc_ids:
                relevant_ids = [doc_ids[0]]
            reference_answer = "; ".join(ground_truth) if ground_truth else "No answer provided."

            cases.append(
                EvaluationCase(
                    query_id=query_id,
                    query=_to_text(row.get("question")),
                    reference_answer=reference_answer,
                    ground_truth_doc_ids=relevant_ids,
                )
            )
            metadata[query_id] = {
                "intent": _to_text(row.get("data_source")) or "general",
                "expected_style": "short_answer",
                "must_include": ground_truth,
                "must_not_include": [],
            }

        return LoadedDataset(
            record=record,
            dataset=EvaluationDataset(name=record.name, documents=documents, cases=cases),
            query_metadata=metadata,
        )

    def _load_ragcare(self, record: DatasetRecord, sample_size: int, seed: int) -> LoadedDataset:
        payload = _safe_json_load(record.source_path)
        rows = payload if isinstance(payload, list) else []
        indices = list(range(len(rows)))
        selected_indices = set(_deterministic_subset_ids([str(i) for i in indices], sample_size, seed))
        filtered_rows = [row for i, row in enumerate(rows) if str(i) in selected_indices]

        documents: list[Document] = []
        cases: list[EvaluationCase] = []
        metadata: dict[str, dict[str, Any]] = {}

        for idx, row in enumerate(filtered_rows):
            query_id = f"ragcare_{idx:04d}"
            doc_id = f"ragcare_doc_{idx:04d}"
            question = _to_text(row.get("Question"))
            answer = _to_text(row.get("Text Answer") or row.get("Answer"))
            context = _to_text(row.get("Context"))
            intent = _to_text(row.get("Type")) or "medical"

            documents.append(
                Document(
                    doc_id=doc_id,
                    title=f"{intent} reference p.{_to_text(row.get('Page'))}",
                    text=context,
                    tags=["ragcare", intent.lower()],
                )
            )
            cases.append(
                EvaluationCase(
                    query_id=query_id,
                    query=question,
                    reference_answer=answer,
                    ground_truth_doc_ids=[doc_id],
                )
            )
            metadata[query_id] = {
                "intent": intent.lower(),
                "expected_style": "short_answer",
                "must_include": [answer] if answer else [],
                "must_not_include": [],
            }

        return LoadedDataset(
            record=record,
            dataset=EvaluationDataset(name=record.name, documents=documents, cases=cases),
            query_metadata=metadata,
        )

    def _load_open_ragbench(self, record: DatasetRecord, sample_size: int, seed: int) -> LoadedDataset:
        root = record.source_path
        queries = _safe_json_load(root / "queries.json")
        qrels = _safe_json_load(root / "qrels.json")
        answers = _safe_json_load(root / "answers.json")
        query_ids = list(queries.keys())
        selected_query_ids = _deterministic_subset_ids(query_ids, sample_size, seed)

        relevant_doc_ids = {qrels[qid]["doc_id"] for qid in selected_query_ids if qid in qrels}
        corpus_paths = sorted((root / "corpus").glob("*.json"))
        all_doc_ids = [path.stem for path in corpus_paths]

        rng = random.Random(seed)
        negative_pool = [doc_id for doc_id in all_doc_ids if doc_id not in relevant_doc_ids]
        rng.shuffle(negative_pool)
        negative_count = min(max(len(relevant_doc_ids), 80), len(negative_pool))
        selected_doc_ids = relevant_doc_ids | set(negative_pool[:negative_count])

        documents: list[Document] = []
        for doc_id in selected_doc_ids:
            path = root / "corpus" / f"{doc_id}.json"
            if not path.exists():
                continue
            payload = _safe_json_load(path)
            abstract = _to_text(payload.get("abstract"))
            sections = payload.get("sections", [])
            section_text = "\n".join(_to_text(section.get("text")) for section in sections[:3])
            text = (abstract + "\n" + section_text).strip()
            if len(text) > 3500:
                text = text[:3500]
            documents.append(
                Document(
                    doc_id=doc_id,
                    title=_to_text(payload.get("title") or doc_id),
                    text=text,
                    tags=["open_ragbench"],
                )
            )

        cases: list[EvaluationCase] = []
        metadata: dict[str, dict[str, Any]] = {}
        for qid in selected_query_ids:
            query_data = queries.get(qid, {})
            qrel = qrels.get(qid, {})
            answer = _to_text(answers.get(qid))
            doc_id = _to_text(qrel.get("doc_id"))
            if not doc_id:
                continue
            cases.append(
                EvaluationCase(
                    query_id=qid,
                    query=_to_text(query_data.get("query")),
                    reference_answer=answer,
                    ground_truth_doc_ids=[doc_id],
                )
            )
            metadata[qid] = {
                "intent": _to_text(query_data.get("type") or "general").lower(),
                "source": _to_text(query_data.get("source") or "text"),
                "expected_style": _to_text(query_data.get("type") or "short_answer").lower(),
                "must_include": [],
                "must_not_include": [],
                "gold_spans": [_safe_int(qrel.get("section_id"), -1)],
            }

        return LoadedDataset(
            record=record,
            dataset=EvaluationDataset(name=record.name, documents=documents, cases=cases),
            query_metadata=metadata,
        )

    def _load_natural_questions(self, record: DatasetRecord, sample_size: int, seed: int) -> LoadedDataset:
        import pandas as pd

        frame = pd.read_parquet(record.source_path, columns=["query", "answer"])
        indices = _deterministic_subset_ids([str(i) for i in range(len(frame))], sample_size, seed)

        documents: list[Document] = []
        cases: list[EvaluationCase] = []
        metadata: dict[str, dict[str, Any]] = {}

        for idx_str in indices:
            idx = int(idx_str)
            row = frame.iloc[idx]
            query_id = f"nq_{idx:06d}"
            doc_id = f"nq_doc_{idx:06d}"
            answer = _to_text(row.get("answer"))
            query = _to_text(row.get("query"))

            documents.append(
                Document(
                    doc_id=doc_id,
                    title=f"Natural Questions reference {idx}",
                    text=answer,
                    tags=["natural_questions"],
                )
            )
            cases.append(
                EvaluationCase(
                    query_id=query_id,
                    query=query,
                    reference_answer=answer,
                    ground_truth_doc_ids=[doc_id],
                )
            )
            metadata[query_id] = {
                "intent": "open_domain_qa",
                "expected_style": "short_answer",
                "must_include": [answer[:80]] if answer else [],
                "must_not_include": [],
                "dataset_notice": "pair_only_baseline",
            }

        return LoadedDataset(
            record=record,
            dataset=EvaluationDataset(name=record.name, documents=documents, cases=cases),
            query_metadata=metadata,
        )

    def _load_finder(self, record: DatasetRecord, sample_size: int, seed: int) -> LoadedDataset:
        import pandas as pd

        frame = pd.read_parquet(
            record.source_path,
            columns=["_id", "text", "reasoning", "category", "references", "answer", "type"],
        )
        indices = _deterministic_subset_ids([str(i) for i in range(len(frame))], sample_size, seed)

        documents: list[Document] = []
        cases: list[EvaluationCase] = []
        metadata: dict[str, dict[str, Any]] = {}

        for idx_str in indices:
            idx = int(idx_str)
            row = frame.iloc[idx]
            query_id = f"finder_{_to_text(row.get('_id') or idx)}"
            query = _to_text(row.get("text"))
            answer = _to_text(row.get("answer"))
            category = _to_text(row.get("category") or "finance").lower()
            references = [item for item in _as_list(row.get("references")) if item][:3]

            doc_ids: list[str] = []
            if not references:
                continue

            for ref_idx, reference in enumerate(references):
                doc_id = _make_reference_doc_id("finder_doc", idx, ref_idx)
                doc_ids.append(doc_id)
                documents.append(
                    Document(
                        doc_id=doc_id,
                        title=f"FinDER evidence {idx}-{ref_idx}",
                        text=reference,
                        tags=["finder", category],
                    )
                )

            cases.append(
                EvaluationCase(
                    query_id=query_id,
                    query=query,
                    reference_answer=answer,
                    ground_truth_doc_ids=doc_ids,
                )
            )
            metadata[query_id] = {
                "intent": category,
                "expected_style": "memo" if bool(row.get("reasoning")) else "short_answer",
                "must_include": [answer[:120]] if answer else [],
                "must_not_include": [],
                "question_type": _to_text(row.get("type") or "unknown"),
            }

        return LoadedDataset(
            record=record,
            dataset=EvaluationDataset(name=record.name, documents=documents, cases=cases),
            query_metadata=metadata,
        )

    def _load_paperzilla(self, record: DatasetRecord, sample_size: int, seed: int) -> LoadedDataset:
        payload = _safe_json_load(record.source_path)
        papers = payload.get("papers", [])
        ids = [paper.get("paper_id", f"paper_{idx}") for idx, paper in enumerate(papers)]
        selected_ids = set(_deterministic_subset_ids(ids, sample_size, seed))
        selected_papers = [paper for paper in papers if paper.get("paper_id") in selected_ids]

        documents: list[Document] = []
        cases: list[EvaluationCase] = []
        metadata: dict[str, dict[str, Any]] = {}

        for idx, paper in enumerate(selected_papers):
            paper_id = _to_text(paper.get("paper_id") or f"paper_{idx}")
            title = _to_text(paper.get("title"))
            abstract = _to_text(paper.get("abstract"))
            annotations = paper.get("annotations", [])
            rationale = " ".join(_to_text(item.get("reasoning")) for item in annotations[:2]).strip()
            doc_text = (abstract + "\n" + rationale).strip()

            documents.append(
                Document(
                    doc_id=paper_id,
                    title=title,
                    text=doc_text,
                    tags=["paperzilla", "research"],
                )
            )
            query_id = f"paperzilla_{idx:04d}"
            query = f"Summarize the core contribution and retrieval relevance of '{title}'."
            reference_answer = abstract or rationale or "No abstract available."
            cases.append(
                EvaluationCase(
                    query_id=query_id,
                    query=query,
                    reference_answer=reference_answer,
                    ground_truth_doc_ids=[paper_id],
                )
            )
            metadata[query_id] = {
                "intent": "research_relevance",
                "expected_style": "memo",
                "must_include": [],
                "must_not_include": [],
                "published_date": _to_text(paper.get("published_date")),
            }

        return LoadedDataset(
            record=record,
            dataset=EvaluationDataset(name=record.name, documents=documents, cases=cases),
            query_metadata=metadata,
        )

    def _load_custom_package(self, record: DatasetRecord, sample_size: int, seed: int) -> LoadedDataset:
        corpus_path = record.source_path / "corpus.jsonl"
        eval_path = record.source_path / "eval_set.jsonl"

        doc_index: dict[str, Document] = {}
        with corpus_path.open("r", encoding="utf-8") as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                doc_id = _to_text(row.get("doc_id"))
                if not doc_id:
                    continue
                tags = [
                    _to_text(row.get("source_type")),
                    _to_text(row.get("effective_date")),
                ]
                tags = [tag for tag in tags if tag]
                doc_index[doc_id] = Document(
                    doc_id=doc_id,
                    title=_to_text(row.get("title") or doc_id),
                    text=_to_text(row.get("text")),
                    tags=tags,
                )

        eval_rows: list[dict[str, Any]] = []
        with eval_path.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    eval_rows.append(json.loads(line))

        candidate_ids = [str(row.get("query_id", f"custom_{idx:05d}")) for idx, row in enumerate(eval_rows)]
        selected_ids = set(_deterministic_subset_ids(candidate_ids, sample_size, seed))
        selected_rows = [
            row for idx, row in enumerate(eval_rows) if str(row.get("query_id", f"custom_{idx:05d}")) in selected_ids
        ]

        cases: list[EvaluationCase] = []
        metadata: dict[str, dict[str, Any]] = {}
        referenced_doc_ids: set[str] = set()

        for idx, row in enumerate(selected_rows):
            query_id = _to_text(row.get("query_id") or f"custom_{idx:05d}")
            query = _to_text(row.get("query"))
            gold_doc_ids = _as_list(row.get("gold_doc_ids"))
            gold_doc_ids = [doc_id for doc_id in gold_doc_ids if doc_id in doc_index]
            if not gold_doc_ids:
                continue

            referenced_doc_ids.update(gold_doc_ids)
            must_include = _as_list(row.get("must_include"))
            reference_answer = "; ".join(must_include).strip()
            if not reference_answer:
                reference_answer = _to_text(row.get("reference_answer"))
            if not reference_answer:
                reference_answer = doc_index[gold_doc_ids[0]].text[:220]

            cases.append(
                EvaluationCase(
                    query_id=query_id,
                    query=query,
                    reference_answer=reference_answer,
                    ground_truth_doc_ids=gold_doc_ids,
                )
            )
            metadata[query_id] = {
                "intent": _to_text(row.get("intent") or "custom"),
                "history": _as_list(row.get("history")),
                "expected_style": _to_text(row.get("expected_style") or "short_answer"),
                "must_include": must_include,
                "must_not_include": _as_list(row.get("must_not_include")),
                "gold_spans": _as_list(row.get("gold_spans")),
            }

        documents = [doc_index[doc_id] for doc_id in sorted(referenced_doc_ids)]
        return LoadedDataset(
            record=record,
            dataset=EvaluationDataset(name=record.name, documents=documents, cases=cases),
            query_metadata=metadata,
        )
