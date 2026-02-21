# Third-Party Datasets

This project uses third-party datasets for evaluation.

## Redistribution policy

Do not redistribute these datasets from this repository unless you have explicit permission from each dataset owner/licensor.

Clone/download directly from each official Hugging Face dataset page and follow each dataset's license and usage terms.

## Supported datasets

| Dataset | Size | Entries | Domain | Source |
| --- | --- | --- | --- | --- |
| Open RAGBench | 716 MB | 1,000 docs, 3,045 queries | Scientific | [vectara/open_ragbench](https://huggingface.co/datasets/vectara/open_ragbench) |
| RetrievalQA | 16 MB | 1,271 questions | General | [zihanz/RetrievalQA](https://huggingface.co/datasets/zihanz/RetrievalQA) |
| Natural Questions | 43 MB | 100,231 pairs | General | [sentence-transformers/natural-questions](https://huggingface.co/datasets/sentence-transformers/natural-questions) |
| Paperzilla RAG | 1.3 MB | 250 papers | CS Research | [paperzilla/paperzilla-rag-retrieval-250](https://huggingface.co/datasets/paperzilla/paperzilla-rag-retrieval-250) |
| FinDER | 149 MB | 5,703 triplets | Financial | [Linq-AI-Research/FinDER](https://huggingface.co/datasets/Linq-AI-Research/FinDER) |
| RAGCare-QA | 1.7 MB | 420 questions | Medical | [ChatMED-Project/RAGCare-QA](https://huggingface.co/datasets/ChatMED-Project/RAGCare-QA) |

## Download command

```bash
python scripts/download_hf_datasets.py
```

Optional flags:

```bash
python scripts/download_hf_datasets.py --list
python scripts/download_hf_datasets.py --datasets open_ragbench,finder
python scripts/download_hf_datasets.py --data-root /absolute/path/to/data
```
