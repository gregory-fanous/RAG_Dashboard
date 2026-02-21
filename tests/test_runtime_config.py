from rag_eval.runtime import RuntimeSettings


def test_validate_real_mode_rejects_placeholder_key():
    settings = RuntimeSettings(
        execution_mode="real",
        allow_synthetic_mode=False,
        openai_api_key="YOUR_OPENAI_API_KEY_HERE",
        openai_base_url="https://api.openai.com/v1",
        openai_chat_model="gpt-4.1-mini",
        openai_embedding_model="text-embedding-3-large",
        openai_verifier_model="gpt-4.1-mini",
        request_timeout_sec=120,
        max_generation_tokens=700,
        max_context_chars=1400,
        embedding_batch_size=48,
        top_candidate_multiplier=5,
        max_rerank_candidates=20,
        max_verifier_claims=8,
    )

    try:
        settings.validate_for_real_mode()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_validate_synthetic_mode_requires_explicit_opt_in():
    settings = RuntimeSettings(
        execution_mode="synthetic",
        allow_synthetic_mode=False,
        openai_api_key="",
        openai_base_url="https://api.openai.com/v1",
        openai_chat_model="gpt-4.1-mini",
        openai_embedding_model="text-embedding-3-large",
        openai_verifier_model="gpt-4.1-mini",
        request_timeout_sec=120,
        max_generation_tokens=700,
        max_context_chars=1400,
        embedding_batch_size=48,
        top_candidate_multiplier=5,
        max_rerank_candidates=20,
        max_verifier_claims=8,
    )

    try:
        settings.validate_for_synthetic_mode()
        assert False, "expected ValueError"
    except ValueError:
        pass
