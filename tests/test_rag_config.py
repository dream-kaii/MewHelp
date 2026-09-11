from app.config import get_settings


def test_rag_settings_defaults():
    s = get_settings()
    assert s.milvus_uri.startswith("http")
    assert s.knowledge_collection == "knowledge"
    assert s.milvus_test_collection == "knowledge_test"
    assert s.embed_model  # 非空即可:本地 .env 可合法覆盖为缓存快照路径
    assert s.rag_top_k == 5 and 0 < s.rag_score_threshold < 1
    assert s.chunk_max_chars == 800 and s.chunk_overlap == 120
    assert 0 < s.dedupe_sim_threshold <= 1
