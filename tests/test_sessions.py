from app.sessions import SessionStore


def test_get_or_create_new_without_id():
    s, created = SessionStore().get_or_create(None)
    assert s.session_id and created


def test_get_or_create_returns_existing():
    store = SessionStore()
    sid = "abc"
    s1, created = store.get_or_create(sid)
    assert created is True
    s2, created2 = store.get_or_create(sid)
    assert created2 is False and s2.session_id == sid


def test_append_turn_orders_pairs():
    store = SessionStore()
    store.get_or_create("s1")
    store.append_turn("s1", "问1", "答1")
    store.append_turn("s1", "问2", "答2")
    assert [m["role"] for m in store.get("s1").turns] == ["user", "assistant", "user", "assistant"]
    assert store.get("s1").turns[0]["content"] == "问1"


def test_append_turn_caps_to_max_turns_keeping_pairs():
    store = SessionStore(max_turns=2)
    store.get_or_create("cap")
    for i in range(5):
        store.append_turn("cap", f"问{i}", f"答{i}")
    assert len(store.get("cap").turns) == 4  # 仅保留最近 2 轮(user+assistant)
    assert store.get("cap").turns[0] == {"role": "user", "content": "问3"}


def test_store_evicts_oldest_when_over_max():
    store = SessionStore(max_sessions=2)
    store.get_or_create("a")
    store.get_or_create("b")
    store.get_or_create("c")  # 触发淘汰
    assert store.get("a") is None
    assert store.get("b") is not None and store.get("c") is not None


def test_lock_is_stable_per_session_and_eviction_drops_lock():
    store = SessionStore(max_sessions=2)
    store.get_or_create("a")
    store.get_or_create("b")
    assert store.lock("a") is store.lock("a")  # 同会话返回同一把锁
    store.get_or_create("c")  # 淘汰 "a"
    assert store.get("a") is None
    assert store.lock("c") is store.lock("c")


def test_append_turn_tolerates_evicted_session():
    store = SessionStore(max_sessions=1)
    store.get_or_create("gone")
    store.get_or_create("other")  # 淘汰 "gone"
    store.append_turn("gone", "问", "答")  # 不应抛 KeyError
    assert store.get("other").turns == []
