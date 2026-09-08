import asyncio
import logging
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field

from app.config import get_settings

logger = logging.getLogger("mewhelp.sessions")


@dataclass
class Session:
    session_id: str
    created_at: float = field(default_factory=time.time)
    turns: list[dict] = field(default_factory=list)  # [{"role":"user"|"assistant","content":str}, ...]


class SessionStore:
    """进程内存会话存储。LRU 淘汰最旧会话;单会话超轮数时删最旧整对。"""

    def __init__(self, max_sessions: int | None = None, max_turns: int | None = None):
        s = get_settings()
        self._max_sessions = max_sessions or s.session_max_count
        self._max_turns = max_turns or s.session_max_turns
        self._sessions: "OrderedDict[str, Session]" = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}

    def lock(self, session_id: str) -> asyncio.Lock:
        """返回该会话的串行化锁。整轮对话持锁,防止同会话并发请求互相踩历史。"""
        lk = self._locks.get(session_id)
        if lk is None:
            lk = self._locks[session_id] = asyncio.Lock()
        return lk

    def new_session(self) -> Session:
        return self._put(Session(session_id=uuid.uuid4().hex))

    def get(self, session_id: str) -> Session | None:
        return self._sessions.get(session_id)

    def get_or_create(self, session_id: str | None) -> tuple[Session, bool]:
        if session_id:
            existing = self._sessions.get(session_id)
            if existing is not None:
                self._sessions.move_to_end(session_id)
                return existing, False
            return self._put(Session(session_id=session_id)), True
        return self.new_session(), True

    def append_turn(self, session_id: str, user_msg: str, assistant_msg: str) -> None:
        s = self._sessions.get(session_id)
        if s is None:
            # 会话可能已在流式期间被 LRU 淘汰:静默丢弃本次落盘,避免打断 SSE 流。
            logger.warning("append_turn: session %s 已被淘汰,丢弃本次写入", session_id)
            return
        s.turns.append({"role": "user", "content": user_msg})
        s.turns.append({"role": "assistant", "content": assistant_msg})
        max_msgs = self._max_turns * 2
        while len(s.turns) > max_msgs:
            del s.turns[:2]  # 删最旧一整对,保持 user/assistant 配对

    def clear(self) -> None:
        self._sessions.clear()
        self._locks.clear()

    def _put(self, session: Session) -> Session:
        self._sessions[session.session_id] = session
        while len(self._sessions) > self._max_sessions:
            _, evicted = self._sessions.popitem(last=False)
            self._locks.pop(evicted.session_id, None)
        return session


_store = SessionStore()


def get_store() -> SessionStore:
    return _store
