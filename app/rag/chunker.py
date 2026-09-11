"""结构感知切分:Markdown 标题层级 → 表格按行(复制表头) → 超长递归 → 重叠裁到句号。"""
import re
from dataclasses import dataclass

_SENT_END = "。!?！？;；"
_HEADING = re.compile(r"^(#{1,6})\s+(.*)$")
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$")


@dataclass
class Chunk:
    doc_id: str
    category: str
    questions: str
    answer: str
    text: str
    section_path: str
    content_type: str
    is_key_clause: bool
    order_index: int


def build_text(category: str, questions: str, answer: str) -> str:
    return "\n".join(p for p in (category, questions, answer) if p)


def _is_table_sep(line: str) -> bool:
    return bool(re.fullmatch(r"\s*\|[\s:\-|]+\|\s*", line))


def _split_sections(markdown: str) -> list[tuple[str, str, str]]:
    """→ [(上级路径, 章节标题, 正文)];正文不含标题行。"""
    sections: list[tuple[str, str, str]] = []
    stack: list[str] = []
    cur_title, cur_lines = "", []
    for line in markdown.splitlines():
        m = _HEADING.match(line)
        if m:
            if cur_title or cur_lines:
                sections.append((stack[-2] if len(stack) >= 2 else "", cur_title, "\n".join(cur_lines).strip()))
            level, title = len(m.group(1)), m.group(2).strip()
            stack = stack[: level - 1]
            stack.append(title)
            cur_title, cur_lines = title, []
        else:
            cur_lines.append(line)
    if cur_title or cur_lines:
        sections.append((stack[-2] if len(stack) >= 2 else "", cur_title, "\n".join(cur_lines).strip()))
    return sections


def _split_table(body: str, max_chars: int, overlap: int) -> list[str]:
    """表格:表头(表头行 + 分隔行)复制到每个按行切出的块。

    - 每个数据行单独成块,块首复制表头;
    - 单行超宽时只对该行做子切分,每个子块**仍带完整表头**;
    - 正文行(非表格行)不与表头混拼,单独成块。
    """
    lines = [ln for ln in body.splitlines() if ln.strip()]
    start = next((i for i, ln in enumerate(lines) if _TABLE_ROW.match(ln)), None)
    header: list[str] = []
    header_idx: set[int] = set()
    if start is not None:
        header.append(lines[start])
        header_idx.add(start)
        # 只有紧随其后的分隔行才算表头第二行(避免吞掉数据行)
        if start + 1 < len(lines) and _is_table_sep(lines[start + 1]):
            header.append(lines[start + 1])
            header_idx.add(start + 1)
    prefix = "\n".join(header)

    pieces: list[str] = []
    buf: list[str] = []

    def flush() -> None:
        if not buf:
            return
        block = "\n".join(buf)
        buf.clear()
        pieces.extend([block] if len(block) <= max_chars else _force_split(block, max_chars, overlap))

    def emit_row(row: str) -> None:
        text = f"{prefix}\n{row}" if prefix else row
        if len(text) <= max_chars:
            pieces.append(text)
            return
        # 超宽行:子切分正文,但每个子块都重新带上表头
        budget = max(1, max_chars - len(prefix) - 1)
        for seg in _force_split(row, budget, 0):
            pieces.append(f"{prefix}\n{seg}" if prefix else seg)

    for i, ln in enumerate(lines):
        if i in header_idx:
            continue
        if _TABLE_ROW.match(ln):
            flush()
            emit_row(ln)
        else:
            buf.append(ln)
    flush()
    return pieces or [body]


def _overlap_tail(text: str, overlap: int) -> str:
    """取 text 末尾不超过 overlap 字符、且落在句子边界上的一段,作为下一块的开头。

    返回的是**完整句子的后缀**(末尾即 text 末尾,开头紧跟在某个终止符之后),
    所以相邻块共享的这段文字不含半截话;窗口内没有终止符时返回空串。
    """
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    idx = min((i for i, ch in enumerate(tail) if ch in _SENT_END), default=-1)
    if idx == -1:
        return ""
    return tail[idx + 1 :]


def _recursive_split(body: str, max_chars: int, overlap: int) -> list[str]:
    if len(body) <= max_chars:
        return [body]
    # 1) 优先在表格行边界切;2) 其次句子边界;3) 兜底按长度
    if any(_TABLE_ROW.match(ln) for ln in body.splitlines()):
        return _split_table(body, max_chars, overlap)
    pieces: list[str] = []
    buf = ""
    for sent in re.split(r"(?<=[。!?！？;；])", body):
        if not sent:
            continue
        if buf and len(buf) + len(sent) > max_chars:
            pieces.append(buf)
            # 下一块以"上一块的句子后缀"开头,且整块长度仍不超过 max_chars
            buf = _overlap_tail(buf, min(overlap, max(0, max_chars - len(sent)))) + sent
        else:
            buf += sent
    if buf:
        pieces.append(buf)
    out: list[str] = []
    for p in pieces:
        out.extend(_force_split(p, max_chars, overlap) if len(p) > max_chars else [p])
    return out


def _force_split(text: str, max_chars: int, overlap: int) -> list[str]:
    # overlap >= max_chars 时旧写法步长为 1(逐字符切,块数爆炸)→ 钳制 overlap
    stride = max(1, max_chars - min(overlap, max_chars // 2))
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + max_chars])
        i += stride
    return out


def chunk_markdown(
    doc_id: str,
    markdown: str,
    *,
    content_type: str = "政策",
    max_chars: int = 800,
    overlap: int = 120,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    idx = 0
    for parent, title, body in _split_sections(markdown):
        if not body.strip():
            continue
        section_path = f"{parent} > {title}" if parent else title
        category = parent or title
        questions = title
        for piece in _recursive_split(body.strip(), max_chars, overlap):
            piece = piece.strip()
            if not piece:
                continue
            chunks.append(
                Chunk(
                    doc_id=doc_id,
                    category=category,
                    questions=questions,
                    answer=piece,
                    text=build_text(category, questions, piece),
                    section_path=section_path,
                    content_type=content_type,
                    is_key_clause=("关键" in piece or "必须" in piece),
                    order_index=idx,
                )
            )
            idx += 1
    # 前后块指针
    return chunks
