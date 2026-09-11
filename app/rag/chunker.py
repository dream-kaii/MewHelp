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


def _split_table(body: str) -> list[str]:
    """表格:表头(含分隔行)复制到每个按行切出的块。"""
    lines = [ln for ln in body.splitlines() if ln.strip()]
    header, rows = [], []
    for ln in lines:
        if _TABLE_ROW.match(ln):
            if len(header) < 2:
                header.append(ln)
            elif _is_table_sep(ln):
                header.append(ln)
            else:
                rows.append(ln)
        else:
            rows.append(ln)
    if not rows:
        return [body]
    prefix = "\n".join(header[:2])
    out = []
    for r in rows:
        out.append(f"{prefix}\n{r}" if prefix else r)
    return out


def _trim_to_sentence(text: str, overlap: int) -> str:
    """取末尾 overlap 字符做重叠,并回退到最近句号,避免半截话。"""
    if overlap <= 0:
        return ""
    tail = text[-overlap:]
    idx = max((tail.rfind(ch) for ch in _SENT_END), default=-1)
    if idx == -1:
        return tail
    return tail[idx + 1 :]


def _recursive_split(body: str, max_chars: int, overlap: int) -> list[str]:
    if len(body) <= max_chars:
        return [body]
    # 1) 优先在表格行边界切;2) 其次句子边界;3) 兜底按长度
    pieces: list[str] = []
    if any(_TABLE_ROW.match(ln) for ln in body.splitlines()):
        return _split_table(body) if max(len(x) for x in _split_table(body)) <= max_chars else _force_split(body, max_chars, overlap)
    buf = ""
    for sent in re.split(r"(?<=[。!?！？;；])", body):
        if not sent:
            continue
        if len(buf) + len(sent) > max_chars and buf:
            pieces.append(buf)
            buf = _trim_to_sentence(buf, overlap) + sent
        else:
            buf += sent
    if buf:
        pieces.append(buf)
    out: list[str] = []
    for p in pieces:
        out.extend(_force_split(p, max_chars, overlap) if len(p) > max_chars * 1.5 else [p])
    return out


def _force_split(text: str, max_chars: int, overlap: int) -> list[str]:
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + max_chars])
        i += max(1, max_chars - overlap)
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
