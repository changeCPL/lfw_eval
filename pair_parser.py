"""将文本行解析为 FacePair（支持同人三列、异人四列格式）。

常见布局（与 LFW 单折结构一致时）：
- 可选首行表头，例如 ``name,imagenum1,imagenum2,`` —— 需用 ``skip_header_data_rows`` 跳过。
- 同人若干行：``Name,num1,num2`` 或句末多一个逗号 ``Name,num1,num2,``（空字段会被忽略）。
- 异人若干行：``Name1,num1,Name2,num2``。

是否与「10 段连续块」兼容取决于 **6000 对在内存中的顺序**，见 ``evaluate_10fold`` 文档说明。
"""

from __future__ import annotations

import re
from typing import Iterable

from .types import FacePair


def _split_fields(line: str) -> list[str]:
    # 同时兼容英文逗号与中文逗号
    parts = [p.strip() for p in re.split(r"[,，]", line) if p.strip()]
    return parts


def _make_key(person: str, index: int) -> str:
    return f"{person}_{index:04d}"


def parse_pair_line(line: str) -> FacePair:
    """
    解析一行配对定义。

    同人（3 列）：``Person, i, j`` → ``Person_iiii`` 与 ``Person_jjjj``。
    异人（4 列）：``P1, i, P2, j`` → ``P1_iiii`` 与 ``P2_jjjj``。

    空行、仅空白、以 ``#`` 开头的行视为无效，抛出 ValueError。
    """
    s = line.strip()
    if not s or s.startswith("#"):
        raise ValueError(f"无效行（空或注释）: {line!r}")

    parts = _split_fields(s)
    if len(parts) == 3:
        person, a, b = parts[0], int(parts[1]), int(parts[2])
        return FacePair(
            key_a=_make_key(person, a),
            key_b=_make_key(person, b),
            is_same=True,
        )
    if len(parts) == 4:
        p1, i1, p2, i2 = parts[0], int(parts[1]), parts[2], int(parts[3])
        return FacePair(
            key_a=_make_key(p1, i1),
            key_b=_make_key(p2, i2),
            is_same=False,
        )

    raise ValueError(
        f"期望 3 列（同人）或 4 列（异人），得到 {len(parts)} 列: {line!r}"
    )


def parse_pair_lines(
    lines: Iterable[str],
    *,
    skip_header_data_rows: int = 0,
) -> list[FacePair]:
    """
    逐行解析，跳过空行与 ``#`` 注释行。

    ``skip_header_data_rows``：跳过开头若干条「有效内容行」（非空、非注释）。
    例如首行为 ``name,imagenum1,imagenum2,`` 时设为 ``1``，避免被当成同人三元组解析。
    """
    out: list[FacePair] = []
    skip_left = max(0, int(skip_header_data_rows))
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if skip_left > 0:
            skip_left -= 1
            continue
        out.append(parse_pair_line(line))
    return out
