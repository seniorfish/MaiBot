from typing import Any, List, Tuple

from json_repair import repair_json

import json
import re

from src.common.logger import get_logger

from .expression_style_utils import normalize_expression_style_for_learning

logger = get_logger("expression_utils")


def _normalize_repair_json_result(repaired_result: Any) -> str:
    """将 `repair_json` 的返回结果统一转换为字符串。"""
    if isinstance(repaired_result, str):
        return repaired_result
    if isinstance(repaired_result, tuple) and repaired_result:
        first_item = repaired_result[0]
        if isinstance(first_item, str):
            return first_item
        return json.dumps(first_item, ensure_ascii=False)
    raise TypeError(f"repair_json 返回了无法处理的结果类型: {type(repaired_result)}")


def _strip_markdown_code_fence(text: str) -> str:
    """移除 LLM 可能附带的 Markdown 代码块包裹。"""
    raw = text.strip()
    if match := re.search(r"```json\s*(.*?)\s*```", raw, re.DOTALL):
        return match[1].strip()
    raw = re.sub(r"^```\s*", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"```\s*$", "", raw, flags=re.MULTILINE)
    return raw.strip()


def parse_expression_response(response: str) -> Tuple[List[Tuple[str, str, str]], List[Tuple[str, str]]]:
    """
    解析 LLM 返回的表达方式总结和黑话 JSON，提取两个列表。

    Returns:
        第一个列表是表达方式 (situation, style, id)
        第二个列表是黑话 (content, id)
    """
    if not response:
        return [], []

    raw = _strip_markdown_code_fence(response)

    parsed = _try_parse(raw)
    if parsed is None:
        fixed = fix_chinese_quotes_in_json(raw)
        parsed = _try_parse(fixed)
    if parsed is None:
        logger.error(f"处理后的 JSON 字符串（前500字符）：{raw[:500]}")
        return [], []

    if isinstance(parsed, dict):
        parsed_list = [parsed]
    elif isinstance(parsed, list):
        parsed_list = parsed
    else:
        logger.error(f"表达风格解析结果类型异常: {type(parsed)}, 内容: {parsed}")
        return [], []

    expressions: List[Tuple[str, str, str]] = []
    jargon_entries: List[Tuple[str, str]] = []

    for item in parsed_list:
        if not isinstance(item, dict):
            continue

        situation = str(item.get("situation", "")).strip()
        style = normalize_expression_style_for_learning(str(item.get("style", "")).strip())
        # 模型输入和输出使用短字段 id，内部元组及插件 Hook 仍沿用来源编号的既有约定。
        source_id = str(item.get("id", "")).strip()

        if situation and style and source_id:
            expressions.append((situation, style, source_id))
            continue

        content = str(item.get("content", "")).strip()
        if content and source_id:
            jargon_entries.append((content, source_id))

    return expressions, jargon_entries


def parse_jargon_response(response: str) -> List[Tuple[str, str]]:
    """解析独立黑话学习 LLM 响应，提取 (content, id) 列表。"""

    _, jargon_entries = parse_expression_response(response)
    return jargon_entries


def is_single_char_jargon(content: str) -> bool:
    """判断是否是单字黑话（单个汉字、英文或数字）。"""
    if not content or len(content) != 1:
        return False

    char = content[0]
    return (
        "\u4e00" <= char <= "\u9fff"
        or "a" <= char <= "z"
        or "A" <= char <= "Z"
        or "0" <= char <= "9"
    )


def _try_parse(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        try:
            repaired = _normalize_repair_json_result(repair_json(text))
            return json.loads(repaired)
        except Exception:
            return None
