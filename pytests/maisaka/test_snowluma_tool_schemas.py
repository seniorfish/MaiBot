"""SnowLuma 适配器的工具声明必须能被宿主正常规范化。

插件里的 ``@Tool`` 只在导入时构造 schema，宿主在之后的 normalize 阶段才会
发现 ``array`` 缺少 ``items`` 之类的错误——那条路径曾让整个 Focus 组停摆。
这里在测试阶段把插件当包导入，逐个跑一遍宿主侧的规范化。
"""

import importlib
import inspect
import os
import shutil
import sys
from pathlib import Path

import pytest

from src.llm_models.payload_content.tool_option import normalize_tool_option

PLUGIN_DIR = Path(
    os.environ.get("MAIBOT_INSTANCE_DIR", "")
) / "plugins" / "MaiBot-SnowLuma-Adapter"


def _load_plugin_module(tmp_path: Path):
    if not (PLUGIN_DIR / "plugin.py").is_file():
        pytest.skip(f"适配器插件不存在: {PLUGIN_DIR}")
    package = tmp_path / "snowluma_probe"
    shutil.copytree(
        PLUGIN_DIR,
        package,
        ignore=shutil.ignore_patterns("__pycache__", "tests", "*.pyc"),
    )
    sys.path.insert(0, str(tmp_path))
    try:
        return importlib.import_module("snowluma_probe.plugin")
    finally:
        sys.path.remove(str(tmp_path))


def _tool_infos(module) -> list:
    plugin_class = next(
        obj
        for obj in vars(module).values()
        if inspect.isclass(obj) and obj.__module__ == module.__name__ and hasattr(obj, "tool_onebot_api")
    )
    infos = []
    for _, func in inspect.getmembers(plugin_class, predicate=inspect.isfunction):
        info = getattr(func, "__maibot_component_info__", None)
        if info is not None and type(info).__name__ == "ToolComponentInfo":
            infos.append(info)
    return infos


def test_every_adapter_tool_schema_normalizes(tmp_path: Path) -> None:
    module = _load_plugin_module(tmp_path)
    infos = _tool_infos(module)
    assert len(infos) >= 10

    for info in infos:
        normalize_tool_option(
            {
                "name": info.name,
                "description": info.description or "",
                "parameters": info.get_parameters_schema(),
            }
        )


def test_qzone_feed_rendering_hides_html_and_ads(tmp_path: Path) -> None:
    module = _load_plugin_module(tmp_path)
    plugin = object.__new__(module.SnowLumaAdapterPlugin)
    payload = {
        "feeds": [
            {
                "uin": 1001,
                "nickname": "同学",
                "time": 1793889840,
                "appid": 311,
                "tid": "tid-a",
                "content": "作业太多了",
                "content_complete": True,
                "images": ["https://a1.qpic.cn/x"],
                "like_count": 3,
                "comment_count": 1,
                "html": "<li>非常长的原始 HTML</li>",
            },
            {"uin": 0, "nickname": "", "appid": 6600, "tid": None, "html": "<li>广告</li>"},
        ]
    }

    content = plugin._qzone_list_result({"success": True, "data": payload}, kind="feeds")["content"]

    assert "tid=tid-a" in content
    assert "作业太多了" in content
    assert "https://a1.qpic.cn/x" in content
    assert "非常长的原始 HTML" not in content
    assert "已过滤 1 条广告" in content


def test_qzone_feed_filter_by_qq(tmp_path: Path) -> None:
    module = _load_plugin_module(tmp_path)
    plugin = object.__new__(module.SnowLumaAdapterPlugin)
    payload = {
        "feeds": [
            {"uin": 1001, "nickname": "甲", "time": 1793889840, "appid": 311, "tid": "a", "content": "甲的说说"},
            {"uin": 1002, "nickname": "乙", "time": 1793889800, "appid": 311, "tid": "b", "content": "乙的说说"},
        ]
    }

    content = plugin._qzone_list_result({"success": True, "data": payload}, kind="feeds", qq="1002")["content"]

    assert "乙的说说" in content
    assert "甲的说说" not in content
    assert "共检查 2 条动态" in content


def test_qzone_feed_filter_without_match(tmp_path: Path) -> None:
    module = _load_plugin_module(tmp_path)
    plugin = object.__new__(module.SnowLumaAdapterPlugin)
    payload = {"feeds": [{"uin": 1001, "tid": "a", "appid": 311, "content": "x"}]}

    result = plugin._qzone_list_result({"success": True, "data": payload}, kind="feeds", qq="9999")

    assert result["success"] is True
    assert result["count"] == 0
    assert "没有来自 QQ 9999" in result["content"]
