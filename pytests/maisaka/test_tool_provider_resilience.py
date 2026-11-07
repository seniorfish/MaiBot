"""单个 Provider 声明失败时只跳过它，不能拖垮整个 Agent。"""

import pytest

from src.core.tooling import ToolRegistry, ToolSpec


class _GoodProvider:
    provider_name = "good"
    provider_type = "builtin"

    async def list_tools(self, context=None):
        return [
            ToolSpec(
                name="good_tool",
                description="ok",
                parameters_schema={"type": "object", "properties": {}},
                provider_name=self.provider_name,
                provider_type=self.provider_type,
            )
        ]

    async def invoke(self, invocation, context=None):  # pragma: no cover - not used
        raise NotImplementedError


class _BrokenProvider:
    provider_name = "broken"
    provider_type = "plugin"

    async def list_tools(self, context=None):
        raise ValueError("数组参数必须提供 items_schema")

    async def invoke(self, invocation, context=None):  # pragma: no cover - not used
        raise NotImplementedError


@pytest.mark.asyncio
async def test_broken_provider_is_skipped() -> None:
    registry = ToolRegistry()
    registry.register_provider(_BrokenProvider())
    registry.register_provider(_GoodProvider())

    specs = await registry.list_tools(None)

    assert [spec.name for spec in specs] == ["good_tool"]
