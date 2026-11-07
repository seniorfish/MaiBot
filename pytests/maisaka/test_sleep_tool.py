"""sleep tool enters an indefinite wait and never schedules a timeout."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from src.core.tooling import ToolExecutionResult, ToolInvocation
from src.maisaka.builtin_tool.sleep import get_tool_spec, handle_tool


class _FakeContext:
    def __init__(self) -> None:
        self.runtime = SimpleNamespace(_enter_counted_wait_state=Mock(return_value=1))
        self.engine = SimpleNamespace(active_logical_turn_id="turn-1")

    @staticmethod
    def build_success_result(name, content, metadata=None):
        return ToolExecutionResult(tool_name=name, success=True, content=content, metadata=dict(metadata or {}))


def test_sleep_spec_has_no_required_parameters() -> None:
    spec = get_tool_spec()
    assert spec.name == "sleep"
    assert spec.parameters_schema["properties"] == {}
    assert spec.parameters_schema.get("required", []) == []


@pytest.mark.asyncio
async def test_sleep_enters_indefinite_wait() -> None:
    context = _FakeContext()
    invocation = ToolInvocation(
        tool_name="sleep",
        arguments={},
        call_id="call-sleep",
        session_id="sess",
        stream_id="sess",
    )

    result = await handle_tool(context, invocation, None)

    context.runtime._enter_counted_wait_state.assert_called_once_with(
        seconds=None,
        tool_call_id="call-sleep",
        logical_turn_id="turn-1",
    )
    assert result.success is True
    assert result.metadata["pause_execution"] is True
    assert result.metadata["sleeping"] is True
