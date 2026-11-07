"""sleep builtin tool.

``sleep`` replaces ``wait`` in the event-driven core: it has no timeout and
returns only when the next event wakes the agent. Calling it is also the way to
declare the current event handled or ignored.
"""

from typing import Optional

from src.core.tooling import ToolExecutionContext, ToolExecutionResult, ToolInvocation, ToolSpec

from .context import BuiltinToolRuntimeContext


def get_tool_spec() -> ToolSpec:
    """Return the tool declaration for sleep."""

    return ToolSpec(
        name="sleep",
        description=(
            "结束当前事件并进入待机，直到下一个事件把你唤醒。"
            "不需要回复、没有更多动作、或者决定忽略这条消息时，直接调用它即可；"
            "被唤醒时会告诉你待机了多久。不要为了显得勤快而先做无意义的操作。"
        ),
        parameters_schema={
            "type": "object",
            "properties": {},
            "additionalProperties": False,
        },
        provider_name="maisaka_builtin",
        provider_type="builtin",
    )


async def handle_tool(
    tool_ctx: BuiltinToolRuntimeContext,
    invocation: ToolInvocation,
    context: Optional[ToolExecutionContext] = None,
) -> ToolExecutionResult:
    """Enter an indefinite wait that the next event can end."""

    del context
    tool_ctx.runtime._enter_counted_wait_state(
        seconds=None,
        tool_call_id=invocation.call_id,
        logical_turn_id=tool_ctx.engine.active_logical_turn_id,
    )
    return tool_ctx.build_success_result(
        invocation.tool_name,
        "已进入待机，等待下一个事件；期间不会主动查看聊天。",
        metadata={
            "pause_execution": True,
            "sleeping": True,
        },
    )
