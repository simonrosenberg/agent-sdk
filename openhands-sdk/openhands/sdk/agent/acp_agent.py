"""ACPAgent — an AgentBase subclass that delegates to an ACP server.

The Agent Client Protocol (ACP) lets OpenHands power conversations using
ACP-compatible servers (Claude Code, Gemini CLI, etc.) instead of direct
LLM calls.  The ACP server manages its own LLM, tools, and execution;
the ACPAgent simply relays user messages and collects the response.

See https://agentclientprotocol.com/protocol/overview
"""

from __future__ import annotations

import os
from collections.abc import Generator
from typing import TYPE_CHECKING, Any

from pydantic import Field, PrivateAttr

from openhands.sdk.agent.base import AgentBase
from openhands.sdk.conversation.state import ConversationExecutionStatus
from openhands.sdk.event import MessageEvent, SystemPromptEvent
from openhands.sdk.llm import LLM, Message, TextContent
from openhands.sdk.logger import get_logger
from openhands.sdk.tool import Tool  # noqa: TC002


if TYPE_CHECKING:
    from openhands.sdk.conversation import (
        ConversationCallbackType,
        ConversationState,
        ConversationTokenCallbackType,
        LocalConversation,
    )


logger = get_logger(__name__)


def _make_sentinel_llm() -> LLM:
    """Create a sentinel LLM that should never be called."""
    return LLM(model="acp-managed")


# ---------------------------------------------------------------------------
# ACP Client implementation
# ---------------------------------------------------------------------------


async def _filter_jsonrpc_lines(
    source: Any, dest: Any
) -> None:
    """Read lines from *source* and forward only JSON-RPC lines to *dest*.

    Some ACP servers (e.g. ``claude-code-acp`` v0.1.x) emit log messages
    like ``[ACP] ...`` to stdout alongside JSON-RPC traffic.  This coroutine
    strips those non-protocol lines so the JSON-RPC connection is not confused.
    """
    try:
        while True:
            line = await source.readline()
            if not line:
                dest.feed_eof()
                break
            # JSON-RPC messages are single-line JSON objects containing
            # "jsonrpc". Filter out multi-line pretty-printed JSON from
            # debug logs that also start with '{'.
            stripped = line.lstrip()
            if stripped.startswith(b"{") and b'"jsonrpc"' in line:
                dest.feed_data(line)
            else:
                # Log non-JSON lines at debug level
                try:
                    logger.debug(
                        "ACP stdout (non-JSON): %s",
                        line.decode(errors="replace").rstrip(),
                    )
                except Exception:
                    pass
    except Exception:
        dest.feed_eof()


class _OpenHandsACPClient:
    """ACP Client that accumulates session updates and emits OpenHands events.

    Implements the ``Client`` protocol from ``agent_client_protocol``.
    """

    def __init__(self) -> None:
        self.accumulated_text: list[str] = []
        self.accumulated_thoughts: list[str] = []
        self.on_token: Any = None  # ConversationTokenCallbackType | None

    def reset(self) -> None:
        self.accumulated_text.clear()
        self.accumulated_thoughts.clear()
        self.on_token = None

    # -- Client protocol methods ------------------------------------------

    async def session_update(
        self,
        session_id: str,  # noqa: ARG002
        update: Any,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        from acp.schema import (
            AgentMessageChunk,
            AgentThoughtChunk,
            TextContentBlock,
            ToolCallProgress,
            ToolCallStart,
        )

        if isinstance(update, AgentMessageChunk):
            if isinstance(update.content, TextContentBlock):
                text = update.content.text
                self.accumulated_text.append(text)
                if self.on_token is not None:
                    try:
                        self.on_token(text)
                    except Exception:
                        pass
        elif isinstance(update, AgentThoughtChunk):
            if isinstance(update.content, TextContentBlock):
                self.accumulated_thoughts.append(update.content.text)
        elif isinstance(update, (ToolCallStart, ToolCallProgress)):
            logger.debug("ACP tool call event: %s", type(update).__name__)
        else:
            logger.debug("ACP session update: %s", type(update).__name__)

    async def request_permission(
        self,
        options: list[Any],
        session_id: str,  # noqa: ARG002
        tool_call: Any,  # noqa: ARG002
        **kwargs: Any,  # noqa: ARG002
    ) -> Any:
        """Auto-approve all permission requests from the ACP server."""
        from acp.schema import AllowedOutcome, RequestPermissionResponse

        # Pick the first option (usually "allow once")
        option_id = options[0].option_id if options else "allow_once"
        return RequestPermissionResponse(
            result=AllowedOutcome(outcome="selected", option_id=option_id),
        )

    # fs/terminal methods — raise NotImplementedError; ACP server handles its own
    async def write_text_file(self, **kwargs: Any) -> None:
        raise NotImplementedError("ACP server handles file operations")

    async def read_text_file(self, **kwargs: Any) -> Any:
        raise NotImplementedError("ACP server handles file operations")

    async def create_terminal(self, **kwargs: Any) -> Any:
        raise NotImplementedError("ACP server handles terminal operations")

    async def terminal_output(self, **kwargs: Any) -> Any:
        raise NotImplementedError("ACP server handles terminal operations")

    async def release_terminal(self, **kwargs: Any) -> None:
        raise NotImplementedError("ACP server handles terminal operations")

    async def wait_for_terminal_exit(self, **kwargs: Any) -> Any:
        raise NotImplementedError("ACP server handles terminal operations")

    async def kill_terminal(self, **kwargs: Any) -> None:
        raise NotImplementedError("ACP server handles terminal operations")

    async def ext_method(
        self,
        method: str,  # noqa: ARG002
        params: dict[str, Any],  # noqa: ARG002
    ) -> dict[str, Any]:
        return {}

    async def ext_notification(
        self,
        method: str,  # noqa: ARG002
        params: dict[str, Any],  # noqa: ARG002
    ) -> None:
        pass

    def on_connect(self, conn: Any) -> None:  # noqa: ARG002
        pass


# ---------------------------------------------------------------------------
# ACPAgent
# ---------------------------------------------------------------------------


class ACPAgent(AgentBase):
    """Agent that delegates to an ACP (Agent Client Protocol) server.

    Instead of calling an LLM directly, this agent spawns an ACP-compatible
    server (e.g. ``claude-code-acp``) as a subprocess and communicates with
    it via the ACP protocol.  The server manages its own LLM, tools, and
    execution lifecycle.

    Example::

        from openhands.sdk.agent import ACPAgent
        from openhands.sdk.conversation import Conversation

        agent = ACPAgent(acp_command=["npx", "-y", "claude-code-acp"])
        conversation = Conversation(agent=agent, workspace="./workspace")
        conversation.send_message("Hello! What is 2+2?")
        conversation.run()
    """

    # Override required fields with ACP-appropriate defaults
    llm: LLM = Field(default_factory=_make_sentinel_llm)
    tools: list[Tool] = Field(default_factory=list)
    include_default_tools: list[str] = Field(default_factory=list)

    # ACP-specific configuration
    acp_command: list[str] = Field(
        ...,
        description=(
            "Command to start the ACP server, "
            "e.g. ['npx', '-y', 'claude-code-acp']"
        ),
    )
    acp_args: list[str] = Field(
        default_factory=list,
        description="Additional arguments for the ACP server command",
    )
    acp_env: dict[str, str] = Field(
        default_factory=dict,
        description="Additional environment variables for the ACP server process",
    )

    # Private runtime state
    _executor: Any = PrivateAttr(default=None)
    _conn: Any = PrivateAttr(default=None)  # ClientSideConnection
    _session_id: str | None = PrivateAttr(default=None)
    _process: Any = PrivateAttr(default=None)  # asyncio subprocess
    _client: Any = PrivateAttr(default=None)  # _OpenHandsACPClient
    _filtered_reader: Any = PrivateAttr(default=None)  # StreamReader
    _closed: bool = PrivateAttr(default=False)

    # -- Override base properties to be no-ops for ACP ---------------------

    @property
    def system_message(self) -> str:
        return "ACP-managed agent"

    def get_all_llms(self) -> Generator[LLM, None, None]:
        yield from ()

    # -- Lifecycle ---------------------------------------------------------

    def init_state(
        self,
        state: ConversationState,
        on_event: ConversationCallbackType,
    ) -> None:
        """Spawn the ACP server and initialize a session."""
        # Lazy import guard
        try:
            from acp import spawn_agent_process  # noqa: F401
        except ImportError:
            raise ImportError(
                "The 'agent-client-protocol' package is required for ACPAgent. "
                "Install it with: pip install 'openhands-sdk[acp]' or "
                "pip install agent-client-protocol"
            ) from None

        # Validate no unsupported features
        if self.tools:
            raise NotImplementedError(
                "ACPAgent does not support custom tools; "
                "the ACP server manages its own tools"
            )
        if self.mcp_config:
            raise NotImplementedError(
                "ACPAgent does not support mcp_config; "
                "configure MCP on the ACP server instead"
            )
        if self.condenser is not None:
            raise NotImplementedError(
                "ACPAgent does not support condenser; "
                "the ACP server manages its own context"
            )
        if self.critic is not None:
            raise NotImplementedError(
                "ACPAgent does not support critic; "
                "the ACP server manages its own evaluation"
            )
        if self.agent_context is not None:
            raise NotImplementedError(
                "ACPAgent does not support agent_context; "
                "configure the ACP server directly"
            )

        from openhands.sdk.utils.async_executor import AsyncExecutor

        self._executor = AsyncExecutor()

        try:
            self._start_acp_server(state)
        except Exception as e:
            logger.error("Failed to start ACP server: %s", e)
            self._cleanup()
            raise

        # Emit a minimal SystemPromptEvent
        event = SystemPromptEvent(
            source="agent",
            system_prompt=TextContent(text="ACP-managed agent"),
            tools=[],
        )
        on_event(event)
        self._initialized = True

    def _start_acp_server(self, state: ConversationState) -> None:
        """Start the ACP subprocess and initialize the session."""
        import asyncio

        from acp.client.connection import ClientSideConnection
        from acp.transports import default_environment

        client = _OpenHandsACPClient()
        self._client = client

        # Build environment: inherit current env + ACP extras
        env = default_environment()
        env.update(os.environ)
        env.update(self.acp_env)

        command = self.acp_command[0]
        args = list(self.acp_command[1:]) + list(self.acp_args)

        working_dir = str(state.workspace.working_dir)

        async def _init() -> tuple[Any, Any, Any, str]:
            # Spawn the subprocess directly so we can install a
            # filtering reader that skips non-JSON-RPC lines some
            # ACP servers (e.g. claude-code-acp v0.1.x) write to
            # stdout.
            process = await asyncio.create_subprocess_exec(
                command,
                *args,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                env=env,
            )
            assert process.stdin is not None
            assert process.stdout is not None

            # Wrap the subprocess stdout in a filtering reader that
            # only passes lines starting with '{' (JSON-RPC messages).
            filtered_reader = asyncio.StreamReader()
            asyncio.get_event_loop().create_task(
                _filter_jsonrpc_lines(process.stdout, filtered_reader)
            )

            conn = ClientSideConnection(
                client,
                process.stdin,   # write to subprocess
                filtered_reader,  # read filtered output
            )

            # Initialize the protocol
            await conn.initialize(protocol_version=1)

            # Create a new session
            response = await conn.new_session(cwd=working_dir)
            session_id = response.session_id

            return conn, process, filtered_reader, session_id

        result = self._executor.run_async(_init)
        self._conn, self._process, self._filtered_reader, self._session_id = result

    def step(
        self,
        conversation: LocalConversation,
        on_event: ConversationCallbackType,
        on_token: ConversationTokenCallbackType | None = None,
    ) -> None:
        """Send the latest user message to the ACP server and emit the response."""
        state = conversation.state

        # Find the latest user message
        user_message = None
        for event in reversed(list(state.events)):
            if isinstance(event, MessageEvent) and event.source == "user":
                # Extract text from the message
                for content in event.llm_message.content:
                    if isinstance(content, TextContent) and content.text.strip():
                        user_message = content.text
                        break
                if user_message:
                    break

        if user_message is None:
            logger.warning("No user message found; finishing conversation")
            state.execution_status = ConversationExecutionStatus.FINISHED
            return

        # Reset client accumulators
        self._client.reset()
        self._client.on_token = on_token

        try:
            import asyncio

            from acp.helpers import text_block

            async def _prompt() -> None:
                await self._conn.prompt(
                    [text_block(user_message)],
                    self._session_id,
                )
                # Allow pending session_update notifications to be
                # processed before we read accumulated text.  The ACP
                # server sends notifications and responses through the
                # same connection, but notification handlers run as
                # tasks and may not have completed when prompt() returns.
                await asyncio.sleep(0.1)

            # Send prompt to ACP server
            self._executor.run_async(_prompt)

            # Build response message
            response_text = "".join(self._client.accumulated_text)
            thought_text = "".join(self._client.accumulated_thoughts)

            if not response_text:
                response_text = "(No response from ACP server)"

            message = Message(
                role="assistant",
                content=[TextContent(text=response_text)],
                reasoning_content=thought_text if thought_text else None,
            )

            msg_event = MessageEvent(
                source="agent",
                llm_message=message,
            )
            on_event(msg_event)
            state.execution_status = ConversationExecutionStatus.FINISHED

        except Exception as e:
            logger.error("ACP prompt failed: %s", e, exc_info=True)
            # Emit error as an agent message since AgentErrorEvent requires
            # tool context we don't have
            error_message = Message(
                role="assistant",
                content=[TextContent(text=f"ACP error: {e}")],
            )
            error_event = MessageEvent(
                source="agent",
                llm_message=error_message,
            )
            on_event(error_event)
            state.execution_status = ConversationExecutionStatus.ERROR

    def close(self) -> None:
        """Terminate the ACP subprocess and clean up resources."""
        if self._closed:
            return
        self._closed = True
        self._cleanup()

    def _cleanup(self) -> None:
        """Internal cleanup of ACP resources."""
        # Close the connection first
        if self._conn is not None and self._executor is not None:
            try:
                self._executor.run_async(self._conn.close())
            except Exception as e:
                logger.debug("Error closing ACP connection: %s", e)
            self._conn = None

        # Terminate the subprocess
        if self._process is not None:
            try:
                self._process.terminate()
            except Exception:
                pass
            try:
                self._process.kill()
            except Exception:
                pass
            self._process = None

        if self._executor is not None:
            try:
                self._executor.close()
            except Exception:
                pass
            self._executor = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
