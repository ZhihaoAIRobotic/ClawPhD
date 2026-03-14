"""Subagent manager for background task execution."""

import asyncio
import json
import uuid
from pathlib import Path
from typing import Any

from loguru import logger

from clawphd.agent.tools.filesystem import EditFileTool, ListDirTool, ReadFileTool, WriteFileTool
from clawphd.agent.tools.registry import ToolRegistry
from clawphd.agent.tools.shell import ExecTool
from clawphd.agent.tools.web import WebFetchTool, WebSearchTool
from clawphd.bus.events import InboundMessage
from clawphd.bus.queue import MessageBus
from clawphd.config.schema import ExecToolConfig
from clawphd.providers.base import LLMProvider


class SubagentManager:
    """Manages background subagent execution."""

    def __init__(
        self,
        provider: LLMProvider,
        workspace: Path,
        bus: MessageBus,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        reasoning_effort: str | None = None,
        brave_api_key: str | None = None,
        s2_api_key: str | None = None,
        web_proxy: str | None = None,
        exec_config: "ExecToolConfig | None" = None,
        restrict_to_workspace: bool = False,
        vlm_provider: Any = None,
        image_gen_provider: Any = None,
        reference_store: Any = None,
        fal_api_key: str | None = None,
    ):
        from clawphd.config.schema import ExecToolConfig
        self.provider = provider
        self.workspace = workspace
        self.bus = bus
        self.model = model or provider.get_default_model()
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.reasoning_effort = reasoning_effort
        self.brave_api_key = brave_api_key
        self.s2_api_key = s2_api_key
        self.web_proxy = web_proxy
        self.exec_config = exec_config or ExecToolConfig()
        self.restrict_to_workspace = restrict_to_workspace
        self.vlm_provider = vlm_provider
        self.image_gen_provider = image_gen_provider
        self.reference_store = reference_store
        self.fal_api_key = fal_api_key
        self._running_tasks: dict[str, asyncio.Task[None]] = {}
        self._session_tasks: dict[str, set[str]] = {}  # session_key -> {task_id, ...}

    def _register_clawphd_tools(self, tools: ToolRegistry, allowed_dir: Path | None) -> None:
        """Register ClawPhD-specific research and diagram tools for subagents."""
        try:
            from clawphd.agent.tools.autopage import (
                ExtractTableHTMLTool,
                MatchTemplateTool,
                ParsePaperTool,
                RenderHTMLTool,
                ReviewHTMLVisualTool,
            )
            from clawphd.agent.tools.figureref import (
                ClassifyFiguresTool,
                ExportFigureReferenceTool,
                ExtractPaperFiguresTool,
                SearchInfluentialPapersTool,
            )
            from clawphd.agent.tools.paperbanana import (
                CritiqueImageTool,
                GenerateImageTool,
                OptimizeInputTool,
                PlanDiagramTool,
                SearchReferencesTool,
            )
        except ImportError:
            return

        if self.vlm_provider or self.image_gen_provider:
            output_dir = str(self.workspace / "outputs")
            tools.register(OptimizeInputTool(vlm_provider=self.vlm_provider))
            tools.register(
                PlanDiagramTool(
                    vlm_provider=self.vlm_provider,
                    reference_store=self.reference_store,
                )
            )
            tools.register(
                SearchReferencesTool(
                    vlm_provider=self.vlm_provider,
                    reference_store=self.reference_store,
                )
            )
            tools.register(
                GenerateImageTool(
                    image_gen_provider=self.image_gen_provider,
                    vlm_provider=self.vlm_provider,
                    output_dir=output_dir,
                )
            )
            tools.register(CritiqueImageTool(vlm_provider=self.vlm_provider))

        tools.register(ParsePaperTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(RenderHTMLTool(workspace=self.workspace, allowed_dir=allowed_dir))
        tools.register(MatchTemplateTool(workspace=self.workspace, allowed_dir=allowed_dir))

        if self.vlm_provider:
            tools.register(
                ReviewHTMLVisualTool(vlm_provider=self.vlm_provider, allowed_dir=allowed_dir)
            )
            tools.register(
                ExtractTableHTMLTool(vlm_provider=self.vlm_provider, allowed_dir=allowed_dir)
            )

        tools.register(SearchInfluentialPapersTool(api_key=self.s2_api_key))
        tools.register(
            ExtractPaperFiguresTool(workspace=self.workspace, allowed_dir=allowed_dir)
        )
        tools.register(ExportFigureReferenceTool(workspace=self.workspace))
        if self.vlm_provider:
            tools.register(ClassifyFiguresTool(vlm_provider=self.vlm_provider))

        # AutoFigure: image-to-drawio tools
        try:
            from clawphd.agent.tools.autofigure import (
                CropRemoveBgTool,
                GenerateDrawioTemplateTool,
                GenerateSVGTemplateTool,
                ReplaceIconsDrawioTool,
                ReplaceIconsSVGTool,
                SegmentFigureTool,
            )
        except ImportError:
            pass
        else:
            if self.fal_api_key:
                tools.register(SegmentFigureTool(fal_api_key=self.fal_api_key))
            tools.register(CropRemoveBgTool())
            af_vlm = self._get_autofigure_vlm()
            if af_vlm:
                tools.register(GenerateSVGTemplateTool(vlm_provider=af_vlm))
                tools.register(GenerateDrawioTemplateTool(vlm_provider=af_vlm))
            tools.register(ReplaceIconsSVGTool())
            tools.register(ReplaceIconsDrawioTool())

    def _get_autofigure_vlm(self) -> Any:
        """Get or create a multimodal VLM for autofigure (same logic as AgentLoop)."""
        from clawphd.agent.tools.paperbanana_providers import OpenRouterVLM

        TEXT_ONLY = {"openai/gpt-4.1-mini", "openai/gpt-4o-mini", "openai/gpt-3.5-turbo"}
        DEFAULT = "google/gemini-2.5-flash"

        if self.vlm_provider is None:
            return None

        if isinstance(self.vlm_provider, OpenRouterVLM):
            current = getattr(self.vlm_provider, "_model", "")
            if current in TEXT_ONLY:
                try:
                    from clawphd.config.loader import load_config
                    cfg = load_config()
                    override = cfg.tools.autofigure.vlm_model
                except Exception:
                    override = ""
                model = override or DEFAULT
                return OpenRouterVLM(
                    api_key=self.vlm_provider._api_key,
                    model=model,
                    api_base=self.vlm_provider._api_base,
                )

        return self.vlm_provider

    async def spawn(
        self,
        task: str,
        label: str | None = None,
        origin_channel: str = "cli",
        origin_chat_id: str = "direct",
        session_key: str | None = None,
    ) -> str:
        """Spawn a subagent to execute a task in the background."""
        task_id = str(uuid.uuid4())[:8]
        display_label = label or task[:30] + ("..." if len(task) > 30 else "")
        origin = {"channel": origin_channel, "chat_id": origin_chat_id}

        bg_task = asyncio.create_task(
            self._run_subagent(task_id, task, display_label, origin)
        )
        self._running_tasks[task_id] = bg_task
        if session_key:
            self._session_tasks.setdefault(session_key, set()).add(task_id)

        def _cleanup(_: asyncio.Task) -> None:
            self._running_tasks.pop(task_id, None)
            if session_key and (ids := self._session_tasks.get(session_key)):
                ids.discard(task_id)
                if not ids:
                    del self._session_tasks[session_key]

        bg_task.add_done_callback(_cleanup)

        logger.info("Spawned subagent [{}]: {}", task_id, display_label)
        return f"Subagent [{display_label}] started (id: {task_id}). I'll notify you when it completes."

    async def _run_subagent(
        self,
        task_id: str,
        task: str,
        label: str,
        origin: dict[str, str],
    ) -> None:
        """Execute the subagent task and announce the result."""
        logger.info("Subagent [{}] starting task: {}", task_id, label)

        try:
            # Build subagent tools (no message tool, no spawn tool)
            tools = ToolRegistry()
            allowed_dir = self.workspace if self.restrict_to_workspace else None
            tools.register(ReadFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(WriteFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(EditFileTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(ListDirTool(workspace=self.workspace, allowed_dir=allowed_dir))
            tools.register(ExecTool(
                working_dir=str(self.workspace),
                timeout=self.exec_config.timeout,
                restrict_to_workspace=self.restrict_to_workspace,
                path_append=self.exec_config.path_append,
            ))
            tools.register(WebSearchTool(api_key=self.brave_api_key, proxy=self.web_proxy))
            tools.register(WebFetchTool(proxy=self.web_proxy))
            self._register_clawphd_tools(tools, allowed_dir)
            
            system_prompt = self._build_subagent_prompt()
            messages: list[dict[str, Any]] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": task},
            ]

            # Run agent loop (limited iterations)
            max_iterations = 15
            iteration = 0
            final_result: str | None = None

            while iteration < max_iterations:
                iteration += 1

                response = await self.provider.chat(
                    messages=messages,
                    tools=tools.get_definitions(),
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    reasoning_effort=self.reasoning_effort,
                )

                if response.has_tool_calls:
                    # Add assistant message with tool calls
                    tool_call_dicts = [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.name,
                                "arguments": json.dumps(tc.arguments, ensure_ascii=False),
                            },
                        }
                        for tc in response.tool_calls
                    ]
                    messages.append({
                        "role": "assistant",
                        "content": response.content or "",
                        "tool_calls": tool_call_dicts,
                    })

                    # Execute tools
                    for tool_call in response.tool_calls:
                        args_str = json.dumps(tool_call.arguments, ensure_ascii=False)
                        logger.debug("Subagent [{}] executing: {} with arguments: {}", task_id, tool_call.name, args_str)
                        result = await tools.execute(tool_call.name, tool_call.arguments)
                        messages.append({
                            "role": "tool",
                            "tool_call_id": tool_call.id,
                            "name": tool_call.name,
                            "content": result,
                        })
                else:
                    final_result = response.content
                    break

            if final_result is None:
                final_result = "Task completed but no final response was generated."

            logger.info("Subagent [{}] completed successfully", task_id)
            await self._announce_result(task_id, label, task, final_result, origin, "ok")

        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error("Subagent [{}] failed: {}", task_id, e)
            await self._announce_result(task_id, label, task, error_msg, origin, "error")

    async def _announce_result(
        self,
        task_id: str,
        label: str,
        task: str,
        result: str,
        origin: dict[str, str],
        status: str,
    ) -> None:
        """Announce the subagent result to the main agent via the message bus."""
        status_text = "completed successfully" if status == "ok" else "failed"

        announce_content = f"""[Subagent '{label}' {status_text}]

Task: {task}

Result:
{result}

Summarize this naturally for the user. Keep it brief (1-2 sentences). Do not mention technical details like "subagent" or task IDs."""

        # Inject as system message to trigger main agent
        msg = InboundMessage(
            channel="system",
            sender_id="subagent",
            chat_id=f"{origin['channel']}:{origin['chat_id']}",
            content=announce_content,
        )

        await self.bus.publish_inbound(msg)
        logger.debug("Subagent [{}] announced result to {}:{}", task_id, origin['channel'], origin['chat_id'])
    
    def _build_subagent_prompt(self) -> str:
        """Build a focused system prompt for the subagent."""
        from clawphd.agent.context import ContextBuilder
        from clawphd.agent.skills import SkillsLoader

        time_ctx = ContextBuilder._build_runtime_context(None, None)
        parts = [f"""# Subagent

{time_ctx}

You are a subagent spawned by the main agent to complete a specific task.
Stay focused on the assigned task. Your final response will be reported back to the main agent.

## Workspace
{self.workspace}"""]

        skills_summary = SkillsLoader(self.workspace).build_skills_summary()
        if skills_summary:
            parts.append(f"## Skills\n\nRead SKILL.md with read_file to use a skill.\n\n{skills_summary}")

        return "\n\n".join(parts)
    
    async def cancel_by_session(self, session_key: str) -> int:
        """Cancel all subagents for the given session. Returns count cancelled."""
        tasks = [self._running_tasks[tid] for tid in self._session_tasks.get(session_key, [])
                 if tid in self._running_tasks and not self._running_tasks[tid].done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        return len(tasks)

    def get_running_count(self) -> int:
        """Return the number of currently running subagents."""
        return len(self._running_tasks)
