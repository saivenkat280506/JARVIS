"""LangGraph-based Execution Runtime for JARVIS.

Implements an explicit state-graph workflow for conversational turns and tool dispatch,
integrating with the existing LLMClientProtocol, ToolRegistry, ActionApprovalManager,
ActionLedger, and SQLite storage engines.
"""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, TypedDict
import uuid

from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.base import BaseCheckpointSaver

from agent.ollama_client import LLMClientProtocol, OllamaClient, OllamaError
from agent.runtime import (
    AgentRunResult,
    ToolExecutionRecord,
    extract_action_metadata,
)
from agent.tool_registry import ToolRegistry, create_default_registry
from config.settings import settings
from security.approval import (
    ActionApprovalManager,
    ApprovalStatus,
    classify_confirmation_intent,
    default_approval_manager,
)
from storage.conversation import (
    ConversationStorage,
    default_conversation_storage,
)
from storage.tasks import (
    TaskRecord,
    TaskStatus,
    TaskStorage,
    default_task_storage,
)
from storage.action_ledger import (
    ActionLedger,
    ActionLedgerConflictError,
    ActionLedgerStatus,
    ActionUncertaintyError,
    RetrySafety,
    default_action_ledger,
    normalize_and_hash_arguments,
)
from storage.checkpoint import (
    CheckpointStorage,
    default_checkpoint_storage,
)

logger = logging.getLogger(__name__)


class AgentGraphState(TypedDict):
    """Explicit state representation for a single LangGraph agent conversational turn."""

    prompt: str
    conversation_id: Optional[str]
    task_id: Optional[str]
    options: Optional[Dict[str, Any]]
    model: Optional[str]
    max_rounds: int
    rounds_used: int
    messages: List[Dict[str, Any]]
    turn_new_messages: List[Dict[str, Any]]
    current_tool_calls: List[Dict[str, Any]]
    tool_records: List[ToolExecutionRecord]
    final_response: str
    status: str
    error_message: Optional[str]
    completed: bool
    requires_approval: bool
    pending_action_id: Optional[str]


class LangGraphAgentRuntime:
    """Graph-based agent runtime coordinating conversation history, LLM interaction,
    and bounded tool dispatch using LangGraph.
    """

    DEFAULT_SYSTEM_PROMPT = (
        "You are J.A.R.V.I.S., a secure, fully local AI voice assistant. "
        "Keep internal thinking minimal and brief (under 20 words). "
        "Provide a direct, natural spoken response in one concise sentence. "
        "Never recite raw identifiers, technical codes, UTC offsets, Markdown formatting, or JSON. "
        "When asked for current time, calculations, playing/pausing/resuming/stopping music, "
        "opening websites, interacting with or observing web pages, or creating/listing/cancelling scheduled reminders, "
        "you must invoke the appropriate tool immediately. "
        "Accurately reflect tool outcomes: if a tool succeeds, confirm the action in one brief sentence. "
        "Do not deliberate on internal technical notes or caveats. "
        "If a tool reports an error or failure, clearly report the failure; "
        "never claim success unless the tool result has success: true. "
        "If an action requires user confirmation (such as consequential form submissions or purchases), ask the user for approval. "
        "If a reminder time is ambiguous (such as 'remind me later'), ask the user for clarification."
    )

    def __init__(
        self,
        client: Optional[LLMClientProtocol] = None,
        registry: Optional[ToolRegistry] = None,
        storage: Optional[ConversationStorage] = None,
        conversation_storage: Optional[ConversationStorage] = None,
        approval_manager: Optional[ActionApprovalManager] = None,
        task_storage: Optional[TaskStorage] = None,
        action_ledger: Optional[ActionLedger] = None,
        max_rounds: int = 5,
        system_prompt: Optional[str] = None,
        model: Optional[str] = None,
        max_history_turns: Optional[int] = None,
        max_history_chars: Optional[int] = None,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        checkpoint_storage: Optional[CheckpointStorage] = None,
        checkpoint_db_path: Optional[Any] = None,
        enable_checkpointing: Optional[bool] = None,
        interrupt_before: Optional[List[str]] = None,
    ) -> None:
        self.client: LLMClientProtocol = client or OllamaClient()
        self.registry = registry or create_default_registry()
        self.storage = conversation_storage or storage or default_conversation_storage
        self.approval_manager = approval_manager or default_approval_manager
        self.task_storage = task_storage or default_task_storage
        self.action_ledger = action_ledger or default_action_ledger
        self.max_rounds = max(1, max_rounds)
        self.system_prompt = (
            system_prompt if system_prompt is not None else self.DEFAULT_SYSTEM_PROMPT
        )
        self.model = model
        self.max_history_turns = (
            max_history_turns
            if max_history_turns is not None
            else settings.conversation_max_turns
        )
        self.max_history_chars = (
            max_history_chars
            if max_history_chars is not None
            else settings.conversation_max_chars
        )
        self.interrupt_before = interrupt_before

        # Checkpoint persistence configuration
        if enable_checkpointing is None:
            self.enable_checkpointing = getattr(settings, "enable_checkpoints", True)
        else:
            self.enable_checkpointing = bool(enable_checkpointing)

        self.checkpointer: Optional[BaseCheckpointSaver] = None
        self._checkpoint_storage: Optional[CheckpointStorage] = None

        if self.enable_checkpointing:
            if checkpointer is not None:
                self.checkpointer = checkpointer
                self._checkpoint_storage = checkpoint_storage
            elif checkpoint_storage is not None:
                self._checkpoint_storage = checkpoint_storage
            elif checkpoint_db_path is not None:
                self._checkpoint_storage = CheckpointStorage(checkpoint_db_path)
            else:
                self._checkpoint_storage = default_checkpoint_storage

        self._graph = self._build_graph(checkpointer=self.checkpointer, interrupt_before=self.interrupt_before)

    async def _ensure_active_graph(self) -> None:
        """Ensure checkpointer connection is initialized and graph is compiled with checkpointer."""
        if not self.enable_checkpointing:
            return
        if self._checkpoint_storage is not None:
            active_saver = await self._checkpoint_storage.get_or_create_saver()
            if self.checkpointer is not active_saver:
                self.checkpointer = active_saver
                self._graph = self._build_graph(checkpointer=self.checkpointer, interrupt_before=self.interrupt_before)

    @property
    def graph(self):
        """Return the underlying compiled LangGraph instance."""
        return self._graph

    # -------------------------------------------------------------------------
    # Graph Nodes
    # -------------------------------------------------------------------------

    async def _node_prepare_input(self, state: AgentGraphState) -> Dict[str, Any]:
        """Node 1: Prepare model input, resolve pending approvals, and initialize turn state."""
        prompt = state["prompt"]
        conversation_id = state["conversation_id"]
        task_id = state["task_id"]

        # Check if user is responding to a pending consequential action approval request
        latest_pending = None
        if conversation_id:
            waiting_tasks = self.task_storage.list_tasks(
                status=TaskStatus.WAITING_FOR_APPROVAL,
                conversation_id=conversation_id,
                limit=10,
            )
            for wt in waiting_tasks:
                p_act_id = wt.metadata.get("pending_action_id")
                if p_act_id:
                    pa = self.approval_manager.get_pending_action(p_act_id)
                    if pa and pa.status == ApprovalStatus.PENDING and not pa.is_expired():
                        latest_pending = pa
                        task_id = wt.id
                        break
        elif task_id:
            existing_task = self.task_storage.get_task(task_id)
            if existing_task and existing_task.status == TaskStatus.WAITING_FOR_APPROVAL:
                p_act_id = existing_task.metadata.get("pending_action_id")
                if p_act_id:
                    pa = self.approval_manager.get_pending_action(p_act_id)
                    if pa and pa.status == ApprovalStatus.PENDING and not pa.is_expired():
                        latest_pending = pa
        else:
            cand = self.approval_manager.get_latest_pending()
            if cand and cand.status == ApprovalStatus.PENDING:
                waiting_tasks = self.task_storage.list_tasks(
                    status=TaskStatus.WAITING_FOR_APPROVAL, limit=10
                )
                matching = next(
                    (wt for wt in waiting_tasks if wt.metadata.get("pending_action_id") == cand.action_id),
                    None,
                )
                if not matching or not matching.conversation_id:
                    latest_pending = cand
                    if matching:
                        task_id = matching.id

        # Initialize or retrieve persistent task
        if task_id:
            existing_task = self.task_storage.get_task(task_id)
            if existing_task and existing_task.status == TaskStatus.QUEUED:
                self.task_storage.update_task_status(
                    task_id, TaskStatus.RUNNING, status_message="Executing turn"
                )
        else:
            task = self.task_storage.create_task(
                task_type="agent_turn",
                description=f"Turn: {prompt[:60]}",
                conversation_id=conversation_id,
                status=TaskStatus.RUNNING,
            )
            task_id = task.id

        # Handle pending confirmation response
        if latest_pending and latest_pending.status == ApprovalStatus.PENDING:
            intent, conf = classify_confirmation_intent(prompt)
            if intent == "affirm" and conf >= 0.8:
                if task_id:
                    cur_t = self.task_storage.get_task(task_id)
                    if cur_t and cur_t.status == TaskStatus.WAITING_FOR_APPROVAL:
                        self.task_storage.update_task_status(
                            task_id, TaskStatus.RUNNING, status_message="Executing confirmed action"
                        )
                attempt_rec = None
                try:
                    _, args_hash = normalize_and_hash_arguments(latest_pending.arguments)
                    action_type = (
                        latest_pending.arguments.get("action", "action")
                        if isinstance(latest_pending.arguments, dict)
                        else "action"
                    )
                    logical_step = f"{latest_pending.tool_name}_{action_type}_{args_hash[:12]}"
                    action_meta = extract_action_metadata(
                        latest_pending.tool_name,
                        latest_pending.arguments,
                        conversation_id=conversation_id,
                    )
                    attempt_rec, is_cached = self.action_ledger.prepare_action(
                        task_id=task_id,
                        step_name=logical_step,
                        tool_name=latest_pending.tool_name,
                        arguments=latest_pending.arguments,
                        action_id=latest_pending.action_id,
                        retry_safety=RetrySafety.NON_IDEMPOTENT,
                        initial_status=ActionLedgerStatus.APPROVED,
                        metadata=action_meta,
                    )
                except ActionUncertaintyError as ue:
                    err_msg = f"Action blocked: {ue}"
                    return {
                        "task_id": task_id,
                        "final_response": err_msg,
                        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": err_msg}],
                        "turn_new_messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": err_msg}],
                        "status": "action_uncertainty_blocked",
                        "error_message": str(ue),
                        "completed": False,
                        "rounds_used": 1,
                    }
                except ActionLedgerConflictError as ce:
                    err_msg = f"Action conflict: {ce}"
                    return {
                        "task_id": task_id,
                        "final_response": err_msg,
                        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": err_msg}],
                        "turn_new_messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": err_msg}],
                        "status": "action_conflict_blocked",
                        "error_message": str(ce),
                        "completed": False,
                        "rounds_used": 1,
                    }

                if is_cached and attempt_rec:
                    final_msg = f"Action was already executed and verified (Attempt #{attempt_rec.attempt_number})."
                    return {
                        "task_id": task_id,
                        "final_response": final_msg,
                        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": final_msg}],
                        "turn_new_messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": final_msg}],
                        "status": "success",
                        "completed": True,
                        "rounds_used": 1,
                    }

                ok, msg, token = self.approval_manager.approve_action(
                    latest_pending.action_id, channel="trusted_user_interaction"
                )
                if ok and token:
                    if attempt_rec:
                        self.action_ledger.update_status(attempt_rec.id, ActionLedgerStatus.EXECUTING)
                    t0 = time.perf_counter()
                    exec_res = await self.registry.execute_async(
                        latest_pending.tool_name,
                        {
                            **latest_pending.arguments,
                            "action_id": latest_pending.action_id,
                            "approval_token": token,
                        },
                    )
                    dur = time.perf_counter() - t0
                    rec = ToolExecutionRecord(
                        round_index=1,
                        tool_name=latest_pending.tool_name,
                        arguments=latest_pending.arguments,
                        result=exec_res,
                        duration_sec=round(dur, 4),
                    )
                    exec_payload = exec_res.get("result") if isinstance(exec_res.get("result"), dict) else exec_res
                    success_flag = bool(exec_payload.get("success", exec_res.get("success")))

                    if attempt_rec:
                        if success_flag:
                            self.action_ledger.update_status(
                                attempt_rec.id,
                                ActionLedgerStatus.SUCCEEDED,
                                status_message=exec_payload.get("message") or "Action succeeded",
                            )
                        else:
                            self.action_ledger.update_status(
                                attempt_rec.id,
                                ActionLedgerStatus.FAILED,
                                error_message=exec_payload.get("error") or "Action failed",
                            )

                    self.approval_manager.record_execution_outcome(
                        action_id=latest_pending.action_id,
                        tool_name=latest_pending.tool_name,
                        success=success_flag,
                        details=exec_payload.get("message") or exec_payload.get("error") or ("Action completed" if success_flag else "Action failed"),
                    )

                    if success_flag:
                        summary = exec_payload.get("message") or exec_res.get("message") or f"Action '{latest_pending.action_type}' completed successfully."
                        final_msg = f"Confirmed and executed: {summary}"
                        return {
                            "task_id": task_id,
                            "tool_records": [rec],
                            "final_response": final_msg,
                            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": final_msg}],
                            "turn_new_messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": final_msg}],
                            "status": "success",
                            "completed": True,
                            "rounds_used": 1,
                        }
                    else:
                        err_det = exec_res.get("error") or exec_res.get("message") or "Action execution failed."
                        fail_msg = f"Confirmed, but action could not be completed: {err_det}"
                        return {
                            "task_id": task_id,
                            "tool_records": [rec],
                            "final_response": fail_msg,
                            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": fail_msg}],
                            "turn_new_messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": fail_msg}],
                            "status": "error",
                            "error_message": err_det,
                            "completed": True,
                            "rounds_used": 1,
                        }

            elif intent == "reject" and conf >= 0.8:
                self.approval_manager.reject_action(
                    latest_pending.action_id,
                    channel="trusted_user_interaction",
                    reason=f"User stated: '{prompt}'",
                )
                _, args_hash = normalize_and_hash_arguments(latest_pending.arguments)
                action_type = (
                    latest_pending.arguments.get("action", "action")
                    if isinstance(latest_pending.arguments, dict)
                    else "action"
                )
                logical_step = f"{latest_pending.tool_name}_{action_type}_{args_hash[:12]}"
                latest_attempt = self.action_ledger.get_latest_attempt(f"{task_id}:{logical_step}")
                if latest_attempt and not latest_attempt.is_terminal():
                    self.action_ledger.update_status(
                        latest_attempt.id,
                        ActionLedgerStatus.CANCELLED,
                        status_message="User rejected confirmation",
                    )
                final_msg = "Understood. The action has been cancelled."
                if task_id:
                    self.task_storage.update_task_status(
                        task_id,
                        TaskStatus.CANCELLED,
                        status_message=final_msg,
                    )
                return {
                    "task_id": task_id,
                    "final_response": final_msg,
                    "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": final_msg}],
                    "turn_new_messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": final_msg}],
                    "status": "success",
                    "completed": True,
                    "rounds_used": 1,
                }
            else:
                # Ambiguous confirmation input: fail-closed
                pending_desc = latest_pending.description or f"{latest_pending.tool_name} action"
                final_msg = (
                    f"A consequential action is awaiting your explicit authorization: {pending_desc}. "
                    "Ambient speech or ambiguous responses cannot authorize actions. "
                    "Please state 'confirm' to proceed or 'cancel' to deny."
                )
                return {
                    "task_id": task_id,
                    "final_response": final_msg,
                    "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": final_msg}],
                    "turn_new_messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": final_msg}],
                    "status": "awaiting_approval_ambiguous",
                    "error_message": "Ambiguous input while consequential action awaits approval.",
                    "completed": False,
                    "rounds_used": 1,
                }
        else:
            # Check if user is repeating an affirmation/rejection for an action that was already executed or uncertain
            intent, conf = classify_confirmation_intent(prompt)
            latest_action = None
            if conversation_id:
                recent_tasks = self.task_storage.list_tasks(conversation_id=conversation_id, limit=10)
                for t in recent_tasks:
                    p_act_id = t.metadata.get("pending_action_id")
                    if p_act_id:
                        pa = self.approval_manager.get_pending_action(p_act_id)
                        if pa:
                            latest_action = pa
                            break
            elif task_id:
                t = self.task_storage.get_task(task_id)
                if t:
                    p_act_id = t.metadata.get("pending_action_id")
                    if p_act_id:
                        latest_action = self.approval_manager.get_pending_action(p_act_id)
            else:
                latest_action = self.approval_manager.get_latest_action()

            if intent == "affirm" and conf >= 0.8:
                if latest_action:
                    att_rec = self.action_ledger.get_attempt_by_action_id(latest_action.action_id)
                    if att_rec and att_rec.status == ActionLedgerStatus.SUCCEEDED:
                        final_msg = f"Action was already executed and verified (Attempt #{att_rec.attempt_number})."
                        return {
                            "task_id": task_id,
                            "final_response": final_msg,
                            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": final_msg}],
                            "turn_new_messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": final_msg}],
                            "status": "success",
                            "completed": True,
                            "rounds_used": 1,
                        }
                    elif att_rec and att_rec.status == ActionLedgerStatus.OUTCOME_UNKNOWN:
                        err_msg = f"Action '{latest_action.action_type}' has unknown outcome; cannot retry without reconciliation or explicit user decision."
                        return {
                            "task_id": task_id,
                            "final_response": err_msg,
                            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": err_msg}],
                            "turn_new_messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": err_msg}],
                            "status": "action_uncertainty_blocked",
                            "error_message": err_msg,
                            "completed": False,
                            "rounds_used": 1,
                        }
            elif intent == "reject" and conf >= 0.8:
                if latest_action:
                    att_rec = self.action_ledger.get_attempt_by_action_id(latest_action.action_id)
                    if att_rec and att_rec.status == ActionLedgerStatus.SUCCEEDED:
                        final_msg = f"Action was already executed and verified (Attempt #{att_rec.attempt_number}); cannot be cancelled."
                        return {
                            "task_id": task_id,
                            "final_response": final_msg,
                            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": final_msg}],
                            "turn_new_messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": final_msg}],
                            "status": "success",
                            "completed": True,
                            "rounds_used": 1,
                        }
                    elif att_rec and att_rec.status == ActionLedgerStatus.OUTCOME_UNKNOWN:
                        final_msg = (
                            f"Execution cancelled for future attempts. However, action '{latest_action.action_type}' "
                            f"(attempt #{att_rec.attempt_number}) has an unknown outcome and may have already been "
                            "submitted or taken effect externally; local cancellation cannot reverse external actions. "
                            f"Use '--reconcile {att_rec.id}' or verify external state directly."
                        )
                        if task_id:
                            self.task_storage.update_task_status(
                                task_id,
                                TaskStatus.CANCELLED,
                                status_message="Local execution cancelled; external outcome remains unknown",
                            )
                        return {
                            "task_id": task_id,
                            "final_response": final_msg,
                            "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": final_msg}],
                            "turn_new_messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": final_msg}],
                            "status": "success",
                            "completed": True,
                            "rounds_used": 1,
                        }
            elif latest_action:
                att_rec = self.action_ledger.get_attempt_by_action_id(latest_action.action_id)
                if att_rec and att_rec.status == ActionLedgerStatus.OUTCOME_UNKNOWN:
                    err_msg = (
                        f"Action '{latest_action.action_type}' (attempt #{att_rec.attempt_number}) has an unknown outcome. "
                        f"Your response '{prompt}' is ambiguous. Automatic replay is prohibited. "
                        f"Please specify if you wish to reconcile/verify external state, or stop local execution."
                    )
                    return {
                        "task_id": task_id,
                        "final_response": err_msg,
                        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": err_msg}],
                        "turn_new_messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": err_msg}],
                        "status": "action_uncertainty_blocked",
                        "error_message": err_msg,
                        "completed": False,
                        "rounds_used": 1,
                    }

        # Build initial messages thread
        messages: List[Dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})

        if conversation_id:
            history_msgs = self.storage.get_bounded_history(
                conversation_id=conversation_id,
                max_turns=self.max_history_turns,
                max_chars=self.max_history_chars,
            )
            messages.extend(history_msgs)

        messages.append({"role": "user", "content": prompt})

        return {
            "task_id": task_id,
            "messages": messages,
            "turn_new_messages": [{"role": "user", "content": prompt}],
            "status": "running",
            "completed": False,
            "rounds_used": 0,
        }

    async def _node_call_model(self, state: AgentGraphState) -> Dict[str, Any]:
        """Node 2: Call the LLMClientProtocol model with tool schemas."""
        rounds_consumed = state["rounds_used"] + 1
        tool_schemas = self.registry.get_schemas()

        try:
            response = await self.client.chat(
                messages=state["messages"],
                tools=tool_schemas,
                options=state.get("options"),
                model=state.get("model") or self.model,
            )
        except OllamaError as exc:
            return {
                "rounds_used": rounds_consumed,
                "status": "error",
                "error_message": str(exc),
                "completed": False,
            }
        except Exception as exc:
            return {
                "rounds_used": rounds_consumed,
                "status": "error",
                "error_message": f"Unexpected runtime failure: {exc}",
                "completed": False,
            }

        assistant_msg = response.get("message", {})
        updated_messages = list(state["messages"]) + [assistant_msg]
        updated_turn_msgs = list(state.get("turn_new_messages") or []) + [assistant_msg]
        tool_calls = assistant_msg.get("tool_calls") or []

        if not tool_calls:
            final_content = assistant_msg.get("content", "").strip()
            done_reason = response.get("done_reason")

            if done_reason == "length":
                return {
                    "rounds_used": rounds_consumed,
                    "messages": updated_messages,
                    "turn_new_messages": updated_turn_msgs,
                    "final_response": final_content or "I was unable to complete my response within the token limit.",
                    "status": "truncated",
                    "error_message": "Generation truncated by token limit before response completed.",
                    "completed": False,
                    "current_tool_calls": [],
                }

            if not final_content and state.get("tool_records"):
                return {
                    "rounds_used": rounds_consumed,
                    "messages": updated_messages,
                    "turn_new_messages": updated_turn_msgs,
                    "status": "empty_response",
                    "error_message": "Model returned empty response following tool execution.",
                    "completed": False,
                    "current_tool_calls": [],
                }

            return {
                "rounds_used": rounds_consumed,
                "messages": updated_messages,
                "turn_new_messages": updated_turn_msgs,
                "final_response": final_content,
                "status": "success",
                "completed": True,
                "current_tool_calls": [],
            }

        return {
            "rounds_used": rounds_consumed,
            "messages": updated_messages,
            "turn_new_messages": updated_turn_msgs,
            "current_tool_calls": tool_calls,
        }

    async def _node_execute_tools(self, state: AgentGraphState) -> Dict[str, Any]:
        """Node 3: Execute tool calls through ToolRegistry, preserving approvals and ledger."""
        tool_calls = state.get("current_tool_calls") or []
        task_id = state.get("task_id")
        conversation_id = state.get("conversation_id")
        round_idx = state["rounds_used"]

        tool_records = list(state.get("tool_records") or [])
        messages = list(state["messages"])
        turn_new_messages = list(state.get("turn_new_messages") or [])
        requires_approval = False

        for call in tool_calls:
            fn_info = call.get("function", {})
            fn_name = fn_info.get("name", "").strip()
            raw_args = fn_info.get("arguments", {})
            call_id = call.get("id")

            if isinstance(raw_args, str):
                try:
                    parsed_args = json.loads(raw_args) if raw_args.strip() else {}
                except json.JSONDecodeError:
                    parsed_args = {}
            elif isinstance(raw_args, dict):
                parsed_args = raw_args
            else:
                parsed_args = {}

            # Check if this tool call was already executed (e.g. before interruption or during replay)
            already_recorded: Optional[ToolExecutionRecord] = None
            if call_id:
                already_recorded = next(
                    (r for r in tool_records if r.tool_call_id == call_id), None
                )
            if not already_recorded:
                already_recorded = next(
                    (
                        r
                        for r in tool_records
                        if r.round_index == round_idx
                        and r.tool_name == fn_name
                        and r.arguments == parsed_args
                    ),
                    None,
                )

            # Check ActionLedger for existing execution / outcome unknown
            ledger_cached_result = None
            if task_id and isinstance(parsed_args, dict):
                try:
                    _, args_hash = normalize_and_hash_arguments(parsed_args)
                    action_type = parsed_args.get("action", "action")
                    logical_step = f"{fn_name}_{action_type}_{args_hash[:12]}"
                    idemp_key = f"{task_id}:{logical_step}"
                    latest_att = self.action_ledger.get_latest_attempt(idemp_key)
                    if latest_att:
                        if latest_att.status == ActionLedgerStatus.SUCCEEDED:
                            ledger_cached_result = {
                                "success": True,
                                "message": latest_att.status_message or "Action already executed successfully",
                                "cached": True,
                            }
                        elif latest_att.status == ActionLedgerStatus.OUTCOME_UNKNOWN:
                            err_msg = (
                                f"Action '{fn_name}' has an unknown outcome in the action ledger. "
                                "Automatic replay is prohibited upon resumption. Please reconcile external state."
                            )
                            return {
                                "status": "action_uncertainty_blocked",
                                "error_message": err_msg,
                                "final_response": err_msg,
                                "completed": False,
                                "messages": messages,
                                "turn_new_messages": turn_new_messages,
                                "tool_records": tool_records,
                                "current_tool_calls": [],
                            }
                except Exception as exc:
                    logger.debug("Ledger check exception in execute_tools: %s", exc)

            if already_recorded is not None:
                execution_result = already_recorded.result
                duration = already_recorded.duration_sec
            elif ledger_cached_result is not None:
                execution_result = ledger_cached_result
                duration = 0.0
                tool_records.append(
                    ToolExecutionRecord(
                        round_index=round_idx,
                        tool_name=fn_name,
                        tool_call_id=call_id,
                        arguments=parsed_args,
                        result=execution_result,
                        duration_sec=duration,
                    )
                )
            else:
                t0 = time.perf_counter()
                execution_result = await self.registry.execute_async(fn_name, parsed_args)
                duration = time.perf_counter() - t0

                # Enforce consequential action approval boundary
                payload = (
                    execution_result.get("result")
                    if isinstance(execution_result.get("result"), dict)
                    else execution_result
                )
                if payload.get("requires_confirmation") or execution_result.get("requires_confirmation"):
                    act_id = payload.get("action_id") or execution_result.get("action_id")
                    reason = (
                        payload.get("consequential_reason")
                        or execution_result.get("consequential_reason")
                        or "Confirmation required"
                    )
                    requires_approval = True

                    if task_id:
                        current_t = self.task_storage.get_task(task_id)
                        if current_t and current_t.status != TaskStatus.WAITING_FOR_APPROVAL:
                            self.task_storage.update_task_status(
                                task_id,
                                TaskStatus.WAITING_FOR_APPROVAL,
                                status_message=f"Waiting for approval: {reason} (action: {act_id})",
                                metadata_updates={
                                    "pending_action_id": act_id,
                                    "conversation_id": conversation_id,
                                },
                            )

                    # Prepare in action ledger
                    try:
                        _, args_hash = normalize_and_hash_arguments(parsed_args)
                        action_type = (
                            parsed_args.get("action", "action")
                            if isinstance(parsed_args, dict)
                            else "action"
                        )
                        logical_step = f"{fn_name}_{action_type}_{args_hash[:12]}"
                        action_meta = extract_action_metadata(
                            fn_name, parsed_args, conversation_id=conversation_id
                        )
                        self.action_ledger.prepare_action(
                            task_id=task_id,
                            step_name=logical_step,
                            tool_name=fn_name,
                            arguments=parsed_args,
                            action_id=act_id,
                            retry_safety=RetrySafety.NON_IDEMPOTENT,
                            initial_status=ActionLedgerStatus.AWAITING_APPROVAL,
                            metadata=action_meta,
                        )
                    except (ActionLedgerConflictError, ActionUncertaintyError):
                        pass

                tool_records.append(
                    ToolExecutionRecord(
                        round_index=round_idx,
                        tool_name=fn_name,
                        tool_call_id=call_id,
                        arguments=parsed_args,
                        result=execution_result,
                        duration_sec=round(duration, 4),
                    )
                )

            tool_msg_exists = False
            if call_id:
                tool_msg_exists = any(
                    m.get("role") == "tool" and m.get("tool_call_id") == call_id
                    for m in messages
                )
            if not tool_msg_exists:
                tool_msg: Dict[str, Any] = {
                    "role": "tool",
                    "content": json.dumps(execution_result),
                }
                if call_id:
                    tool_msg["tool_call_id"] = call_id
                if fn_name:
                    tool_msg["name"] = fn_name

                messages.append(tool_msg)
                turn_new_messages.append(tool_msg)

        updates: Dict[str, Any] = {
            "messages": messages,
            "turn_new_messages": turn_new_messages,
            "tool_records": tool_records,
            "current_tool_calls": [],
        }

        if requires_approval:
            updates["requires_approval"] = True
            updates["status"] = "waiting_for_approval"
            updates["completed"] = False
            for rec in tool_records:
                res = rec.result.get("result") if isinstance(rec.result.get("result"), dict) else rec.result
                if res.get("requires_confirmation") or rec.result.get("requires_confirmation"):
                    c_reason = res.get("consequential_reason") or rec.result.get("consequential_reason") or "Confirmation required"
                    updates["final_response"] = f"A consequential action requires your explicit approval: {c_reason}."
                    break

        return updates

    async def _node_handle_limit(self, state: AgentGraphState) -> Dict[str, Any]:
        """Node 4: Handle safe termination when execution round limit is reached."""
        return {
            "final_response": "Maximum tool execution rounds reached without final answer.",
            "status": "max_rounds_reached",
            "completed": False,
        }

    async def _node_finalize(self, state: AgentGraphState) -> Dict[str, Any]:
        """Node 5: Atomically persist conversation turns and synchronize task statuses."""
        task_id = state.get("task_id")
        conversation_id = state.get("conversation_id")
        status = state.get("status", "pending")
        final_content = state.get("final_response", "")

        # Persist conversation turn if successful or awaiting approval ambiguous
        if conversation_id and status in ("success", "awaiting_approval_ambiguous"):
            turn_msgs = state.get("turn_new_messages") or []
            if turn_msgs:
                self.storage.append_turn(conversation_id, turn_msgs)

        # Update persistent task status
        if task_id:
            cur_task = self.task_storage.get_task(task_id)
            if cur_task:
                if status == "success" and cur_task.status not in (TaskStatus.WAITING_FOR_APPROVAL, TaskStatus.CANCELLED):
                    self.task_storage.update_task_status(
                        task_id,
                        TaskStatus.SUCCEEDED,
                        status_message=final_content or "Turn completed successfully",
                    )
                elif status in ("error", "truncated", "empty_response", "max_rounds_reached"):
                    self.task_storage.update_task_status(
                        task_id,
                        TaskStatus.FAILED,
                        error_message=state.get("error_message") or final_content or status,
                    )

        return {}

    # -------------------------------------------------------------------------
    # Routing Logic (Conditional Edges)
    # -------------------------------------------------------------------------

    def _route_after_prepare(self, state: AgentGraphState) -> str:
        """Route conditionally after input preparation."""
        if state.get("completed") or state.get("status") in (
            "success",
            "action_uncertainty_blocked",
            "action_conflict_blocked",
            "awaiting_approval_ambiguous",
            "error",
        ):
            return "finalize"
        return "call_model"

    def _route_after_model(self, state: AgentGraphState) -> str:
        """Route conditionally after model inference."""
        status = state.get("status")
        if state.get("completed") or status in ("success", "truncated", "empty_response", "error"):
            return "finalize"

        if state.get("current_tool_calls"):
            if state["rounds_used"] >= state["max_rounds"]:
                return "handle_limit"
            return "execute_tools"

        return "finalize"

    def _route_after_tools(self, state: AgentGraphState) -> str:
        """Route conditionally after tool execution."""
        if (
            state.get("requires_approval")
            or state.get("status") in ("waiting_for_approval", "action_uncertainty_blocked", "error")
            or state.get("completed")
        ):
            return "finalize"

        if state["rounds_used"] >= state["max_rounds"]:
            return "handle_limit"

        return "call_model"

    # -------------------------------------------------------------------------
    # Graph Construction
    # -------------------------------------------------------------------------

    def _build_graph(
        self,
        checkpointer: Optional[BaseCheckpointSaver] = None,
        interrupt_before: Optional[List[str]] = None,
    ):
        """Construct and compile the LangGraph workflow."""
        builder = StateGraph(AgentGraphState)

        builder.add_node("prepare_input", self._node_prepare_input)
        builder.add_node("call_model", self._node_call_model)
        builder.add_node("execute_tools", self._node_execute_tools)
        builder.add_node("handle_limit", self._node_handle_limit)
        builder.add_node("finalize", self._node_finalize)

        builder.add_edge(START, "prepare_input")

        builder.add_conditional_edges(
            "prepare_input",
            self._route_after_prepare,
            {"finalize": "finalize", "call_model": "call_model"},
        )

        builder.add_conditional_edges(
            "call_model",
            self._route_after_model,
            {
                "finalize": "finalize",
                "execute_tools": "execute_tools",
                "handle_limit": "handle_limit",
            },
        )

        builder.add_conditional_edges(
            "execute_tools",
            self._route_after_tools,
            {
                "finalize": "finalize",
                "call_model": "call_model",
                "handle_limit": "handle_limit",
            },
        )

        builder.add_edge("handle_limit", "finalize")
        builder.add_edge("finalize", END)

        return builder.compile(checkpointer=checkpointer, interrupt_before=interrupt_before)

    # -------------------------------------------------------------------------
    # Public Execution Entry Points & Resumption Controls
    # -------------------------------------------------------------------------

    async def run(
        self,
        prompt: str,
        conversation_id: Optional[str] = None,
        options: Optional[Dict[str, Any]] = None,
        task_id: Optional[str] = None,
    ) -> AgentRunResult:
        """Execute a conversational turn through the compiled LangGraph workflow."""
        await self._ensure_active_graph()

        effective_cid = conversation_id or f"conv_{uuid.uuid4().hex[:12]}"
        config = {"configurable": {"thread_id": effective_cid}}

        initial_state: AgentGraphState = {
            "prompt": prompt,
            "conversation_id": conversation_id,
            "task_id": task_id,
            "options": options,
            "model": self.model,
            "max_rounds": self.max_rounds,
            "rounds_used": 0,
            "messages": [],
            "turn_new_messages": [],
            "current_tool_calls": [],
            "tool_records": [],
            "final_response": "",
            "status": "pending",
            "error_message": None,
            "completed": False,
            "requires_approval": False,
            "pending_action_id": None,
        }

        final_state = await self._graph.ainvoke(initial_state, config=config)

        return AgentRunResult(
            prompt=prompt,
            final_response=final_state.get("final_response", ""),
            messages=final_state.get("messages", []),
            tool_records=final_state.get("tool_records", []),
            rounds_used=final_state.get("rounds_used", 0),
            completed=final_state.get("completed", False),
            status=final_state.get("status", "error"),
            error_message=final_state.get("error_message"),
            conversation_id=conversation_id,
            task_id=final_state.get("task_id"),
        )

    async def get_state(self, conversation_id: str) -> Optional[Any]:
        """Retrieve the persisted state snapshot for a given conversation thread."""
        if not self.enable_checkpointing:
            return None
        await self._ensure_active_graph()
        if self.checkpointer is None:
            return None

        config = {"configurable": {"thread_id": conversation_id}}
        snapshot = await self._graph.aget_state(config)
        if not snapshot or not snapshot.values:
            return None
        return snapshot

    async def has_interrupted_turn(self, conversation_id: str) -> bool:
        """Check if the conversation thread has an uncompleted, interrupted turn pending resumption."""
        if not self.enable_checkpointing:
            return False
        await self._ensure_active_graph()
        if self.checkpointer is None:
            return False

        config = {"configurable": {"thread_id": conversation_id}}
        snapshot = await self._graph.aget_state(config)
        if not snapshot or not snapshot.values:
            return False
        return bool(snapshot.next)

    async def resume(self, conversation_id: str) -> Optional[AgentRunResult]:
        """Resume execution of an interrupted conversational turn from persisted checkpoint."""
        if not self.enable_checkpointing:
            return None
        await self._ensure_active_graph()
        if self.checkpointer is None:
            return None

        config = {"configurable": {"thread_id": conversation_id}}
        snapshot = await self._graph.aget_state(config)
        if not snapshot or not snapshot.values:
            return None

        # If turn already reached END, return the final result without re-executing
        if not snapshot.next:
            vals = snapshot.values
            return AgentRunResult(
                prompt=vals.get("prompt", ""),
                final_response=vals.get("final_response", ""),
                messages=vals.get("messages", []),
                tool_records=vals.get("tool_records", []),
                rounds_used=vals.get("rounds_used", 0),
                completed=vals.get("completed", False),
                status=vals.get("status", "success" if vals.get("completed") else "error"),
                error_message=vals.get("error_message"),
                conversation_id=vals.get("conversation_id") or conversation_id,
                task_id=vals.get("task_id"),
            )

        final_state = await self._graph.ainvoke(None, config=config)
        return AgentRunResult(
            prompt=final_state.get("prompt", ""),
            final_response=final_state.get("final_response", ""),
            messages=final_state.get("messages", []),
            tool_records=final_state.get("tool_records", []),
            rounds_used=final_state.get("rounds_used", 0),
            completed=final_state.get("completed", False),
            status=final_state.get("status", "error"),
            error_message=final_state.get("error_message"),
            conversation_id=final_state.get("conversation_id") or conversation_id,
            task_id=final_state.get("task_id"),
        )

    async def close(self) -> None:
        """Cleanly close any open checkpoint storage or saver connection."""
        if self._checkpoint_storage is not None:
            await self._checkpoint_storage.close()
            self.checkpointer = None
        elif self.checkpointer is not None and hasattr(self.checkpointer, "close"):
            try:
                import asyncio
                if asyncio.iscoroutinefunction(self.checkpointer.close):
                    await self.checkpointer.close()
                else:
                    self.checkpointer.close()
            except Exception as exc:
                logger.debug("Error closing checkpointer: %s", exc)
            self.checkpointer = None
