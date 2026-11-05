"""JARVIS Foundation Entry Point & Agent Core Demonstration."""
from __future__ import annotations

import argparse
import asyncio
import json
import platform
import sys
import httpx

from config.settings import settings
from agent.ollama_client import OllamaClient
from agent.runtime import AgentRuntime, create_agent_runtime
from agent.tool_registry import create_default_registry


def check_ollama_runtime(base_url: str, expected_model: str) -> dict[str, str | bool]:
    """Check whether local Ollama daemon is reachable and whether the model is present."""
    try:
        response = httpx.get(f"{base_url}/api/tags", timeout=1.5)
        if response.status_code == 200:
            data = response.json()
            installed_models = [m.get("name", "") for m in data.get("models", [])]
            model_present = any(
                expected_model in m or m.startswith(expected_model.split(":")[0])
                for m in installed_models
            )
            return {
                "reachable": True,
                "model_present": model_present,
                "installed_models": ", ".join(installed_models) or "none",
            }
    except Exception as exc:
        return {
            "reachable": False,
            "model_present": False,
            "error": str(exc),
        }
    return {"reachable": False, "model_present": False}


def print_health_report() -> int:
    """Print system configuration and verified asset health report."""
    print("=" * 68)
    print("  J.A.R.V.I.S. Foundation Diagnostics")
    print("  Fully Local, Autonomous AI Agent (macOS / Apple Silicon)")
    print("=" * 68)
    print(f"System:       {platform.system()} {platform.release()} ({platform.machine()})")
    print(f"Python:       {sys.version.split()[0]} ({sys.executable})")
    print(f"Project Root: {settings.project_root}")
    print("-" * 68)

    # Configuration summary
    print("[1] Configuration & Model Runtime:")
    print(f"  • Ollama Base URL:    {settings.ollama_base_url}")
    print(f"  • Primary LLM:        {settings.llm_model} (num_ctx: {settings.llm_num_ctx})")
    print(f"  • Vision Model:       {settings.vision_model}")
    print(f"  • Audio Sample Rate:  {settings.sample_rate} Hz")

    ollama_status = check_ollama_runtime(settings.ollama_base_url, settings.llm_model)
    if ollama_status["reachable"]:
        model_flag = "READY" if ollama_status["model_present"] else "MODEL MISSING"
        print(f"  • Ollama Daemon:      ONLINE ({model_flag})")
    else:
        print(f"  • Ollama Daemon:      OFFLINE or UNREACHABLE")

    print("-" * 68)

    # Asset verification summary
    print("[2] Local Asset Verification:")
    asset_catalog = settings.get_asset_catalog()
    all_ready = True

    for key, info in asset_catalog.items():
        size_kb = info["size_bytes"] / 1024
        status_tag = "[OK]" if info["status"] == "ready" else "[MISSING]"
        if info["status"] != "ready":
            all_ready = False
        print(f"  {status_tag} {key.upper():<12} : {info['name']} ({size_kb:.1f} KB)")
        print(f"       Path: {info['path']}")

    print("-" * 68)

    # SQLite Database Summary
    print("[3] Persistent Storage (SQLite):")
    db_exists = settings.database_path.is_file()
    db_size = settings.database_path.stat().st_size if db_exists else 0
    db_status = f"READY ({db_size / 1024:.1f} KB)" if db_exists else "INITIALIZED ON FIRST USE"
    print(f"  • Database Path:      {settings.database_path}")
    print(f"  • Database State:     {db_status}")

    print("=" * 68)
    if all_ready:
        print("  Status: FOUNDATION ASSETS VERIFIED & READY FOR MODULE IMPLEMENTATION")
        print("=" * 68)
        return 0
    else:
        print("  Status: WARNING - SOME ASSETS ARE MISSING OR UNREADABLE")
        print("=" * 68)
        return 1



async def run_agent_demonstration(prompt: str, conversation_id: Optional[str] = None) -> int:
    """Execute live local agent demonstration with explicit runtime/model output separation."""
    print("=" * 68)
    print("  J.A.R.V.I.S. Local Agent Demonstration")
    print("  End-to-End Tool Calling Loop (Ollama + Python Runtime)")
    if conversation_id:
        print(f"  Conversation ID:      {conversation_id}")
    print("=" * 68)
    print(f"[User Prompt]\n  \"{prompt}\"\n")

    client = OllamaClient(
        base_url=settings.ollama_base_url,
        default_model=settings.llm_model,
        default_num_ctx=settings.llm_num_ctx,
    )

    if not await client.is_available():
        print("[Error] Local Ollama service is not reachable at", settings.ollama_base_url)
        return 1

    registry = create_default_registry()
    runtime = create_agent_runtime(client=client, registry=registry, max_rounds=5)

    print("--------------------------------------------------------------------")
    print("[Agent Execution Pipeline]")
    print(f"• Registered Tools: {', '.join(registry.list_tools())}")
    print("• Initiating Ollama inference turn...")

    result = await runtime.run(prompt, conversation_id=conversation_id)
    await client.close()

    print("\n--------------------------------------------------------------------")
    if result.tool_records:
        print(f"[Python Runtime Executions] ({len(result.tool_records)} tool call(s) executed):")
        for idx, rec in enumerate(result.tool_records, start=1):
            print(f"  • Tool Call #{idx}:")
            print(f"    - Requested Tool Name : {rec.tool_name}")
            print(f"    - Model Arguments     : {json.dumps(rec.arguments)}")
            print(f"    - Python Return Value : {json.dumps(rec.result)}")
            print(f"    - Execution Latency   : {rec.duration_sec * 1000:.2f} ms")
    else:
        print("[Python Runtime Executions]: None (Model replied without requesting tools)")

    print("\n--------------------------------------------------------------------")
    print("[Model Final Response]")
    print(f"  {result.final_response}")

    print("\n--------------------------------------------------------------------")
    print(f"[Execution Summary]")
    if result.task_id:
        print(f"  • Task ID         : {result.task_id}")
    print(f"  • Rounds Consumed : {result.rounds_used}")
    print(f"  • Completed Status: {result.status}")
    print("=" * 68)

    return 0 if (result.completed and len(result.tool_records) > 0) else 1


def handle_music_command(action: str) -> int:
    """Handle manual music commands without automatic playback in tests."""
    import time
    from tools.music import default_music_controller

    print("=" * 68)
    print(f"  J.A.R.V.I.S. Music Controller (Command: {action.upper()})")
    print("=" * 68)

    if action == "test":
        print("[1/4] Starting playback of garage_music.mp3...")
        res_play = default_music_controller.play()
        print(f"      Result: {res_play}")
        time.sleep(2.0)

        print("[2/4] Pausing playback...")
        res_pause = default_music_controller.pause()
        print(f"      Result: {res_pause}")
        time.sleep(1.0)

        print("[3/4] Resuming playback...")
        res_resume = default_music_controller.resume()
        print(f"      Result: {res_resume}")
        time.sleep(1.0)

        print("[4/4] Stopping playback...")
        res_stop = default_music_controller.stop()
        print(f"      Result: {res_stop}")
        print("=" * 68)
        print("  Music lifecycle test completed successfully.")
        print("=" * 68)
        return 0

    elif action == "play":
        res = default_music_controller.play()
        print(f"Result: {json.dumps(res, indent=2)}")
        return 0 if res.get("success") else 1
    elif action == "pause":
        res = default_music_controller.pause()
        print(f"Result: {json.dumps(res, indent=2)}")
        return 0 if res.get("success") else 1
    elif action == "resume":
        res = default_music_controller.resume()
        print(f"Result: {json.dumps(res, indent=2)}")
        return 0 if res.get("success") else 1
    elif action == "stop":
        res = default_music_controller.stop()
        print(f"Result: {json.dumps(res, indent=2)}")
        return 0 if res.get("success") else 1
    elif action == "status":
        state = default_music_controller.get_state()
        print(f"Current State: {state.value}")
        print(f"Track:         {default_music_controller.music_path.name}")
        print(f"Path:          {default_music_controller.music_path}")
        return 0
    return 1


def handle_reminder_create(text: str, due: str) -> int:
    """CLI handler for creating a reminder."""
    from tools.reminders import create_reminder

    res = create_reminder(text=text, due_time=due)
    print("=" * 68)
    print("  J.A.R.V.I.S. Scheduled Reminders: CREATE")
    print("=" * 68)
    if res.get("success"):
        rem = res["reminder"]
        print(f"  • ID:       {rem['id']}")
        print(f"  • Due:      {rem['due_at']} ({rem['timezone']})")
        print(f"  • Message:  {rem['text']}")
        print(f"  • Status:   {rem['status']}")
        print(f"\n  Result: {res['message']}")
        return 0
    else:
        print(f"  [Error] {res.get('error')}")
        return 1


def handle_reminder_list(status: str = "all") -> int:
    """CLI handler for listing reminders."""
    from tools.reminders import list_reminders

    res = list_reminders(status=status)
    print("=" * 68)
    print(f"  J.A.R.V.I.S. Scheduled Reminders: LIST (Filter: {status.upper()})")
    print("=" * 68)
    reminders = res.get("reminders", [])
    if not reminders:
        print("  No reminders found matching filter.")
    else:
        for r in reminders:
            print(f"  [{r['status'].upper():<9}] ID: {r['id']} | Due: {r['due_at']} | Text: {r['text']}")
    print("=" * 68)
    return 0


def handle_reminder_cancel(reminder_id: str) -> int:
    """CLI handler for cancelling a reminder."""
    from tools.reminders import cancel_reminder

    res = cancel_reminder(reminder_id=reminder_id)
    print("=" * 68)
    print("  J.A.R.V.I.S. Scheduled Reminders: CANCEL")
    print("=" * 68)
    if res.get("success"):
        print(f"  [OK] {res['message']}")
        return 0
    else:
        print(f"  [Error] {res.get('error')}")
        return 1


async def handle_run_scheduler() -> int:
    """CLI handler for running the reminder scheduler."""
    from scheduler.reminder_scheduler import default_scheduler

    print("=" * 68)
    print("  J.A.R.V.I.S. Reminder Scheduler Service")
    print("  Monitoring SQLite for due alerts. Press Ctrl+C to stop.")
    print("=" * 68)
    default_scheduler.start()
    try:
        while True:
            await asyncio.sleep(1.0)
    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\nStopping scheduler...")
        await default_scheduler.stop()
        print("Scheduler stopped cleanly.")
        return 0


async def handle_poll_scheduler() -> int:
    """CLI handler for running a single scheduler poll."""
    from scheduler.reminder_scheduler import default_scheduler

    print("=" * 68)
    print("  J.A.R.V.I.S. Reminder Scheduler: Single Poll Check")
    print("=" * 68)
    results = await default_scheduler.poll_once()
    if not results:
        print("  No pending reminders are currently due.")
    else:
        print(f"  Delivered {len(results)} due reminder(s):")
        for item in results:
            missed_tag = " [MISSED]" if item.get("was_missed") else ""
            print(f"  • {item['id']}: '{item['text']}' -> {item['status']}{missed_tag}")
    print("=" * 68)
    return 0


def handle_list_conversations() -> int:
    """List recent persistent conversations from SQLite database."""
    from storage.conversation import default_conversation_storage
    convs = default_conversation_storage.list_conversations(limit=25)
    print("=" * 68)
    print("  J.A.R.V.I.S. Persistent Conversation History")
    print("=" * 68)
    if not convs:
        print("  No stored conversations found.")
    else:
        print(f"  {'ID':<30} | {'Msgs':<5} | {'Updated At':<19} | Title")
        print("  " + "-" * 64)
        for c in convs:
            title = (c.title[:24] + "...") if c.title and len(c.title) > 24 else (c.title or "Untitled")
            print(f"  {c.id:<30} | {c.message_count:<5} | {c.updated_at[:19]:<19} | {title}")
    print("=" * 68)
    return 0


def handle_clear_conversation(conversation_id: str) -> int:
    """Delete all messages and remove specified conversation ID."""
    from storage.conversation import default_conversation_storage
    existed = default_conversation_storage.delete_conversation(conversation_id)
    print("=" * 68)
    if existed:
        print(f"  [Deleted] Conversation '{conversation_id}' and all associated messages.")
    else:
        print(f"  [Not Found] Conversation '{conversation_id}' was not found.")
    print("=" * 68)
    return 0 if existed else 1


def handle_task_list(status: str = "all") -> int:
    """List persistent tasks from SQLite database with optional status filter."""
    from storage.tasks import TaskStatus, default_task_storage

    filter_status = None if status == "all" else TaskStatus(status)
    tasks = default_task_storage.list_tasks(status=filter_status, limit=30)
    print("=" * 68)
    print(f"  J.A.R.V.I.S. Persistent Task Lifecycle: LIST (Filter: {status.upper()})")
    print("=" * 68)
    if not tasks:
        print("  No persistent tasks found matching filter.")
    else:
        print(f"  {'ID':<18} | {'Type':<12} | {'Status':<21} | Description")
        print("  " + "-" * 64)
        for t in tasks:
            desc = (t.description[:26] + "...") if len(t.description) > 26 else t.description
            print(f"  {t.id:<18} | {t.task_type:<12} | {t.status.value:<21} | {desc}")
    print("=" * 68)
    return 0


def handle_task_show(task_id: str) -> int:
    """Display comprehensive details and metadata for a specific task ID."""
    from storage.tasks import default_task_storage

    task = default_task_storage.get_task(task_id)
    print("=" * 68)
    print(f"  J.A.R.V.I.S. Task Details: {task_id}")
    print("=" * 68)
    if not task:
        print(f"  [Not Found] Task '{task_id}' does not exist.")
        print("=" * 68)
        return 1

    print(f"  • ID:            {task.id}")
    print(f"  • Type:          {task.task_type}")
    print(f"  • Status:        {task.status.value.upper()}")
    print(f"  • Description:   {task.description}")
    print(f"  • Conversation:  {task.conversation_id or 'None'}")
    print(f"  • Created At:    {task.created_at}")
    print(f"  • Started At:    {task.started_at or 'Not started'}")
    print(f"  • Updated At:    {task.updated_at}")
    print(f"  • Completed At:  {task.completed_at or 'Incomplete'}")
    if task.status_message:
        print(f"  • Status Msg:    {task.status_message}")
    if task.error_message:
        print(f"  • Error Msg:     {task.error_message}")
    if task.metadata:
        print(f"  • Metadata:      {json.dumps(task.metadata, indent=4)}")
    print("=" * 68)
    return 0


def handle_task_cancel(task_id: str) -> int:
    """Cancel an active task by ID."""
    from storage.tasks import InvalidTaskStateTransitionError, TaskNotFoundError, default_task_storage

    print("=" * 68)
    print(f"  J.A.R.V.I.S. Task Cancellation: {task_id}")
    print("=" * 68)
    try:
        updated = default_task_storage.cancel_task(task_id, reason="Cancelled via CLI")
        print(f"  [OK] Task '{task_id}' successfully transitioned to CANCELLED.")
        print(f"  • Status Message: {updated.status_message}")
        print("=" * 68)
        return 0
    except TaskNotFoundError:
        print(f"  [Error] Task '{task_id}' not found.")
        print("=" * 68)
        return 1
    except InvalidTaskStateTransitionError as exc:
        print(f"  [Rejected] Cannot cancel task: {exc}")
        print("=" * 68)
        return 1


def handle_task_recover() -> int:
    """Explicitly trigger crash recovery on abandoned active tasks."""
    from storage.tasks import default_task_storage

    print("=" * 68)
    print("  J.A.R.V.I.S. Task Crash Recovery Audit")
    print("  Scanning database for active tasks abandoned by process shutdown...")
    print("=" * 68)
    recovered = default_task_storage.recover_interrupted_tasks()
    if not recovered:
        print("  All tasks are in clean terminal states. No recovery actions needed.")
    else:
        print(f"  Recovered {len(recovered)} abandoned task(s) -> marked INTERRUPTED:")
        for t in recovered:
            from_status = t.metadata.get("recovered_from_status", "unknown")
            print(f"  • {t.id} ({t.task_type}): was '{from_status}' -> INTERRUPTED")
            print(f"    Reason: {t.status_message}")
    print("=" * 68)
    return 0


def handle_ledger_list(status: str = "all") -> int:
    """List action attempts from persistent action ledger with optional filter."""
    from storage.action_ledger import ActionLedgerStatus, default_action_ledger

    filter_status = None if status == "all" else ActionLedgerStatus(status)
    attempts = default_action_ledger.list_attempts(status=filter_status, limit=30)
    print("=" * 68)
    print(f"  J.A.R.V.I.S. Action Idempotency Ledger: LIST (Filter: {status.upper()})")
    print("=" * 68)
    if not attempts:
        print("  No action attempts found matching filter.")
    else:
        print(f"  {'Attempt ID':<18} | {'Tool':<16} | {'Att#':<4} | {'Status':<16} | Idempotency Key")
        print("  " + "-" * 72)
        for a in attempts:
            print(f"  {a.id:<18} | {a.tool_name:<16} | #{a.attempt_number:<3} | {a.status.value:<16} | {a.idempotency_key}")
    print("=" * 68)
    return 0


def handle_ledger_show(attempt_id: str) -> int:
    """Display comprehensive details and reconciliation state for an action attempt."""
    from storage.action_ledger import default_action_ledger

    attempt = default_action_ledger.get_attempt(attempt_id)
    print("=" * 68)
    print(f"  J.A.R.V.I.S. Action Ledger Details: {attempt_id}")
    print("=" * 68)
    if not attempt:
        print(f"  [Not Found] Action attempt '{attempt_id}' does not exist.")
        print("=" * 68)
        return 1

    print(f"  • Attempt ID:     {attempt.id}")
    print(f"  • Idempotency Key: {attempt.idempotency_key}")
    print(f"  • Task ID:        {attempt.task_id}")
    print(f"  • Step Name:      {attempt.step_name}")
    print(f"  • Tool Name:      {attempt.tool_name}")
    print(f"  • Attempt Number: #{attempt.attempt_number}")
    print(f"  • Status:         {attempt.status.value.upper()}")
    print(f"  • Retry Safety:   {attempt.retry_safety.value}")
    print(f"  • Arguments Hash: {attempt.arguments_hash}")
    print(f"  • Created At:     {attempt.created_at}")
    print(f"  • Started At:     {attempt.started_at or 'Not started'}")
    print(f"  • Completed At:   {attempt.completed_at or 'Incomplete'}")
    if attempt.action_id:
        print(f"  • Linked Action:  {attempt.action_id}")
    if attempt.status_message:
        print(f"  • Status Message: {attempt.status_message}")
    if attempt.error_message:
        print(f"  • Error Message:  {attempt.error_message}")
    if attempt.reconciliation_notes:
        print(f"  • Reconciliation: {attempt.reconciliation_notes}")
    if attempt.metadata:
        print(f"  • Metadata:       {json.dumps(attempt.metadata, indent=4)}")
    print("=" * 68)
    return 0


async def handle_ledger_reconcile(attempt_id: str) -> int:
    """Run read-only outcome reconciliation on an uncertain action attempt."""
    from storage.action_ledger import ActionLedgerStatus, default_action_ledger
    from tools.browser_automation import default_browser_automation

    attempt = default_action_ledger.get_attempt(attempt_id)
    print("=" * 68)
    print(f"  J.A.R.V.I.S. Action Outcome Reconciliation: {attempt_id}")
    print("=" * 68)
    if not attempt:
        print(f"  [Not Found] Action attempt '{attempt_id}' does not exist.")
        print("=" * 68)
        return 1

    if attempt.status != ActionLedgerStatus.OUTCOME_UNKNOWN:
        print(f"  [Skipped] Attempt '{attempt_id}' is in status '{attempt.status.value}', not 'outcome_unknown'.")
        print("  Reconciliation is only necessary for uncertain outcomes.")
        print("=" * 68)
        return 0

    print(f"  Target Tool:   {attempt.tool_name}")
    print(f"  Attempt Key:   {attempt.idempotency_key}")
    print(f"  Status Message: {attempt.status_message}")
    print("  Executing read-only inspection...")

    if attempt.tool_name == "browser_interact":
        meta = attempt.metadata or {}
        action = meta.get("action")
        if not action:
            print(f"  [Error] Action attempt '{attempt_id}' is missing required action metadata.")
            print("  Cannot safely perform reconciliation without action context. Preserving OUTCOME_UNKNOWN.")
            print("=" * 68)
            return 1

        target_url = meta.get("url")
        selector = meta.get("selector")
        expected_text = meta.get("text")

        status, evidence = await default_browser_automation.reconcile_action_outcome(
            action=action,
            target_url=target_url,
            selector=selector,
            expected_text=expected_text,
        )
    else:
        status = ActionLedgerStatus.OUTCOME_UNKNOWN
        evidence = f"No automated read-only reconciliation provider for tool '{attempt.tool_name}'."

    if status in (ActionLedgerStatus.SUCCEEDED, ActionLedgerStatus.FAILED):
        updated = default_action_ledger.reconcile_outcome(attempt_id, status, evidence)
        print(f"  [Reconciled] Outcome resolved to: {updated.status.value.upper()}")
        print(f"  • Evidence: {evidence}")
    else:
        print(f"  [Uncertain] Could not resolve outcome: {evidence}")
        print("  Status remains OUTCOME_UNKNOWN; requiring explicit user decision.")

    print("=" * 68)
    return 0


async def handle_voice_session(
    push_to_talk: bool = True,
    conversation_id: Optional[str] = None,
) -> int:
    """Run interactive local voice loop."""
    import threading
    import uuid
    from voice import VoicePipeline

    active_cid = conversation_id or f"conv_voice_{uuid.uuid4().hex[:8]}"

    print("=" * 68)
    print("  J.A.R.V.I.S. Local Voice Interaction Session")
    print(f"  Conversation ID: {active_cid}")
    print("  Microphone → Whisper STT → Ollama Agent → Pocket TTS → Audio")
    print("=" * 68)

    client = OllamaClient(
        base_url=settings.ollama_base_url,
        default_model=settings.llm_model,
        default_num_ctx=settings.llm_num_ctx,
    )
    if not await client.is_available():
        print(f"[Error] Ollama service not reachable at {settings.ollama_base_url}")
        return 1

    registry = create_default_registry()
    runtime = create_agent_runtime(client=client, registry=registry, max_rounds=5)
    pipeline = VoicePipeline(runtime=runtime, conversation_id=active_cid)

    print("• Initializing local speech models (Whisper + Pocket TTS)...")
    warmup_res = await asyncio.to_thread(pipeline.warmup)
    print(f"• Voice models ready (Warmup: STT {warmup_res['stt_warmup_sec']:.2f}s, TTS {warmup_res['tts_warmup_sec']:.2f}s)")

    if not pipeline.recorder.is_input_available():
        print("[Error] No microphone / audio input devices detected.")
        await client.close()
        return 1

    devices = pipeline.recorder.get_input_devices()
    print(f"• Active microphone: {devices[0].get('name', 'Default Microphone')}")
    if not push_to_talk:
        print("-" * 68)
        print("  Continuous Hands-Free Voice Mode Active")
        print("  • Automatic speech detection & dynamic noise-floor calibration.")
        print("  • Speak freely; JARVIS listens, responds, and returns to listening.")
        print("  • Speak 'exit', 'quit', 'stop listening' or press Ctrl+C to stop.")
        print("-" * 68)

        def on_turn_completed(turn_res) -> None:
            if turn_res.error == "No speech detected" or not turn_res.user_query:
                return

            if turn_res.error == "Transcript suppressed by reflection guard":
                print("  [Acoustic Safety] Assistant speech reflection suppressed.")
                return

            if not turn_res.success and turn_res.error:
                print(f"  [Notice]: Turn incomplete: {turn_res.error}")

            print(f"\n  [User Spoke] : \"{turn_res.user_query}\"")
            if turn_res.tool_records:
                tools_used = ", ".join(f"{r.tool_name}" for r in turn_res.tool_records)
                print(f"  [Tools Used] : {tools_used}")
            if turn_res.final_response:
                print(f"  [JARVIS Spoke]: \"{turn_res.final_response}\"")
            if turn_res.metrics:
                print(f"  [Latency Breakdown]:")
                print(f"    • STT (Whisper)  : {turn_res.stt_latency_ms:.1f} ms")
                print(f"    • LLM (Qwen)     : {turn_res.pure_llm_latency_ms:.1f} ms")
                print(f"    • Tools ({len(turn_res.tool_records)})     : {turn_res.tool_latency_ms:.1f} ms")
                print(f"    • TTS (Pocket)   : {turn_res.tts_latency_ms:.1f} ms")
                print(f"    • Total Turn     : {turn_res.total_latency_ms:.1f} ms")
            print("-" * 68)

        def on_state_change(state: str) -> None:
            if state == "listening":
                print("  [Listening...] (Speak freely or press Ctrl+C to quit)", end="\r", flush=True)

        try:
            await pipeline.run_hands_free_loop(
                conversation_id=active_cid,
                silence_duration_sec=settings.stt_silence_duration_sec,
                max_duration_sec=settings.stt_max_record_sec,
                energy_threshold=settings.stt_energy_threshold,
                speak_response=True,
                on_turn_completed=on_turn_completed,
                on_state_change=on_state_change,
            )
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n[Hands-Free Voice Session Interrupted] Shutting down cleanly.")
        finally:
            pipeline.tts_engine.stop_playback()
            await client.close()

        return 0

    # Push-to-talk interactive mode
    print("-" * 68)
    print("Controls:")
    print("  [Enter]       : Start speaking (Press [Enter] again to finish)")
    print("  'q' + [Enter] : Quit voice session")
    print("-" * 68)

    try:
        while True:
            try:
                cmd = await asyncio.to_thread(input, "\n[Ready] Press [Enter] to speak ('q' to quit) > ")
                if cmd.strip().lower() == "q":
                    print("Exiting voice session. Goodbye, sir.")
                    break

                stop_event = threading.Event()
                print("  [Recording...] Speak your command now. Press [Enter] when done...")

                def wait_for_enter():
                    try:
                        input()
                    except (EOFError, KeyboardInterrupt):
                        pass
                    stop_event.set()

                enter_thread = threading.Thread(target=wait_for_enter, daemon=True)
                enter_thread.start()

                turn_res = await pipeline.execute_push_to_talk_turn(
                    stop_event=stop_event,
                    max_duration_sec=settings.stt_max_record_sec,
                    speak_response=True,
                )

                if turn_res.error == "No speech detected" or not turn_res.user_query:
                    print("  [STT]: No speech detected.")
                    continue

                if not turn_res.success and turn_res.error:
                    print(f"  [Notice]: Turn incomplete: {turn_res.error}")

                print(f"\n  [User Spoke] : \"{turn_res.user_query}\"")
                if turn_res.tool_records:
                    tools_used = ", ".join(f"{r.tool_name}" for r in turn_res.tool_records)
                    print(f"  [Tools Used] : {tools_used}")
                print(f"  [JARVIS Spoke]: \"{turn_res.final_response}\"")
                print(f"  [Latency Breakdown]:")
                print(f"    • STT (Whisper)  : {turn_res.stt_latency_ms:.1f} ms")
                print(f"    • LLM (Qwen)     : {turn_res.pure_llm_latency_ms:.1f} ms")
                print(f"    • Tools ({len(turn_res.tool_records)})     : {turn_res.tool_latency_ms:.1f} ms")
                print(f"    • TTS (Pocket)   : {turn_res.tts_latency_ms:.1f} ms")
                print(f"    • Total Turn     : {turn_res.total_latency_ms:.1f} ms")
                print("-" * 68)

            except (KeyboardInterrupt, asyncio.CancelledError):
                raise
            except Exception as turn_exc:
                print(f"  [Warning] Turn encountered an issue: {turn_exc}. Session remains active.")

    except (KeyboardInterrupt, asyncio.CancelledError):
        print("\n[Voice Session Interrupted] Shutting down cleanly.")
    finally:
        pipeline.tts_engine.stop_playback()
        await client.close()

    return 0



def handle_voice_speak(text: str) -> int:
    """Synthesize and play a text prompt using Pocket TTS."""
    import time
    from voice import TextToSpeechEngine

    print("=" * 68)
    print("  J.A.R.V.I.S. Text-to-Speech Output Test")
    print("=" * 68)
    print(f"Text: \"{text}\"")
    engine = TextToSpeechEngine()
    t0 = time.perf_counter()
    res = engine.speak(text)
    if res.error:
        print(f"[Error] TTS synthesis failed: {res.error}")
        return 1
    print(f"  • Synthesis Latency : {res.latency_sec * 1000:.1f} ms")
    print(f"  • Audio Duration    : {res.audio_duration_sec:.2f} s")
    print(f"  • Playback Status   : {'SUCCESS' if res.played else 'FAILED'}")
    print("=" * 68)
    return 0 if res.played else 1


async def handle_voice_benchmark() -> int:
    """Benchmark local STT, LLM, TTS, and total response latency separately."""
    import time
    from voice import VoicePipeline

    print("=" * 68)
    print("  J.A.R.V.I.S. Local Voice Interaction Benchmark")
    print("  Independent latency profiling on Apple Silicon")
    print("=" * 68)

    client = OllamaClient(
        base_url=settings.ollama_base_url,
        default_model=settings.llm_model,
        default_num_ctx=settings.llm_num_ctx,
    )
    if not await client.is_available():
        print(f"[Error] Ollama service not reachable at {settings.ollama_base_url}")
        return 1

    registry = create_default_registry()
    runtime = create_agent_runtime(client=client, registry=registry, max_rounds=5)
    pipeline = VoicePipeline(runtime=runtime)

    print("[1/4] Warming up local neural models...")
    warmup_res = await asyncio.to_thread(pipeline.warmup)
    print(f"      • STT Model Load Time: {warmup_res['stt_warmup_sec']:.2f} s")
    print(f"      • TTS Model Load Time: {warmup_res['tts_warmup_sec']:.2f} s")

    print("\n[2/4] Benchmarking STT Transcription (Speech-to-Text)...")
    stt_res = await asyncio.to_thread(
        pipeline.stt_engine.transcribe_file,
        settings.voice_clean_path,
    )
    stt_rtf = stt_res.latency_sec / stt_res.audio_duration_sec if stt_res.audio_duration_sec > 0 else 0
    print(f"      • Audio Duration : {stt_res.audio_duration_sec:.2f} s")
    print(f"      • Transcription  : \"{stt_res.text[:60]}...\"")
    print(f"      • STT Latency    : {stt_res.latency_sec * 1000:.1f} ms")
    print(f"      • Real-Time Factor (RTF): {stt_rtf:.3f}x (lower is faster)")

    print("\n[3/4] Benchmarking LLM Reasoning & Tool Execution...")
    test_prompt = "Calculate (125 * 8) + 40."
    t0 = time.perf_counter()
    llm_res = await runtime.run(test_prompt)
    llm_latency_ms = (time.perf_counter() - t0) * 1000.0
    print(f"      • Prompt         : \"{test_prompt}\"")
    print(f"      • Tools Executed : {[r.tool_name for r in llm_res.tool_records]}")
    print(f"      • Final Output   : \"{llm_res.final_response.strip()}\"")
    print(f"      • LLM Latency    : {llm_latency_ms:.1f} ms")

    print("\n[4/4] Benchmarking Pocket TTS Synthesis (Text-to-Speech)...")
    tts_text = llm_res.final_response.strip() or "All systems are operational, sir."
    t0 = time.perf_counter()
    audio_np, sr, tts_latency_sec = await asyncio.to_thread(
        pipeline.tts_engine.synthesize,
        tts_text,
    )
    tts_dur_sec = len(audio_np) / sr if len(audio_np) > 0 else 0
    tts_rtf = tts_latency_sec / tts_dur_sec if tts_dur_sec > 0 else 0
    print(f"      • Text to Speak  : \"{tts_text}\"")
    print(f"      • Audio Duration : {tts_dur_sec:.2f} s")
    print(f"      • TTS Latency    : {tts_latency_sec * 1000:.1f} ms")
    print(f"      • Real-Time Factor (RTF): {tts_rtf:.3f}x (lower is faster)")

    total_pipeline_ms = (stt_res.latency_sec * 1000.0) + llm_latency_ms + (tts_latency_sec * 1000.0)
    print("\n" + "=" * 68)
    print("  Summary Latency Profile (Fully Local Inference)")
    print("=" * 68)
    print(f"  Component          | Latency     | Metric")
    print(f"  -------------------+-------------+------------------------------")
    print(f"  STT (Whisper base) | {stt_res.latency_sec * 1000:7.1f} ms | RTF: {stt_rtf:.3f}x")
    print(f"  LLM (Qwen 4B+Tool) | {llm_latency_ms:7.1f} ms | 1 tool round + synthesis")
    print(f"  TTS (Pocket TTS)   | {tts_latency_sec * 1000:7.1f} ms | RTF: {tts_rtf:.3f}x")
    print(f"  -------------------+-------------+------------------------------")
    print(f"  Total Turn Latency | {total_pipeline_ms:7.1f} ms | Complete roundtrip")
    print("=" * 68)

    await client.close()
    return 0



def main() -> int:
    """Entry point dispatcher."""
    parser = argparse.ArgumentParser(description="JARVIS Autonomous Agent Entry Point")
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run live agent arithmetic tool demonstration: 'Calculate (125 * 8) + 40.'",
    )
    parser.add_argument(
        "--demo-web",
        action="store_true",
        help="Run live agent website opening demonstration: 'Open the website apple.com'",
    )
    parser.add_argument(
        "--music",
        choices=["test", "play", "pause", "resume", "stop", "status"],
        help="Manual music playback controller commands for hardware testing.",
    )
    parser.add_argument(
        "--reminders",
        nargs="?",
        const="all",
        choices=["all", "pending", "delivered", "cancelled"],
        help="List reminders from SQLite database (default: all).",
    )
    parser.add_argument(
        "--create-reminder",
        type=str,
        help="Text message for a new scheduled reminder.",
    )
    parser.add_argument(
        "--due",
        type=str,
        help="Due timestamp or relative duration for --create-reminder (e.g. 'in 10 minutes' or '2026-10-10 15:00').",
    )
    parser.add_argument(
        "--cancel-reminder",
        type=str,
        help="Cancel an active reminder by ID.",
    )
    parser.add_argument(
        "--run-scheduler",
        action="store_true",
        help="Run the background reminder scheduler service.",
    )
    parser.add_argument(
        "--poll-scheduler",
        action="store_true",
        help="Run a single poll pass for currently due reminders.",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="Run live agent with a custom user prompt.",
    )
    parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Run foundation and asset discovery diagnostics.",
    )
    parser.add_argument(
        "--bridge",
        "--server",
        action="store_true",
        help="Run the FastAPI REST bridge server for the frontend Command Deck on port 8000.",
    )
    parser.add_argument(
        "--desktop",
        action="store_true",
        help="Launch JARVIS via the Desktop Process Supervisor and Lifecycle Manager.",
    )
    parser.add_argument(
        "--voice",
        action="store_true",
        help="Run interactive local voice loop (push-to-talk microphone mode).",
    )
    parser.add_argument(
        "--voice-auto",
        action="store_true",
        help="Run interactive local voice loop with automatic silence endpointing.",
    )
    parser.add_argument(
        "--conversation-id",
        type=str,
        default=None,
        help="Explicit conversation ID to resume or assign for multi-turn history.",
    )
    parser.add_argument(
        "--conversations",
        action="store_true",
        help="List recent persistent conversations stored in SQLite database.",
    )
    parser.add_argument(
        "--clear-conversation",
        type=str,
        help="Delete all messages and remove specified conversation ID.",
    )
    parser.add_argument(
        "--voice-speak",
        type=str,
        help="Test TTS output by synthesizing and speaking the specified text.",
    )
    parser.add_argument(
        "--voice-benchmark",
        action="store_true",
        help="Run comprehensive local benchmark across STT, LLM, and TTS with latency metrics.",
    )
    parser.add_argument(
        "--tasks",
        nargs="?",
        const="all",
        choices=["all", "queued", "running", "waiting_for_approval", "succeeded", "failed", "cancelled", "interrupted"],
        help="List persistent tasks from SQLite database (default: all).",
    )
    parser.add_argument(
        "--task-show",
        type=str,
        help="Display detailed status and metadata for a specific task ID.",
    )
    parser.add_argument(
        "--cancel-task",
        type=str,
        help="Cancel an active task by ID.",
    )
    parser.add_argument(
        "--recover-tasks",
        action="store_true",
        help="Audit and recover abandoned active tasks left by prior process crashes.",
    )
    parser.add_argument(
        "--ledger",
        nargs="?",
        const="all",
        choices=["all", "prepared", "awaiting_approval", "approved", "executing", "succeeded", "failed", "outcome_unknown", "cancelled"],
        help="List consequential action attempts from persistent SQLite ledger (default: all).",
    )
    parser.add_argument(
        "--ledger-show",
        type=str,
        help="Display comprehensive details for a specific action attempt ID.",
    )
    parser.add_argument(
        "--reconcile",
        type=str,
        help="Run read-only outcome reconciliation on an uncertain action attempt ID.",
    )

    args = parser.parse_args()

    if args.ledger is not None:
        return handle_ledger_list(status=args.ledger)
    elif args.ledger_show:
        return handle_ledger_show(args.ledger_show)
    elif args.reconcile:
        return asyncio.run(handle_ledger_reconcile(args.reconcile))
    elif args.tasks is not None:
        return handle_task_list(status=args.tasks)
    elif args.task_show:
        return handle_task_show(args.task_show)
    elif args.cancel_task:
        return handle_task_cancel(args.cancel_task)
    elif args.recover_tasks:
        return handle_task_recover()
    elif args.conversations:
        return handle_list_conversations()
    elif args.clear_conversation:
        return handle_clear_conversation(args.clear_conversation)
    elif args.voice:
        return asyncio.run(handle_voice_session(push_to_talk=True, conversation_id=args.conversation_id))
    elif args.voice_auto:
        return asyncio.run(handle_voice_session(push_to_talk=False, conversation_id=args.conversation_id))
    elif args.voice_speak:
        return handle_voice_speak(args.voice_speak)
    elif args.voice_benchmark:
        return asyncio.run(handle_voice_benchmark())
    elif args.music:
        return handle_music_command(args.music)
    elif args.reminders is not None:
        return handle_reminder_list(status=args.reminders)
    elif args.create_reminder:
        if not args.due:
            print("[Error] --due is required when using --create-reminder (e.g. --due 'in 10 minutes')")
            return 1
        return handle_reminder_create(text=args.create_reminder, due=args.due)
    elif args.cancel_reminder:
        return handle_reminder_cancel(args.cancel_reminder)
    elif args.run_scheduler:
        return asyncio.run(handle_run_scheduler())
    elif args.poll_scheduler:
        return asyncio.run(handle_poll_scheduler())
    elif args.demo:
        return asyncio.run(run_agent_demonstration("Calculate (125 * 8) + 40.", conversation_id=args.conversation_id))
    elif args.demo_web:
        return asyncio.run(run_agent_demonstration("Open the website apple.com", conversation_id=args.conversation_id))
    elif args.prompt:
        return asyncio.run(run_agent_demonstration(args.prompt, conversation_id=args.conversation_id))
    elif args.desktop:
        from desktop.lifecycle import main as desktop_main
        return desktop_main()
    elif args.bridge:
        import uvicorn
        from bridge import app
        uvicorn.run(app, host="127.0.0.1", port=8000)
        return 0
    else:
        return print_health_report()



if __name__ == "__main__":
    sys.exit(main())


