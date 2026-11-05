"""Secure tool registration, validation, and dispatch for JARVIS."""
from __future__ import annotations

import ast
import asyncio
import concurrent.futures
from datetime import datetime
import inspect
import operator
from typing import Any, Callable, Dict, List, Optional, Type
from pydantic import BaseModel, ConfigDict, Field, ValidationError

# Allowed arithmetic operators for safe AST evaluation
_ALLOWED_OPERATORS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_MAX_EXPRESSION_LENGTH = 200
_MAX_AST_DEPTH = 20
_MAX_EXPONENT = 100
_MAX_RESULT_MAGNITUDE = 1e15


def safe_calculate(expression: str) -> float | int:
    """Safely evaluate arithmetic expressions without eval, exec, or shell.

    Supports: numeric literals (int, float), parentheses, +, -, *, /, **.
    Rejects: non-arithmetic syntax, variables, calls, attributes, huge exponents, division by zero.
    """
    clean_expr = expression.strip()
    if not clean_expr:
        raise ValueError("Expression cannot be empty.")
    if len(clean_expr) > _MAX_EXPRESSION_LENGTH:
        raise ValueError(
            f"Expression exceeds maximum length of {_MAX_EXPRESSION_LENGTH} characters."
        )

    try:
        root_node = ast.parse(clean_expr, mode="eval")
    except SyntaxError as exc:
        raise ValueError(f"Invalid mathematical syntax: {exc}") from exc

    def _eval_node(node: ast.AST, depth: int = 0) -> float | int:
        if depth > _MAX_AST_DEPTH:
            raise ValueError("Expression nesting depth exceeds safe limit.")

        if isinstance(node, ast.Expression):
            return _eval_node(node.body, depth + 1)

        elif isinstance(node, ast.Constant):
            if isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
                return node.value
            raise ValueError(f"Disallowed constant type: {type(node.value).__name__}")

        elif isinstance(node, ast.UnaryOp):
            op_cls = type(node.op)
            if op_cls in _ALLOWED_OPERATORS:
                val = _eval_node(node.operand, depth + 1)
                return _ALLOWED_OPERATORS[op_cls](val)
            raise ValueError(f"Unsupported unary operator: {op_cls.__name__}")

        elif isinstance(node, ast.BinOp):
            op_cls = type(node.op)
            if op_cls in _ALLOWED_OPERATORS:
                left = _eval_node(node.left, depth + 1)
                right = _eval_node(node.right, depth + 1)

                if op_cls is ast.Div:
                    if right == 0:
                        raise ZeroDivisionError("Division by zero is undefined.")

                elif op_cls is ast.Pow:
                    if abs(right) > _MAX_EXPONENT:
                        raise ValueError(
                            f"Exponent {right} exceeds safe maximum of {_MAX_EXPONENT}."
                        )
                    if abs(left) > 1e5 and right > 5:
                        raise ValueError("Power computation exceeds safe numerical limit.")

                result = _ALLOWED_OPERATORS[op_cls](left, right)

                if isinstance(result, (int, float)):
                    if abs(result) > _MAX_RESULT_MAGNITUDE:
                        raise ValueError("Result magnitude exceeds safe limit (1e15).")
                return result
            raise ValueError(f"Unsupported binary operator: {op_cls.__name__}")

        else:
            raise ValueError(f"Disallowed syntax element: {type(node).__name__}")

    return _eval_node(root_node)


def get_local_time() -> Dict[str, str]:
    """Retrieve the machine's local date, time, and timezone information."""
    now = datetime.now().astimezone()
    tz_name = now.tzname() or "Local"
    raw_offset = now.strftime("%z")
    formatted_offset = (
        f"UTC{raw_offset[:3]}:{raw_offset[3:]}" if raw_offset else ""
    )
    full_timezone = f"{tz_name} ({formatted_offset})" if formatted_offset else tz_name

    return {
        "iso_timestamp": now.isoformat(),
        "readable_date": now.strftime("%A, %B %d, %Y"),
        "readable_time": now.strftime("%I:%M:%S %p"),
        "timezone": full_timezone,
    }


# Pydantic argument models for strict validation
class NoArgs(BaseModel):
    """Schema for tools that accept no arguments."""

    model_config = ConfigDict(extra="forbid")


class CalculateArgs(BaseModel):
    """Schema for the calculate tool."""

    expression: str = Field(
        ...,
        min_length=1,
        max_length=_MAX_EXPRESSION_LENGTH,
        description="Mathematical arithmetic expression, e.g. (125 * 8) + 40",
    )
    model_config = ConfigDict(extra="forbid")


from storage.action_ledger import RetrySafety


class ToolDefinition:
    """Specification and executable handler for an agent tool."""

    def __init__(
        self,
        name: str,
        description: str,
        parameters_schema: Dict[str, Any],
        handler: Callable[..., Any],
        args_model: Optional[Type[BaseModel]] = None,
        retry_safety: Optional[RetrySafety] = None,
    ) -> None:
        self.name = name.strip()
        self.description = description.strip()
        self.parameters_schema = parameters_schema
        self.handler = handler
        self.args_model = args_model
        self.retry_safety = retry_safety or RetrySafety.NON_IDEMPOTENT


class ToolRegistry:
    """Registry maintaining authorized tools and executing them with strict validation."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolDefinition] = {}

    def register(self, tool: ToolDefinition) -> None:
        """Register a tool. Fails on duplicate registration."""
        if not tool.name:
            raise ValueError("Tool name cannot be empty.")
        if tool.name in self._tools:
            raise ValueError(f"Tool '{tool.name}' is already registered.")
        self._tools[tool.name] = tool

    def get(self, name: str) -> Optional[ToolDefinition]:
        """Retrieve a registered tool by name."""
        return self._tools.get(name)

    def list_tools(self) -> List[str]:
        """List names of all registered tools."""
        return sorted(list(self._tools.keys()))

    def get_retry_safety(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> RetrySafety:
        """Determine retry safety classification for a named tool and invocation arguments."""
        tool = self.get(name)
        if not tool:
            return RetrySafety.NON_IDEMPOTENT
        if name == "browser_interact" and arguments:
            action = arguments.get("action", "")
            if action == "read_page":
                return RetrySafety.READ_ONLY
            elif action in ("navigate", "scroll"):
                return RetrySafety.NATURALLY_IDEMPOTENT
            return RetrySafety.NON_IDEMPOTENT
        return tool.retry_safety

    def get_schemas(self) -> List[Dict[str, Any]]:
        """Return tool definitions in Ollama / OpenAI function format."""
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters_schema,
                },
            })
        return schemas

    async def execute_async(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate arguments and safely execute the named tool asynchronously."""
        tool = self._tools.get(name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{name}' is not registered.",
                "available_tools": self.list_tools(),
            }

        args = arguments if arguments is not None else {}
        if not isinstance(args, dict):
            return {
                "success": False,
                "error": f"Tool arguments must be a JSON object, got {type(args).__name__}.",
            }

        kwargs: Dict[str, Any] = {}
        if tool.args_model is not None:
            try:
                validated_model = tool.args_model.model_validate(args)
                kwargs = validated_model.model_dump()
            except ValidationError as exc:
                return {
                    "success": False,
                    "error": f"Invalid arguments for tool '{name}': {exc.errors()}",
                }
        else:
            kwargs = args

        try:
            if inspect.iscoroutinefunction(tool.handler):
                result = await tool.handler(**kwargs)
            else:
                result = tool.handler(**kwargs)
                if inspect.iscoroutine(result):
                    result = await result
            return {"success": True, "result": result}
        except ZeroDivisionError as exc:
            return {"success": False, "error": f"Math error: {exc}"}
        except ValueError as exc:
            return {"success": False, "error": f"Calculation error: {exc}"}
        except Exception as exc:
            return {"success": False, "error": f"Execution error in '{name}': {exc}"}

    def execute(
        self, name: str, arguments: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Validate arguments and safely execute the named tool synchronously."""
        tool = self._tools.get(name)
        if not tool:
            return {
                "success": False,
                "error": f"Tool '{name}' is not registered.",
                "available_tools": self.list_tools(),
            }

        args = arguments if arguments is not None else {}
        if not isinstance(args, dict):
            return {
                "success": False,
                "error": f"Tool arguments must be a JSON object, got {type(args).__name__}.",
            }

        kwargs: Dict[str, Any] = {}
        if tool.args_model is not None:
            try:
                validated_model = tool.args_model.model_validate(args)
                kwargs = validated_model.model_dump()
            except ValidationError as exc:
                return {
                    "success": False,
                    "error": f"Invalid arguments for tool '{name}': {exc.errors()}",
                }
        else:
            kwargs = args

        try:
            if inspect.iscoroutinefunction(tool.handler):
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None
                if loop and loop.is_running():
                    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                        result = pool.submit(asyncio.run, tool.handler(**kwargs)).result()
                else:
                    result = asyncio.run(tool.handler(**kwargs))
            else:
                result = tool.handler(**kwargs)
                if inspect.iscoroutine(result):
                    try:
                        loop = asyncio.get_running_loop()
                    except RuntimeError:
                        loop = None
                    if loop and loop.is_running():
                        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                            result = pool.submit(asyncio.run, result).result()
                    else:
                        result = asyncio.run(result)
            return {"success": True, "result": result}
        except ZeroDivisionError as exc:
            return {"success": False, "error": f"Math error: {exc}"}
        except ValueError as exc:
            return {"success": False, "error": f"Calculation error: {exc}"}
        except Exception as exc:
            return {"success": False, "error": f"Execution error in '{name}': {exc}"}


class OpenWebsiteArgs(BaseModel):
    """Schema for the open_website tool."""

    url: str = Field(
        ...,
        min_length=1,
        max_length=2048,
        description="URL or domain to open in default browser, e.g. 'google.com' or 'https://github.com'",
    )
    model_config = ConfigDict(extra="forbid")


def create_default_registry() -> ToolRegistry:
    """Build a ToolRegistry loaded with the authorized tools."""
    from tools.browser import open_website
    from tools.music import pause_music, play_music, resume_music, stop_music

    registry = ToolRegistry()

    # 1. get_local_time tool
    registry.register(
        ToolDefinition(
            name="get_local_time",
            description="Returns the current local date, time, and timezone of the host system.",
            parameters_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=get_local_time,
            args_model=NoArgs,
        )
    )

    # 2. calculate tool
    registry.register(
        ToolDefinition(
            name="calculate",
            description=(
                "Safely evaluate basic arithmetic expressions. Supports numeric literals, "
                "parentheses, +, -, *, /, and **. Rejects code execution or arbitrary syntax."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "Mathematical expression, e.g. (125 * 8) + 40",
                    }
                },
                "required": ["expression"],
            },
            handler=safe_calculate,
            args_model=CalculateArgs,
        )
    )

    # 3. play_music tool
    registry.register(
        ToolDefinition(
            name="play_music",
            description="Start playing the local background music track.",
            parameters_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=play_music,
            args_model=NoArgs,
        )
    )

    # 4. pause_music tool
    registry.register(
        ToolDefinition(
            name="pause_music",
            description="Pause the currently playing music track.",
            parameters_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=pause_music,
            args_model=NoArgs,
        )
    )

    # 5. resume_music tool
    registry.register(
        ToolDefinition(
            name="resume_music",
            description="Resume playback of the paused music track.",
            parameters_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=resume_music,
            args_model=NoArgs,
        )
    )

    # 6. stop_music tool
    registry.register(
        ToolDefinition(
            name="stop_music",
            description="Stop music playback completely.",
            parameters_schema={
                "type": "object",
                "properties": {},
                "required": [],
            },
            handler=stop_music,
            args_model=NoArgs,
        )
    )

    # 7. open_website tool
    registry.register(
        ToolDefinition(
            name="open_website",
            description="Open a validated HTTP or HTTPS web address in the user's default browser.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The website address or domain to open (e.g. 'google.com' or 'https://github.com')",
                    }
                },
                "required": ["url"],
            },
            handler=open_website,
            args_model=OpenWebsiteArgs,
        )
    )

    # 8. create_reminder tool
    from tools.reminders import (
        CancelReminderArgs,
        CreateReminderArgs,
        ListRemindersArgs,
        cancel_reminder,
        create_reminder,
        list_reminders,
    )

    registry.register(
        ToolDefinition(
            name="create_reminder",
            description="Create a persistent scheduled reminder with unambiguous future due time.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Content of the reminder",
                    },
                    "due_time": {
                        "type": "string",
                        "description": "Due time: relative (e.g. 'in 15 minutes') or ISO format (e.g. '2026-10-10 16:00')",
                    },
                },
                "required": ["text", "due_time"],
            },
            handler=create_reminder,
            args_model=CreateReminderArgs,
        )
    )

    # 9. list_reminders tool
    registry.register(
        ToolDefinition(
            name="list_reminders",
            description="List scheduled reminders filtered by status ('pending', 'delivered', 'cancelled', 'all').",
            parameters_schema={
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "description": "Status filter: 'pending', 'delivered', 'cancelled', or 'all'",
                    }
                },
                "required": [],
            },
            handler=list_reminders,
            args_model=ListRemindersArgs,
        )
    )

    # 10. cancel_reminder tool
    registry.register(
        ToolDefinition(
            name="cancel_reminder",
            description="Cancel an active pending reminder by its unique reminder ID.",
            parameters_schema={
                "type": "object",
                "properties": {
                    "reminder_id": {
                        "type": "string",
                        "description": "Unique ID of the reminder to cancel (e.g. 'rem_12345678')",
                    }
                },
                "required": ["reminder_id"],
            },
            handler=cancel_reminder,
            args_model=CancelReminderArgs,
        )
    )

    # 11. browser_interact tool (Chrome DevTools Protocol)
    from tools.browser_automation import (
        BrowserInteractArgs,
        BrowserObserveArgs,
        browser_interact_handler,
        browser_observe_handler,
    )

    registry.register(
        ToolDefinition(
            name="browser_interact",
            description=(
                "Safely interact with an active browser page using Chrome DevTools Protocol (CDP). "
                "Supports allowlisted actions: navigate (URL), click (element by selector/index), "
                "type (text into input), press_key (e.g. Enter), scroll, and read_page. "
                "Consequential actions (submit, pay, purchase, send, delete) require explicit server approval: "
                "an unapproved action returns a pending action_id; once approved via trusted user interaction, "
                "provide the action_id and approval_token."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "action": {
                        "type": "string",
                        "enum": ["navigate", "click", "type", "press_key", "scroll", "read_page"],
                        "description": "Permitted browser action",
                    },
                    "url": {
                        "type": "string",
                        "description": "Target HTTP or HTTPS address for 'navigate'",
                    },
                    "selector": {
                        "type": "string",
                        "description": "CSS selector of element to click or type into",
                    },
                    "element_index": {
                        "type": "integer",
                        "description": "Numeric index of element from prior observation",
                    },
                    "text": {
                        "type": "string",
                        "description": "Text to enter for 'type'",
                    },
                    "key": {
                        "type": "string",
                        "description": "Key name for 'press_key' (default: Enter)",
                    },
                    "action_id": {
                        "type": "string",
                        "description": "Unique identifier of server-side pending consequential action awaiting approval",
                    },
                    "approval_token": {
                        "type": "string",
                        "description": "Single-use cryptographic approval token issued by trusted user approval channel",
                    },
                    "user_confirmed": {
                        "type": "boolean",
                        "description": "Deprecated untrusted parameter (cannot authorize consequential actions)",
                    },
                },
                "required": ["action"],
            },
            handler=browser_interact_handler,
            args_model=BrowserInteractArgs,
        )
    )

    # 12. browser_observe tool (Chrome DevTools Protocol)
    registry.register(
        ToolDefinition(
            name="browser_observe",
            description=(
                "Observe the current browser page: returns URL, title, bounded visible text, "
                "and allowlisted interactive elements with indices and selectors."
            ),
            parameters_schema={
                "type": "object",
                "properties": {
                    "target_id": {
                        "type": "string",
                        "description": "Optional specific tab target ID to observe",
                    }
                },
                "required": [],
            },
            handler=browser_observe_handler,
            args_model=BrowserObserveArgs,
        )
    )

    return registry


