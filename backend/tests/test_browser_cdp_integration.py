"""Integration tests for JARVIS browser automation against a controlled local test server."""
import asyncio
from http.server import BaseHTTPRequestHandler, HTTPServer
import threading
from typing import Optional
import pytest

from tools.browser_automation import (
    BrowserAutomationController,
    is_consequential_action,
)
from tools.cdp_client import CDPClient, CDPConnectionError
from security.approval import default_approval_manager


CONTROLLED_HTML = """<!DOCTYPE html>
<html>
<head><title>JARVIS Integration Test Page</title></head>
<body>
    <h1>JARVIS Controlled Test Page</h1>
    <p id="page-description">This is a local, controlled test page for browser automation.</p>
    
    <div id="counter-section">
        <span id="counter-val">0</span>
        <button id="counter-btn" onclick="document.getElementById('counter-val').innerText = '1';">Increment Counter</button>
    </div>

    <form id="order-form">
        <input id="user-input" type="text" placeholder="Your name" value="" />
        <input id="pass-input" type="password" placeholder="Your password" value="" />
        <button id="submit-order" type="submit">Submit Order</button>
    </form>

    <a id="second-page-link" href="/target-page">Proceed to Target Page</a>
</body>
</html>
"""

TARGET_HTML = """<!DOCTYPE html>
<html>
<head><title>Target Page Title</title></head>
<body>
    <h1>Arrived at Target Page</h1>
</body>
</html>
"""


class ControlledTestServerHandler(BaseHTTPRequestHandler):
    """HTTP handler serving the controlled test pages."""

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(CONTROLLED_HTML.encode("utf-8"))
        elif self.path == "/target-page":
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.end_headers()
            self.wfile.write(TARGET_HTML.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Quiet logging in test runs


@pytest.fixture(scope="module")
def local_test_server():
    """Spin up a local test HTTP server on an ephemeral loopback port."""
    server = HTTPServer(("127.0.0.1", 0), ControlledTestServerHandler)
    port = server.server_address[1]
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    base_url = f"http://127.0.0.1:{port}"
    yield base_url
    server.shutdown()
    server.server_close()


class MockControlledCDPClient(CDPClient):
    """Mock CDP simulating interactions against the controlled test DOM."""

    def __init__(self, base_url: str) -> None:
        super().__init__(host="127.0.0.1", port=9222)
        self.base_url = base_url
        self.current_url = f"{base_url}/"
        self.current_title = "JARVIS Integration Test Page"
        self.user_input_val = ""
        self.counter_val = "0"
        self.disconnected = False

    async def is_available(self):
        if self.disconnected:
            return False, "Browser unavailable"
        return True, ""

    async def connect_to_target(self, target_id=None):
        if self.disconnected:
            raise CDPConnectionError("Disconnected from Chrome")
        return None

    async def send_command(self, method: str, params=None, timeout=None):
        if self.disconnected:
            raise CDPConnectionError("Connection to Chrome closed")

        if method == "Page.navigate":
            target = params.get("url", "")
            self.current_url = target
            if "/target-page" in target:
                self.current_title = "Target Page Title"
            return {"frameId": "mock_frame_1"}

        elif method == "Runtime.evaluate":
            expr = params.get("expression", "")
            # Observation call
            if "visibleText" in expr and "interactive_elements" in expr:
                return {
                    "result": {
                        "value": {
                            "url": self.current_url,
                            "title": self.current_title,
                            "visible_text": f"JARVIS Controlled Test Page Counter: {self.counter_val}",
                            "interactive_elements": [
                                {
                                    "index": 1,
                                    "tag": "button",
                                    "role": "button",
                                    "text": "Increment Counter",
                                    "selector": "#counter-btn",
                                },
                                {
                                    "index": 2,
                                    "tag": "input",
                                    "role": "input",
                                    "text": "Your name",
                                    "selector": "#user-input",
                                },
                                {
                                    "index": 3,
                                    "tag": "input",
                                    "role": "input",
                                    "type": "password",
                                    "text": "Your password",
                                    "selector": "#pass-input",
                                },
                                {
                                    "index": 4,
                                    "tag": "button",
                                    "role": "button",
                                    "text": "Submit Order",
                                    "selector": "#submit-order",
                                },
                                {
                                    "index": 5,
                                    "tag": "a",
                                    "role": "link",
                                    "text": "Proceed to Target Page",
                                    "selector": "#second-page-link",
                                },
                            ],
                        }
                    }
                }
            elif "return el ? el.value : null" in expr:
                return {"result": {"value": self.user_input_val}}
            elif "el.value =" in expr:
                import re
                m = re.search(r'el\.value = "([^"]*)"', expr)
                if m:
                    self.user_input_val = m.group(1)
                return {"result": {"value": {"found": True, "value": self.user_input_val}}}
            elif "el.click()" in expr:
                if "#counter-btn" in expr:
                    self.counter_val = "1"
                    return {"result": {"value": {"found": True}}}
                elif "#second-page-link" in expr:
                    self.current_url = f"{self.base_url}/target-page"
                    self.current_title = "Target Page Title"
                    return {"result": {"value": {"found": True}}}
                elif "#submit-order" in expr:
                    return {"result": {"value": {"found": True}}}
                elif "nonexistent" in expr:
                    return {"result": {"value": {"found": False}}}
                return {"result": {"value": {"found": True}}}

        return {}


@pytest.mark.anyio
async def test_controlled_page_observe_act_verify_flow(local_test_server: str):
    cdp = MockControlledCDPClient(base_url=local_test_server)
    controller = BrowserAutomationController(cdp_client=cdp)

    # 1. Observe controlled page
    obs_res = await controller.observe_page()
    assert obs_res["success"] is True
    obs = obs_res["observation"]
    assert obs["url"] == f"{local_test_server}/"
    assert obs["title"] == "JARVIS Integration Test Page"
    assert len(obs["interactive_elements"]) == 5

    # 2. Type into text input (#user-input)
    type_res = await controller.execute_action(
        action="type",
        selector="#user-input",
        text="TestUser",
    )
    assert type_res["success"] is True
    assert type_res["verified"] is True
    assert "TestUser" in type_res["verification_evidence"]

    # 3. Click non-consequential button (#counter-btn)
    click_res = await controller.execute_action(
        action="click",
        selector="#counter-btn",
    )
    assert click_res["success"] is True
    assert click_res["verified"] is True

    # 4. Attempt consequential action (#submit-order) without confirmation -> must be blocked
    submit_res = await controller.execute_action(
        action="click",
        element_index=4,  # #submit-order
        user_confirmed=False,
    )
    assert submit_res["success"] is False
    assert submit_res["requires_confirmation"] is True
    assert "action_id" in submit_res
    action_id = submit_res["action_id"]

    # Model cannot self-approve via user_confirmed=True
    submit_self = await controller.execute_action(
        action="click",
        element_index=4,
        user_confirmed=True,
    )
    assert submit_self["success"] is False
    assert submit_self["requires_confirmation"] is True

    # 5. Execute consequential action with trusted user approval -> allowed
    ok, msg, token = default_approval_manager.approve_action(action_id, channel="trusted_user")
    assert ok is True
    submit_confirmed = await controller.execute_action(
        action="click",
        element_index=4,
        action_id=action_id,
        approval_token=token,
    )
    assert submit_confirmed["success"] is True
    assert submit_confirmed["verified"] is True

    # 6. Click navigation link to proceed to target page
    nav_res = await controller.execute_action(
        action="click",
        selector="#second-page-link",
    )
    assert nav_res["success"] is True
    assert nav_res["verified"] is True
    assert "target-page" in nav_res["verification_evidence"]


@pytest.mark.anyio
async def test_controlled_page_browser_disconnection(local_test_server: str):
    cdp = MockControlledCDPClient(base_url=local_test_server)
    controller = BrowserAutomationController(cdp_client=cdp)

    # Simulate Chrome sudden crash or disconnect
    cdp.disconnected = True

    obs_res = await controller.observe_page()
    assert obs_res["success"] is False
    assert obs_res.get("available") is False

    act_res = await controller.execute_action("click", selector="#counter-btn")
    assert act_res["success"] is False
    assert act_res.get("available") is False
