import { spawn } from "node:child_process";
import fs from "node:fs";
import path from "node:path";

const CHROME_PATH = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome";
const PORT = 9222;
const FRONTEND_URL = "http://127.0.0.1:5173/";
const SCREENSHOTS_DIR = "/Users/saivenkat/Projects/JARVIS/screenshots";
const ARTIFACTS_DIR = "/Users/saivenkat/.gemini/antigravity-ide/brain/15ee3771-51b7-41e7-9bab-556e343410c0";

fs.mkdirSync(SCREENSHOTS_DIR, { recursive: true });
fs.mkdirSync(ARTIFACTS_DIR, { recursive: true });

// Purge any redundant / obsolete screenshots from screenshots/ directory
for (const file of fs.readdirSync(SCREENSHOTS_DIR)) {
  if (file.endsWith(".png")) {
    fs.unlinkSync(path.join(SCREENSHOTS_DIR, file));
  }
}
console.log("Cleared redundant screenshots from screenshots directory.");

function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

class CDPClient {
  constructor(wsUrl) {
    this.wsUrl = wsUrl;
    this.id = 0;
    this.callbacks = new Map();
  }

  async connect() {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(this.wsUrl);
      this.ws.onopen = () => resolve();
      this.ws.onerror = (err) => reject(err);
      this.ws.onmessage = (event) => {
        const msg = JSON.parse(event.data);
        if (msg.id && this.callbacks.has(msg.id)) {
          const { resolve, reject } = this.callbacks.get(msg.id);
          this.callbacks.delete(msg.id);
          if (msg.error) {
            reject(new Error(msg.error.message || JSON.stringify(msg.error)));
          } else {
            resolve(msg.result);
          }
        }
      };
    });
  }

  send(method, params = {}) {
    return new Promise((resolve, reject) => {
      const id = ++this.id;
      this.callbacks.set(id, { resolve, reject });
      this.ws.send(JSON.stringify({ id, method, params }));
    });
  }

  close() {
    if (this.ws) {
      this.ws.close();
    }
  }
}

async function capture() {
  console.log("Launching Headless Chrome...");
  const chromeProc = spawn(
    CHROME_PATH,
    [
      "--headless=new",
      `--remote-debugging-port=${PORT}`,
      "--window-size=1150,800",
      "--hide-scrollbars",
      "--disable-gpu",
      "--no-sandbox",
      FRONTEND_URL,
    ],
    { stdio: "ignore" }
  );

  try {
    let connected = false;
    let targets = null;
    for (let i = 0; i < 20; i++) {
      await sleep(500);
      try {
        const res = await fetch(`http://127.0.0.1:${PORT}/json/list`);
        if (res.ok) {
          targets = await res.json();
          if (targets && targets.length > 0) {
            connected = true;
            break;
          }
        }
      } catch (e) {
        // retry
      }
    }

    if (!connected || !targets) {
      throw new Error("Failed to connect to Chrome DevTools endpoint");
    }

    const pageTarget = targets.find((t) => t.type === "page") || targets[0];
    console.log("Connecting to target:", pageTarget.title, pageTarget.webSocketDebuggerUrl);

    const client = new CDPClient(pageTarget.webSocketDebuggerUrl);
    await client.connect();

    await client.send("Page.enable");
    await client.send("Runtime.enable");
    await client.send("DOM.enable");

    // Configure 2x retina device metrics override
    await client.send("Emulation.setDeviceMetricsOverride", {
      width: 1150,
      height: 800,
      deviceScaleFactor: 2,
      mobile: false,
    });

    console.log("Waiting for backend bridge data & initial render...");
    await sleep(2000);

    async function takeScreenshot(filename, clip = null) {
      const params = { format: "png" };
      if (clip) params.clip = clip;
      const res = await client.send("Page.captureScreenshot", params);
      const buffer = Buffer.from(res.data, "base64");
      
      const mainPath = path.join(SCREENSHOTS_DIR, filename);
      const artifactPath = path.join(ARTIFACTS_DIR, filename);
      fs.writeFileSync(mainPath, buffer);
      fs.writeFileSync(artifactPath, buffer);
      console.log(`Saved screenshot: ${filename} (${buffer.length} bytes)`);
    }

    // 1. Standby State: Full window with new side-by-side squircle mic and speaker controls in header
    console.log("Capturing 01_jarvis_standby.png...");
    await takeScreenshot("01_jarvis_standby.png");

    // 2. User Live Voice Shimmer Transcription: Voice input active with shimmering transcription bar above composer
    console.log("Capturing 02_user_voice_shimmer.png...");
    await client.send("Runtime.evaluate", {
      expression: `(() => {
        const reactor = document.getElementById("reactor");
        const statusDot = document.getElementById("status-dot");
        const telemetryStatus = document.getElementById("telemetry-status-text");
        const centerState = document.getElementById("center-state-text");
        const centerSub = document.getElementById("center-substate-text");
        const micBtn = document.getElementById("mic-btn");
        const transcriptionContainer = document.getElementById("transcription-container");
        const transcriptionText = document.getElementById("transcription-text");

        reactor.className = "reactor reactor-listening";
        reactor.setAttribute("data-state", "listening");
        centerState.textContent = "LISTENING";
        centerSub.textContent = "VOICE INPUT";
        statusDot.className = "status-dot listening";
        telemetryStatus.textContent = "LISTENING";
        
        // Mic button in active ON state (cyan squircle with clean unslashed mic icon)
        micBtn.className = "squircle-btn active";
        micBtn.setAttribute("aria-pressed", "true");
        const micIconBox = document.getElementById("mic-icon-box");
        if (micIconBox) {
          micIconBox.innerHTML = \`
            <svg id="mic-icon" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path>
              <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
              <line x1="12" y1="19" x2="12" y2="22"></line>
            </svg>
          \`;
        }

        // Show Floating Shimmer Transcription above composer with Word Wrap (No box, no border)
        transcriptionContainer.style.display = "flex";
        transcriptionText.textContent = "JARVIS, initiate full system diagnostic scan and inspect neural routing status across local LLM bridge";
      })()`,
    });
    await sleep(600);
    await takeScreenshot("02_user_voice_shimmer.png");

    // 3. JARVIS Streaming Response: Live word streaming with blinking glowing cursor & speaking state
    console.log("Capturing 03_jarvis_streaming_speaking.png...");
    await client.send("Runtime.evaluate", {
      expression: `(() => {
        const reactor = document.getElementById("reactor");
        const statusDot = document.getElementById("status-dot");
        const telemetryStatus = document.getElementById("telemetry-status-text");
        const centerState = document.getElementById("center-state-text");
        const centerSub = document.getElementById("center-substate-text");
        const micBtn = document.getElementById("mic-btn");
        const transcriptionContainer = document.getElementById("transcription-container");
        const chatScroll = document.getElementById("chat-scroll");

        // Reactor in speaking state
        reactor.className = "reactor reactor-speaking";
        reactor.setAttribute("data-state", "speaking");
        centerState.textContent = "SPEAKING";
        centerSub.textContent = "SYNTHESIZING";
        statusDot.className = "status-dot speaking";
        telemetryStatus.textContent = "SPEAKING";

        // Mic back to muted / standby (coral single border, NO blue border)
        micBtn.className = "squircle-btn muted";
        micBtn.setAttribute("aria-pressed", "false");
        const micIconBox = document.getElementById("mic-icon-box");
        if (micIconBox) {
          micIconBox.innerHTML = \`
            <svg id="mic-icon" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
              <line x1="2" y1="2" x2="22" y2="22"></line>
              <path d="M18.89 13.23A7.12 7.12 0 0 0 19 12v-2"></path>
              <path d="M5 10v2a7 7 0 0 0 12 5"></path>
              <path d="M15 9.34V5a3 3 0 0 0-5.68-1.33"></path>
              <path d="M9 9v3a3 3 0 0 0 5.12 2.12"></path>
              <line x1="12" y1="19" x2="12" y2="22"></line>
            </svg>
          \`;
        }

        // Hide temporary transcription bar (dispatched to JARVIS)
        transcriptionContainer.style.display = "none";

        // User message bubble
        const userWrap = document.createElement("div");
        userWrap.className = "user-message-wrap";
        userWrap.innerHTML = \`
          <div class="user-message">JARVIS, run full diagnostic scan and inspect neural routing status</div>
          <div class="message-timestamp">1:48 PM</div>
        \`;
        chatScroll.appendChild(userWrap);

        // Streaming agent response bubble with animated streaming cursor
        const agentWrap = document.createElement("div");
        agentWrap.className = "agent-message-wrap";
        agentWrap.innerHTML = \`
          <div class="agent-message">
            <div class="agent-label">
              <div class="agent-avatar">
                <svg xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                  <path d="M2 10v3"></path>
                  <path d="M6 6v11"></path>
                  <path d="M10 3v18"></path>
                  <path d="M14 8v7"></path>
                  <path d="M18 5v13"></path>
                  <path d="M22 10v3"></path>
                </svg>
              </div>
              <span>JARVIS <span>• STREAMING RESPONSE</span></span>
            </div>
            <div class="agent-content">
              <p>Diagnostics sequence complete, sir. All neural routing channels and localized subroutines are performing within nominal parameters. System telemetry is actively synchronized with the FastAPI bridge on port 8000.<span class="streaming-cursor"></span></p>
            </div>
          </div>
          <div class="message-timestamp agent-timestamp">1:48 PM</div>
        \`;
        chatScroll.appendChild(agentWrap);
        chatScroll.scrollTop = chatScroll.scrollHeight;
      })()`,
    });
    await sleep(600);
    await takeScreenshot("03_jarvis_streaming_speaking.png");

    // 4. Operational Workspace Chat Panel Close-up
    console.log("Capturing 04_workspace_chat_panel.png...");
    // Finish stream by removing cursor and adding tool badges to the latest message
    await client.send("Runtime.evaluate", {
      expression: `(() => {
        const cursor = document.querySelector(".streaming-cursor");
        if (cursor) cursor.remove();
        const agentContents = document.querySelectorAll(".agent-message .agent-content");
        const latestP = agentContents[agentContents.length - 1];
        if (latestP) {
          const badgeWrap = document.createElement("div");
          badgeWrap.innerHTML = \`
            <div style="display:flex; flex-wrap:wrap; gap:6px; margin-top:8px;">
              <span style="font-family:var(--font-mono); font-size:10px; background:rgba(34,211,238,0.12); color:var(--primary); padding:2px 8px; border-radius:4px; border:1px solid rgba(34,211,238,0.25);">TOOL: SYSTEM_DIAGNOSTICS (2ms)</span>
              <span style="font-family:var(--font-mono); font-size:10px; background:rgba(34,211,238,0.12); color:var(--primary); padding:2px 8px; border-radius:4px; border:1px solid rgba(34,211,238,0.25);">ROUTER: OLLAMA / LOCAL</span>
            </div>
          \`;
          latestP.appendChild(badgeWrap);
        }
      })()`,
    });
    await sleep(400);
    await takeScreenshot("04_workspace_chat_panel.png", {
      x: 480,
      y: 29, // below titlebar
      width: 670,
      height: 771,
      scale: 1,
    });

    console.log("All 4 curated snapshots captured successfully!");
    client.close();
  } finally {
    chromeProc.kill("SIGTERM");
  }
}

capture().catch((err) => {
  console.error("Snapshot capture failed:", err);
  process.exit(1);
});
