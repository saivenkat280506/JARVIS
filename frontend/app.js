/**
 * JARVIS Operational Workspace — Client Engine
 * 
 * Features:
 *  - 180-tick Concentric SVG Arc Reactor HUD
 *  - Responsive State Machine: STANDBY, LISTENING, THINKING, SPEAKING
 *  - Inner Ring Bounces on response (clean, bar-free center)
 *  - Speech Recognition (Microphone / STT via Web Speech API)
 *  - Speech Synthesis (Audio response / TTS with inner ring bounce sync)
 *  - Real-time Backend API Bridge to http://127.0.0.1:8000
 *  - Dynamic Telemetry HUD with live diagnostic telemetry
 */

document.addEventListener("DOMContentLoaded", () => {
  // --- DOM References ---
  const reactor = document.getElementById("reactor");
  const ticksGroup = document.getElementById("ticks-group");
  const centerStateText = document.getElementById("center-state-text");
  const centerSubstateText = document.getElementById("center-substate-text");
  const statusDot = document.getElementById("status-dot");
  const telemetryStatusText = document.getElementById("telemetry-status-text");
  const telemetryLatency = document.getElementById("telemetry-latency");
  const telemetryVoice = document.getElementById("telemetry-voice");
  const telemetryRouter = document.getElementById("telemetry-router");
  const telemetryMemory = document.getElementById("telemetry-memory");
  const telemetryBridge = document.getElementById("telemetry-bridge");
  
  const headerRoutingText = document.getElementById("header-routing-text");
  const headerBridgeText = document.getElementById("header-bridge-text");

  const chatScroll = document.getElementById("chat-scroll");
  const composerForm = document.getElementById("composer-form");
  const commandInput = document.getElementById("command-input");
  const sendBtn = document.getElementById("send-btn");
  const micBtn = document.getElementById("mic-btn");
  const micIconBox = document.getElementById("mic-icon-box");
  const speakerBtn = document.getElementById("speaker-btn");
  const speakerIconBox = document.getElementById("speaker-icon-box");
  const clearChatBtn = document.getElementById("clear-chat-btn");
  const voiceNote = document.getElementById("voice-note");
  const transcriptionContainer = document.getElementById("transcription-container");
  const transcriptionText = document.getElementById("transcription-text");

  // --- Configuration & State ---
  const BRIDGE_URL = "http://127.0.0.1:8000";
  let bridgeConnected = false;
  let currentState = "standby";
  let isProcessing = false;
  let currentUtterance = null;
  let speechRecognition = null;
  let isListeningWithMic = false;
  let speakerEnabled = true;
  let latestMeasuredLatency = null;
  let lastDiagnostics = null;

  const stateSubtitles = {
    standby: "READY",
    listening: "MIC ACTIVE",
    thinking: "PROCESSING",
    speaking: "SYNTHESIZING"
  };

  // --- 1. Generate Arc Reactor 180 Precision Ticks ---
  function generateReactorTicks() {
    if (!ticksGroup) return;
    const count = 180;
    const cx = 220;
    const cy = 220;
    const outerR = 215;

    let svgHtml = "";
    for (let i = 0; i < count; i++) {
      const angle = (i * Math.PI) / 90; // 2 deg per tick
      const isMajor = (i % 15 === 0);
      const isMedium = (i % 5 === 0);
      const innerR = isMajor ? 204 : (isMedium ? 208 : 211);
      const strokeWidth = isMajor ? 1.5 : 0.7;

      const x1 = cx + Math.sin(angle) * innerR;
      const y1 = cy - Math.cos(angle) * innerR;
      const x2 = cx + Math.sin(angle) * outerR;
      const y2 = cy - Math.cos(angle) * outerR;

      svgHtml += `<line x1="${x1.toFixed(3)}" y1="${y1.toFixed(3)}" x2="${x2.toFixed(3)}" y2="${y2.toFixed(3)}" class="ring-tick" stroke-width="${strokeWidth}"></line>`;
    }
    ticksGroup.innerHTML = svgHtml;
  }
  generateReactorTicks();

  // --- 2. Live Telemetry HUD Synchronization ---
  function updateTelemetryUI() {
    if (bridgeConnected && lastDiagnostics) {
      if (telemetryBridge) telemetryBridge.textContent = "READY";
      if (headerBridgeText) {
        headerBridgeText.innerHTML = `<span style="color:var(--primary); font-weight:600;">ONLINE (PORT ${lastDiagnostics.bridge?.port || 8000})</span>`;
      }

      // Router provider & model
      const routerData = lastDiagnostics.router || {};
      const modelName = (routerData.model || lastDiagnostics.llm_model || "UNKNOWN").toUpperCase();
      const providerName = (routerData.provider || "OLLAMA").toUpperCase();
      const routerDisplay = routerData.display || `${providerName} / ${modelName}`;
      if (telemetryRouter) telemetryRouter.textContent = routerDisplay;
      if (headerRoutingText) {
        headerRoutingText.textContent = routerData.ollama_online ? `AUTO-RESOLVE (${modelName})` : `LOCAL AGENT (${modelName})`;
      }

      // Voice engine state
      const voiceState = lastDiagnostics.voice_engine?.state || lastDiagnostics.voice_engine_state || "UNAVAILABLE";
      if (telemetryVoice) telemetryVoice.textContent = voiceState;

      // Memory metrics
      const memDisplay = lastDiagnostics.memory?.display || "N/A";
      if (telemetryMemory) telemetryMemory.textContent = memDisplay;

      // Measured latency
      if (latestMeasuredLatency !== null) {
        if (telemetryLatency) telemetryLatency.textContent = `${latestMeasuredLatency}ms`;
      } else if (lastDiagnostics.latency?.display) {
        if (telemetryLatency) telemetryLatency.textContent = lastDiagnostics.latency.display;
      } else if (lastDiagnostics.latest_latency_ms !== null && lastDiagnostics.latest_latency_ms !== undefined) {
        if (telemetryLatency) telemetryLatency.textContent = `${Math.round(lastDiagnostics.latest_latency_ms)}ms`;
      } else {
        if (telemetryLatency) telemetryLatency.textContent = "N/A";
      }

      // System lifecycle status
      if (telemetryStatusText) telemetryStatusText.textContent = currentState.toUpperCase();
      if (statusDot) statusDot.className = `status-dot ${currentState}`;
      if (centerStateText) centerStateText.textContent = currentState.toUpperCase();
      if (centerSubstateText) centerSubstateText.textContent = stateSubtitles[currentState] || "READY";
      if (reactor) {
        reactor.className = `reactor reactor-${currentState}`;
        reactor.setAttribute("data-state", currentState);
        reactor.setAttribute("aria-label", `JARVIS ${currentState}`);
      }
    } else {
      // Offline / Unreachable backend state
      if (telemetryBridge) telemetryBridge.textContent = "UNAVAILABLE";
      if (headerBridgeText) {
        headerBridgeText.innerHTML = `<span style="color:var(--destructive);">UNAVAILABLE</span>`;
      }
      if (headerRoutingText) headerRoutingText.textContent = "OFFLINE";
      if (telemetryRouter) telemetryRouter.textContent = "UNKNOWN";
      if (telemetryVoice) telemetryVoice.textContent = "UNKNOWN";
      if (telemetryMemory) telemetryMemory.textContent = "N/A";
      if (telemetryLatency) {
        telemetryLatency.textContent = latestMeasuredLatency !== null ? `${latestMeasuredLatency}ms` : "N/A";
      }
      if (statusDot) statusDot.className = "status-dot unavailable";
      if (telemetryStatusText) telemetryStatusText.textContent = "UNAVAILABLE";
      if (centerStateText) centerStateText.textContent = "OFFLINE";
      if (centerSubstateText) centerSubstateText.textContent = "BRIDGE DISCONNECTED";
      if (reactor) {
        reactor.className = "reactor reactor-standby reactor-offline";
        reactor.setAttribute("data-state", "offline");
        reactor.setAttribute("aria-label", "JARVIS offline");
      }
    }
  }

  // --- 3. State Machine Transitions ---
  function setState(state) {
    currentState = state;
    if (reactor) {
      reactor.className = `reactor reactor-${state}${!bridgeConnected ? " reactor-offline" : ""}`;
      reactor.setAttribute("data-state", state);
      reactor.setAttribute("aria-label", `JARVIS ${state}`);
    }

    if (centerStateText) centerStateText.textContent = state.toUpperCase();
    if (centerSubstateText) centerSubstateText.textContent = stateSubtitles[state] || "READY";

    if (statusDot) {
      statusDot.className = bridgeConnected ? `status-dot ${state}` : "status-dot unavailable";
    }
    if (telemetryStatusText) {
      telemetryStatusText.textContent = bridgeConnected ? state.toUpperCase() : "UNAVAILABLE";
    }

    if (state === "listening") {
      if (micBtn) {
        micBtn.classList.add("active");
        micBtn.setAttribute("aria-pressed", "true");
      }
      if (micToggleLabel) micToggleLabel.textContent = "Microphone on";
    } else {
      if (micBtn) {
        micBtn.classList.remove("active");
        micBtn.setAttribute("aria-pressed", "false");
      }
      if (micToggleLabel) micToggleLabel.textContent = "Microphone off";
    }
  }

  // --- 4. Speech Synthesis (TTS Audio Output) ---
  function speakResponse(text, onComplete) {
    if (!speakerEnabled) {
      if (onComplete) onComplete();
      return;
    }

    if (!window.speechSynthesis) {
      setState("speaking");
      const duration = Math.min(6000, Math.max(2500, text.length * 45));
      setTimeout(() => {
        setState("standby");
        if (onComplete) onComplete();
      }, duration);
      return;
    }

    try {
      window.speechSynthesis.cancel();

      const spokenText = text
        .replace(/```[\s\S]*?```/g, "Code block omitted.")
        .replace(/`([^`]+)`/g, "$1")
        .replace(/\*\*([^*]+)\*\*/g, "$1")
        .replace(/\*([^*]+)\*/g, "$1")
        .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
        .trim();

      const utterance = new SpeechSynthesisUtterance(spokenText);
      currentUtterance = utterance;

      const voices = window.speechSynthesis.getVoices();
      const preferredVoice = voices.find(v => 
        (v.name.includes("Daniel") || v.name.includes("Oliver") || v.name.includes("Google UK English Male") || v.name.includes("Arthur")) && v.lang.startsWith("en")
      ) || voices.find(v => v.lang.startsWith("en"));

      if (preferredVoice) {
        utterance.voice = preferredVoice;
      }

      utterance.rate = 1.05;
      utterance.pitch = 0.96;

      utterance.onstart = () => {
        setState("speaking");
      };

      utterance.onend = () => {
        currentUtterance = null;
        setState("standby");
        if (onComplete) onComplete();
      };

      utterance.onerror = (e) => {
        console.warn("TTS error:", e);
        currentUtterance = null;
        setState("standby");
        if (onComplete) onComplete();
      };

      window.speechSynthesis.speak(utterance);
    } catch (err) {
      console.warn("Speech synthesis failed:", err);
      setState("speaking");
      setTimeout(() => {
        setState("standby");
        if (onComplete) onComplete();
      }, 3000);
    }
  }

  if (window.speechSynthesis) {
    window.speechSynthesis.onvoiceschanged = () => {
      window.speechSynthesis.getVoices();
    };
  }

  // --- 5. Speech Recognition & Live Shimmer Transcription ---
  const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;

  function updateMicUI() {
    if (!micBtn) return;
    if (isListeningWithMic) {
      micBtn.classList.add("active");
      micBtn.classList.remove("muted", "off");
      micBtn.setAttribute("aria-pressed", "true");
      micBtn.setAttribute("title", "Microphone: Active listening (Click to mute)");
      if (micIconBox) {
        micIconBox.innerHTML = `
          <svg id="mic-icon" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M12 2a3 3 0 0 0-3 3v7a3 3 0 0 0 6 0V5a3 3 0 0 0-3-3Z"></path>
            <path d="M19 10v2a7 7 0 0 1-14 0v-2"></path>
            <line x1="12" y1="19" x2="12" y2="22"></line>
          </svg>
        `;
      }
    } else {
      micBtn.classList.remove("active");
      micBtn.classList.add("muted");
      micBtn.setAttribute("aria-pressed", "false");
      micBtn.setAttribute("title", "Microphone: Muted / Idle (Click to activate)");
      if (micIconBox) {
        micIconBox.innerHTML = `
          <svg id="mic-icon" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <line x1="2" y1="2" x2="22" y2="22"></line>
            <path d="M18.89 13.23A7.12 7.12 0 0 0 19 12v-2"></path>
            <path d="M5 10v2a7 7 0 0 0 12 5"></path>
            <path d="M15 9.34V5a3 3 0 0 0-5.68-1.33"></path>
            <path d="M9 9v3a3 3 0 0 0 5.12 2.12"></path>
            <line x1="12" y1="19" x2="12" y2="22"></line>
          </svg>
        `;
      }
    }
  }

  function updateSpeakerUI() {
    if (!speakerBtn) return;
    if (speakerEnabled) {
      speakerBtn.classList.add("active");
      speakerBtn.classList.remove("muted", "off");
      speakerBtn.setAttribute("aria-pressed", "true");
      speakerBtn.setAttribute("title", "Voice output speaker: Active (Click to mute)");
      if (speakerIconBox) {
        speakerIconBox.innerHTML = `
          <svg id="speaker-icon" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
            <path d="M15.54 8.46a5 5 0 0 1 0 7.07"></path>
            <path d="M19.07 4.93a10 10 0 0 1 0 14.14"></path>
          </svg>
        `;
      }
    } else {
      speakerBtn.classList.remove("active");
      speakerBtn.classList.add("muted");
      speakerBtn.setAttribute("aria-pressed", "false");
      speakerBtn.setAttribute("title", "Voice output speaker: Muted (Click to enable)");
      if (speakerIconBox) {
        speakerIconBox.innerHTML = `
          <svg id="speaker-icon" xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
            <line x1="22" y1="9" x2="16" y2="15"></line>
            <line x1="16" y1="9" x2="22" y2="15"></line>
          </svg>
        `;
      }
    }
  }

  if (SpeechRecognition) {
    speechRecognition = new SpeechRecognition();
    speechRecognition.continuous = false;
    speechRecognition.interimResults = true;
    speechRecognition.lang = "en-US";

    speechRecognition.onstart = () => {
      isListeningWithMic = true;
      setState("listening");
      updateMicUI();
      if (transcriptionContainer) {
        transcriptionContainer.style.display = "flex";
      }
      if (transcriptionText) {
        transcriptionText.textContent = "Listening for speech command…";
      }
    };

    speechRecognition.onresult = (event) => {
      let interim = "";
      let finalTranscript = "";

      for (let i = event.resultIndex; i < event.results.length; ++i) {
        if (event.results[i].isFinal) {
          finalTranscript += event.results[i][0].transcript;
        } else {
          interim += event.results[i][0].transcript;
        }
      }

      const activeText = (finalTranscript || interim).trim();
      if (activeText && transcriptionText) {
        transcriptionText.textContent = activeText;
      }

      // Once synthesized/finalized, auto-send directly to JARVIS
      if (finalTranscript.trim()) {
        const captured = finalTranscript.trim();
        if (transcriptionText) {
          transcriptionText.textContent = captured;
        }
        setTimeout(() => {
          if (transcriptionContainer) {
            transcriptionContainer.style.display = "none";
          }
          if (captured && !isProcessing) {
            submitCommand(captured);
          }
        }, 380);
      }
    };

    speechRecognition.onerror = (event) => {
      console.warn("Speech recognition error:", event.error);
      isListeningWithMic = false;
      updateMicUI();
      if (transcriptionText) {
        transcriptionText.textContent = `Mic status: ${event.error || "idle"}`;
      }
      setTimeout(() => {
        if (currentState === "listening") setState("standby");
        if (transcriptionContainer) transcriptionContainer.style.display = "none";
      }, 1800);
    };

    speechRecognition.onend = () => {
      isListeningWithMic = false;
      updateMicUI();
      if (currentState === "listening" && !isProcessing) {
        setState("standby");
        if (transcriptionContainer) transcriptionContainer.style.display = "none";
      }
    };
  }

  micBtn.addEventListener("click", () => {
    if (isProcessing) return;

    if (currentUtterance && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }

    if (speechRecognition) {
      if (isListeningWithMic) {
        speechRecognition.stop();
        isListeningWithMic = false;
        updateMicUI();
        setState("standby");
        if (transcriptionContainer) transcriptionContainer.style.display = "none";
      } else {
        try {
          speechRecognition.start();
        } catch (e) {
          console.warn("Mic start exception:", e);
        }
      }
    } else {
      // Fallback / simulation mode when browser STT is unsupported
      if (isListeningWithMic) {
        isListeningWithMic = false;
        updateMicUI();
        setState("standby");
        if (transcriptionContainer) transcriptionContainer.style.display = "none";
      } else {
        isListeningWithMic = true;
        updateMicUI();
        setState("listening");
        if (transcriptionContainer) transcriptionContainer.style.display = "flex";
        if (transcriptionText) transcriptionText.textContent = "Simulating voice input: \"Run system diagnostic status\"…";
        setTimeout(() => {
          if (isListeningWithMic) {
            isListeningWithMic = false;
            updateMicUI();
            if (transcriptionContainer) transcriptionContainer.style.display = "none";
            submitCommand("Run system diagnostic status");
          }
        }, 2200);
      }
    }
  });

  if (speakerBtn) {
    speakerBtn.addEventListener("click", () => {
      speakerEnabled = !speakerEnabled;
      if (!speakerEnabled && window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
      updateSpeakerUI();
    });
  }

  // Initialize button visual states
  updateMicUI();
  updateSpeakerUI();

  // --- 6. Backend Bridge Client (Real Diagnostic Telemetry) ---
  async function checkBridgeStatus() {
    try {
      // Try /api/diagnostics first, then fall back to /api/status
      let resp = await fetch(`${BRIDGE_URL}/api/diagnostics`, {
        method: "GET",
        headers: { "Accept": "application/json" },
        cache: "no-store",
        signal: AbortSignal.timeout(2200),
      });

      if (!resp.ok) {
        resp = await fetch(`${BRIDGE_URL}/api/status`, {
          method: "GET",
          headers: { "Accept": "application/json" },
          cache: "no-store",
          signal: AbortSignal.timeout(2200),
        });
      }

      if (resp.ok) {
        const data = await resp.json();
        bridgeConnected = true;
        lastDiagnostics = data;

        if (data.latency?.latest_ms !== undefined && data.latency.latest_ms !== null) {
          latestMeasuredLatency = Math.round(data.latency.latest_ms);
        } else if (data.latest_latency_ms !== undefined && data.latest_latency_ms !== null) {
          latestMeasuredLatency = Math.round(data.latest_latency_ms);
        }

        updateTelemetryUI();
        return data;
      } else {
        throw new Error(`Status ${resp.status}`);
      }
    } catch {
      bridgeConnected = false;
      lastDiagnostics = null;
      updateTelemetryUI();
    }
    return null;
  }

  // Initial check and periodic polling (reasonable 4-second interval)
  checkBridgeStatus();
  setInterval(checkBridgeStatus, 4000);

  // --- 7. Genuine Chat Message Rendering with Timestamps ---
  let chatHistory = [];

  function formatTimestamp(timeValue) {
    if (!timeValue) {
      return new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    }
    const d = new Date(timeValue);
    if (isNaN(d.getTime())) {
      return new Date().toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
    }
    return d.toLocaleTimeString([], { hour: "numeric", minute: "2-digit" });
  }

  function renderMessages(messages) {
    chatScroll.innerHTML = "";
    if (!messages || messages.length === 0) {
      chatScroll.innerHTML = `
        <div class="chat-empty-state">
          <svg xmlns="http://www.w3.org/2000/svg" width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z"></path>
          </svg>
          <div style="font-weight: 500;">Private local workspace ready.</div>
          <div style="font-size: 11px; font-family: var(--font-mono); opacity: 0.7;">Type a message or use microphone to command JARVIS</div>
        </div>
      `;
      return;
    }

    messages.forEach(msg => {
      const timeStr = formatTimestamp(msg.created_at || msg.timestamp);
      if (msg.role === "user") {
        const wrap = document.createElement("div");
        wrap.className = "user-message-wrap";

        const bubble = document.createElement("div");
        bubble.className = "user-message";
        bubble.textContent = msg.text || msg.content;

        const stamp = document.createElement("div");
        stamp.className = "message-timestamp";
        stamp.textContent = timeStr;

        wrap.appendChild(bubble);
        wrap.appendChild(stamp);
        chatScroll.appendChild(wrap);
      } else {
        const wrap = document.createElement("div");
        wrap.className = "agent-message-wrap";

        const agentDiv = document.createElement("div");
        agentDiv.className = "agent-message";

        // Render tool badges only when actual tools were invoked
        let toolHtml = "";
        if (msg.tools && msg.tools.length > 0) {
          toolHtml = `
            <div style="display:flex; flex-wrap:wrap; gap:6px; margin-top:8px;">
              ${msg.tools.map(t => `<span style="font-family:var(--font-mono); font-size:10px; background:rgba(34,211,238,0.12); color:var(--primary); padding:2px 8px; border-radius:4px; border:1px solid rgba(34,211,238,0.25);">TOOL: ${t.tool || "EXEC"} (${t.latency_ms || 1}ms)</span>`).join("")}
            </div>
          `;
        }

        agentDiv.innerHTML = `
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
            <span>JARVIS</span>
          </div>
          <div class="agent-content">
            <p>${msg.text || msg.content}</p>
            ${toolHtml}
          </div>
        `;

        const stamp = document.createElement("div");
        stamp.className = "message-timestamp agent-timestamp";
        stamp.textContent = timeStr;

        wrap.appendChild(agentDiv);
        wrap.appendChild(stamp);
        chatScroll.appendChild(wrap);
      }
    });

    chatScroll.scrollTop = chatScroll.scrollHeight;
  }

  // Clear Chat Button Handler
  if (clearChatBtn) {
    clearChatBtn.addEventListener("click", async () => {
      if (currentUtterance && window.speechSynthesis) {
        window.speechSynthesis.cancel();
      }
      chatHistory = [];
      renderMessages([]);
      try {
        await fetch(`${BRIDGE_URL}/api/clear`, {
          method: "POST",
          signal: AbortSignal.timeout(2000),
        });
      } catch (err) {
        console.warn("Could not clear backend history:", err);
      }
    });
  }

  // Restore genuine persisted history from SQLite if available
  async function loadPersistedHistory() {
    try {
      const resp = await fetch(`${BRIDGE_URL}/api/history?conversation_id=deck_session`, {
        signal: AbortSignal.timeout(2000),
      });
      if (resp.ok) {
        const msgs = await resp.json();
        if (Array.isArray(msgs) && msgs.length > 0) {
          chatHistory = msgs.map(m => ({
            role: m.role === "assistant" ? "agent" : m.role,
            text: m.content,
            created_at: m.created_at,
          }));
          renderMessages(chatHistory);
        }
      }
    } catch {
      // Leave clean operational workspace
    }
  }

  renderMessages(chatHistory);
  loadPersistedHistory();

  // --- 8. Command Input Handlers ---
  commandInput.addEventListener("input", () => {
    sendBtn.disabled = !commandInput.value.trim() || isProcessing;
  });

  // Local deterministic fallback when backend bridge is disconnected
  function generateFallbackResponse(query) {
    const q = query.toLowerCase();
    if (/status|diagnostics|health|bridge/.test(q)) {
      return "Local bridge is currently offline. Start backend/bridge.py on port 8000 to enable real-time telemetry.";
    } else if (/who are you|jarvis|identity/.test(q)) {
      return "I am J.A.R.V.I.S., your autonomous localized intelligence agent. How may I assist you, sir?";
    } else if (/calculate|math|\+|\-|\*|\//.test(q)) {
      return `Offline arithmetic preview for "${query}". Reconnect the API bridge for verified tool execution.`;
    } else {
      return `Command received: "${query}". Offline fallback active — reconnect API bridge on port 8000 for agent processing.`;
    }
  }

  // --- 9. Interactive Command Submission & Execution ---
  async function submitCommand(query) {
    if (!query || isProcessing) return;

    if (currentUtterance && window.speechSynthesis) {
      window.speechSynthesis.cancel();
    }

    // Append User Message with timestamp
    const nowIso = new Date().toISOString();
    chatHistory.push({ role: "user", text: query, created_at: nowIso });
    renderMessages(chatHistory);
    commandInput.value = "";
    sendBtn.disabled = true;

    // Set Thinking State
    isProcessing = true;
    setState("thinking");

    // Temporary thinking spinner in chat
    const thinkingIndicator = document.createElement("div");
    thinkingIndicator.id = "thinking-indicator";
    thinkingIndicator.className = "agent-message";
    thinkingIndicator.innerHTML = `
      <div class="agent-label">
        <div class="agent-avatar">
          <svg class="animate-spin" xmlns="http://www.w3.org/2000/svg" width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
            <path d="M21 12a9 9 0 1 1-6.219-8.56"></path>
          </svg>
        </div>
        <span>JARVIS <span>• PROCESSING</span></span>
      </div>
      <div class="agent-content" style="padding-top: 6px; color: var(--muted-foreground); font-size: 13px;">
        Executing agent reasoning turn…
      </div>
    `;
    chatScroll.appendChild(thinkingIndicator);
    chatScroll.scrollTop = chatScroll.scrollHeight;

    let responseText = "";
    let toolsUsed = [];

    // Real execution via Backend Bridge
    if (bridgeConnected) {
      const tReq = performance.now();
      try {
        const resp = await fetch(`${BRIDGE_URL}/api/chat`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: query, conversation_id: "deck_session" }),
          signal: AbortSignal.timeout(25000),
        });

        if (resp.ok) {
          const result = await resp.json();
          responseText = result.response || "Task completed.";
          toolsUsed = result.tools_used || [];
          const measuredMs = result.latency_ms || Math.round(performance.now() - tReq);
          latestMeasuredLatency = Math.round(measuredMs);
          if (telemetryLatency) telemetryLatency.textContent = `${latestMeasuredLatency}ms`;
        } else {
          responseText = generateFallbackResponse(query);
        }
      } catch (err) {
        console.warn("Backend bridge request timed out or failed:", err);
        bridgeConnected = false;
        responseText = generateFallbackResponse(query);
      }
    } else {
      await new Promise(r => setTimeout(r, 600));
      responseText = generateFallbackResponse(query);
    }

    // Remove thinking indicator
    const ind = document.getElementById("thinking-indicator");
    if (ind) ind.remove();

    // Stream JARVIS response progressively into DOM with streaming cursor & synchronized TTS
    const replyIso = new Date().toISOString();
    const timeStr = formatTimestamp(replyIso);

    const wrap = document.createElement("div");
    wrap.className = "agent-message-wrap";

    const agentDiv = document.createElement("div");
    agentDiv.className = "agent-message";

    agentDiv.innerHTML = `
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
        <span>JARVIS</span>
      </div>
      <div class="agent-content">
        <p><span class="stream-text"></span><span class="streaming-cursor"></span></p>
        <div class="tool-badges-slot"></div>
      </div>
    `;

    const stamp = document.createElement("div");
    stamp.className = "message-timestamp agent-timestamp";
    stamp.textContent = timeStr;

    wrap.appendChild(agentDiv);
    wrap.appendChild(stamp);
    chatScroll.appendChild(wrap);
    chatScroll.scrollTop = chatScroll.scrollHeight;

    const streamText = agentDiv.querySelector(".stream-text");
    const streamCursor = agentDiv.querySelector(".streaming-cursor");
    const toolBadgesSlot = agentDiv.querySelector(".tool-badges-slot");

    // Start voice response playback
    speakResponse(responseText);

    // Progressive streaming of words with animated cursor
    const words = responseText.split(" ");
    let wordIdx = 0;
    let accumulated = "";

    await new Promise((resolve) => {
      const streamInterval = setInterval(() => {
        if (wordIdx < words.length) {
          accumulated += (wordIdx === 0 ? "" : " ") + words[wordIdx];
          if (streamText) streamText.textContent = accumulated;
          chatScroll.scrollTop = chatScroll.scrollHeight;
          wordIdx++;
        } else {
          clearInterval(streamInterval);
          if (streamCursor) streamCursor.remove();

          if (toolsUsed && toolsUsed.length > 0 && toolBadgesSlot) {
            toolBadgesSlot.innerHTML = `
              <div style="display:flex; flex-wrap:wrap; gap:6px; margin-top:8px;">
                ${toolsUsed.map(t => `<span style="font-family:var(--font-mono); font-size:10px; background:rgba(34,211,238,0.12); color:var(--primary); padding:2px 8px; border-radius:4px; border:1px solid rgba(34,211,238,0.25);">TOOL: ${t.tool || "EXEC"} (${t.latency_ms || 1}ms)</span>`).join("")}
              </div>
            `;
          }

          chatHistory.push({
            role: "agent",
            text: responseText,
            tools: toolsUsed,
            created_at: replyIso,
          });

          isProcessing = false;
          updateTelemetryUI();
          resolve();
        }
      }, 26);
    });
  }

  composerForm.addEventListener("submit", (e) => {
    e.preventDefault();
    submitCommand(commandInput.value.trim());
  });
});
