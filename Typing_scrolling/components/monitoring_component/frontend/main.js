(function () {
  function safeString(value) {
    if (value === null || value === undefined) return "";
    return String(value);
  }

  function getEl(id) {
    return document.getElementById(id);
  }

  const el = {
    startBtn: getEl("startBtn"),
    stopBtn: getEl("stopBtn"),
    clearBtn: getEl("clearBtn"),
    status: getEl("status"),
    typingArea: getEl("typingArea"),
    typedLen: getEl("typedLen"),
    typingCount: getEl("typingCount"),
    duration: getEl("duration"),
    warning: getEl("warning"),
  };

  function showWarning(message) {
    if (!el.warning) return;
    if (!message) {
      el.warning.style.display = "none";
      el.warning.textContent = "";
      return;
    }
    el.warning.textContent = String(message);
    el.warning.style.display = "block";
  }

  const missingIds = Object.entries(el)
    .filter(([, v]) => v === null)
    .map(([k]) => k);

  if (missingIds.length > 0) {
    // Try to surface a visible error but never throw (avoid blank iframe).
    showWarning(
      `Monitoring widget failed to initialize. Missing elements: ${missingIds.join(", ")}.`
    );
    return;
  }

  const state = {
    monitoringStatus: "idle",
    typingEvents: [],
    scrollEvents: [],
    startedAtMs: null,
    stoppedAtMs: null,
    componentVersion: "0.4.0",
  };

  function durationSeconds() {
    if (state.startedAtMs === null) return 0.0;
    const end = state.stoppedAtMs === null ? Date.now() : state.stoppedAtMs;
    const d = Math.max(0, end - state.startedAtMs);
    return d / 1000.0;
  }

  function setStatus(status) {
    state.monitoringStatus = status;
    el.status.textContent = status;
    el.status.classList.remove("idle", "monitoring", "stopped");
    el.status.classList.add(status);
  }

  function toPayload() {
    return {
      typed_text: safeString(el.typingArea.value),
      typing_events: state.typingEvents,
      scroll_events: state.scrollEvents,
      started_at_ms: state.startedAtMs,
      stopped_at_ms: state.stoppedAtMs,
      monitoring_status: state.monitoringStatus,
      component_version: state.componentVersion,
      event_counts: {
        typing: state.typingEvents.length,
        scroll: state.scrollEvents.length,
      },
    };
  }

  function updateSummary() {
    el.typedLen.textContent = String(safeString(el.typingArea.value).length);
    el.typingCount.textContent = String(state.typingEvents.length);
    el.duration.textContent = durationSeconds().toFixed(1);
  }

  function hasStreamlitBridge() {
    const s = window.Streamlit;
    return (
      !!s &&
      typeof s.setComponentReady === "function" &&
      typeof s.setComponentValue === "function" &&
      typeof s.setFrameHeight === "function"
    );
  }

  function pushValue() {
    if (!hasStreamlitBridge()) return;
    try {
      window.Streamlit.setComponentValue(toPayload());
    } catch (e) {
      // ignore
    }
  }

  function resetToIdle(shouldPush) {
    state.monitoringStatus = "idle";
    state.typingEvents = [];
    state.scrollEvents = [];
    state.startedAtMs = null;
    state.stoppedAtMs = null;
    el.typingArea.value = "";
    el.startBtn.disabled = false;
    el.stopBtn.disabled = true;
    setStatus("idle");
    updateSummary();
    if (shouldPush) {
      pushValue();
    }
  }

  el.startBtn.addEventListener("click", function () {
    showWarning("");
    state.typingEvents = [];
    state.scrollEvents = [];
    state.startedAtMs = Date.now();
    state.stoppedAtMs = null;
    el.typingArea.value = "";
    el.startBtn.disabled = true;
    el.stopBtn.disabled = false;
    setStatus("monitoring");
    updateSummary();
    pushValue();
  });

  el.stopBtn.addEventListener("click", function () {
    if (state.monitoringStatus !== "monitoring") {
      showWarning("Stop clicked before starting. Click Start Monitoring first.");
      return;
    }
    showWarning("");
    state.stoppedAtMs = Date.now();
    el.stopBtn.disabled = true;
    el.startBtn.disabled = false;
    setStatus("stopped");
    updateSummary();
    pushValue();
  });

  el.clearBtn.addEventListener("click", function () {
    showWarning("");
    resetToIdle(true);
  });

  el.typingArea.addEventListener("keydown", function (event) {
    if (state.monitoringStatus !== "monitoring") return;
    state.typingEvents.push({
      type: "keydown",
      key: safeString(event.key),
      code: safeString(event.code || ""),
      timestamp_ms: Date.now(),
      text_length: safeString(el.typingArea.value).length,
    });
    updateSummary();
  });

  el.typingArea.addEventListener("keyup", function (event) {
    if (state.monitoringStatus !== "monitoring") return;
    state.typingEvents.push({
      type: "keyup",
      key: safeString(event.key),
      code: safeString(event.code || ""),
      timestamp_ms: Date.now(),
      text_length: safeString(el.typingArea.value).length,
    });
    updateSummary();
  });

  // Streamlit bridge: keep UI usable even if the bridge doesn't load.
  if (hasStreamlitBridge()) {
    try {
      window.Streamlit.setComponentReady();
      window.Streamlit.setFrameHeight(430);

      if (window.Streamlit.events && window.Streamlit.RENDER_EVENT) {
        window.Streamlit.events.addEventListener(window.Streamlit.RENDER_EVENT, function () {
          updateSummary();
          try {
            window.Streamlit.setFrameHeight(430);
          } catch (e) {
            // ignore
          }
        });
      }
    } catch (e) {
      // ignore
    }
  } else {
    showWarning(
      "Streamlit bridge not loaded. Automatic transfer to Python may not work."
    );
  }

  resetToIdle(false);
  updateSummary();
})();
