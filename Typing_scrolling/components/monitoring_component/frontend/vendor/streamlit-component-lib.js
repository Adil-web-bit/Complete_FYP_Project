// Minimal Streamlit component bridge (no bundler / no npm).
//
// This file intentionally implements only the small subset of the Streamlit
// component API needed by this project:
// - Streamlit.setComponentReady()
// - Streamlit.setFrameHeight(height?)
// - Streamlit.setComponentValue(value)
// - Streamlit.events + Streamlit.RENDER_EVENT
//
// It uses the official postMessage protocol:
// - "streamlit:componentReady"
// - "streamlit:setFrameHeight"
// - "streamlit:setComponentValue"
//
// Notes:
// - Values are sent as JSON ("dataType": "json").
// - Theme injection is best-effort (safe if missing).
(function () {
  "use strict";

  var COMPONENT_READY = "streamlit:componentReady";
  var SET_COMPONENT_VALUE = "streamlit:setComponentValue";
  var SET_FRAME_HEIGHT = "streamlit:setFrameHeight";

  function _createEventTargetShim() {
    var listeners = {};
    return {
      addEventListener: function (type, callback) {
        if (!listeners[type]) listeners[type] = [];
        listeners[type].push(callback);
      },
      removeEventListener: function (type, callback) {
        if (!listeners[type]) return;
        listeners[type] = listeners[type].filter(function (cb) {
          return cb !== callback;
        });
      },
      dispatchEvent: function (event) {
        var type = event && event.type;
        if (!type || !listeners[type]) return true;
        listeners[type].forEach(function (cb) {
          try {
            cb(event);
          } catch (e) {
            // ignore listener errors
          }
        });
        return true;
      },
    };
  }

  function _makeEventTarget() {
    try {
      return new EventTarget();
    } catch (e) {
      return _createEventTargetShim();
    }
  }

  function _sendBackMsg(type, data) {
    var msg = { isStreamlitMessage: true, type: type };
    if (data && typeof data === "object") {
      for (var k in data) {
        if (Object.prototype.hasOwnProperty.call(data, k)) msg[k] = data[k];
      }
    }
    try {
      window.parent.postMessage(msg, "*");
    } catch (e) {
      // ignore
    }
  }

  function _injectTheme(theme) {
    if (!theme || typeof theme !== "object") return;
    try {
      var style = document.createElement("style");
      document.head.appendChild(style);
      style.innerHTML =
        "\n:root{" +
        "--primary-color:" +
        String(theme.primaryColor || "") +
        ";" +
        "--background-color:" +
        String(theme.backgroundColor || "") +
        ";" +
        "--secondary-background-color:" +
        String(theme.secondaryBackgroundColor || "") +
        ";" +
        "--text-color:" +
        String(theme.textColor || "") +
        ";" +
        "--font:" +
        String(theme.font || "") +
        ";" +
        "}\n";
    } catch (e) {
      // ignore
    }
  }

  var Streamlit = {
    API_VERSION: 1,
    RENDER_EVENT: "streamlit:render",
    events: _makeEventTarget(),
    _registered: false,
    _lastFrameHeight: null,

    setComponentReady: function () {
      if (!Streamlit._registered) {
        try {
          window.addEventListener("message", Streamlit._onMessageEvent);
        } catch (e) {
          // ignore
        }
        Streamlit._registered = true;
      }
      _sendBackMsg(COMPONENT_READY, { apiVersion: Streamlit.API_VERSION });
    },

    setFrameHeight: function (height) {
      var h = height;
      if (h === undefined || h === null) {
        try {
          h = document.body.scrollHeight;
        } catch (e) {
          h = 0;
        }
      }
      if (h === Streamlit._lastFrameHeight) return;
      Streamlit._lastFrameHeight = h;
      _sendBackMsg(SET_FRAME_HEIGHT, { height: h });
    },

    setComponentValue: function (value) {
      _sendBackMsg(SET_COMPONENT_VALUE, { value: value, dataType: "json" });
    },

    _onMessageEvent: function (event) {
      try {
        var data = event && event.data;
        if (!data || typeof data !== "object") return;
        if (data.type !== Streamlit.RENDER_EVENT) return;
        Streamlit._onRenderMessage(data);
      } catch (e) {
        // ignore
      }
    },

    _onRenderMessage: function (data) {
      var args = data.args;
      if (args == null || typeof args !== "object") args = {};
      var disabled = Boolean(data.disabled);
      var theme = data.theme;
      if (theme) _injectTheme(theme);

      var detail = { disabled: disabled, args: args, theme: theme };
      try {
        Streamlit.events.dispatchEvent(
          new CustomEvent(Streamlit.RENDER_EVENT, { detail: detail })
        );
      } catch (e) {
        // CustomEvent can fail in very old browsers; ignore.
      }
    },
  };

  window.Streamlit = Streamlit;
})();

