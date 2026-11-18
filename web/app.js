(function () {
  "use strict";

  var MAX_QUESTION = 500;
  var form = document.getElementById("ask-form");
  var input = document.getElementById("question");
  var send = document.getElementById("send");
  var status = document.getElementById("status");
  var quota = document.getElementById("quota");
  var transcript = document.getElementById("transcript");
  var sources = document.getElementById("sources");
  var emptyState = document.getElementById("empty-state");
  var busy = false;
  var locked = false;
  var lockTimer = null;
  var lockDeadline = 0;

  function setStatus(message, state) {
    status.textContent = message;
    if (state) {
      status.setAttribute("data-state", state);
    } else {
      status.removeAttribute("data-state");
    }
  }

  function scrollToLatest() {
    transcript.scrollTop = transcript.scrollHeight;
  }

  function prefersReducedMotion() {
    return Boolean(
      window.matchMedia &&
        window.matchMedia("(prefers-reduced-motion: reduce)").matches
    );
  }

  function detailText(payload) {
    if (!payload || payload.detail === undefined || payload.detail === null) {
      return "";
    }
    if (typeof payload.detail === "string") {
      return payload.detail;
    }
    if (
      typeof payload.detail === "object" &&
      typeof payload.detail.message === "string"
    ) {
      return payload.detail.message;
    }
    return JSON.stringify(payload.detail);
  }

  function rateLimitMessage(payload) {
    var detail = payload && payload.detail;
    if (detail && typeof detail === "object") {
      if (
        detail.reason === "per_ip" &&
        typeof detail.limit === "number" &&
        typeof detail.resets_in_seconds === "number"
      ) {
        var minutes = Math.max(1, Math.ceil(detail.resets_in_seconds / 60));
        return (
          "You've used all " +
          detail.limit +
          " questions this hour. Try again in " +
          minutes +
          " min."
        );
      }
      if (detail.reason === "daily") {
        return "The demo's daily question limit has been reached. Try again tomorrow.";
      }
      if (typeof detail.message === "string") {
        return detail.message;
      }
    }
    return detailText(payload) || "This demo is rate limited.";
  }

  function setQuotaText(text, modifier) {
    if (!quota) {
      return;
    }
    quota.classList.remove("quota--low", "quota--locked");
    if (modifier) {
      quota.classList.add(modifier);
    }
    quota.textContent = text;
    quota.hidden = false;
  }

  function clearQuota() {
    if (!quota) {
      return;
    }
    quota.classList.remove("quota--low", "quota--locked");
    quota.textContent = "";
    quota.hidden = true;
  }

  function unlock() {
    locked = false;
    if (lockTimer !== null) {
      clearInterval(lockTimer);
      lockTimer = null;
    }
    lockDeadline = 0;
    input.disabled = false;
    send.disabled = false;
    send.textContent = busy ? "Thinking…" : "Ask";
    setQuotaText("Your questions are available again", null);
  }

  function updateLockedCountdown() {
    if (!locked) {
      return;
    }
    var remainingMs = lockDeadline - Date.now();
    if (remainingMs <= 0) {
      unlock();
      return;
    }
    var minutes = Math.max(1, Math.ceil(remainingMs / 60000));
    setQuotaText("Next question available in " + minutes + " min", "quota--locked");
  }

  function lockForSeconds(seconds) {
    var safeSeconds =
      typeof seconds === "number" && isFinite(seconds) ? Math.max(0, seconds) : 0;
    locked = true;
    if (lockTimer !== null) {
      clearInterval(lockTimer);
      lockTimer = null;
    }
    lockDeadline = Date.now() + safeSeconds * 1000;
    input.disabled = true;
    send.disabled = true;
    updateLockedCountdown();
    if (locked) {
      lockTimer = window.setInterval(updateLockedCountdown, 30000);
    }
  }

  function lockForTheDay() {
    locked = true;
    if (lockTimer !== null) {
      clearInterval(lockTimer);
      lockTimer = null;
    }
    lockDeadline = 0;
    input.disabled = true;
    send.disabled = true;
    setQuotaText(
      "The demo's daily limit has been reached. Try again tomorrow.",
      "quota--locked"
    );
  }

  function handleRateLimit(payload) {
    var detail = payload && payload.detail;
    if (!detail || typeof detail !== "object") {
      return;
    }
    if (detail.reason === "daily") {
      lockForTheDay();
      return;
    }
    if (typeof detail.resets_in_seconds === "number") {
      lockForSeconds(detail.resets_in_seconds);
    }
  }

  function renderQuota(payload) {
    if (!quota) {
      return;
    }
    if (typeof payload.limit !== "number") {
      clearQuota();
      return;
    }
    var remaining = typeof payload.remaining === "number" ? payload.remaining : 0;
    if (remaining <= 0) {
      if (typeof payload.resets_in_seconds === "number") {
        lockForSeconds(payload.resets_in_seconds);
      } else {
        // No reset hint: wait for the top of the hour window to roll over.
        var now = new Date();
        var nextHour = new Date(now.getTime());
        nextHour.setMinutes(0, 0, 0);
        nextHour.setHours(nextHour.getHours() + 1);
        lockForSeconds(Math.max(1, Math.ceil((nextHour - now) / 1000)));
      }
      return;
    }
    if (remaining <= 2) {
      setQuotaText(
        "Only " + remaining + (remaining === 1 ? " question" : " questions") +
          " left this hour",
        "quota--low"
      );
      return;
    }
    // "1 of 8 questions left": the noun agrees with the limit, not the remainder.
    var noun = payload.limit === 1 ? "question" : "questions";
    setQuotaText(
      remaining + " of " + payload.limit + " " + noun + " left this hour",
      null
    );
  }

  function addMessage(role, text) {
    if (emptyState && emptyState.parentNode === transcript) {
      emptyState.remove();
    }
    var node = document.createElement("div");
    node.className = "message " + role;
    node.textContent = text;
    transcript.appendChild(node);
    scrollToLatest();
    return node;
  }

  function setBusy(value) {
    busy = value;
    send.disabled = value || locked;
    input.disabled = locked;
    send.textContent = value ? "Thinking…" : "Ask";
  }

  function renderAnswer(text, citations) {
    var node = addMessage("answer", "");
    node.textContent = "";

    if (!text) {
      node.textContent = "No answer was returned.";
      return;
    }

    // Each citation carries the marker the model wrote (e.g. 2 for "[2]"), so a
    // button exists only for a marker with a matching citation and the card is
    // found by that marker, never by ordinal position. Repeated markers reuse
    // the same card.
    var citedMarkers = {};
    citations.forEach(function (citation) {
      if (citation && citation.marker) {
        citedMarkers[citation.marker] = true;
      }
    });

    // Keep this answer's citations on its own node: clicking a marker on an
    // older answer must re-render the pane from that answer's sources.
    node._citations = citations;

    var markerPattern = /\[(\d{1,2})\]/g;
    var lastIndex = 0;
    var match;

    while ((match = markerPattern.exec(text)) !== null) {
      if (match.index > lastIndex) {
        node.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
      }

      var marker = parseInt(match[1], 10);
      if (citedMarkers[marker]) {
        var button = document.createElement("button");
        button.type = "button";
        button.className = "citation";
        button.textContent = "[" + marker + "]";
        button.setAttribute("data-marker", String(marker));
        button.setAttribute("aria-label", "Jump to source [" + marker + "]");
        node.appendChild(button);
      } else {
        node.appendChild(document.createTextNode(match[0]));
      }

      lastIndex = markerPattern.lastIndex;
    }

    if (lastIndex < text.length) {
      node.appendChild(document.createTextNode(text.slice(lastIndex)));
    }

    scrollToLatest();
  }

  function renderSources(citations) {
    sources.textContent = "";

    if (!citations.length) {
      var none = document.createElement("p");
      none.className = "empty-state muted";
      none.textContent = "No sources were cited for this answer.";
      sources.appendChild(none);
      return;
    }

    citations.forEach(function (citation) {
      var marker = citation.marker;

      var card = document.createElement("article");
      card.className = "source-card";
      card.id = "source-" + marker;
      card.setAttribute("data-marker", String(marker));

      var head = document.createElement("div");
      head.className = "source-head";

      var file = document.createElement("span");
      file.className = "source-file";
      file.textContent = citation.file || "unknown file";
      file.title = citation.file || "";

      var page = document.createElement("span");
      page.className = "source-page";
      page.textContent = "page " + citation.page;

      head.appendChild(file);
      head.appendChild(page);

      var snippet = document.createElement("p");
      snippet.className = "source-snippet";
      snippet.textContent = citation.snippet || "";

      card.appendChild(head);
      card.appendChild(snippet);
      sources.appendChild(card);
    });
  }

  function revealSource(marker) {
    var card = sources.querySelector('[data-marker="' + marker + '"]');
    if (!card) {
      return;
    }

    card.scrollIntoView({
      behavior: prefersReducedMotion() ? "auto" : "smooth",
      block: "center"
    });
    card.classList.add("is-active");
    card.setAttribute("tabindex", "-1");
    card.focus({ preventScroll: true });

    window.setTimeout(function () {
      card.classList.remove("is-active");
    }, 1600);
  }

  function ask(question) {
    addMessage("user", question);
    input.value = "";
    setBusy(true);
    setStatus("Searching the documents…", "busy");

    fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: question })
    })
      .then(function (response) {
        return response
          .json()
          .catch(function () {
            return {};
          })
          .then(function (payload) {
            return { response: response, payload: payload };
          });
      })
      .then(function (result) {
        if (!result.response.ok) {
          if (result.response.status === 429) {
            handleRateLimit(result.payload);
            throw new Error(rateLimitMessage(result.payload));
          }
          var detail = detailText(result.payload);
          throw new Error(
            detail || "Request failed (" + result.response.status + ")."
          );
        }
        return result.payload;
      })
      .then(function (payload) {
        var citations = payload.citations || [];
        renderAnswer(payload.answer || "", citations);
        renderSources(citations);
        var count = citations.length;
        setStatus(
          "Answer ready · " + count + (count === 1 ? " source" : " sources"),
          "ok"
        );
        renderQuota(payload);
      })
      .catch(function (error) {
        var message = error && error.message ? error.message : "Something went wrong.";
        addMessage("error", message);
        setStatus(message, "error");
      })
      .then(function () {
        setBusy(false);
        input.focus();
      });
  }

  function submitQuestion(event) {
    if (event) {
      event.preventDefault();
    }
    if (busy || locked) {
      return;
    }

    var question = input.value.trim();
    if (!question) {
      setStatus("Type a question first.", "error");
      input.focus();
      return;
    }
    if (question.length > MAX_QUESTION) {
      setStatus("Questions are limited to " + MAX_QUESTION + " characters.", "error");
      return;
    }

    ask(question);
  }

  form.addEventListener("submit", submitQuestion);

  input.addEventListener("keydown", function (event) {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      if (typeof form.requestSubmit === "function") {
        form.requestSubmit();
      } else {
        submitQuestion();
      }
    }
  });

  transcript.addEventListener("click", function (event) {
    if (!(event.target instanceof Element)) {
      return;
    }
    var button = event.target.closest("button.citation");
    if (!button) {
      return;
    }
    // Resolve the card through this specific answer's citations, never through
    // whatever answer happens to own the sources pane right now.
    var message = button.closest(".message.answer");
    var citations = message && message._citations ? message._citations : [];
    var marker = button.getAttribute("data-marker");
    if (!citations.length || !marker) {
      return;
    }
    renderSources(citations);
    revealSource(marker);
  });
})();
