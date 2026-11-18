(function () {
  "use strict";

  var MAX_QUESTION = 500;
  var POWERED_BY_FALLBACK =
    "Free open model via OpenRouter · answers can take a few seconds";
  var form = document.getElementById("ask-form");
  var input = document.getElementById("question");
  var send = document.getElementById("send");
  var status = document.getElementById("status");
  var quota = document.getElementById("quota");
  var transcript = document.getElementById("transcript");
  var sources = document.getElementById("sources");
  var howTo = document.getElementById("how-to");
  var documentList = document.getElementById("document-list");
  var exampleChips = document.getElementById("example-chips");
  var howToFallback = document.getElementById("how-to-fallback");
  var poweredBy = document.getElementById("powered-by");
  var exampleButtons = [];
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

  function shortModelName(model) {
    if (typeof model !== "string" || !model) {
      return "";
    }
    var name = model.slice(model.lastIndexOf("/") + 1);
    return name.replace(/:free$/, "");
  }

  function documentCode(file) {
    if (typeof file !== "string" || !file) {
      return "document";
    }
    var name = file.split(/[\\/]/).pop() || file;
    var base = name.replace(/\.(pdf|md|markdown|html?|txt)$/i, "");
    var match = base.match(
      /^nist[-_ ]?sp[-_ ]?(\d+)[-_ ]?(\d+)(?:[-_ ]?([a-z0-9]+))?$/i
    );
    if (!match) {
      return name;
    }
    var suffix = match[3] ? match[3].toUpperCase() : "";
    var separator = suffix && /^\d+$/.test(suffix) ? "-" : "";
    return "SP " + match[1] + "-" + match[2] + separator + suffix;
  }

  function renderInfo(payload) {
    if (!poweredBy) {
      return;
    }
    var model = payload && typeof payload.model === "string" ? payload.model : "";
    var shortName = shortModelName(model);
    if (!shortName) {
      return;
    }
    var free = payload && payload.free === true;
    poweredBy.textContent = free
      ? "Powered by " +
        shortName +
        " · free open model via OpenRouter · answers can take a few seconds"
      : "Powered by " + shortName + " · answers can take a few seconds";
    poweredBy.hidden = false;
  }

  function loadInfo() {
    if (!poweredBy) {
      return;
    }
    fetch("/api/info", { headers: { Accept: "application/json" } })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("info unavailable");
        }
        return response.json();
      })
      .then(renderInfo)
      .catch(function () {
        poweredBy.textContent = POWERED_BY_FALLBACK;
        poweredBy.hidden = false;
      });
  }

  function updateExampleButtons() {
    exampleButtons.forEach(function (button) {
      button.disabled = busy || locked;
    });
  }

  function askExample(question) {
    if (busy || locked || typeof question !== "string" || !question.trim()) {
      return;
    }
    input.value = question;
    if (typeof form.requestSubmit === "function") {
      form.requestSubmit();
    } else {
      submitQuestion();
    }
  }

  function renderHowTo(payload) {
    if (!documentList || !exampleChips) {
      return;
    }
    documentList.textContent = "";
    exampleChips.textContent = "";
    exampleButtons = [];

    var documents = payload && Array.isArray(payload.documents) ? payload.documents : [];
    documents.forEach(function (doc) {
      var file = doc && typeof doc.file === "string" ? doc.file : "";
      var title =
        doc && typeof doc.title === "string" && doc.title ? doc.title : file;
      if (!title) {
        title = "Unknown document";
      }
      var hasUrl = Boolean(doc && typeof doc.url === "string" && doc.url);

      var item = document.createElement("li");
      item.className = "document-spine";

      var link = document.createElement(hasUrl ? "a" : "div");
      link.className = "spine-link";
      if (hasUrl) {
        link.href = doc.url;
        link.target = "_blank";
        link.rel = "noopener";
      }

      var top = document.createElement("span");
      top.className = "spine-top";

      var code = document.createElement("span");
      code.className = "document-code";
      code.textContent = documentCode(file || title);

      var pages = document.createElement("span");
      pages.className = "doc-pages";
      var pageCount = doc && typeof doc.pages === "number" ? doc.pages : 0;
      pages.textContent =
        pageCount + (pageCount === 1 ? " page" : " pages");

      var name = document.createElement("span");
      name.className = "document-title";
      name.textContent = title;

      top.appendChild(code);
      top.appendChild(pages);
      link.appendChild(top);
      link.appendChild(name);
      item.appendChild(link);
      documentList.appendChild(item);
    });

    var examples = payload && Array.isArray(payload.examples) ? payload.examples : [];
    examples.forEach(function (example) {
      if (
        !example ||
        typeof example.label !== "string" ||
        typeof example.question !== "string"
      ) {
        return;
      }
      var button = document.createElement("button");
      button.type = "button";
      button.className = "example-chip";
      button.textContent = example.label;
      button.disabled = busy || locked;
      button.addEventListener("click", function () {
        askExample(example.question);
      });
      exampleButtons.push(button);
      exampleChips.appendChild(button);
    });

    if (howToFallback) {
      howToFallback.hidden = true;
    }
  }

  function showHowToFallback() {
    if (documentList) {
      documentList.textContent = "";
    }
    if (exampleChips) {
      exampleChips.textContent = "";
    }
    exampleButtons = [];
    if (howToFallback) {
      howToFallback.hidden = false;
    }
  }

  function loadHowTo() {
    if (!howTo) {
      return;
    }
    fetch("/api/documents", { headers: { Accept: "application/json" } })
      .then(function (response) {
        if (!response.ok) {
          throw new Error("documents unavailable");
        }
        return response.json();
      })
      .then(renderHowTo)
      .catch(showHowToFallback);
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
          " questions in the last hour. Try again in " +
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
    updateExampleButtons();
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
    updateExampleButtons();
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
    updateExampleButtons();
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
        // No reset hint: wait for the rolling window to free a slot.
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
          " left in this rolling hour",
        "quota--low"
      );
      return;
    }
    // "1 of 8 questions left": the noun agrees with the limit, not the remainder.
    var noun = payload.limit === 1 ? "question" : "questions";
    setQuotaText(
      remaining + " of " + payload.limit + " " + noun + " left (rolling hour)",
      null
    );
  }

  function addMessage(role, text) {
    if (howTo && howTo.parentNode === transcript) {
      howTo.remove();
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
    updateExampleButtons();
  }

  function appendAnswerMeta(node, model, sourceCount) {
    var shortName = shortModelName(model);
    if (!shortName) {
      return;
    }
    var count = typeof sourceCount === "number" ? sourceCount : 0;
    var label =
      count === 0 ? "no sources" : count + (count === 1 ? " source" : " sources");
    var meta = document.createElement("p");
    meta.className = "answer-meta";
    meta.textContent = "via " + shortName + " · " + label;
    node.appendChild(meta);
  }

  function makeCitationButton(marker) {
    var button = document.createElement("button");
    button.type = "button";
    button.className = "citation";
    button.textContent = "[" + marker + "]";
    button.setAttribute("data-marker", String(marker));
    button.setAttribute("aria-label", "Jump to source [" + marker + "]");
    return button;
  }

  // Build the allowed inline markdown as real DOM nodes. Text always goes
  // through createTextNode/textContent, so model output is never parsed as HTML.
  function appendInline(parent, text, citedMarkers) {
    var pattern = /(\*\*[^*\n]+\*\*|__[^_\n]+__|\*[^*\n]+\*|_[^_\n]+_|`[^`\n]+`|\[\d{1,2}\])/g;
    var lastIndex = 0;
    var match;

    while ((match = pattern.exec(text)) !== null) {
      if (match.index > lastIndex) {
        parent.appendChild(document.createTextNode(text.slice(lastIndex, match.index)));
      }

      var token = match[0];
      if (token.charAt(0) === "[") {
        var marker = parseInt(token.slice(1, -1), 10);
        if (citedMarkers[marker]) {
          parent.appendChild(makeCitationButton(marker));
        } else {
          parent.appendChild(document.createTextNode(token));
        }
      } else if (token.slice(0, 2) === "**" || token.slice(0, 2) === "__") {
        var strong = document.createElement("strong");
        appendInline(strong, token.slice(2, -2), citedMarkers);
        parent.appendChild(strong);
      } else if (token.charAt(0) === "*" || token.charAt(0) === "_") {
        var emphasis = document.createElement("em");
        appendInline(emphasis, token.slice(1, -1), citedMarkers);
        parent.appendChild(emphasis);
      } else if (token.charAt(0) === "`") {
        var code = document.createElement("code");
        code.textContent = token.slice(1, -1);
        parent.appendChild(code);
      } else {
        parent.appendChild(document.createTextNode(token));
      }

      lastIndex = pattern.lastIndex;
    }

    if (lastIndex < text.length) {
      parent.appendChild(document.createTextNode(text.slice(lastIndex)));
    }
  }

  function renderFormattedText(parent, text, citedMarkers) {
    var lines = String(text).replace(/\r\n?/g, "\n").split("\n");
    var bulletPattern = /^\s*-\s+(.*)$/;
    var numberedPattern = /^\s*\d+\.\s+(.*)$/;
    var index = 0;

    while (index < lines.length) {
      var line = lines[index];
      if (!line.trim()) {
        index += 1;
        continue;
      }

      if (bulletPattern.test(line)) {
        var list = document.createElement("ul");
        while (index < lines.length) {
          var itemMatch = bulletPattern.exec(lines[index]);
          if (!itemMatch) {
            break;
          }
          var item = document.createElement("li");
          appendInline(item, itemMatch[1], citedMarkers);
          list.appendChild(item);
          index += 1;
        }
        parent.appendChild(list);
        continue;
      }

      if (numberedPattern.test(line)) {
        var ordered = document.createElement("ol");
        while (index < lines.length) {
          var orderedMatch = numberedPattern.exec(lines[index]);
          if (!orderedMatch) {
            break;
          }
          var orderedItem = document.createElement("li");
          appendInline(orderedItem, orderedMatch[1], citedMarkers);
          ordered.appendChild(orderedItem);
          index += 1;
        }
        parent.appendChild(ordered);
        continue;
      }

      var paragraphLines = [];
      while (
        index < lines.length &&
        lines[index].trim() &&
        !bulletPattern.test(lines[index]) &&
        !numberedPattern.test(lines[index])
      ) {
        paragraphLines.push(lines[index].trim());
        index += 1;
      }
      var paragraph = document.createElement("p");
      appendInline(paragraph, paragraphLines.join(" "), citedMarkers);
      parent.appendChild(paragraph);
    }
  }

  function renderAnswer(text, citations, model) {
    var node = addMessage("answer", "");
    node.textContent = "";

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

    if (!text) {
      var missing = document.createElement("p");
      missing.textContent = "No answer was returned.";
      node.appendChild(missing);
    } else {
      renderFormattedText(node, text, citedMarkers);
    }

    appendAnswerMeta(node, model, citations.length);
    scrollToLatest();
  }

  function renderSources(citations) {
    sources.textContent = "";

    if (!citations.length) {
      var none = document.createElement("p");
      none.id = "sources-empty";
      none.className = "empty-state muted";
      none.textContent = "No sources were cited for this answer.";
      sources.appendChild(none);
      return;
    }

    citations.forEach(function (citation) {
      var marker = citation.marker;
      var fileName = citation.file || "unknown file";
      var pageValue =
        citation.page === undefined || citation.page === null
          ? ""
          : String(citation.page);

      var card = document.createElement("article");
      card.className = "source-card";
      card.id = "source-" + marker;
      card.setAttribute("data-marker", String(marker));
      card.setAttribute("data-file", fileName);
      card.setAttribute("data-page", pageValue);
      card.setAttribute(
        "aria-label",
        "Source [" + marker + "]: " + fileName + ", page " + (pageValue || "unknown")
      );

      var head = document.createElement("div");
      head.className = "source-head";

      var file = document.createElement("span");
      file.className = "source-file";
      file.textContent = documentCode(fileName);
      file.title = fileName;

      var page = document.createElement("span");
      page.className = "source-page";
      page.textContent = pageValue ? "p. " + pageValue : "page —";

      head.appendChild(file);
      head.appendChild(page);

      var snippet = document.createElement("p");
      snippet.className = "source-snippet";
      // Inline span so the highlighter paints each wrapped line, like a marker.
      var mark = document.createElement("span");
      mark.className = "source-mark";
      mark.textContent = citation.snippet || "";
      snippet.appendChild(mark);

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
    // Commit the unhighlighted state first so the background-size transition runs.
    card.getBoundingClientRect();
    card.classList.add("is-active");
    card.setAttribute("tabindex", "-1");
    card.focus({ preventScroll: true });

    window.setTimeout(function () {
      card.classList.remove("is-active");
    }, 1600);
  }

  function citationFor(button) {
    var message = button.closest(".message.answer");
    var citations = message && message._citations ? message._citations : [];
    var marker = button.getAttribute("data-marker");
    for (var i = 0; i < citations.length; i += 1) {
      var citation = citations[i];
      if (citation && String(citation.marker) === marker) {
        return citation;
      }
    }
    return null;
  }

  function cardForCitation(button) {
    var citation = citationFor(button);
    var marker = button.getAttribute("data-marker");
    if (!citation || !marker) {
      return null;
    }
    var card = sources.querySelector('[data-marker="' + marker + '"]');
    if (!card) {
      return null;
    }
    var file = citation.file || "";
    var page =
      citation.page === undefined || citation.page === null
        ? ""
        : String(citation.page);
    if (card.getAttribute("data-file") !== file || card.getAttribute("data-page") !== page) {
      return null;
    }
    return card;
  }

  function setCitationHover(button, active) {
    var card = button ? cardForCitation(button) : null;
    if (!card) {
      return;
    }
    card.classList.toggle("is-hover", active);
  }

  function citationFromEvent(event) {
    if (!(event.target instanceof Element)) {
      return null;
    }
    return event.target.closest("button.citation");
  }

  function ask(question) {
    addMessage("user", question);
    input.value = "";
    setBusy(true);
    setStatus("Searching the documents…", "busy");
    var slowTimer = window.setTimeout(function () {
      setStatus(
        "Waiting for the model… free models can take a few seconds",
        "busy"
      );
    }, 1500);

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
        renderAnswer(payload.answer || "", citations, payload.model);
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
        window.clearTimeout(slowTimer);
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

  transcript.addEventListener("mouseover", function (event) {
    var button = citationFromEvent(event);
    if (button) {
      setCitationHover(button, true);
    }
  });

  transcript.addEventListener("mouseout", function (event) {
    var button = citationFromEvent(event);
    if (button) {
      setCitationHover(button, false);
    }
  });

  transcript.addEventListener("focusin", function (event) {
    var button = citationFromEvent(event);
    if (button) {
      setCitationHover(button, true);
    }
  });

  transcript.addEventListener("focusout", function (event) {
    var button = citationFromEvent(event);
    if (button) {
      setCitationHover(button, false);
    }
  });

  loadHowTo();
  loadInfo();
})();
