"use strict";

const $ = (id) => document.getElementById(id);
const samples = {
  prize:
    "Congratulations! You have won a £1000 cash prize. Call 09061701461 now to claim your FREE reward. Limited time offer!",
  everyday:
    "Hey, are we still meeting for coffee tomorrow? I can be there around 6. Let me know what works for you.",
  urgent:
    "URGENT! You have been selected for a guaranteed £5000 cash award. Call 09050001234 now to claim your prize. Offer expires tonight!",
};
const aiConfigured = document.body.dataset.aiConfigured === "true";
const history = [];
let localRequest, aiRequest, batchRequest;
let currentResult = null;
let analyzedText = "";
let batchData = null;
let toastTimer;

function show(id, visible = true) {
  $(id).hidden = !visible;
}
function setText(id, text) {
  $(id).textContent = text;
}
function toast(text) {
  clearTimeout(toastTimer);
  setText("toast", text);
  show("toast");
  toastTimer = setTimeout(() => show("toast", false), 3500);
}
function node(tag, text, className = "") {
  const element = document.createElement(tag);
  element.textContent = text;
  if (className) element.className = className;
  return element;
}
async function post(path, body, signal) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
    signal,
  });
  let data;
  try {
    data = await response.json();
  } catch {
    throw new Error(
      "The server returned an unreadable response. Please try again.",
    );
  }
  if (!response.ok)
    throw new Error(data.error || "This request could not be completed.");
  return data;
}

function navigate() {
  const names = {
    analyzer: "Message analyzer",
    batch: "Batch check",
    model: "Model insights",
    about: "How it works",
  };
  const requested = location.hash.slice(1);
  const view = Object.hasOwn(names, requested) ? requested : "analyzer";
  for (const name of Object.keys(names)) show(`view-${name}`, name === view);
  document.querySelectorAll("[data-view]").forEach((link) => {
    const active = link.dataset.view === view;
    link.classList.toggle("active", active);
    if (active) link.setAttribute("aria-current", "page");
    else link.removeAttribute("aria-current");
  });
  setText("breadcrumb-view", names[view]);
  document.title = `${names[view]} — Signal`;
  $("toast").hidden = true;
}
document.querySelectorAll("[data-view]").forEach((link) => {
  link.dataset.mobileLabel = {
    analyzer: "Analyze",
    batch: "Batch",
    model: "Model",
    about: "Guide",
  }[link.dataset.view];
  link.setAttribute("aria-label", link.textContent.trim());
});
window.addEventListener("hashchange", () => {
  navigate();
  $("main").focus({ preventScroll: true });
  window.scrollTo(0, 0);
});
navigate();

function resetAnalysis() {
  localRequest?.abort();
  aiRequest?.abort();
  currentResult = null;
  analyzedText = "";
  show("empty-result");
  show("loading-result", false);
  show("result-content", false);
  show("ai-panel", false);
  show("ai-output", false);
  show("ai-tool-output", false);
  show("ai-tools", false);
  show("input-error", false);
  $("ai-consent").checked = false;
  $("analyze-button").disabled = false;
  $("analyze-button").querySelector("span").textContent = "Analyze message";
  $("result-content").closest(".result-card").classList.remove("is-spam");
  setText(
    "message-count",
    `${$("message").value.length.toLocaleString("en-US")} / 5,000 characters`,
  );
  aiButtons(false);
}
$("message").addEventListener("input", resetAnalysis);
document.querySelectorAll("[data-sample]").forEach((button) =>
  button.addEventListener("click", () => {
    $("message").value = samples[button.dataset.sample];
    resetAnalysis();
    $("message").focus();
  }),
);
$("message").addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") {
    event.preventDefault();
    if (!$("analyze-button").disabled) $("analyze-form").requestSubmit();
  }
});

function renderResult(data) {
  show("empty-result", false);
  show("loading-result", false);
  show("result-content");
  show("ai-panel");
  const spam = data.label === "spam";
  $("result-content").closest(".result-card").classList.toggle("is-spam", spam);
  $("verdict-icon")
    .querySelector("use")
    .setAttribute("href", spam ? "#warning" : "#check");
  setText("prediction", data.prediction);
  setText("result-reason", data.reason);
  setText("spam-score", `${(data.spam_probability * 100).toFixed(1)}%`);
  $("score-fill").style.width = `${data.spam_probability * 100}%`;
  $("threshold-marker").style.left = `${data.threshold * 100}%`;
  setText("threshold-label", `Threshold ${(data.threshold * 100).toFixed(0)}%`);
  setText(
    "processing-time",
    `${data.processing_time_ms.toFixed(1)} ms · local model`,
  );
  show("review-note", data.needs_review);
  setText(
    "review-note-text",
    data.unfamiliar_text
      ? "Few familiar word features. This message needs a human review, especially if it uses another language."
      : "This score is close to the decision threshold. Consider reviewing the message yourself.",
  );
  const evidence = $("evidence");
  evidence.replaceChildren();
  const features = data.word_evidence.length
    ? data.word_evidence
    : data.evidence.slice(0, 6);
  for (const feature of features) {
    const chip = node(
      "span",
      feature.feature,
      `evidence-chip ${feature.direction === "spam" ? "spam" : ""}`,
    );
    chip.title = `${feature.kind} feature · ${feature.contribution >= 0 ? "+" : ""}${feature.contribution.toFixed(3)} log-odds contribution`;
    chip.setAttribute(
      "aria-label",
      `${feature.feature}, toward ${feature.direction === "spam" ? "spam" : "not spam"}`,
    );
    evidence.append(chip);
  }
  if (!features.length)
    evidence.append(
      node(
        "span",
        "No recognized text features. The model intercept drives this score.",
        "score-note",
      ),
    );
  aiButtons(false);
}

function recordHistory(text, result) {
  history.unshift({ text, result, time: new Date() });
  history.splice(8);
  setText("history-count", history.length);
  const list = $("history-list");
  list.replaceChildren();
  history.forEach((entry) => {
    const button = node("button", "", "history-item");
    button.type = "button";
    button.title = "Load this message";
    button.append(
      node(
        "span",
        entry.result.prediction,
        `label-pill ${entry.result.label === "spam" ? "spam" : ""}`,
      ),
      node("span", entry.text, "history-text"),
      node(
        "time",
        entry.time.toLocaleTimeString([], {
          hour: "2-digit",
          minute: "2-digit",
        }),
      ),
    );
    button.addEventListener("click", () => {
      $("message").value = entry.text;
      resetAnalysis();
      analyzedText = entry.text;
      currentResult = entry.result;
      renderResult(entry.result);
    });
    list.append(button);
  });
}

$("analyze-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const text = $("message").value.trim();
  if (!text) {
    setText("input-error", "Add a message to get started.");
    show("input-error");
    $("message").focus();
    return;
  }
  if ($("message").value.length > 5000) {
    setText("input-error", "Keep your message within 5,000 characters.");
    show("input-error");
    return;
  }
  resetAnalysis();
  const controller = new AbortController();
  localRequest = controller;
  $("analyze-button").disabled = true;
  $("analyze-button").querySelector("span").textContent = "Analyzing…";
  show("empty-result", false);
  show("loading-result");
  try {
    const data = await post("/predict", { text }, controller.signal);
    if (controller.signal.aborted) return;
    currentResult = data;
    analyzedText = text;
    renderResult(data);
    recordHistory(text, data);
  } catch (error) {
    if (error.name === "AbortError") return;
    show("loading-result", false);
    show("empty-result");
    setText(
      "input-error",
      error.message === "Failed to fetch"
        ? "Cannot reach the server. Check that the app is running, then try again."
        : error.message,
    );
    show("input-error");
  } finally {
    if (localRequest === controller) {
      $("analyze-button").disabled = false;
      $("analyze-button").querySelector("span").textContent = "Analyze message";
    }
  }
});

async function copyText(text) {
  try {
    await navigator.clipboard.writeText(text);
    toast("Copied to clipboard");
  } catch {
    toast("Clipboard access unavailable. You can select and copy the text.");
  }
}
$("copy-result").addEventListener("click", () => {
  if (currentResult)
    copyText(
      `Signal · ${currentResult.prediction}\nSpam model score: ${(currentResult.spam_probability * 100).toFixed(1)}%\n${currentResult.reason}\nLocal ML result. A model score is not a safety rating.`,
    );
});

function aiButtons(busy) {
  const enabled =
    aiConfigured && $("ai-consent").checked && Boolean(currentResult) && !busy;
  $("ai-review-button").disabled = !enabled;
  $("reply-button").disabled = !enabled;
  $("polish-button").disabled = !enabled;
  show("ai-tools", enabled || busy);
  $("reply-button").hidden = currentResult?.label !== "ham";
}
$("ai-consent").addEventListener("change", () => {
  if (!$("ai-consent").checked) aiRequest?.abort();
  aiButtons(false);
});

async function aiAction(path, target, render) {
  if (!currentResult || !$("ai-consent").checked || !aiConfigured) return;
  aiRequest?.abort();
  const controller = new AbortController();
  aiRequest = controller;
  const text = analyzedText;
  aiButtons(true);
  $(target).replaceChildren(node("span", "Asking Gemini…"));
  show(target);
  try {
    const data = await post(path, { text, consent: true }, controller.signal);
    if (controller.signal.aborted || analyzedText !== text) return;
    $(target).replaceChildren();
    render(data, $(target));
  } catch (error) {
    if (error.name !== "AbortError")
      $(target).replaceChildren(
        node(
          "span",
          error.message === "Failed to fetch"
            ? "AI review could not connect. Your local result remains available."
            : error.message,
        ),
      );
  } finally {
    if (aiRequest === controller) {
      if (controller.signal.aborted && analyzedText === text) {
        $(target).replaceChildren();
        show(target, false);
      }
      aiButtons(false);
    }
  }
}
$("ai-review-button").addEventListener("click", () =>
  aiAction("/ai_review", "ai-output", (data, target) => {
    const comparison =
      data.prediction === "Uncertain"
        ? "Needs more context"
        : data.agrees_with_ml
          ? "Agrees with local model"
          : "Differs from local model";
    target.append(
      node("strong", `${data.prediction} · ${comparison}`),
      node("p", data.reason),
      node("p", "Your local ML verdict is unchanged.", "score-note"),
    );
  }),
);
$("reply-button").addEventListener("click", () =>
  aiAction("/generate_reply", "ai-tool-output", (data, target) => {
    target.append(node("strong", "Suggested replies"));
    data.replies.forEach((reply) =>
      target.append(node("p", reply, "reply-suggestion")),
    );
  }),
);
$("polish-button").addEventListener("click", () =>
  aiAction("/polish_text", "ai-tool-output", (data, target) => {
    target.append(
      node("strong", "Neutral rewrite"),
      node("p", data.polished_text, "reply-suggestion"),
      node(
        "p",
        "A rewrite does not prove that the original message is legitimate.",
        "score-note",
      ),
    );
  }),
);

function batchTexts() {
  return $("batch-input")
    .value.split(/\r?\n/)
    .map((text) => text.trim())
    .filter(Boolean);
}
function resetBatch() {
  batchRequest?.abort();
  batchData = null;
  setText("batch-count", `${batchTexts().length} / 50 messages`);
  show("batch-results", false);
  show("batch-error", false);
  $("batch-button").disabled = false;
}
$("batch-input").addEventListener("input", resetBatch);
$("batch-example").addEventListener("click", () => {
  $("batch-input").value = Object.values(samples).join("\n");
  resetBatch();
});
$("batch-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const texts = batchTexts();
  if (
    !texts.length ||
    texts.length > 50 ||
    texts.some((text) => text.length > 5000)
  ) {
    setText(
      "batch-error",
      !texts.length
        ? "Add at least one message, with one message per line."
        : texts.length > 50
          ? "Check up to 50 messages at a time."
          : "Each message must be 5,000 characters or fewer.",
    );
    show("batch-error");
    return;
  }
  resetBatch();
  const controller = new AbortController();
  batchRequest = controller;
  $("batch-button").disabled = true;
  try {
    const data = await post("/predict_batch", { texts }, controller.signal);
    if (controller.signal.aborted) return;
    batchData = { texts, ...data };
    setText(
      "batch-summary",
      `${data.summary.total} messages · ${data.summary.spam} spam · ${data.summary.ham} not spam`,
    );
    $("batch-rows").replaceChildren();
    data.results.forEach((result, index) => {
      const row = node("tr", "");
      row.append(node("td", index + 1), node("td", texts[index]));
      const verdict = node("td", "");
      verdict.append(
        node(
          "span",
          result.prediction,
          `label-pill ${result.label === "spam" ? "spam" : ""}`,
        ),
      );
      row.append(
        verdict,
        node("td", `${(result.spam_probability * 100).toFixed(1)}%`),
        node("td", result.needs_review ? "Suggested" : "—"),
      );
      $("batch-rows").append(row);
    });
    show("batch-results");
  } catch (error) {
    if (error.name !== "AbortError") {
      setText(
        "batch-error",
        error.message === "Failed to fetch"
          ? "Cannot reach the server. Try again when the app is running."
          : error.message,
      );
      show("batch-error");
    }
  } finally {
    if (batchRequest === controller) $("batch-button").disabled = false;
  }
});

function csvCell(value) {
  let text = String(value);
  if (/^[\s]*[=+\-@]/.test(text) || /^[\t\r]/.test(text)) text = `'${text}`;
  return `"${text.replaceAll('"', '""')}"`;
}
$("export-csv").addEventListener("click", () => {
  if (!batchData) return;
  const rows = [["message", "prediction", "spam_model_score", "needs_review"]];
  batchData.results.forEach((result, index) =>
    rows.push([
      batchData.texts[index],
      result.prediction,
      result.spam_probability,
      result.needs_review,
    ]),
  );
  const blob = new Blob(
    ["\uFEFF" + rows.map((row) => row.map(csvCell).join(",")).join("\r\n")],
    { type: "text/csv;charset=utf-8" },
  );
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "signal-results.csv";
  document.body.append(link);
  link.click();
  link.remove();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
  toast("CSV exported");
});
