const state = {
  sessionId: null,
};

const uploadView = document.getElementById("uploadView");
const chatView = document.getElementById("chatView");
const fileInput = document.getElementById("fileInput");
const uploadButton = document.getElementById("uploadButton");
const uploadStatus = document.getElementById("uploadStatus");
const selectedFiles = document.getElementById("selectedFiles");
const uploadSpinner = document.getElementById("uploadSpinner");
const chatSpinner = document.getElementById("chatSpinner");
const newDatasetButton = document.getElementById("newDatasetButton");
const sessionBadge = document.getElementById("sessionBadge");
const datasetSummary = document.getElementById("datasetSummary");
const welcomePanel = document.getElementById("welcomePanel");
const chatHistory = document.getElementById("chatHistory");
const chatForm = document.getElementById("chatForm");
const questionInput = document.getElementById("questionInput");

function escapeHtml(text) {
  return String(text)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function setUploadLoading(isLoading) {
  uploadSpinner.classList.toggle("hidden", !isLoading);
  uploadButton.disabled = isLoading;
}

function setChatLoading(isLoading) {
  chatSpinner.classList.toggle("hidden", !isLoading);
}

function resetChatWorkspace() {
  chatHistory.innerHTML = "";
  welcomePanel.classList.remove("hidden");
}

function showUploadView() {
  uploadView.classList.remove("hidden");
  chatView.classList.add("hidden");
  state.sessionId = null;
  fileInput.value = "";
  uploadStatus.textContent = "No dataset uploaded yet.";
  selectedFiles.innerHTML = "<p>No files selected.</p>";
  datasetSummary.innerHTML = "";
  sessionBadge.textContent = "Ready";
  resetChatWorkspace();
}

function showChatView() {
  uploadView.classList.add("hidden");
  chatView.classList.remove("hidden");
  questionInput.focus();
}

function autoresizeTextarea() {
  questionInput.style.height = "auto";
  questionInput.style.height = `${Math.min(questionInput.scrollHeight, 180)}px`;
}

function renderSelectedFiles() {
  const files = Array.from(fileInput.files);
  if (!files.length) {
    selectedFiles.innerHTML = "<p>No files selected.</p>";
    return;
  }
  selectedFiles.innerHTML = files.map((file) => `
    <div class="file-pill">
      <strong>${escapeHtml(file.name)}</strong>
      <span class="dataset-meta">${Math.max(1, Math.round(file.size / 1024))} KB</span>
    </div>
  `).join("");
}

function addMessage(role, message, meta = "") {
  const article = document.createElement("article");
  article.className = `message ${role}`;
  article.innerHTML = `
    <div class="message-role">${role === "user" ? "You" : "Assistant"}</div>
    <div class="message-bubble">${message}</div>
    ${meta ? `<div class="message-meta">${escapeHtml(meta)}</div>` : ""}
  `;
  chatHistory.appendChild(article);
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function renderSessionSummary(payload) {
  const cards = payload.source_files.map((file) => `
    <div class="dataset-card">
      <strong>${escapeHtml(file.filename)}</strong>
      <div class="dataset-meta">${file.stats.sentences} sentences</div>
      <div class="dataset-meta">${file.stats.tokens} tokens</div>
    </div>
  `).join("");

  datasetSummary.innerHTML = `
    <div class="dataset-card">
      <strong>${payload.source_files.length} file(s)</strong>
      <div class="dataset-meta">${payload.chunk_count} chunks indexed</div>
    </div>
    ${cards}
  `;
}

fileInput.addEventListener("change", renderSelectedFiles);
questionInput.addEventListener("input", autoresizeTextarea);

newDatasetButton.addEventListener("click", showUploadView);

uploadButton.addEventListener("click", async () => {
  if (!fileInput.files.length) {
    uploadStatus.textContent = "Choose at least one PDF, CSV, or TXT file.";
    return;
  }

  const formData = new FormData();
  Array.from(fileInput.files).forEach((file) => formData.append("files", file));
  setUploadLoading(true);
  uploadStatus.textContent = "Processing files and building the domain knowledge base...";

  try {
    const response = await fetch("/upload", {
      method: "POST",
      body: formData,
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Upload failed.");
    }

    state.sessionId = payload.session_id;
    sessionBadge.textContent = payload.session_id.slice(0, 8);
    renderSessionSummary(payload);
    resetChatWorkspace();
    addMessage(
      "assistant",
      "Your dataset is ready. Ask a question and I will answer only from the uploaded files."
    );
    showChatView();
  } catch (error) {
    uploadStatus.textContent = error.message;
  } finally {
    setUploadLoading(false);
  }
});

chatForm.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = questionInput.value.trim();
  if (!question || !state.sessionId) {
    return;
  }

  welcomePanel.classList.add("hidden");
  addMessage("user", escapeHtml(question));
  questionInput.value = "";
  autoresizeTextarea();
  setChatLoading(true);

  try {
    const response = await fetch("/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        session_id: state.sessionId,
        question,
        top_k: 4,
      }),
    });
    const payload = await response.json();
    if (!response.ok) {
      throw new Error(payload.detail || "Question answering failed.");
    }

    addMessage(
      "assistant",
      escapeHtml(payload.answer),
      `confidence ${payload.confidence} | model ${payload.model}`
    );
  } catch (error) {
    addMessage("assistant", escapeHtml(error.message));
  } finally {
    setChatLoading(false);
  }
});

autoresizeTextarea();
