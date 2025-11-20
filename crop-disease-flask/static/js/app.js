document.addEventListener("DOMContentLoaded", function () {
  const form = document.getElementById("predictForm");
  const resultCard = document.getElementById("resultCard");
  const resultText = document.getElementById("resultText");
  const resultTitle = document.getElementById("resultTitle");
  const confWrap = document.getElementById("confidenceWrap");
  const confBar = document.getElementById("confidenceBar");
  const errorWrap = document.getElementById("errorWrap");
  const debugInputs = document.getElementById("debugInputs");
  const submitBtn = document.getElementById("submitBtn");
  const submitSpinner = document.getElementById("submitSpinner");
  const want_b_hidden = document.getElementById("want_b_hidden");

  function bindFilter(inputId, selectId) {
    const input = document.getElementById(inputId);
    const select = document.getElementById(selectId);
    if (!input || !select) return;
    input.addEventListener("input", function () {
      const q = input.value.toLowerCase().trim();
      Array.from(select.options).forEach(opt => {
        if (!opt.value) {
          opt.hidden = false;
          return;
        }
        opt.hidden = q && !opt.text.toLowerCase().includes(q);
      });
    });
    input.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter") {
        ev.preventDefault();
        select.focus();
      }
    });
  }
  bindFilter("stage_search", "stage");
  bindFilter("region_search", "region");
  bindFilter("cause_search", "cause");

  document.getElementById("modelOpts").addEventListener("change", function (ev) {
    const model = document.querySelector('input[name="model"]:checked').value;
    if (model === "B") {
      want_b_hidden.value = "on";
    } else {
      want_b_hidden.value = "";
    }
  });

  document.querySelectorAll(".preset").forEach(btn => {
    btn.addEventListener("click", function () {
      const s = this.getAttribute("data-stage");
      const r = this.getAttribute("data-region");
      if (s) {
        setSelectValueByText("stage", s);
      }
      if (r) {
        setSelectValueByText("region", r);
      }
    });
  });

  document.getElementById("clearHistory").addEventListener("click", function () {
    localStorage.removeItem("pred_history");
    renderHistory();
  });

  function renderHistory() {
    const list = document.getElementById("historyList");
    list.innerHTML = "";
    const hist = JSON.parse(localStorage.getItem("pred_history") || "[]");
    if (!hist.length) {
      list.innerHTML = "<li class='text-muted'>No history yet</li>";
      return;
    }
    hist.slice().reverse().forEach(item => {
      const li = document.createElement("li");
      li.innerHTML = `<strong>${escapeHtml(item.disease)}</strong> — ${escapeHtml(item.model)} (${item.time})`;
      list.appendChild(li);
    });
  }
  renderHistory();

  function pushHistory(entry) {
    const hist = JSON.parse(localStorage.getItem("pred_history") || "[]");
    hist.push(entry);
    localStorage.setItem("pred_history", JSON.stringify(hist.slice(-25)));
    renderHistory();
  }

  function setSelectValueByText(selectId, text) {
    const s = document.getElementById(selectId);
    for (const opt of s.options) {
      if (opt.text === text) {
        s.value = opt.value;
        s.dispatchEvent(new Event('change'));
        return;
      }
    }
    if (selectId === "cause" || selectId === "stage" || selectId === "region") {
      s.value = text;
    }
  }

  function getModelChoice() {
    const model = document.querySelector('input[name="model"]:checked').value;
    if (model === "auto") return "auto";
    if (model === "A") return "A";
    if (model === "B") return "B";
    if (model === "C") return "C";
    return "auto";
  }

  form.addEventListener("submit", async function (e) {
    e.preventDefault();
    if (!form.checkValidity()) {
      form.classList.add("was-validated");
      return;
    }

    const modelChoice = getModelChoice();
    const stage = document.getElementById("stage").value.trim();
    const region = document.getElementById("region").value.trim();
    const cause = document.getElementById("cause").value.trim();

    let want_b = false;
    if (modelChoice === "B") {
      want_b = true;
    } else if (modelChoice === "A") {
      document.getElementById("cause").value = "";
    } else if (modelChoice === "C") {
      if (!cause) {
        showError("Model C requires a cause. Please select it or switch model.");
        return;
      }
    }

    const payload = {
      stage: stage,
      region: region,
      cause: cause,
      want_b: want_b
    };

    errorWrap.classList.add("d-none");
    resultCard.classList.remove("d-none");
    resultTitle.textContent = "Predicting…";
    resultText.textContent = "";
    confWrap.classList.add("d-none");
    confBar.style.width = "0%";
    confBar.textContent = "0%";
    debugInputs.textContent = `${stage || "—"} / ${region || "—"} / ${cause || "—"} (${modelChoice})`;

    submitBtn.disabled = true;
    submitSpinner.classList.remove("d-none");

    try {
      const resp = await fetch("/api/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      const data = await resp.json();

      if (data.error) {
        throw new Error(data.error);
      }

      resultTitle.textContent = "Prediction";
      if (data.disease && data.cause) {
        resultText.innerHTML = `<strong>Disease:</strong> ${escapeHtml(data.disease)}<br/><strong>Cause:</strong> ${escapeHtml(data.cause)}`;
      } else if (data.disease) {
        resultText.innerHTML = `<strong>Disease:</strong> ${escapeHtml(data.disease)}`;
      } else {
        resultText.textContent = JSON.stringify(data);
      }

      if (data.confidence !== undefined) {
        confWrap.classList.remove("d-none");
        const pct = Math.round(100 * Number(data.confidence));
        confBar.style.width = pct + "%";
        confBar.textContent = pct + "%";
        confBar.classList.remove("bg-danger", "bg-success");
        if (pct > 66) confBar.classList.add("bg-success");
        else if (pct < 33) confBar.classList.add("bg-danger");
      } else {
        confWrap.classList.add("d-none");
      }

      const now = new Date().toLocaleString();
      pushHistory({ disease: data.disease || "(none)", model: modelChoice === "auto" ? (cause ? "C (auto)" : (want_b ? "B (auto)" : "A (auto)")) : modelChoice, time: now });
      debugInputs.textContent = `${stage || "—"} / ${region || "—"} / ${cause || "—"} (${modelChoice})`;

    } catch (err) {
      showError(err.message || "Unknown error");
    } finally {
      submitBtn.disabled = false;
      submitSpinner.classList.add("d-none");
    }
  });

  function showError(msg) {
    errorWrap.classList.remove("d-none");
    errorWrap.textContent = msg;
    resultCard.classList.remove("d-none");
    resultTitle.textContent = "Error";
    resultText.textContent = "";
  }

  function escapeHtml(s) {
    if (!s) return s;
    return String(s).replaceAll("&","&amp;").replaceAll("<","&lt;").replaceAll(">","&gt;").replaceAll('"',"&quot;");
  }
});