document.addEventListener("DOMContentLoaded", () => {
  const scenarioList = document.getElementById("scenario-list");
  const timelineFeed = document.getElementById("timeline-feed");
  const accountContext = document.getElementById("account-context");
  const replayStatus = document.getElementById("replay-status");
  const txForm = document.getElementById("tx-form");
  const btnReset = document.getElementById("btn-reset");
  const btnOpenMetrics = document.getElementById("btn-open-metrics");
  const btnCloseModal = document.getElementById("btn-close-modal");
  const metricsModal = document.getElementById("metrics-modal");
  const modalMetricsBody = document.getElementById("modal-metrics-body");

  // Dials & Scores
  const finalScoreVal = document.getElementById("final-score-val");
  const scoreRing = document.getElementById("score-ring");
  const riskTierBadge = document.getElementById("risk-tier-badge");
  const policyDecisionText = document.getElementById("policy-decision-text");
  const actionRecommendationText = document.getElementById("action-recommendation-text");
  const mlScoreVal = document.getElementById("ml-score-val");
  const seqScoreVal = document.getElementById("seq-score-val");
  const mlBar = document.getElementById("ml-bar");
  const seqBar = document.getElementById("seq-bar");

  // Investigation Panel
  const invBody = document.getElementById("inv-body");
  const invSource = document.getElementById("inv-source");

  // Status pills
  const valModel = document.getElementById("val-model");
  const valGemma = document.getElementById("val-gemma");

  let isReplaying = false;

  // 1. Fetch Health & Status
  async function checkHealth() {
    try {
      const res = await fetch("/api/health");
      const data = await res.json();
      if (data.model_loaded) {
        valModel.textContent = data.model_version || "RandomForest Ready";
      }
      if (data.gemma_available) {
        valGemma.textContent = (data.gemma_model || "Gemma Active").replace("models/", "");
      } else {
        valGemma.textContent = "Offline Fallback";
        document.getElementById("pill-gemma").querySelector(".status-dot").className = "status-dot amber";
      }
    } catch (e) {
      console.warn("Could not fetch health:", e);
    }
  }

  // 2. Fetch Scenarios
  async function loadScenarios() {
    try {
      const res = await fetch("/api/scenarios");
      const scenarios = await res.json();
      scenarioList.innerHTML = "";

      scenarios.forEach((sc, idx) => {
        const card = document.createElement("div");
        card.className = "scenario-card" + (idx === 0 ? " active" : "");
        card.dataset.id = sc.id;
        card.innerHTML = `
          <div class="scenario-top">
            <span class="scenario-title">${sc.title}</span>
            <span class="scenario-badge ${sc.badge_color}">${sc.badge}</span>
          </div>
          <div class="scenario-desc">${sc.description}</div>
        `;

        card.addEventListener("click", () => {
          if (isReplaying) return;
          document.querySelectorAll(".scenario-card").forEach(c => c.classList.remove("active"));
          card.classList.add("active");
          runScenarioReplay(sc.id);
        });

        scenarioList.appendChild(card);
      });
    } catch (e) {
      scenarioList.innerHTML = `<div class="empty-desc text-danger">Failed to load scenarios.</div>`;
    }
  }

  // 3. Replay Scenario with Step-by-Step Animation
  async function runScenarioReplay(scenarioId) {
    if (isReplaying) return;
    isReplaying = true;
    replayStatus.textContent = "Replaying...";
    replayStatus.style.color = "var(--color-amber)";

    // Clear feed & reset dials immediately
    timelineFeed.innerHTML = "";
    resetScoreDial();
    renderEmptyInvestigation("✦ Gemma 4-26B analyzing sequence in real time...");

    try {
      const res = await fetch(`/api/scenarios/replay/${scenarioId}`, { method: "POST" });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Server error ${res.status}`);
      }
      const data = await res.json();
      if (!data || !Array.isArray(data.steps)) {
        throw new Error("Invalid sequence data received");
      }

      accountContext.innerHTML = `Monitoring: <span class="highlight">${data.account_id}</span> (${data.scenario.title})`;

      // Animate step by step
      for (let i = 0; i < data.steps.length; i++) {
        const step = data.steps[i];
        await new Promise(r => setTimeout(r, 220));
        appendTimelineItem(step, i + 1);
        updateScoreDial(step);
      }

      // Render Final Investigation
      if (data.final_investigation) {
        renderInvestigation(data.final_investigation);
      } else {
        renderEmptyInvestigation("Transaction cleared within baseline parameters. Verified legitimate sequence profile.");
      }

      replayStatus.textContent = "Completed";
      replayStatus.style.color = "var(--color-green)";
    } catch (e) {
      console.error("Replay error:", e);
      replayStatus.textContent = "Error";
      replayStatus.style.color = "var(--color-rose)";
      renderEmptyInvestigation(`Notice: ${e.message || "Failed to load sequence."} Click again to retry.`);
    } finally {
      isReplaying = false;
    }
  }

  // 4. Submit Custom Transaction
  txForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (isReplaying) return;

    const payload = {
      account_id: document.getElementById("acc-id").value,
      type: document.getElementById("tx-type").value,
      amount: parseFloat(document.getElementById("tx-amount").value),
      recipient: document.getElementById("tx-recipient").value,
      step: 1,
      auto_investigate: true,
    };

    accountContext.innerHTML = `Monitoring: <span class="highlight">${payload.account_id}</span>`;
    replayStatus.textContent = "Scoring...";

    try {
      const res = await fetch("/api/score", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await res.json();

      const stepData = {
        step: payload.step,
        type: payload.type,
        amount: payload.amount,
        recipient: payload.recipient,
        label: "Manual injection",
        ml_risk: data.ml_risk,
        sequence_score: data.sequence_evaluation.sequence_risk_score,
        final_risk_score: data.hybrid_risk.final_risk_score,
        risk_tier: data.hybrid_risk.risk_tier,
        policy_decision: data.hybrid_risk.policy_decision,
        action_label: data.hybrid_risk.recommended_action_label,
        signals: data.sequence_evaluation.signals,
        is_micro_probing: data.sequence_evaluation.is_micro_probing,
        behavioral_features: data.sequence_evaluation.behavioral_features || {},
      };

      appendTimelineItem(stepData);
      updateScoreDial(stepData);

      if (data.investigation) {
        renderInvestigation(data.investigation);
      } else {
        renderEmptyInvestigation("Risk score below threshold. No investigation warranted.");
      }

      replayStatus.textContent = "Scored";
      replayStatus.style.color = "var(--color-green)";
    } catch (e) {
      console.error("Scoring error:", e);
      replayStatus.textContent = "Error";
      replayStatus.style.color = "var(--color-rose)";
    }
  });

  // Helper: Append Timeline Card
  function appendTimelineItem(step, index) {
    // If empty state exists, remove it
    const empty = timelineFeed.querySelector(".empty-state");
    if (empty) empty.remove();

    const isCritical = step.risk_tier === "CRITICAL" || step.is_micro_probing;
    const isSafe = step.risk_tier === "LOW";

    const bf = step.behavioral_features || {};
    let bfHtml = "";
    if (Object.keys(bf).length > 0) {
      bfHtml = `
        <div class="timeline-features">
          <span class="feat-pill" title="Current amount vs 10-tx moving average"><span class="feat-k">amount_vs_avg:</span> <span class="feat-v">${bf.amount_vs_avg_ratio || 1}x</span></span>
          <span class="feat-pill" title="Distinct recipients in last 10m"><span class="feat-k">unique_payees_10m:</span> <span class="feat-v">${bf.unique_recipients_10m || 1}</span></span>
          <span class="feat-pill" title="Seconds since prior transaction"><span class="feat-k">time_delta:</span> <span class="feat-v">${bf.time_since_last_tx || 0}s</span></span>
          ${bf.micro_tx_before_large_tx ? '<span class="feat-pill critical" title="Micro-probing sequence detected"><span class="feat-k">micro_probing:</span> <span class="feat-v">TRUE</span></span>' : ''}
          ${bf.new_recipient ? '<span class="feat-pill warning" title="First-time recipient"><span class="feat-k">new_payee:</span> <span class="feat-v">YES</span></span>' : ''}
        </div>
      `;
    }

    const item = document.createElement("div");
    item.className = `timeline-item ${isCritical ? "critical" : isSafe ? "safe" : ""}`;

    const signalsHtml = (step.signals || [])
      .map(s => `<span class="signal-chip">[${s.severity}] ${s.title}</span>`)
      .join("");

    item.innerHTML = `
      <div class="timeline-step-badge">
        <span>#${index || step.step}</span>
      </div>
      <div class="timeline-content">
        <div class="timeline-top">
          <span class="timeline-amount">$${step.amount.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</span>
          <span class="timeline-type-pill">${step.type}</span>
        </div>
        <div class="timeline-meta">
          <span>To: <strong>${step.recipient}</strong></span> • <span>${step.label || 'Transaction'}</span>
        </div>
        ${bfHtml}
        ${signalsHtml ? `<div class="timeline-signals">${signalsHtml}</div>` : ""}
      </div>
    `;

    timelineFeed.appendChild(item);
    timelineFeed.scrollTop = timelineFeed.scrollHeight;
  }

  // Helper: Update Risk Dial & Bars
  function updateScoreDial(step) {
    const finalScore = step.final_risk_score;
    finalScoreVal.textContent = finalScore.toFixed(2);
    riskTierBadge.textContent = step.risk_tier;
    riskTierBadge.className = `risk-tier ${step.risk_tier}`;
    policyDecisionText.textContent = step.policy_decision.replace(/_/g, " ");
    actionRecommendationText.textContent = step.action_label || "No action required";

    // Dial ring class
    scoreRing.parentElement.className = `score-dial-wrap ${step.risk_tier.toLowerCase()}`;

    // Bars
    mlScoreVal.textContent = step.ml_risk.toFixed(2);
    mlBar.style.width = `${Math.min(100, step.ml_risk * 100)}%`;

    seqScoreVal.textContent = step.sequence_score.toFixed(2);
    seqBar.style.width = `${Math.min(100, step.sequence_score * 100)}%`;
  }

  function resetScoreDial() {
    finalScoreVal.textContent = "0.00";
    riskTierBadge.textContent = "IDLE";
    riskTierBadge.className = "risk-tier";
    policyDecisionText.textContent = "Awaiting input";
    actionRecommendationText.textContent = "—";
    scoreRing.parentElement.className = "score-dial-wrap";
    mlScoreVal.textContent = "0.00";
    seqScoreVal.textContent = "0.00";
    mlBar.style.width = "0%";
    seqBar.style.width = "0%";
  }

  // Helper: Render Gemma Investigation
  function renderInvestigation(inv) {
    invSource.textContent = inv.source || "Gemma 4-26B";
    const signalsList = (inv.observed_signals || [])
      .map(s => `<li>${s}</li>`)
      .join("");
    const counterList = (inv.counter_evidence || [])
      .map(c => `<li>${c}</li>`)
      .join("");

    invBody.innerHTML = `
      <div class="inv-section-title">Executive Threat Rationale</div>
      <div class="inv-threat-text">${inv.threat_summary}</div>

      <div class="inv-section-title">Observed Sequence Evidence</div>
      <ul class="inv-bullets">${signalsList || "<li>No major signals flagged.</li>"}</ul>

      <div class="inv-section-title">Mitigating Counter-Evidence</div>
      <ul class="inv-bullets">${counterList || "<li>Standard transactional profile.</li>"}</ul>

      <div class="inv-section-title">Recommended Policy Action</div>
      <div class="inv-action-box">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
          <circle cx="12" cy="12" r="10"></circle>
          <line x1="12" y1="8" x2="12" y2="12"></line>
          <line x1="12" y1="16" x2="12.01" y2="16"></line>
        </svg>
        <span>${inv.recommended_action}</span>
      </div>
    `;
  }

  function renderEmptyInvestigation(msg) {
    invSource.textContent = "Monitoring";
    invBody.innerHTML = `<div class="inv-placeholder">${msg}</div>`;
  }

  // 5. Reset Button
  btnReset.addEventListener("click", async () => {
    try {
      await fetch("/api/reset", { method: "POST" });
      timelineFeed.innerHTML = `
        <div class="empty-state">
          <div class="empty-icon">📊</div>
          <div class="empty-title">State Reset Complete</div>
          <div class="empty-desc">Sequence history and known recipients have been flushed.</div>
        </div>
      `;
      accountContext.innerHTML = `Monitoring: <span class="highlight">State cleared</span>`;
      resetScoreDial();
      renderEmptyInvestigation("Engine memory reset. Ready for next evaluation.");
    } catch (e) {
      console.error("Reset error:", e);
    }
  });

  // 6. Metrics Modal
  btnOpenMetrics.addEventListener("click", async () => {
    try {
      const res = await fetch("/api/metrics");
      const m = await res.json();

      let tableRows = (m.threshold_reports || [])
        .map(r => `
          <tr>
            <td><strong>${r.threshold.toFixed(2)}</strong></td>
            <td>${(r.precision * 100).toFixed(1)}%</td>
            <td>${(r.recall * 100).toFixed(1)}%</td>
            <td>${r.f1.toFixed(3)}</td>
            <td style="color: var(--color-rose);">${r.false_positives}</td>
            <td style="color: var(--color-amber);">${r.false_negatives}</td>
            <td>${(r.false_positive_rate * 100).toFixed(2)}%</td>
          </tr>
        `).join("");

      modalMetricsBody.innerHTML = `
        <p><strong>Dataset:</strong> ${m.dataset}</p>
        <p><strong>Test Split:</strong> ${m.test_rows.toLocaleString()} rows (${m.test_fraud_rows} fraud cases, prevalence: ${(m.fraud_prevalence_test * 100).toFixed(2)}%)</p>
        <p><strong>Average Precision (PR-AUC):</strong> <span style="color: var(--color-blue); font-weight: 700;">${m.average_precision_pr_auc.toFixed(4)}</span> | <strong>ROC-AUC:</strong> <span style="color: var(--color-green); font-weight: 700;">${m.roc_auc.toFixed(4)}</span></p>

        <table class="metrics-table">
          <thead>
            <tr>
              <th>Threshold</th>
              <th>Precision</th>
              <th>Recall</th>
              <th>F1 Score</th>
              <th>False Positives</th>
              <th>Missed Fraud</th>
              <th>FPR</th>
            </tr>
          </thead>
          <tbody>
            ${tableRows}
          </tbody>
        </table>

        <div style="margin-top: 16px; font-size: 0.76rem; color: var(--text-dim); line-height: 1.4;">
          <strong>Data Leakage Safeguard:</strong> All post-transaction balances (<code>newbalanceOrig</code>, <code>newbalanceDest</code>) and <code>isFlaggedFraud</code> were strictly excluded during training to preserve realistic inference at payment decision time.
        </div>
      `;

      metricsModal.classList.add("show");
    } catch (e) {
      modalMetricsBody.textContent = "Metrics unavailable.";
      metricsModal.classList.add("show");
    }
  });

  btnCloseModal.addEventListener("click", () => metricsModal.classList.remove("show"));
  metricsModal.addEventListener("click", (e) => {
    if (e.target === metricsModal) metricsModal.classList.remove("show");
  });

  // Initial load
  checkHealth();
  loadScenarios();
});
