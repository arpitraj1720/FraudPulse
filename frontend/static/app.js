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
  let activeTimerInterval = null;

  function clearTimerInterval() {
    if (activeTimerInterval) {
      clearInterval(activeTimerInterval);
      activeTimerInterval = null;
    }
  }

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

  // 3. Replay Scenario with Two-Tier Async Decoupling
  async function runScenarioReplay(scenarioId) {
    if (isReplaying) return;
    isReplaying = true;
    clearTimerInterval();

    replayStatus.textContent = "⚡ Tier 1 Hot Path running...";
    replayStatus.style.color = "var(--color-blue)";

    // Clear feed & reset dials immediately
    timelineFeed.innerHTML = "";
    resetScoreDial();
    renderEmptyInvestigation("⚡ Tier 1: Evaluating transaction sequence in-memory...");

    try {
      // Tier 1: Superfast synchronous score (< 10ms)
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

      // Swift animated playback of sequence
      for (let i = 0; i < data.steps.length; i++) {
        const step = data.steps[i];
        await new Promise(r => setTimeout(r, 120));
        appendTimelineItem(step, i + 1);
        updateScoreDial(step);
      }

      // Tier 1 Decision Path
      if (!data.requires_investigation) {
        // SAFE TRANSACTION: Auto-Cleared immediately!
        renderBypassedInvestigation(data.tier1_latency_ms);
        replayStatus.textContent = `Auto-Cleared (${data.tier1_latency_ms}ms)`;
        replayStatus.style.color = "var(--color-green)";
        isReplaying = false;
      } else {
        // SUSPICIOUS TRANSACTION: Escalate to Tier 2 Gemma asynchronously!
        renderTier2Loading(data.tier1_latency_ms);
        replayStatus.textContent = `⚡ Flagged (${data.tier1_latency_ms}ms) | 🤖 Tier 2 Generating...`;
        replayStatus.style.color = "var(--color-amber)";
        isReplaying = false;

        // Async Background fetch to Tier 2 Gemma
        try {
          const invRes = await fetch("/api/investigate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data.investigation_payload),
          });
          if (!invRes.ok) throw new Error(`Tier 2 API error ${invRes.status}`);
          const brief = await invRes.json();
          renderInvestigation(brief, data.tier1_latency_ms);
          replayStatus.textContent = `Completed (Tier 1 + Tier 2 Ready)`;
          replayStatus.style.color = "var(--color-green)";
        } catch (invErr) {
          console.error("Tier 2 investigation error:", invErr);
          clearTimerInterval();
          renderEmptyInvestigation("Notice: Tier 2 investigation offline. Policy enforcement remains active.");
        }
      }
    } catch (e) {
      console.error("Replay error:", e);
      replayStatus.textContent = "Error";
      replayStatus.style.color = "var(--color-rose)";
      renderEmptyInvestigation(`Notice: ${e.message || "Failed to load sequence."} Click again to retry.`);
      isReplaying = false;
    }
  }

  // 4. Submit Custom Transaction
  txForm.addEventListener("submit", async (e) => {
    e.preventDefault();
    if (isReplaying) return;
    clearTimerInterval();

    const payload = {
      account_id: document.getElementById("acc-id").value,
      type: document.getElementById("tx-type").value,
      amount: parseFloat(document.getElementById("tx-amount").value),
      recipient: document.getElementById("tx-recipient").value,
      step: 1,
      auto_investigate: false,
    };

    accountContext.innerHTML = `Monitoring: <span class="highlight">${payload.account_id}</span>`;
    replayStatus.textContent = "⚡ Tier 1 Scoring...";
    replayStatus.style.color = "var(--color-blue)";

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

      if (!data.requires_investigation) {
        renderBypassedInvestigation(data.tier1_latency_ms);
        replayStatus.textContent = `Auto-Cleared (${data.tier1_latency_ms}ms)`;
        replayStatus.style.color = "var(--color-green)";
      } else {
        renderTier2Loading(data.tier1_latency_ms);
        replayStatus.textContent = `⚡ Flagged (${data.tier1_latency_ms}ms) | 🤖 Tier 2 Generating...`;
        replayStatus.style.color = "var(--color-amber)";

        try {
          const invRes = await fetch("/api/investigate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(data.investigation_payload),
          });
          const brief = await invRes.json();
          renderInvestigation(brief, data.tier1_latency_ms);
          replayStatus.textContent = `Completed (Tier 1 + Tier 2 Ready)`;
          replayStatus.style.color = "var(--color-green)";
        } catch (invErr) {
          clearTimerInterval();
          renderEmptyInvestigation("Notice: Tier 2 investigation offline. Policy enforcement remains active.");
        }
      }
    } catch (e) {
      console.error("Scoring error:", e);
      replayStatus.textContent = "Error";
      replayStatus.style.color = "var(--color-rose)";
    }
  });

  // Helper: Append Timeline Card
  function appendTimelineItem(step, index) {
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

    scoreRing.parentElement.className = `score-dial-wrap ${step.risk_tier.toLowerCase()}`;

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
  function renderInvestigation(inv, tier1LatencyMs) {
    clearTimerInterval();
    invSource.textContent = inv.source || "Gemma 4-26B";
    const signalsList = (inv.observed_signals || [])
      .map(s => `<li>${s}</li>`)
      .join("");
    const counterList = (inv.counter_evidence || [])
      .map(c => `<li>${c}</li>`)
      .join("");

    const genTimeSec = inv.generation_time_ms ? (inv.generation_time_ms / 1000).toFixed(1) : "7.2";

    invBody.innerHTML = `
      <div class="tier-path-banner escalated">
        <span>⚡ TIER 1: ESCALATED (${tier1LatencyMs || 3.9}ms)</span>
        <span>🤖 TIER 2: SYNTHESIZED (${genTimeSec}s)</span>
      </div>

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
      <span class="generation-time-tag">⏱️ Incident brief synthesized in ${genTimeSec}s via ${inv.source || 'models/gemma-4-26b-a4b-it'}</span>
    `;
  }

  function renderTier2Loading(tier1LatencyMs) {
    clearTimerInterval();
    invSource.textContent = "Gemma 27B";
    const startSec = Date.now();

    invBody.innerHTML = `
      <div class="tier-path-banner escalated">
        <span>⚡ TIER 1 HOT PATH: ESCALATED</span>
        <span>${tier1LatencyMs || 3.8}ms SLA</span>
      </div>
      <div class="tier2-loader-box">
        <div class="pulsing-radar"></div>
        <div class="tier2-loader-title">Tier 2: Gemma 27B Autonomous Copilot</div>
        <div class="tier2-loader-sub">Synthesizing behavioral velocity, multi-wallet graph & compliance SAR brief in the background...</div>
        <div class="tier2-timer-badge" id="tier2-timer">Calling Google AI Studio (0.0s)</div>
      </div>
    `;

    activeTimerInterval = setInterval(() => {
      const elapsed = ((Date.now() - startSec) / 1000).toFixed(1);
      const badge = document.getElementById("tier2-timer");
      if (badge) {
        badge.textContent = `Calling Google AI Studio (${elapsed}s)`;
      }
    }, 100);
  }

  function renderBypassedInvestigation(tier1LatencyMs) {
    clearTimerInterval();
    invSource.textContent = "Tier 1 Cleared";
    invBody.innerHTML = `
      <div class="tier-path-banner cleared">
        <span>⚡ TIER 1 HOT PATH: AUTO-CLEARED</span>
        <span>${tier1LatencyMs || 2.4}ms SLA</span>
      </div>
      <div class="tier2-bypassed-box">
        <div class="tier2-bypassed-title">✓ Tier 2 LLM Escalation Bypassed</div>
        <div class="tier2-bypassed-desc">
          Payment passed in-line ML validation and payee trust discount. Zero LLM latency overhead incurred. Transaction authorized at gateway speed.
        </div>
      </div>
    `;
  }

  function renderEmptyInvestigation(msg) {
    clearTimerInterval();
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

  // ========================================================================
  // FUTURISTIC 3D TRANSACTION VISUAL (THREE.JS)
  // Two matching 3D objects moving symmetrically toward each other
  // ========================================================================
  function initHero3DAnimation() {
    const canvas = document.getElementById("hero-3d-canvas");
    const container = document.getElementById("canvas-3d-wrapper");
    const distanceMeter = document.getElementById("tx-distance-meter");
    if (!canvas || !container || typeof THREE === "undefined") {
      console.warn("Three.js or canvas not available for 3D hero.");
      return;
    }

    const scene = new THREE.Scene();

    const camera = new THREE.PerspectiveCamera(
      42,
      container.clientWidth / container.clientHeight,
      0.1,
      100
    );
    camera.position.set(0, 0, 8.5);

    const renderer = new THREE.WebGLRenderer({
      canvas: canvas,
      antialias: true,
      alpha: true,
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.1;

    // --- Studio Lighting ---
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.85);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.2);
    keyLight.position.set(4, 7, 5);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x94a3b8, 0.7);
    fillLight.position.set(-5, -2, 4);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0xffffff, 0.9);
    rimLight.position.set(0, 8, -4);
    scene.add(rimLight);

    // Contact Plasma Point Light
    const contactLight = new THREE.PointLight(0x38bdf8, 0, 4);
    contactLight.position.set(0, 0, 0);
    scene.add(contactLight);

    // --- Materials (Ceramic White & Dark Graphite Precision) ---
    const ceramicMat = new THREE.MeshStandardMaterial({
      color: 0xf8fafc,
      roughness: 0.18,
      metalness: 0.08,
    });

    const graphiteMat = new THREE.MeshStandardMaterial({
      color: 0x1e2430,
      roughness: 0.35,
      metalness: 0.85,
    });

    const titaniumMat = new THREE.MeshStandardMaterial({
      color: 0xd1d5db,
      roughness: 0.25,
      metalness: 0.92,
    });

    const contactRingMat = new THREE.MeshStandardMaterial({
      color: 0x0f172a,
      roughness: 0.4,
      metalness: 0.7,
    });

    const glowAccentMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
    });

    // Helper: Build a Modular Cryptographic Terminal Cylinder (Image 2 Aesthetic)
    function createTransactionCore(isTop) {
      const group = new THREE.Group();

      // Main Outer Sleeve
      const mainGeom = new THREE.CylinderGeometry(1.05, 1.25, 2.8, 48, 1);
      const mainMesh = new THREE.Mesh(mainGeom, ceramicMat);
      mainMesh.position.y = isTop ? 1.4 : -1.4;
      group.add(mainMesh);

      // Chamfered Back Cap
      const capGeom = new THREE.CylinderGeometry(
        isTop ? 1.3 : 0.9,
        isTop ? 0.9 : 1.3,
        0.3,
        48
      );
      const capMesh = new THREE.Mesh(capGeom, graphiteMat);
      capMesh.position.y = isTop ? 2.85 : -2.85;
      group.add(capMesh);

      // Dark Precision Joint Collar
      const collarGeom = new THREE.CylinderGeometry(1.08, 1.08, 0.45, 48);
      const collarMesh = new THREE.Mesh(collarGeom, graphiteMat);
      collarMesh.position.y = isTop ? 0.35 : -0.35;
      group.add(collarMesh);

      // Fine Inset Grooves
      const grooveGeom = new THREE.TorusGeometry(1.085, 0.018, 16, 48);
      const groove1 = new THREE.Mesh(grooveGeom, titaniumMat);
      groove1.rotation.x = Math.PI / 2;
      groove1.position.y = isTop ? 0.45 : -0.45;
      group.add(groove1);

      // Connector Face / Interface Plate
      const faceGeom = new THREE.CylinderGeometry(0.98, 1.04, 0.2, 48);
      const faceMesh = new THREE.Mesh(faceGeom, contactRingMat);
      faceMesh.position.y = isTop ? 0.08 : -0.08;
      group.add(faceMesh);

      // Matching Latch / Pin Mechanism (Inspired by MDXLAB connector)
      if (isTop) {
        // Male Cryptographic Key / Probe
        const probeBase = new THREE.CylinderGeometry(0.38, 0.45, 0.4, 32);
        const probeBaseMesh = new THREE.Mesh(probeBase, titaniumMat);
        probeBaseMesh.position.y = -0.15;
        group.add(probeBaseMesh);

        const pinGeom = new THREE.BoxGeometry(0.18, 0.35, 0.48);
        const pinMesh = new THREE.Mesh(pinGeom, graphiteMat);
        pinMesh.position.y = -0.22;
        group.add(pinMesh);

        // Core Glowing Aperture
        const glowRingGeom = new THREE.TorusGeometry(0.68, 0.025, 16, 36);
        const glowRing = new THREE.Mesh(glowRingGeom, glowAccentMat);
        glowRing.rotation.x = Math.PI / 2;
        glowRing.position.y = 0.01;
        group.add(glowRing);
      } else {
        // Female Receptor Socket
        const socketRim = new THREE.CylinderGeometry(0.48, 0.48, 0.35, 32);
        const socketRimMesh = new THREE.Mesh(socketRim, graphiteMat);
        socketRimMesh.position.y = 0.12;
        group.add(socketRimMesh);

        // Interlocking Notch Aperture
        const notchGeom = new THREE.CylinderGeometry(0.3, 0.3, 0.2, 32);
        const notchMesh = new THREE.Mesh(notchGeom, contactRingMat);
        notchMesh.position.y = 0.18;
        group.add(notchMesh);

        // Matching Emissive Ring
        const glowRingGeom = new THREE.TorusGeometry(0.68, 0.025, 16, 36);
        const glowRing = new THREE.Mesh(glowRingGeom, glowAccentMat);
        glowRing.rotation.x = Math.PI / 2;
        glowRing.position.y = -0.01;
        group.add(glowRing);
      }

      return group;
    }

    // Two Matching Symmetric 3D Objects (Sender Core Top & Receiver Core Bottom)
    const topCore = createTransactionCore(true);
    const bottomCore = createTransactionCore(false);
    scene.add(topCore);
    scene.add(bottomCore);

    // Convergence Energy Pulse Ring
    const pulseRingGeom = new THREE.RingGeometry(0.2, 0.85, 32);
    const pulseRingMat = new THREE.MeshBasicMaterial({
      color: 0x38bdf8,
      side: THREE.DoubleSide,
      transparent: true,
      opacity: 0,
      blending: THREE.AdditiveBlending,
    });
    const pulseRing = new THREE.Mesh(pulseRingGeom, pulseRingMat);
    pulseRing.rotation.x = Math.PI / 2;
    scene.add(pulseRing);

    // Floating Ambient Data Sparks
    const particleCount = 40;
    const particleGeom = new THREE.BufferGeometry();
    const particlePositions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount * 3; i += 3) {
      particlePositions[i] = (Math.random() - 0.5) * 3.5;
      particlePositions[i + 1] = (Math.random() - 0.5) * 1.5;
      particlePositions[i + 2] = (Math.random() - 0.5) * 3.5;
    }
    particleGeom.setAttribute(
      "position",
      new THREE.BufferAttribute(particlePositions, 3)
    );
    const particleMat = new THREE.PointsMaterial({
      color: 0x60a5fa,
      size: 0.035,
      transparent: true,
      opacity: 0.6,
    });
    const particleSystem = new THREE.Points(particleGeom, particleMat);
    scene.add(particleSystem);

    // --- Interactive Mouse & Drag Rotation ---
    let isDragging = false;
    let prevMouseX = 0;
    let prevMouseY = 0;
    let targetRotY = 0.35;
    let targetRotX = 0.12;
    let currentRotY = 0.35;
    let currentRotX = 0.12;

    container.addEventListener("mousedown", (e) => {
      isDragging = true;
      prevMouseX = e.clientX;
      prevMouseY = e.clientY;
    });

    window.addEventListener("mouseup", () => {
      isDragging = false;
    });

    window.addEventListener("mousemove", (e) => {
      if (isDragging) {
        const deltaX = e.clientX - prevMouseX;
        const deltaY = e.clientY - prevMouseY;
        targetRotY += deltaX * 0.007;
        targetRotX += deltaY * 0.007;
        targetRotX = Math.max(-0.4, Math.min(0.4, targetRotX));
        prevMouseX = e.clientX;
        prevMouseY = e.clientY;
      } else {
        // Subtle Parallax on Hover
        const rect = container.getBoundingClientRect();
        if (
          e.clientX >= rect.left &&
          e.clientX <= rect.right &&
          e.clientY >= rect.top &&
          e.clientY <= rect.bottom
        ) {
          const normX = (e.clientX - rect.left) / rect.width - 0.5;
          const normY = (e.clientY - rect.top) / rect.height - 0.5;
          targetRotY = 0.35 + normX * 0.45;
          targetRotX = 0.12 + normY * 0.3;
        }
      }
    });

    // Touch support for mobile / tablets
    container.addEventListener("touchstart", (e) => {
      if (e.touches.length === 1) {
        isDragging = true;
        prevMouseX = e.touches[0].clientX;
        prevMouseY = e.touches[0].clientY;
      }
    }, { passive: true });

    window.addEventListener("touchend", () => {
      isDragging = false;
    });

    window.addEventListener("touchmove", (e) => {
      if (isDragging && e.touches.length === 1) {
        const deltaX = e.touches[0].clientX - prevMouseX;
        const deltaY = e.touches[0].clientY - prevMouseY;
        targetRotY += deltaX * 0.008;
        targetRotX += deltaY * 0.008;
        prevMouseX = e.touches[0].clientX;
        prevMouseY = e.touches[0].clientY;
      }
    }, { passive: true });

    // Handle Resize
    function onResize() {
      if (!container || !renderer || !camera) return;
      const width = container.clientWidth;
      const height = container.clientHeight;
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
      renderer.setSize(width, height);
    }
    window.addEventListener("resize", onResize);

    // --- Animation Loop ---
    let clock = new THREE.Clock();

    function animate() {
      requestAnimationFrame(animate);
      const elapsedTime = clock.getElapsedTime();

      // Smooth Rotation Damping
      currentRotY += (targetRotY - currentRotY) * 0.08;
      currentRotX += (targetRotX - currentRotX) * 0.08;

      topCore.rotation.y = currentRotY + elapsedTime * 0.15;
      topCore.rotation.x = currentRotX;

      bottomCore.rotation.y = -currentRotY - elapsedTime * 0.15;
      bottomCore.rotation.x = currentRotX;

      // Symmetrical Convergence Motion:
      // Both objects move smoothly toward each other from top (+y) and bottom (-y)
      // Base travel amplitude
      const cycle = Math.sin(elapsedTime * 1.5);
      const normalizedApproach = (cycle + 1) / 2; // 0 to 1

      // Top moves from 1.6 down to 0.18
      const topY = 1.6 - normalizedApproach * 1.42;
      // Bottom moves from -1.6 up to -0.18
      const bottomY = -1.6 + normalizedApproach * 1.42;

      topCore.position.y = topY;
      bottomCore.position.y = bottomY;

      // Convergence Distance Telemetry readout
      const distance = topY - bottomY - 0.36;
      const distanceMicrons = Math.max(0.01, distance * 0.12).toFixed(3);
      if (distanceMeter) {
        if (distance < 0.08) {
          distanceMeter.textContent = "CONVERGENCE: ENGAGED";
          distanceMeter.style.color = "#059669";
        } else {
          distanceMeter.textContent = `CONVERGENCE: ${distanceMicrons}μm`;
          distanceMeter.style.color = "#000";
        }
      }

      // Energy Pulse & Flare on Closest Approach
      if (distance < 0.25) {
        const proximityFactor = 1 - distance / 0.25;
        pulseRingMat.opacity = proximityFactor * 0.85;
        pulseRing.scale.setScalar(1 + proximityFactor * 0.6);
        contactLight.intensity = proximityFactor * 2.2;
      } else {
        pulseRingMat.opacity = 0;
        contactLight.intensity = 0;
      }

      // Slowly rotate particle field
      particleSystem.rotation.y = elapsedTime * 0.05;

      renderer.render(scene, camera);
    }

    animate();
  }

  // ========================================================================
  // MINI TRANSACTION STREAM OSCILLOSCOPE (CANVAS 2D)
  // Displayed in the left hero media card
  // ========================================================================
  function initMiniStreamOscilloscope() {
    const canvas = document.getElementById("mini-stream-canvas");
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    let frame = 0;

    function renderWave() {
      requestAnimationFrame(renderWave);
      frame++;
      const w = canvas.width;
      const h = canvas.height;

      ctx.fillStyle = "#070a0e";
      ctx.fillRect(0, 0, w, h);

      // Grid background
      ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
      ctx.lineWidth = 1;
      for (let x = 0; x < w; x += 20) {
        ctx.beginPath();
        ctx.moveTo(x, 0);
        ctx.lineTo(x, h);
        ctx.stroke();
      }
      for (let y = 0; y < h; y += 20) {
        ctx.beginPath();
        ctx.moveTo(0, y);
        ctx.lineTo(w, y);
        ctx.stroke();
      }

      // Draw real-time frequency pulse wave
      ctx.beginPath();
      ctx.strokeStyle = "#38bdf8";
      ctx.lineWidth = 2;
      ctx.shadowColor = "#38bdf8";
      ctx.shadowBlur = 8;

      const mid = h / 2;
      for (let x = 0; x < w; x++) {
        // Multi-frequency harmonic with intermittent sharp transaction spikes
        const t = (x + frame * 2.5) * 0.04;
        let y = mid + Math.sin(t) * 12 + Math.sin(t * 2.2) * 6;

        // Transaction spike trigger every ~80 frames
        const spikeTrigger = (x + frame * 3) % 110;
        if (spikeTrigger > 50 && spikeTrigger < 58) {
          y += Math.sin((spikeTrigger - 50) * Math.PI / 8) * -28;
        }

        if (x === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      }
      ctx.stroke();
      ctx.shadowBlur = 0;
    }

    renderWave();
  }

  // Smooth Scroll Trigger for Hero CTAs
  const heroScrollBtn = document.getElementById("hero-scroll-btn");
  if (heroScrollBtn) {
    heroScrollBtn.addEventListener("click", (e) => {
      e.preventDefault();
      const target = document.getElementById("simulator-section");
      if (target) {
        target.scrollIntoView({ behavior: "smooth" });
      }
    });
  }

  // Initialize Hero Animations
  initHero3DAnimation();
  initMiniStreamOscilloscope();

  // Initial load
  checkHealth();
  loadScenarios();
});
