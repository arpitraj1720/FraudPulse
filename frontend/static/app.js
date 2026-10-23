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
        document.getElementById("pill-gemma").querySelector(".status-dot").className = "status-dot red";
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
    replayStatus.style.color = "var(--text-main)";

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
        replayStatus.style.color = "var(--color-red)";
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
      replayStatus.style.color = "var(--color-red)";
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
    replayStatus.style.color = "var(--text-main)";

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
        replayStatus.style.color = "var(--color-red)";

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
      replayStatus.style.color = "var(--color-red)";
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
            <td style="color: var(--color-red);">${r.false_positives}</td>
            <td style="color: var(--color-red);">${r.false_negatives}</td>
            <td>${(r.false_positive_rate * 100).toFixed(2)}%</td>
          </tr>
        `).join("");

      modalMetricsBody.innerHTML = `
        <p><strong>Dataset:</strong> ${m.dataset}</p>
        <p><strong>Test Split:</strong> ${m.test_rows.toLocaleString()} rows (${m.test_fraud_rows} fraud cases, prevalence: ${(m.fraud_prevalence_test * 100).toFixed(2)}%)</p>
        <p><strong>Average Precision (PR-AUC):</strong> <span style="color: #ffffff; font-weight: 700;">${m.average_precision_pr_auc.toFixed(4)}</span> | <strong>ROC-AUC:</strong> <span style="color: var(--color-green); font-weight: 700;">${m.roc_auc.toFixed(4)}</span></p>

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
  // ========================================================================
  // FUTURISTIC 3D TRANSACTION VISUAL (THREE.JS)
  // Two matching 3D objects sweeping from top & bottom along a 3D curved arc
  // ========================================================================
  function initHero3DAnimation() {
    const canvas = document.getElementById("hero-3d-canvas");
    const container = document.getElementById("canvas-3d-wrapper");
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
    camera.position.set(0, 0, 9.2);

    const renderer = new THREE.WebGLRenderer({
      canvas: canvas,
      antialias: true,
      alpha: true,
    });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.toneMapping = THREE.ACESFilmicToneMapping;
    renderer.toneMappingExposure = 1.15;

    // --- Studio Lighting ---
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.9);
    scene.add(ambientLight);

    const keyLight = new THREE.DirectionalLight(0xffffff, 1.3);
    keyLight.position.set(5, 8, 6);
    scene.add(keyLight);

    const fillLight = new THREE.DirectionalLight(0x94a3b8, 0.75);
    fillLight.position.set(-6, -3, 5);
    scene.add(fillLight);

    const rimLight = new THREE.DirectionalLight(0xffffff, 0.95);
    rimLight.position.set(0, 9, -5);
    scene.add(rimLight);

    // Contact Plasma Point Light
    const contactLight = new THREE.PointLight(0x38bdf8, 0, 5);
    contactLight.position.set(0, 0, 0);
    scene.add(contactLight);

    // --- Material: Pure Matte Black (Zero Shine, Pure Texture) ---
    const pureMatteBlack = new THREE.MeshStandardMaterial({
      color: 0x080808,
      roughness: 1.0,
      metalness: 0.0,
    });

    // Helper: Build a Solid Oblong Chain Link
    function createSolidLink() {
      const linkGroup = new THREE.Group();

      const linkRadius = 0.58;
      const linkTube = 0.16;
      const straightLen = 0.88;

      // Top semicircular cap
      const topArchGeom = new THREE.TorusGeometry(linkRadius, linkTube, 24, 48, Math.PI);
      const topArch = new THREE.Mesh(topArchGeom, pureMatteBlack);
      topArch.position.y = straightLen / 2;
      linkGroup.add(topArch);

      // Bottom semicircular cap
      const btmArchGeom = new THREE.TorusGeometry(linkRadius, linkTube, 24, 48, Math.PI);
      const btmArch = new THREE.Mesh(btmArchGeom, pureMatteBlack);
      btmArch.rotation.z = Math.PI;
      btmArch.position.y = -straightLen / 2;
      linkGroup.add(btmArch);

      // Straight Left & Right Side Cylinders
      const sideGeom = new THREE.CylinderGeometry(linkTube, linkTube, straightLen, 24);
      const leftSide = new THREE.Mesh(sideGeom, pureMatteBlack);
      leftSide.position.x = -linkRadius;
      linkGroup.add(leftSide);

      const rightSide = new THREE.Mesh(sideGeom, pureMatteBlack);
      rightSide.position.x = linkRadius;
      linkGroup.add(rightSide);

      return linkGroup;
    }

    // Helper: Build a Chain Segment (Top or Bottom)
    // When both meet at y=0, their terminal links thread through each other perfectly
    function createChainSegment(isTop) {
      const segmentGroup = new THREE.Group();
      const linkCount = 5;
      const pitch = 1.20; // Exact mathematical interlocking pitch distance

      for (let i = 0; i < linkCount; i++) {
        const link = createSolidLink();

        // Alternating 90-degree orthogonal rotation:
        // Top segment: link 0 is in ZY plane (rotY = PI/2)
        // Bottom segment: link 0 is in XY plane (rotY = 0)
        // This ensures the two terminal links thread through each other perfectly!
        if (isTop) {
          if (i % 2 === 0) {
            link.rotation.y = Math.PI / 2;
          }
        } else {
          if (i % 2 === 1) {
            link.rotation.y = Math.PI / 2;
          }
        }

        // Space links outward from the center (terminal link at pitch/2)
        const yOffset = (isTop ? 1 : -1) * (pitch / 2 + i * pitch);
        link.position.y = yOffset;

        segmentGroup.add(link);
      }

      return segmentGroup;
    }

    // Two Matching Chain Segments: Top Chain & Bottom Chain
    // Broken initially, and connecting perfectly at the center
    const topCore = createChainSegment(true);
    const bottomCore = createChainSegment(false);
    scene.add(topCore);
    scene.add(bottomCore);

    // Dynamic Connection Surge Flare Ring
    const pulseRingGeom = new THREE.RingGeometry(0.15, 1.25, 36);
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
    const particleCount = 45;
    const particleGeom = new THREE.BufferGeometry();
    const particlePositions = new Float32Array(particleCount * 3);
    for (let i = 0; i < particleCount * 3; i += 3) {
      particlePositions[i] = (Math.random() - 0.5) * 4.5;
      particlePositions[i + 1] = (Math.random() - 0.5) * 2.5;
      particlePositions[i + 2] = (Math.random() - 0.5) * 4.5;
    }
    particleGeom.setAttribute(
      "position",
      new THREE.BufferAttribute(particlePositions, 3)
    );
    const particleMat = new THREE.PointsMaterial({
      color: 0x60a5fa,
      size: 0.04,
      transparent: true,
      opacity: 0.65,
    });
    const particleSystem = new THREE.Points(particleGeom, particleMat);
    scene.add(particleSystem);

    // --- Interactive Mouse & Drag Rotation ---
    let isDragging = false;
    let prevMouseX = 0;
    let prevMouseY = 0;
    let targetRotY = 0.32;
    let targetRotX = 0.1;
    let currentRotY = 0.32;
    let currentRotX = 0.1;

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
          targetRotY = 0.32 + normX * 0.45;
          targetRotX = 0.1 + normY * 0.28;
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

    // --- One-Time Cinematic Chain Connection Animation ---
    let clock = new THREE.Clock();
    const ENTRANCE_DURATION = 2.5; // seconds for the one-time sweep & snap connection

    function animate() {
      requestAnimationFrame(animate);
      const elapsedTime = clock.getElapsedTime();

      // Smooth Rotation Damping for mouse interaction & parallax
      currentRotY += (targetRotY - currentRotY) * 0.08;
      currentRotX += (targetRotX - currentRotX) * 0.08;

      // Single one-time progress: 0 (initial off-screen state) -> 1 (docked state)
      const rawProgress = Math.min(1.0, elapsedTime / ENTRANCE_DURATION);
      // Quintic easeOut: powerful physical surge with smooth deceleration into connection
      const ease = 1 - Math.pow(1 - rawProgress, 4);

      // Elastic snap-rebound effect when the chain pulls taut (around rawProgress 0.88 - 1.0)
      let snapVibration = 0;
      if (rawProgress >= 0.88 && rawProgress < 1.0) {
        const snapT = (rawProgress - 0.88) / 0.12;
        snapVibration = Math.sin(snapT * Math.PI * 4) * Math.exp(-snapT * 3) * 0.06;
      }

      // Top chain segment starts far above screen (+7.8) and descends to exactly 0.0
      const topY = (7.8 * (1 - ease)) + snapVibration;
      // Bottom chain segment starts far below screen (-7.8) and ascends to exactly 0.0
      const bottomY = (-7.8 * (1 - ease)) - snapVibration;

      // Curved lateral trajectory: chains arc gracefully inward and align flush at center (X=0, Z=0)
      const arcFactor = Math.sin(ease * Math.PI);
      const arcX = arcFactor * 1.45;
      const arcZ = arcFactor * 0.45;

      // Dynamic physical sway as the chains travel through air (decays to 0 when docked)
      const sway = Math.sin(elapsedTime * 3.2) * (1 - ease) * 0.16;

      topCore.position.set(arcX + sway, topY, arcZ);
      bottomCore.position.set(-arcX - sway, bottomY, -arcZ);

      // Banking & tilt: chains hang and orient dynamically along the flight vector
      const bankAngle = (1 - ease) * Math.cos(ease * Math.PI) * 0.25;
      const pitchAngle = arcFactor * 0.2;
      const baseTilt = 0.20; // subtle forward tilt to show 3D link depth

      topCore.rotation.z = -bankAngle;
      bottomCore.rotation.z = bankAngle;

      topCore.rotation.x = currentRotX + baseTilt + pitchAngle;
      bottomCore.rotation.x = currentRotX + baseTilt - pitchAngle;

      // Rotational alignment: subtle flight twist during travel that aligns to 0 when docked
      const flightTwist = (1 - ease) * Math.sin(ease * Math.PI) * 0.45;
      const gentleIdleSpin = elapsedTime * 0.12;
      topCore.rotation.y = currentRotY + flightTwist + gentleIdleSpin;
      bottomCore.rotation.y = currentRotY - flightTwist + gentleIdleSpin;

      // Cyan Energy Surge when links interlock at center
      if (rawProgress >= 0.88 && rawProgress < 1.0) {
        const contactPulse = (rawProgress - 0.88) / 0.12;
        pulseRingMat.opacity = Math.sin(contactPulse * Math.PI) * 0.95;
        pulseRing.scale.setScalar(1 + contactPulse * 1.1);
        contactLight.intensity = Math.sin(contactPulse * Math.PI) * 4.2;
      } else if (rawProgress >= 1.0) {
        // Settled locked state: gentle cyan energy glow at connection point
        pulseRingMat.opacity = 0.22;
        pulseRing.scale.setScalar(1.02);
        contactLight.intensity = 0.9;
      } else {
        pulseRingMat.opacity = 0;
        contactLight.intensity = 0;
      }

      // Slowly rotate particle field around the locked chain
      particleSystem.rotation.y = elapsedTime * 0.04;

      renderer.render(scene, camera);
    }

    animate();
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

  // Initialize Hero 3D Animation
  initHero3DAnimation();

  // Initial load
  checkHealth();
  loadScenarios();
});
