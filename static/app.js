/**
 * ceVAE+ Brain MRI Anomaly Detection Visualizer
 * Frontend Application Controller
 */

document.addEventListener('DOMContentLoaded', () => {
  // Global State
  let currentCase = null;
  let activeViewMode = '5panel';
  let activePresetId = 'gbm_parietal';
  let lastUploadedFile = null;

  // DOM Elements
  const elDevice = document.getElementById('deviceVal');
  const elPresetsRow = document.getElementById('presetsRow');
  const elDropzone = document.getElementById('dropzone');
  const elFileInput = document.getElementById('fileInput');

  const elThresholdSlider = document.getElementById('thresholdSlider');
  const elThreshVal = document.getElementById('threshVal');
  const elSigmaSelect = document.getElementById('sigmaSelect');
  const elBtnRun = document.getElementById('btnRunDiagnosis');
  const elBtnRandomNormal = document.getElementById('btnRandomNormal');
  const elBtnRandomPatho = document.getElementById('btnRandomPatho');

  // Verdict Elements
  const elVerdictBanner = document.getElementById('verdictBanner');
  const elVerdictTitle = document.getElementById('verdictTitle');
  const elConfidenceVal = document.getElementById('confidenceVal');
  const elBurdenVal = document.getElementById('burdenVal');

  // Metrics Elements
  const elMetric95th = document.getElementById('metric95th');
  const elMetricMax = document.getElementById('metricMax');
  const elMetricKL = document.getElementById('metricKL');
  const elMetricParenchyma = document.getElementById('metricParenchyma');

  // Image Views
  const elImgInput = document.getElementById('imgInput');
  const elImgRecon = document.getElementById('imgRecon');
  const elImgResidual = document.getElementById('imgResidual');
  const elImgKLSaliency = document.getElementById('imgKLSaliency');
  const elImgOverlay = document.getElementById('imgOverlay');

  const elImgInspectBase = document.getElementById('imgInspectBase');
  const elInspectCentroid = document.getElementById('inspectCentroid');
  const elInspectBbox = document.getElementById('inspectBbox');
  const elInspectPixels = document.getElementById('inspectPixels');
  const elInspectBurden = document.getElementById('inspectBurden');

  const elImgCompareInput = document.getElementById('imgCompareInput');
  const elImgCompareRecon = document.getElementById('imgCompareRecon');
  const elCompareContainer = document.getElementById('compareContainer');
  const elCompareAfterLayer = document.getElementById('compareAfterLayer');
  const elCompareHandle = document.getElementById('compareHandle');

  // View Containers
  const elPanel5View = document.getElementById('panel5View');
  const elPanelInspectView = document.getElementById('panelInspectView');
  const elPanelCompareView = document.getElementById('panelCompareView');
  const elReasoningList = document.getElementById('reasoningList');

  // ==========================================
  // 1. Initial Telemetry & Status
  // ==========================================
  async function fetchStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      elDevice.textContent = `${data.device.toUpperCase()} (${(data.total_parameters / 1e6).toFixed(2)}M Params)`;
    } catch (e) {
      console.warn('Status fetch error:', e);
      elDevice.textContent = 'CPU (ONLINE)';
    }
  }

  // ==========================================
  // 2. Fetch Sample Presets
  // ==========================================
  async function fetchPresets() {
    try {
      const res = await fetch('/api/presets');
      const presets = await res.json();
      renderPresets(presets);
    } catch (e) {
      console.error('Failed to load presets:', e);
      elPresetsRow.innerHTML = '<div class="preset-error">Failed to load presets.</div>';
    }
  }

  function renderPresets(presets) {
    elPresetsRow.innerHTML = '';
    presets.forEach(p => {
      const card = document.createElement('div');
      card.className = `preset-card ${p.id === activePresetId ? 'active' : ''}`;
      card.dataset.id = p.id;

      const tagClass = p.pathological ? 'tag-patho' : 'tag-normal';
      const tagText = p.pathological ? 'PATHOLOGICAL TUMOR' : 'HEALTHY CONTROL';

      card.innerHTML = `
        <img src="${p.thumbnail}" alt="${p.title}" class="preset-thumb">
        <div class="preset-info">
          <span class="preset-title">${p.title}</span>
          <span class="preset-tag ${tagClass}">${tagText}</span>
        </div>
      `;

      card.addEventListener('click', () => {
        document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
        card.classList.add('active');
        activePresetId = p.id;
        lastUploadedFile = null;
        runDiagnosis({ preset_id: p.id });
      });

      elPresetsRow.appendChild(card);
    });
  }

  // ==========================================
  // 3. Core Diagnosis Execution
  // ==========================================
  async function runDiagnosis(options = {}) {
    elVerdictTitle.textContent = 'ANALYZING NEUROANATOMY...';
    elVerdictBanner.style.opacity = '0.6';

    const formData = new FormData();
    formData.append('threshold', elThresholdSlider.value);
    formData.append('gaussian_sigma', elSigmaSelect.value);

    if (options.file || lastUploadedFile) {
      const fileToUpload = options.file || lastUploadedFile;
      formData.append('file', fileToUpload);
    } else if (options.generate_random) {
      formData.append('generate_random', 'true');
      formData.append('pathological', options.pathological ? 'true' : 'false');
      if (options.seed) formData.append('seed', options.seed);
    } else {
      const presetId = options.preset_id || activePresetId || 'gbm_parietal';
      formData.append('preset_id', presetId);
    }

    try {
      const res = await fetch('/api/diagnose', {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
        const err = await res.json();
        throw new Error(err.detail || 'Server evaluation error');
      }

      const data = await res.json();
      currentCase = data;
      renderDiagnosis(data);
    } catch (e) {
      console.error('Diagnostic error:', e);
      elVerdictTitle.textContent = 'EVALUATION ERROR';
      alert(`Error during model inference: ${e.message}`);
    } finally {
      elVerdictBanner.style.opacity = '1.0';
    }
  }

  // ==========================================
  // 4. Render Diagnosis Results & Visuals
  // ==========================================
  function renderDiagnosis(data) {
    // 1. Verdict Banner
    elVerdictTitle.textContent = data.verdict;
    elConfidenceVal.textContent = `${data.confidence}%`;
    elBurdenVal.textContent = `${data.lesion_area_pct}%`;

    if (data.verdict_color === 'danger') {
      elVerdictBanner.className = 'verdict-banner';
    } else {
      elVerdictBanner.className = 'verdict-banner verdict-success';
    }

    // 2. Quantitative Biomarkers
    elMetric95th.textContent = data.anomaly_score_95th.toFixed(4);
    elMetricMax.textContent = data.max_anomaly_intensity.toFixed(4);
    elMetricKL.textContent = data.kl_divergence.toFixed(2);
    elMetricParenchyma.textContent = `${data.parenchyma_pixels.toLocaleString()} px`;

    // 3. Stage Visualizations
    const vis = data.visualizations;
    elImgInput.src = vis.input_slice;
    elImgRecon.src = vis.reconstruction;
    elImgResidual.src = vis.residual;
    elImgKLSaliency.src = vis.kl_saliency;
    elImgOverlay.src = vis.segmentation_overlay;

    // High Res Inspector
    elImgInspectBase.src = vis.segmentation_overlay;
    if (data.centroid) {
      elInspectCentroid.textContent = `(${data.centroid.x}, ${data.centroid.y})`;
    } else {
      elInspectCentroid.textContent = 'None detected';
    }

    if (data.bbox) {
      elInspectBbox.textContent = `${data.bbox.width}x${data.bbox.height} px`;
    } else {
      elInspectBbox.textContent = 'N/A';
    }
    elInspectPixels.textContent = `${data.lesion_pixels} px`;
    elInspectBurden.textContent = `${data.lesion_area_pct}%`;

    // Comparison Split Wipe
    elImgCompareInput.src = vis.input_slice;
    elImgCompareRecon.src = vis.reconstruction;

    // 4. Clinical Reasoning Accordion
    renderReasoning(data.reasoning_steps);
  }

  function renderReasoning(steps) {
    elReasoningList.innerHTML = '';
    steps.forEach((step, idx) => {
      const card = document.createElement('div');
      card.className = `reason-step-card ${idx === 4 || idx === 5 ? 'open' : ''}`;

      card.innerHTML = `
        <button class="reason-header-btn">
          <div class="step-left">
            <span class="step-badge-num">${step.step}</span>
            <div>
              <div class="step-title-text">${step.title}</div>
              <div class="step-subtitle-text">${step.subtitle}</div>
            </div>
          </div>
          <svg class="step-chevron" viewBox="0 0 24 24" width="16" height="16" fill="none" stroke="currentColor" stroke-width="2">
            <polyline points="6 9 12 15 18 9"/>
          </svg>
        </button>
        <div class="reason-body">
          <span class="step-badge-tag">${step.badge}</span>
          <p>${step.description}</p>
          <div class="step-clinical-box">
            <strong>Clinical Rationale:</strong> ${step.clinical_significance}
          </div>
        </div>
      `;

      const btn = card.querySelector('.reason-header-btn');
      btn.addEventListener('click', () => {
        card.classList.toggle('open');
      });

      elReasoningList.appendChild(card);
    });
  }

  // ==========================================
  // 5. Interactive Threshold Slider
  // ==========================================
  let sliderTimeout = null;
  elThresholdSlider.addEventListener('input', (e) => {
    const val = parseFloat(e.target.value).toFixed(2);
    elThreshVal.textContent = val;

    clearTimeout(sliderTimeout);
    sliderTimeout = setTimeout(() => {
      if (lastUploadedFile) {
        runDiagnosis({ file: lastUploadedFile });
      } else {
        runDiagnosis({ preset_id: activePresetId });
      }
    }, 150);
  });

  elSigmaSelect.addEventListener('change', () => {
    if (lastUploadedFile) {
      runDiagnosis({ file: lastUploadedFile });
    } else {
      runDiagnosis({ preset_id: activePresetId });
    }
  });

  elBtnRun.addEventListener('click', () => {
    if (lastUploadedFile) {
      runDiagnosis({ file: lastUploadedFile });
    } else {
      runDiagnosis({ preset_id: activePresetId });
    }
  });

  // Random Generator Buttons
  elBtnRandomNormal.addEventListener('click', () => {
    document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
    activePresetId = null;
    lastUploadedFile = null;
    runDiagnosis({ generate_random: true, pathological: false, seed: Math.floor(Math.random() * 1000000) });
  });

  elBtnRandomPatho.addEventListener('click', () => {
    document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
    activePresetId = null;
    lastUploadedFile = null;
    runDiagnosis({ generate_random: true, pathological: true, seed: Math.floor(Math.random() * 1000000) });
  });

  // ==========================================
  // 6. File Upload & Drag-and-Drop
  // ==========================================
  elDropzone.addEventListener('click', () => elFileInput.click());

  elFileInput.addEventListener('change', (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFileUpload(e.target.files[0]);
    }
  });

  ['dragenter', 'dragover'].forEach(name => {
    elDropzone.addEventListener(name, (e) => {
      e.preventDefault();
      elDropzone.classList.add('dragover');
    });
  });

  ['dragleave', 'drop'].forEach(name => {
    elDropzone.addEventListener(name, (e) => {
      e.preventDefault();
      elDropzone.classList.remove('dragover');
    });
  });

  elDropzone.addEventListener('drop', (e) => {
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  function handleFileUpload(file) {
    document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
    activePresetId = null;
    lastUploadedFile = file;
    runDiagnosis({ file });
  }

  // ==========================================
  // 7. View Mode Switching
  // ==========================================
  const viewTabs = document.querySelectorAll('.view-tab');
  viewTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      viewTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      activeViewMode = tab.dataset.mode;
      elPanel5View.classList.add('hidden');
      elPanelInspectView.classList.add('hidden');
      elPanelCompareView.classList.add('hidden');

      if (activeViewMode === '5panel') {
        elPanel5View.classList.remove('hidden');
      } else if (activeViewMode === 'inspect') {
        elPanelInspectView.classList.remove('hidden');
      } else if (activeViewMode === 'compare') {
        elPanelCompareView.classList.remove('hidden');
      }
    });
  });

  // Layer Toggles in Inspector
  const chkShowMask = document.getElementById('chkShowMask');
  if (chkShowMask) {
    chkShowMask.addEventListener('change', (e) => {
      if (currentCase) {
        elImgInspectBase.src = e.target.checked
          ? currentCase.visualizations.segmentation_overlay
          : currentCase.visualizations.input_slice;
      }
    });
  }

  // ==========================================
  // 8. Split Wipe Comparison Slider
  // ==========================================
  let isDragging = false;

  function updateCompareWipe(clientX) {
    const rect = elCompareContainer.getBoundingClientRect();
    let x = clientX - rect.left;
    x = Math.max(0, Math.min(x, rect.width));
    const pct = (x / rect.width) * 100;

    elCompareHandle.style.left = `${pct}%`;
    elCompareAfterLayer.style.clipPath = `polygon(${pct}% 0, 100% 0, 100% 100%, ${pct}% 100%)`;
  }

  elCompareContainer.addEventListener('mousedown', (e) => {
    isDragging = true;
    updateCompareWipe(e.clientX);
  });

  window.addEventListener('mousemove', (e) => {
    if (isDragging) {
      updateCompareWipe(e.clientX);
    }
  });

  window.addEventListener('mouseup', () => {
    isDragging = false;
  });

  // Touch Support
  elCompareContainer.addEventListener('touchstart', (e) => {
    isDragging = true;
    if (e.touches && e.touches[0]) updateCompareWipe(e.touches[0].clientX);
  });
  window.addEventListener('touchmove', (e) => {
    if (isDragging && e.touches && e.touches[0]) updateCompareWipe(e.touches[0].clientX);
  });
  window.addEventListener('touchend', () => isDragging = false);

  // Initialize
  fetchStatus();
  fetchPresets().then(() => {
    runDiagnosis({ preset_id: 'gbm_parietal' });
  });
});
