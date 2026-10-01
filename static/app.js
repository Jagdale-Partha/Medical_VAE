/**
 * ceVAE+ Medical AI Clinical Console - Frontend Controller
 * Provides real-time universal image ingestion, 4-stage sequential visualization,
 * subtraction colormap switching, and clinical "What This Means" reasoning.
 */

document.addEventListener('DOMContentLoaded', () => {
  // Global State
  let currentCase = null;
  let activePresetId = 'real_gbm_highres';
  let lastUploadedFile = null;
  let subtractionMode = 'inferno'; // 'inferno' | 'gray'
  let stage1Mode = 'processed'; // 'processed' | 'raw'
  let compareSourceMode = 'processed'; // 'processed' | 'raw'

  // DOM Elements - Telemetry & Ingestion
  const elDevice = document.getElementById('deviceVal');
  const elPresetsRow = document.getElementById('presetsRow');
  const elDropzone = document.getElementById('dropzone');
  const elFileInput = document.getElementById('fileInput');
  const elProcBanner = document.getElementById('fileProcessingBanner');
  const elProcFileName = document.getElementById('procFileName');
  const elProcFileDetails = document.getElementById('procFileDetails');
  const elProcStatusPill = document.getElementById('procStatusPill');

  // Input Harmonization Visualizer Card Elements
  const elImgRawInput = document.getElementById('imgRawInput');
  const elImgProcessedInput = document.getElementById('imgProcessedInput');
  const elHRawPill = document.getElementById('hRawPill');
  const elHProcessedPill = document.getElementById('hProcessedPill');
  const elRawDimTag = document.getElementById('rawDimTag');
  const elRawMetaDim = document.getElementById('rawMetaDim');
  const elRawMetaMode = document.getElementById('rawMetaMode');
  const elRawMetaAspect = document.getElementById('rawMetaAspect');
  const elProcMetaRange = document.getElementById('procMetaRange');
  const elProcMetaPad = document.getElementById('procMetaPad');

  // Controls
  const elThresholdSlider = document.getElementById('thresholdSlider');
  const elThreshVal = document.getElementById('threshVal');
  const elSigmaSelect = document.getElementById('sigmaSelect');
  const elBtnRun = document.getElementById('btnRunDiagnosis');
  const elBtnRandomNormal = document.getElementById('btnRandomNormal');
  const elBtnRandomPatho = document.getElementById('btnRandomPatho');

  // Verdict & "What This Means" Elements
  const elVerdictBanner = document.getElementById('verdictBanner');
  const elVerdictTitle = document.getElementById('verdictTitle');
  const elVerdictDot = document.getElementById('verdictDot');
  const elConfidenceVal = document.getElementById('confidenceVal');
  const elBurdenVal = document.getElementById('burdenVal');
  const elPeakAnomalyVal = document.getElementById('peakAnomalyVal');

  const elMeaningDiseaseText = document.getElementById('meaningDiseaseText');
  const elMeaningDiseaseSub = document.getElementById('meaningDiseaseSub');
  const elMeaningReconText = document.getElementById('meaningReconText');
  const elMeaningSubtractionText = document.getElementById('meaningSubtractionText');
  const elMeaningHighlightText = document.getElementById('meaningHighlightText');

  // Core 4-Stage Visual Cards
  const elImgInput = document.getElementById('imgInput');
  const elImgRecon = document.getElementById('imgRecon');
  const elImgSubtraction = document.getElementById('imgSubtraction');
  const elImgHighlighted = document.getElementById('imgHighlighted');
  const elBtnToggleInferno = document.getElementById('btnToggleInferno');
  const elBtnToggleGray = document.getElementById('btnToggleGray');
  const elSubtractionChip = document.getElementById('subtractionChip');

  // Stage 1 Micro Toggles
  const elBtnToggleStage1Proc = document.getElementById('btnToggleStage1Proc');
  const elBtnToggleStage1Raw = document.getElementById('btnToggleStage1Raw');
  const elStage1Title = document.getElementById('stage1Title');
  const elStage1Chip = document.getElementById('stage1Chip');
  const elStage1Desc = document.getElementById('stage1Desc');

  // Deep-Dive Views
  const elViewTabs = document.querySelectorAll('.view-tab');
  const elPanelCompareView = document.getElementById('panelCompareView');
  const elPanelInspectView = document.getElementById('panelInspectView');
  const elPanelPipeline5View = document.getElementById('panelPipeline5View');
  const elPanelReasoningView = document.getElementById('panelReasoningView');

  // Comparison Split Wipe
  const elImgCompareInput = document.getElementById('imgCompareInput');
  const elImgCompareRecon = document.getElementById('imgCompareRecon');
  const elCompareContainer = document.getElementById('compareContainer');
  const elCompareAfterLayer = document.getElementById('compareAfterLayer');
  const elCompareHandle = document.getElementById('compareHandle');
  const elBtnCompareProc = document.getElementById('btnCompareProc');
  const elBtnCompareRaw = document.getElementById('btnCompareRaw');
  const elCompareBeforeTag = document.getElementById('compareBeforeTag');

  // Inspector
  const elImgInspectBase = document.getElementById('imgInspectBase');
  const elInspectCentroid = document.getElementById('inspectCentroid');
  const elInspectBbox = document.getElementById('inspectBbox');
  const elInspectPixels = document.getElementById('inspectPixels');
  const elInspectBurden = document.getElementById('inspectBurden');

  // Pipeline 5 Stage Images
  const elImg5Input = document.getElementById('img5Input');
  const elImg5Recon = document.getElementById('img5Recon');
  const elImg5Residual = document.getElementById('img5Residual');
  const elImg5KLSaliency = document.getElementById('img5KLSaliency');
  const elImg5Overlay = document.getElementById('img5Overlay');

  const elReasoningList = document.getElementById('reasoningList');

  // ==========================================
  // 1. Initial Telemetry & Status
  // ==========================================
  async function fetchStatus() {
    try {
      const res = await fetch('/api/status');
      const data = await res.json();
      elDevice.textContent = `${data.device.toUpperCase()} (${(data.total_parameters / 1e6).toFixed(2)}M PARAMS)`;

      const elBottleneck = document.getElementById('bottleneckVal');
      if (elBottleneck && data.latent_dimension) {
        elBottleneck.textContent = `SPATIAL (${data.latent_dimension}×16×16)`;
      }

      const elCheckpoint = document.getElementById('checkpointVal');
      if (elCheckpoint) {
        if (data.checkpoint_loaded && data.checkpoint_name) {
          const valLoss = data.checkpoint_val_loss !== null ? ` (Val: ${data.checkpoint_val_loss})` : '';
          elCheckpoint.textContent = `${data.checkpoint_name}${valLoss}`;
          elCheckpoint.title = `Trained Checkpoint: ${data.checkpoint_name}\nEpoch: ${data.checkpoint_epoch}\nBest Val Loss: ${data.checkpoint_val_loss}\nLoss Formula: ${data.loss_formulation || 'N/A'}`;
        } else {
          elCheckpoint.textContent = 'UNTRAINED INITIALIZED';
        }
      }
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
      elPresetsRow.innerHTML = '<div class="preset-error">Failed to load clinical presets.</div>';
    }
  }

  function renderPresets(presets) {
    elPresetsRow.innerHTML = '';
    presets.forEach(p => {
      const card = document.createElement('div');
      card.className = `preset-card ${p.id === activePresetId ? 'active' : ''}`;
      card.dataset.id = p.id;

      const tagClass = p.pathological ? 'tag-patho' : 'tag-normal';
      const tagText = p.pathological ? 'PATHOLOGY' : 'HEALTHY';

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
    elVerdictTitle.textContent = 'ANALYZING TISSUE & RECONSTRUCTING...';
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
    const isDiseased = data.is_pathological;
    const meaning = data.diagnostic_meaning || {};
    const vis = data.visualizations;

    // 1. File Ingestion Telemetry Banner & Harmonization Visualizer
    if (data.file_telemetry) {
      const ft = data.file_telemetry;
      elProcFileName.textContent = ft.filename;
      const padInfo = ft.letterbox_padding && ft.letterbox_padding !== "None" ? ` • Letterbox Padding: ${ft.letterbox_padding}` : "";
      const aspectInfo = ft.aspect_ratio ? ` • Aspect: ${ft.aspect_ratio}` : "";
      elProcFileDetails.textContent = `Original: ${ft.original_dimensions} (${ft.original_mode})${aspectInfo}${padInfo} ➔ Converted: ${ft.processed_format}`;
      elProcStatusPill.textContent = ft.status || "MODEL COMPLIANT (1×128×128)";
      elProcStatusPill.className = "proc-status-pill";

      // Populate Harmonization Preview Card
      if (elHRawPill) elHRawPill.textContent = `Given: ${ft.original_dimensions}`;
      if (elRawDimTag) elRawDimTag.textContent = ft.original_dimensions;
      if (elRawMetaDim) elRawMetaDim.textContent = ft.original_dimensions;
      if (elRawMetaMode) elRawMetaMode.textContent = ft.original_mode;
      if (elRawMetaAspect) elRawMetaAspect.textContent = ft.aspect_ratio;
      if (elProcMetaRange) elProcMetaRange.textContent = ft.intensity_range_after || "[0.0, 1.0]";
      if (elProcMetaPad) elProcMetaPad.textContent = ft.letterbox_padding || "None (Centered)";
    }

    const rawSrc = vis.raw_input || vis.input_slice;
    const procSrc = vis.processed_input || vis.input_slice;
    if (elImgRawInput) elImgRawInput.src = rawSrc;
    if (elImgProcessedInput) elImgProcessedInput.src = procSrc;

    // 2. Main Verdict Banner
    elVerdictTitle.textContent = meaning.headline || data.verdict;
    elConfidenceVal.textContent = `${data.confidence}%`;
    elBurdenVal.textContent = `${data.lesion_area_pct}%`;
    elPeakAnomalyVal.textContent = data.max_anomaly_intensity.toFixed(3);

    if (isDiseased) {
      elVerdictBanner.className = 'verdict-banner verdict-danger';
      elVerdictDot.className = 'status-indicator-circle dot-danger';
    } else {
      elVerdictBanner.className = 'verdict-banner verdict-success';
      elVerdictDot.className = 'status-indicator-circle dot-success';
    }

    // 3. "What This Means" Explainer Cards
    elMeaningDiseaseText.textContent = meaning.summary || "Evaluation completed.";
    elMeaningDiseaseSub.innerHTML = `<strong>Status:</strong> ${meaning.badge || (isDiseased ? "DISEASED" : "HEALTHY")} &bull; <strong>Cutoff:</strong> &tau; = ${data.threshold} &bull; <strong>Action:</strong> ${meaning.clinical_action || "See report."}`;
    
    elMeaningReconText.textContent = meaning.reconstruction_meaning || "Reconstructing normative healthy baseline.";
    elMeaningSubtractionText.textContent = meaning.subtraction_meaning || "Subtracting reconstruction from input reveals residual discrepancy.";
    elMeaningHighlightText.textContent = meaning.highlight_meaning || "Dual-scoring with autograd KL gradient isolates genuine focal pathology.";

    // 4. Update Core 4-Stage Visual Cards
    updateStage1Image(vis, data);
    elImgRecon.src = vis.reconstruction;
    updateSubtractionImage(vis);
    elImgHighlighted.src = vis.highlighted_differences || vis.segmentation_overlay;

    // 5. Update Comparison Split Wipe Slider
    updateCompareSource(vis);
    elImgCompareRecon.src = vis.reconstruction;

    // 6. Update Inspector
    elImgInspectBase.src = vis.highlighted_differences || vis.segmentation_overlay;
    const elCrosshair = document.getElementById('crosshair');
    if (data.centroid) {
      elInspectCentroid.textContent = `(${data.centroid.x}, ${data.centroid.y})`;
      if (elCrosshair) {
        elCrosshair.style.display = 'block';
        elCrosshair.style.left = `${(data.centroid.x / 128) * 100}%`;
        elCrosshair.style.top = `${(data.centroid.y / 128) * 100}%`;
      }
    } else {
      elInspectCentroid.textContent = 'None detected (Normative)';
      if (elCrosshair) {
        elCrosshair.style.display = 'none';
      }
    }

    if (data.bbox) {
      elInspectBbox.textContent = `${data.bbox.width} × ${data.bbox.height} px`;
    } else {
      elInspectBbox.textContent = 'N/A';
    }
    elInspectPixels.textContent = `${data.lesion_pixels} px`;
    elInspectBurden.textContent = `${data.lesion_area_pct}% of tissue`;

    // 7. Update Full 5-Stage Scientific Pipeline
    elImg5Input.src = procSrc;
    elImg5Recon.src = vis.reconstruction;
    elImg5Residual.src = vis.residual;
    elImg5KLSaliency.src = vis.kl_saliency;
    elImg5Overlay.src = vis.segmentation_overlay;

    // 8. Update AI Radiologist Narrative
    renderReasoning(data.reasoning_steps);
  }

  function updateStage1Image(vis, data) {
    if (!vis) return;
    const currentData = data || currentCase;
    const ft = currentData?.file_telemetry;
    if (stage1Mode === 'processed') {
      elImgInput.src = vis.processed_input || vis.input_slice;
      if (elStage1Title) elStage1Title.textContent = "Processed Model Input (x)";
      if (elStage1Chip) {
        elStage1Chip.textContent = "1×128×128 Processed";
        elStage1Chip.className = "vcard-chip chip-cyan";
      }
      if (elStage1Desc) elStage1Desc.textContent = "Standardized 128×128 model input tensor normalized in [0.0, 1.0]";
      if (elBtnToggleStage1Proc) elBtnToggleStage1Proc.classList.add('active');
      if (elBtnToggleStage1Raw) elBtnToggleStage1Raw.classList.remove('active');
    } else {
      elImgInput.src = vis.raw_input || vis.input_slice;
      if (elStage1Title) elStage1Title.textContent = "Original Given Scan";
      if (elStage1Chip) {
        elStage1Chip.textContent = ft ? `Raw (${ft.original_dimensions})` : "Original Scan";
        elStage1Chip.className = "vcard-chip chip-neutral";
      }
      if (elStage1Desc) elStage1Desc.textContent = ft ? `Original patient scan (${ft.original_dimensions}, ${ft.original_mode})` : "Raw unscaled patient scan";
      if (elBtnToggleStage1Raw) elBtnToggleStage1Raw.classList.add('active');
      if (elBtnToggleStage1Proc) elBtnToggleStage1Proc.classList.remove('active');
    }
  }

  function updateCompareSource(vis) {
    if (!vis) return;
    if (compareSourceMode === 'processed') {
      elImgCompareInput.src = vis.processed_input || vis.input_slice;
      if (elCompareBeforeTag) elCompareBeforeTag.textContent = "PROCESSED INPUT (x)";
      if (elBtnCompareProc) elBtnCompareProc.classList.add('active');
      if (elBtnCompareRaw) elBtnCompareRaw.classList.remove('active');
    } else {
      elImgCompareInput.src = vis.raw_input || vis.input_slice;
      if (elCompareBeforeTag) elCompareBeforeTag.textContent = "ORIGINAL GIVEN SCAN";
      if (elBtnCompareRaw) elBtnCompareRaw.classList.add('active');
      if (elBtnCompareProc) elBtnCompareProc.classList.remove('active');
    }
  }

  function updateSubtractionImage(vis) {
    if (!vis) return;
    if (subtractionMode === 'inferno') {
      elImgSubtraction.src = vis.subtracted_reconstruction || vis.residual;
      elSubtractionChip.textContent = "Inferno Heatmap (|x - x̂|)";
      elSubtractionChip.className = "vcard-chip chip-gold";
    } else {
      elImgSubtraction.src = vis.subtracted_reconstruction_gray || vis.residual;
      elSubtractionChip.textContent = "Raw Grayscale Difference";
      elSubtractionChip.className = "vcard-chip chip-cyan";
    }
  }

  function renderReasoning(steps) {
    if (!steps) return;
    elReasoningList.innerHTML = '';
    steps.forEach((step, idx) => {
      const card = document.createElement('div');
      card.className = `reason-step-card ${idx === 0 || idx === 1 || idx === 2 ? 'open' : ''}`;

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
  // 5. Subtraction Colormap Toggle
  // ==========================================
  elBtnToggleInferno.addEventListener('click', () => {
    subtractionMode = 'inferno';
    elBtnToggleInferno.classList.add('active');
    elBtnToggleGray.classList.remove('active');
    if (currentCase) updateSubtractionImage(currentCase.visualizations);
  });

  elBtnToggleGray.addEventListener('click', () => {
    subtractionMode = 'gray';
    elBtnToggleGray.classList.add('active');
    elBtnToggleInferno.classList.remove('active');
    if (currentCase) updateSubtractionImage(currentCase.visualizations);
  });

  // ==========================================
  // 5b. Stage 1 View Mode Toggle (128x128 Processed vs Original)
  // ==========================================
  if (elBtnToggleStage1Proc && elBtnToggleStage1Raw) {
    elBtnToggleStage1Proc.addEventListener('click', () => {
      stage1Mode = 'processed';
      if (currentCase) updateStage1Image(currentCase.visualizations);
    });
    elBtnToggleStage1Raw.addEventListener('click', () => {
      stage1Mode = 'raw';
      if (currentCase) updateStage1Image(currentCase.visualizations);
    });
  }

  // ==========================================
  // 5c. Compare Split Wipe Source Toggle
  // ==========================================
  if (elBtnCompareProc && elBtnCompareRaw) {
    elBtnCompareProc.addEventListener('click', () => {
      compareSourceMode = 'processed';
      if (currentCase) updateCompareSource(currentCase.visualizations);
    });
    elBtnCompareRaw.addEventListener('click', () => {
      compareSourceMode = 'raw';
      if (currentCase) updateCompareSource(currentCase.visualizations);
    });
  }

  // ==========================================
  // 6. Interactive Threshold & Controls
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

  // Procedural Generator Buttons
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
  // 7. Universal Drag-and-Drop & File Upload
  // ==========================================
  elDropzone.addEventListener('click', () => elFileInput.click());

  elDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    elDropzone.classList.add('drag-active');
  });

  elDropzone.addEventListener('dragleave', () => {
    elDropzone.classList.remove('drag-active');
  });

  elDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    elDropzone.classList.remove('drag-active');
    if (e.dataTransfer.files.length > 0) {
      handleFileUpload(e.dataTransfer.files[0]);
    }
  });

  elFileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
      handleFileUpload(e.target.files[0]);
    }
  });

  function handleFileUpload(file) {
    lastUploadedFile = file;
    document.querySelectorAll('.preset-card').forEach(c => c.classList.remove('active'));
    activePresetId = null;

    elProcFileName.textContent = `Processing Upload: ${file.name}`;
    elProcFileDetails.textContent = `File size: ${(file.size / 1024).toFixed(1)} KB &bull; Ingesting and converting to 1×128×128 float tensor...`;
    elProcStatusPill.textContent = "HARMONIZING...";

    runDiagnosis({ file: file });
  }

  // ==========================================
  // 8. Tab Navigation
  // ==========================================
  elViewTabs.forEach(tab => {
    tab.addEventListener('click', () => {
      elViewTabs.forEach(t => t.classList.remove('active'));
      tab.classList.add('active');

      const mode = tab.dataset.mode;
      elPanelCompareView.classList.toggle('hidden', mode !== 'compare');
      elPanelInspectView.classList.toggle('hidden', mode !== 'inspect');
      elPanelPipeline5View.classList.toggle('hidden', mode !== 'pipeline5');
      elPanelReasoningView.classList.toggle('hidden', mode !== 'reasoning');
    });
  });

  // ==========================================
  // 9. Split Wipe Slider Interaction
  // ==========================================
  let isDragging = false;

  function updateSplit(clientX) {
    const rect = elCompareContainer.getBoundingClientRect();
    let x = clientX - rect.left;
    x = Math.max(0, Math.min(x, rect.width));
    const pct = (x / rect.width) * 100;

    elCompareAfterLayer.style.clipPath = `polygon(${pct}% 0, 100% 0, 100% 100%, ${pct}% 100%)`;
    elCompareHandle.style.left = `${pct}%`;
  }

  elCompareContainer.addEventListener('mousedown', (e) => {
    isDragging = true;
    updateSplit(e.clientX);
  });

  window.addEventListener('mousemove', (e) => {
    if (!isDragging) return;
    updateSplit(e.clientX);
  });

  window.addEventListener('mouseup', () => {
    isDragging = false;
  });

  // Touch Support
  elCompareContainer.addEventListener('touchstart', (e) => {
    isDragging = true;
    updateSplit(e.touches[0].clientX);
  }, { passive: true });

  window.addEventListener('touchmove', (e) => {
    if (!isDragging) return;
    updateSplit(e.touches[0].clientX);
  }, { passive: true });

  window.addEventListener('touchend', () => {
    isDragging = false;
  });

  // Set initial split wipe to 50%
  if (elCompareAfterLayer && elCompareHandle) {
    elCompareAfterLayer.style.clipPath = 'polygon(50% 0, 100% 0, 100% 100%, 50% 100%)';
    elCompareHandle.style.left = '50%';
  }

  // ==========================================
  // 10. Bootstrap Application
  // ==========================================
  fetchStatus();
  fetchPresets();
  runDiagnosis({ preset_id: 'real_gbm_highres' });
});
