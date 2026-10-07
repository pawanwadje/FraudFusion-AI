// Revision 2 Client-Side JavaScript Logic
let chartInstances = {};
let sampleData = null;
let currentPrediction = null;
let loadedModelDetails = {};

document.addEventListener('DOMContentLoaded', () => {
  // 1. Initialize SPA Tab Navigation
  initTabNavigation();

  // 2. Fetch Sample Claims & Pre-fill Form
  fetchSamples();

  // 3. Bind Form Submit & Buttons
  const form = document.getElementById('claim-form');
  if (form) form.addEventListener('submit', handleFormSubmit);

  document.getElementById('btn-load-fraud').addEventListener('click', () => loadSample('fraud_sample'));
  document.getElementById('btn-load-legit').addEventListener('click', () => loadSample('legit_sample'));
  document.getElementById('btn-load-borderline').addEventListener('click', () => loadSample('borderline_sample'));
  document.getElementById('btn-clear-form').addEventListener('click', clearForm);
});

// SPA Tab Switching Logic
function initTabNavigation() {
  const navLinks = document.querySelectorAll('.nav-link');
  navLinks.forEach(link => {
    link.addEventListener('click', (e) => {
      e.preventDefault();
      const tabId = link.getAttribute('data-tab');
      switchTab(tabId);
    });
  });
}

function switchTab(tabId) {
  // Update Nav Links Active State
  document.querySelectorAll('.nav-link').forEach(link => {
    if (link.getAttribute('data-tab') === tabId) {
      link.classList.add('active');
    } else {
      link.classList.remove('active');
    }
  });

  // Update Page Views
  document.querySelectorAll('.page-view').forEach(page => {
    page.classList.remove('active');
  });

  const targetPage = document.getElementById(`page-${tabId}`);
  if (targetPage) {
    targetPage.classList.add('active');
  }

  // Load Model Deep-Dive Details if navigating to RF, XGB, or ISO
  if (['rf', 'xgb', 'iso'].includes(tabId)) {
    loadModelDeepDive(tabId);
  }
}

// Fetch Sample Claims from FastAPI
async function fetchSamples() {
  try {
    const res = await fetch('/api/samples');
    if (res.ok) {
      sampleData = await res.json();
      if (sampleData && sampleData.fraud_sample) {
        populateForm(sampleData.fraud_sample);
      }
    }
  } catch (err) {
    console.warn('Could not fetch preset samples:', err);
  }
}

function loadSample(sampleKey) {
  if (sampleData && sampleData[sampleKey]) {
    populateForm(sampleData[sampleKey]);
    showToast(`Loaded ${sampleKey.replace('_sample', '')} claim preset!`);
  }
}

function populateForm(data) {
  for (const [key, value] of Object.entries(data)) {
    const el = document.getElementById(key);
    if (el) {
      el.value = value;
    }
  }
}

function clearForm() {
  const form = document.getElementById('claim-form');
  form.reset();
  showToast('Form cleared.');
}

function showToast(msg) {
  const statusEl = document.getElementById('backend-status');
  if (!statusEl) return;
  const originalText = statusEl.innerText;
  statusEl.innerText = msg;
  statusEl.style.color = '#0EA5E9';
  setTimeout(() => {
    statusEl.innerText = originalText;
    statusEl.style.color = '';
  }, 2500);
}

// Form Submit Handler
async function handleFormSubmit(e) {
  e.preventDefault();
  showLoading(true);

  const formData = new FormData(e.target);
  const claimPayload = {};

  formData.forEach((val, key) => {
    if (['WeekOfMonth', 'WeekOfMonthClaimed', 'Age', 'RepNumber', 'Deductible', 'DriverRating', 'Year'].includes(key)) {
      claimPayload[key] = parseInt(val, 10) || 0;
    } else {
      claimPayload[key] = val;
    }
  });

  try {
    const response = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(claimPayload)
    });

    if (!response.ok) {
      const errData = await response.json();
      throw new Error(errData.detail || 'Prediction failed');
    }

    const data = await response.json();
    currentPrediction = data;

    // Render Predict Tab Results
    renderPredictResults(data);

    // Update Live Prediction State on Deep-Dive Banners
    updateLiveBanners(data);

  } catch (err) {
    alert(`Error running claim evaluation: ${err.message}`);
    console.error(err);
  } finally {
    showLoading(false);
  }
}

function showLoading(isLoading) {
  const overlay = document.getElementById('loading-overlay');
  overlay.style.display = isLoading ? 'flex' : 'none';
}

function toggleMetrics(id) {
  const content = document.getElementById(id);
  if (content) content.classList.toggle('open');
}

// Render Results on Predict Dashboard
function renderPredictResults(data) {
  const resultsArea = document.getElementById('results-area');
  resultsArea.style.display = 'block';

  // 1. Headline Ensemble Verdict Card
  const ensemble = data.ensemble;
  document.getElementById('ensemble-confidence-num').innerText = `${ensemble.confidence}%`;
  
  const badgeEl = document.getElementById('ensemble-verdict-badge');
  badgeEl.innerText = ensemble.decision_fusion;
  badgeEl.className = 'verdict-badge';
  if (ensemble.fusion_badge === 'fraud') badgeEl.classList.add('verdict-fraud');
  else if (ensemble.fusion_badge === 'legit') badgeEl.classList.add('verdict-legit');
  else badgeEl.classList.add('verdict-review');

  document.getElementById('ensemble-explanation').innerText = ensemble.explanation;

  // 2. Base Model Cards
  const rf = data.random_forest;
  document.getElementById('rf-confidence').innerText = `${rf.confidence}%`;
  setTagStyle('rf-verdict-tag', rf.label);

  const xgb = data.xgboost;
  document.getElementById('xgb-confidence').innerText = `${xgb.confidence}%`;
  setTagStyle('xgb-verdict-tag', xgb.label);

  const iso = data.isolation_forest;
  document.getElementById('iso-confidence').innerText = `${iso.confidence}%`;
  setTagStyle('iso-verdict-tag', iso.label);

  // 3. Agreement Note
  document.getElementById('agreement-summary-tag').innerText = data.comparison.agreement_note;

  // 4. Render Predict Charts
  renderConfidenceChart(rf.confidence, xgb.confidence, iso.confidence, ensemble.confidence);
  renderImportanceChart('chart-importance', data.top_features, '#0EA5E9');

  // Scroll to results
  resultsArea.scrollIntoView({ behavior: 'smooth', block: 'start' });
}

function setTagStyle(tagId, label) {
  const tag = document.getElementById(tagId);
  if (!tag) return;
  tag.innerText = label;
  tag.className = 'model-verdict-tag';
  if (label === 'Fraud') tag.classList.add('verdict-fraud');
  else tag.classList.add('verdict-legit');
}

// Update Live Claim Prediction Banners on Model Deep-Dive Pages
function updateLiveBanners(data) {
  // RF Live Banner
  const rf = data.random_forest;
  document.getElementById('rf-live-verdict').innerText = rf.label;
  document.getElementById('rf-live-conf').innerText = `${rf.confidence}%`;
  document.getElementById('rf-live-verdict').style.color = rf.label === 'Fraud' ? 'var(--status-fraud)' : 'var(--color-rf)';

  // XGB Live Banner
  const xgb = data.xgboost;
  document.getElementById('xgb-live-verdict').innerText = xgb.label;
  document.getElementById('xgb-live-conf').innerText = `${xgb.confidence}%`;
  document.getElementById('xgb-live-verdict').style.color = xgb.label === 'Fraud' ? 'var(--status-fraud)' : 'var(--color-xgb)';

  // ISO Live Banner
  const iso = data.isolation_forest;
  document.getElementById('iso-live-verdict').innerText = iso.label;
  document.getElementById('iso-live-conf').innerText = `${iso.confidence}%`;
  document.getElementById('iso-live-verdict').style.color = iso.label === 'Fraud' ? 'var(--status-fraud)' : 'var(--color-iso)';
}

// Fetch & Render Model Deep-Dive Page Data
async function loadModelDeepDive(modelKey) {
  const apiMap = { 'rf': 'random_forest', 'xgb': 'xgboost', 'iso': 'isolation_forest' };
  const apiName = apiMap[modelKey];

  if (loadedModelDetails[apiName]) {
    return; // Already loaded
  }

  try {
    const res = await fetch(`/api/model-details/${apiName}`);
    if (!res.ok) return;
    const details = await res.json();
    loadedModelDetails[apiName] = details;

    if (modelKey === 'rf') {
      renderRocChart('chart-rf-roc', details.roc_curve.fpr, details.roc_curve.tpr, 'Random Forest ROC', '#10B981');
      renderPrChart('chart-rf-pr', details.pr_curve.precision, details.pr_curve.recall, 'Random Forest PR', '#10B981');
    } else if (modelKey === 'xgb') {
      renderRocChart('chart-xgb-roc', details.roc_curve.fpr, details.roc_curve.tpr, 'XGBoost ROC', '#0EA5E9');
      renderPrChart('chart-xgb-pr', details.pr_curve.precision, details.pr_curve.recall, 'XGBoost PR', '#0EA5E9');
    } else if (modelKey === 'iso') {
      const dist = details.score_distribution;
      renderIsoDistributionChart('chart-iso-distribution', dist.bin_centers, dist.legit_counts, dist.fraud_counts, dist.threshold);
    }
  } catch (err) {
    console.error(`Error loading details for ${modelKey}:`, err);
  }
}

// --- Chart Rendering Functions ---

// 1. Model Confidence Bar Chart
function renderConfidenceChart(rfConf, xgbConf, isoConf, metaConf) {
  const ctx = document.getElementById('chart-confidence').getContext('2d');
  if (chartInstances['confidence']) chartInstances['confidence'].destroy();

  chartInstances['confidence'] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: ['Random Forest', 'XGBoost', 'Isolation Forest', 'Meta Ensemble'],
      datasets: [{
        label: 'Confidence / Anomaly Index (%)',
        data: [rfConf, xgbConf, isoConf, metaConf],
        backgroundColor: ['rgba(16, 185, 129, 0.75)', 'rgba(14, 165, 233, 0.75)', 'rgba(245, 158, 11, 0.75)', 'rgba(244, 63, 94, 0.85)'],
        borderColor: ['#10B981', '#0EA5E9', '#F59E0B', '#F43F5E'],
        borderWidth: 1.5,
        borderRadius: 6
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        y: { beginAtZero: true, max: 100, grid: { color: '#334155' }, ticks: { color: '#94A3B8', callback: v => v + '%' } },
        x: { grid: { display: false }, ticks: { color: '#F8FAFC', font: { weight: '600' } } }
      },
      plugins: { legend: { display: false } }
    }
  });
}

// 2. Feature Importance Horizontal Bar Chart
function renderImportanceChart(canvasId, features, color) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  if (chartInstances[canvasId]) chartInstances[canvasId].destroy();

  const labels = features.map(f => f.feature);
  const values = features.map(f => f.importance);

  chartInstances[canvasId] = new Chart(ctx, {
    type: 'bar',
    indexAxis: 'y',
    data: {
      labels: labels,
      datasets: [{
        label: 'Relative Importance (%)',
        data: values,
        backgroundColor: color + 'BB',
        borderColor: color,
        borderWidth: 1,
        borderRadius: 4
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { beginAtZero: true, grid: { color: '#334155' }, ticks: { color: '#94A3B8', callback: v => v + '%' } },
        y: { grid: { display: false }, ticks: { color: '#F8FAFC', font: { size: 11 } } }
      },
      plugins: { legend: { display: false } }
    }
  });
}

// 3. ROC Curve Chart
function renderRocChart(canvasId, fpr, tpr, label, color) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  if (chartInstances[canvasId]) chartInstances[canvasId].destroy();

  const dataPoints = fpr.map((x, i) => ({ x: x, y: tpr[i] }));

  chartInstances[canvasId] = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        {
          label: label,
          data: dataPoints,
          borderColor: color,
          backgroundColor: color + '22',
          borderWidth: 2.5,
          fill: true,
          tension: 0.2
        },
        {
          label: 'Random Chance Baseline',
          data: [{ x: 0, y: 0 }, { x: 1, y: 1 }],
          borderColor: '#64748B',
          borderDash: [5, 5],
          borderWidth: 1.5,
          pointRadius: 0
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'False Positive Rate', color: '#94A3B8' }, grid: { color: '#334155' }, ticks: { color: '#94A3B8' } },
        y: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'True Positive Rate', color: '#94A3B8' }, grid: { color: '#334155' }, ticks: { color: '#94A3B8' } }
      }
    }
  });
}

// 4. Precision-Recall Curve Chart
function renderPrChart(canvasId, precision, recall, label, color) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  if (chartInstances[canvasId]) chartInstances[canvasId].destroy();

  const dataPoints = recall.map((x, i) => ({ x: x, y: precision[i] }));

  chartInstances[canvasId] = new Chart(ctx, {
    type: 'line',
    data: {
      datasets: [
        {
          label: label,
          data: dataPoints,
          borderColor: color,
          backgroundColor: color + '22',
          borderWidth: 2.5,
          fill: true,
          tension: 0.2
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'Recall', color: '#94A3B8' }, grid: { color: '#334155' }, ticks: { color: '#94A3B8' } },
        y: { type: 'linear', min: 0, max: 1, title: { display: true, text: 'Precision', color: '#94A3B8' }, grid: { color: '#334155' }, ticks: { color: '#94A3B8' } }
      }
    }
  });
}

// 5. Isolation Forest Anomaly Score Distribution Chart
function renderIsoDistributionChart(canvasId, binCenters, legitCounts, fraudCounts, threshold) {
  const ctx = document.getElementById(canvasId).getContext('2d');
  if (chartInstances[canvasId]) chartInstances[canvasId].destroy();

  chartInstances[canvasId] = new Chart(ctx, {
    type: 'bar',
    data: {
      labels: binCenters.map(v => v.toFixed(3)),
      datasets: [
        { label: 'Legitimate Claims', data: legitCounts, backgroundColor: 'rgba(16, 185, 129, 0.7)', borderColor: '#10B981', borderWidth: 1 },
        { label: 'Fraudulent Claims', data: fraudCounts, backgroundColor: 'rgba(239, 68, 68, 0.85)', borderColor: '#EF4444', borderWidth: 1 }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        x: { title: { display: true, text: 'Raw Anomaly Score (-score_samples)', color: '#94A3B8' }, grid: { color: '#334155' }, ticks: { color: '#94A3B8' } },
        y: { title: { display: true, text: 'Claim Count', color: '#94A3B8' }, grid: { color: '#334155' }, ticks: { color: '#94A3B8' } }
      }
    }
  });
}
