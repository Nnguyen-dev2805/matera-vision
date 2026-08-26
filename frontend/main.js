/**
 * Matera OMR - Minimalist Client Controller
 */

const API_BASE = window.location.port === '5173' || window.location.port === '3000'
  ? 'http://localhost:8000'
  : window.location.origin;

// Stage Sections
const selectionStage = document.getElementById('selection-stage');
const readyStage = document.getElementById('ready-stage');
const progressStage = document.getElementById('progress-stage');

// Step 1: Selection Dropzone
const uploadDropzone = document.getElementById('upload-dropzone');
const btnBrowseTrigger = document.getElementById('btn-browse-trigger');
const folderFileInput = document.getElementById('folder-file-input');
const btnLoadSample = document.getElementById('btn-load-sample');

// Step 2: Ready Banner
const readyFolderName = document.getElementById('ready-folder-name');
const readyFileCount = document.getElementById('ready-file-count');
const btnChangeFolder = document.getElementById('btn-change-folder');
const btnStartExtract = document.getElementById('btn-start-extract');
const checkboxDebug = document.getElementById('checkbox-debug');

// Step 3: Progress & KPIs
const liveDot = document.getElementById('live-dot');
const progressStatusText = document.getElementById('progress-status-text');
const progressPercent = document.getElementById('progress-percent');
const progressFill = document.getElementById('progress-fill');

const kpiFiles = document.getElementById('kpi-files');
const kpiPages = document.getElementById('kpi-pages');
const kpiAnswers = document.getElementById('kpi-answers');
const kpiReviews = document.getElementById('kpi-reviews');

// Data Table & Tabs
const tableBody = document.getElementById('table-body');
const countAll = document.getElementById('count-all');
const countResolved = document.getElementById('count-resolved');
const countReview = document.getElementById('count-review');
const tabBtns = document.querySelectorAll('.tab-btn');

// Header & Floating Action Bar
const btnNewBatch = document.getElementById('btn-new-batch');
const floatingBar = document.getElementById('floating-bar');
const footerExcelPath = document.getElementById('footer-excel-path');
const btnDownloadExcel = document.getElementById('btn-download-excel');
const btnResetAll = document.getElementById('btn-reset-all');

// Debug Modal
const debugModal = document.getElementById('debug-modal');
const debugModalTitle = document.getElementById('debug-modal-title');
const debugModalContent = document.getElementById('debug-modal-content');
const btnCloseModal = document.getElementById('btn-close-modal');

// State
let selectedFiles = [];
let presetPath = null;
let currentEventSource = null;
let currentFilter = 'all';
let processedRows = [];
let totalAnswers = 0;
let totalReviews = 0;

document.addEventListener('DOMContentLoaded', () => {
  setupEventListeners();
});

function setupEventListeners() {
  // 1. Click Dropzone or Browse Button -> Open Windows Folder Picker
  uploadDropzone.addEventListener('click', (e) => {
    folderFileInput.click();
  });

  btnBrowseTrigger.addEventListener('click', (e) => {
    e.stopPropagation();
    folderFileInput.click();
  });

  // 2. Native Folder Input Change
  folderFileInput.addEventListener('change', (e) => {
    handleFolderSelected(Array.from(e.target.files));
  });

  // 3. Drag and Drop Support
  uploadDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    uploadDropzone.classList.add('drag-over');
  });

  uploadDropzone.addEventListener('dragleave', () => {
    uploadDropzone.classList.remove('drag-over');
  });

  uploadDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    uploadDropzone.classList.remove('drag-over');
    if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
      handleFolderSelected(Array.from(e.dataTransfer.files));
    }
  });

  // 4. Sample Preset Button
  btnLoadSample.addEventListener('click', () => {
    handlePresetSelected('data/example');
  });

  // 5. Change Folder Button
  btnChangeFolder.addEventListener('click', () => {
    folderFileInput.click();
  });

  // 6. Start Extraction
  btnStartExtract.addEventListener('click', handleStartExtract);

  // 7. Reset / New Batch
  btnNewBatch.addEventListener('click', handleResetAll);
  btnResetAll.addEventListener('click', handleResetAll);

  // 8. Filter Tabs
  tabBtns.forEach((btn) => {
    btn.addEventListener('click', () => {
      tabBtns.forEach((b) => b.classList.remove('active'));
      btn.classList.add('active');
      currentFilter = btn.getAttribute('data-filter');
      renderFilteredRows();
    });
  });

  // 9. Close Modal
  btnCloseModal.addEventListener('click', () => {
    debugModal.style.display = 'none';
  });
  debugModal.addEventListener('click', (e) => {
    if (e.target === debugModal) {
      debugModal.style.display = 'none';
    }
  });
}

function handleFolderSelected(files) {
  const pdfs = files.filter((f) => f.name.toLowerCase().endsWith('.pdf'));
  if (pdfs.length === 0) {
    alert('No PDF documents found in the selected folder. Please choose another folder.');
    return;
  }

  selectedFiles = pdfs;
  presetPath = null;

  let folderName = 'Selected Folder';
  if (pdfs[0].webkitRelativePath) {
    const parts = pdfs[0].webkitRelativePath.split('/');
    if (parts.length > 1) {
      folderName = parts[0];
    }
  }

  // Switch to Stage 2: Ready Banner
  readyFolderName.textContent = folderName;
  readyFileCount.textContent = `${pdfs.length} PDF Documents`;
  kpiFiles.textContent = pdfs.length;

  selectionStage.style.display = 'none';
  readyStage.style.display = 'block';
  btnNewBatch.style.display = 'inline-flex';
}

function handlePresetSelected(path) {
  selectedFiles = [];
  presetPath = path;

  readyFolderName.textContent = 'Sample Dataset (data/example)';
  readyFileCount.textContent = '1 PDF (10 Pages)';
  kpiFiles.textContent = '1';

  selectionStage.style.display = 'none';
  readyStage.style.display = 'block';
  btnNewBatch.style.display = 'inline-flex';
}

async function handleStartExtract() {
  if (selectedFiles.length === 0 && !presetPath) {
    alert('Please select a folder first.');
    return;
  }

  // Switch to Stage 3: Live Progress & Table
  readyStage.style.display = 'none';
  progressStage.style.display = 'flex';
  floatingBar.style.display = 'none';

  processedRows = [];
  totalAnswers = 0;
  totalReviews = 0;
  kpiPages.textContent = '0';
  kpiAnswers.textContent = '0';
  kpiReviews.textContent = '0';
  countAll.textContent = '0';
  countResolved.textContent = '0';
  countReview.textContent = '0';

  tableBody.innerHTML = '';
  liveDot.style.display = 'inline-block';
  progressPercent.textContent = '0%';
  progressFill.style.width = '0%';
  progressStatusText.textContent = 'Preparing documents...';

  if (currentEventSource) {
    currentEventSource.close();
  }

  let sseUrl = '';
  const isDebug = checkboxDebug.checked;

  if (selectedFiles.length > 0) {
    progressStatusText.textContent = `Uploading ${selectedFiles.length} PDF documents...`;
    try {
      const formData = new FormData();
      selectedFiles.forEach((file) => {
        formData.append('files', file);
      });

      const res = await fetch(`${API_BASE}/api/upload-session`, {
        method: 'POST',
        body: formData,
      });

      if (!res.ok) {
          const errData = await res.json().catch(() => ({}));
          throw new Error(`Server returned ${res.status}: ${JSON.stringify(errData)}`);
      }

      const uploadData = await res.json();
      sseUrl = `${API_BASE}/api/process-stream?session_id=${encodeURIComponent(uploadData.session_id)}&debug=${isDebug}`;
    } catch (err) {
      alert('Upload error: ' + err.message);
      readyStage.style.display = 'block';
      progressStage.style.display = 'none';
      return;
    }
  } else if (presetPath) {
    sseUrl = `${API_BASE}/api/process-stream?folder_path=${encodeURIComponent(presetPath)}&debug=${isDebug}`;
  }

  progressStatusText.textContent = 'Analyzing and extracting marks...';
  currentEventSource = new EventSource(sseUrl);

  currentEventSource.onmessage = (event) => {
    try {
      const msg = JSON.parse(event.data);
      handleLiveEvent(msg);
    } catch (err) {
      console.error('SSE JSON error:', err);
    }
  };

  currentEventSource.onerror = (err) => {
    console.error('SSE stream closed/error:', err);
    liveDot.style.display = 'none';
    if (currentEventSource) {
      currentEventSource.close();
      currentEventSource = null;
    }
  };
}

function handleLiveEvent(msg) {
  switch (msg.type) {
    case 'init':
      progressStatusText.textContent = `Found ${msg.total_files} PDF files. Processing with V17 Engine...`;
      kpiFiles.textContent = msg.total_files;
      break;

    case 'file_start':
      progressStatusText.textContent = `Extracting [${msg.file_index}/${msg.total_files}] ${msg.file_name}`;
      updateProgress(msg.percent);
      break;

    case 'page_done':
      handlePageDone(msg);
      break;

    case 'page_error':
      handlePageError(msg);
      break;

    case 'complete':
      handleComplete(msg);
      break;

    case 'error':
      liveDot.style.display = 'none';
      progressStatusText.textContent = `Error: ${msg.message}`;
      alert(`Error: ${msg.message}`);
      break;
  }
}

function handlePageDone(msg) {
  updateProgress(msg.percent);
  progressStatusText.textContent = `Processed ${msg.file_name} (Page ${msg.page_number} of ${msg.total_pages_in_file})`;

  processedRows.push(msg);
  totalAnswers += (msg.selected_count || 0);
  totalReviews += (msg.review_count || 0);

  kpiPages.textContent = processedRows.length;
  kpiAnswers.textContent = totalAnswers.toLocaleString();
  kpiReviews.textContent = totalReviews;

  updateFilterCounts();
  appendTableRow(msg, processedRows.length);
}

function updateFilterCounts() {
  countAll.textContent = processedRows.length;
  countResolved.textContent = processedRows.filter((r) => r.status === 'resolved').length;
  countReview.textContent = processedRows.filter((r) => r.status === 'needs_review' || r.status === 'error').length;
}

function handlePageError(msg) {
  // Update progress text
  progressStatusText.textContent = `Error in ${msg.file_name} (Page ${msg.page_number}): ${msg.error}`;
  progressStatusText.style.color = '#DC2626'; // Red
  setTimeout(() => progressStatusText.style.color = '', 3000);
  
  msg.status = 'error';
  processedRows.push(msg);
  updateFilterCounts();
  appendTableRow(msg, processedRows.length);
}

function appendTableRow(item, index) {
  const isMatch = currentFilter === 'all' || item.status === currentFilter;
  if (!isMatch) return;

  const tr = document.createElement('tr');

  let statusBadge = '';
  if (item.status === 'resolved') {
    statusBadge = `<span class="status-pill resolved"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="20 6 9 17 4 12"></polyline></svg> Automated</span>`;
  } else if (item.status === 'error') {
    statusBadge = `<span class="status-pill" style="color: #DC2626; background: #FEF2F2; padding: 4px 10px; border-radius: 9999px; font-weight: 600; font-size: 11px; border: 1px solid #FCA5A5;"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg> Error</span>`;
  } else {
    statusBadge = `<span class="status-pill review"><svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg> Review Required</span>`;
  }

  let debugBtn = '';
  if (item.debug_images && item.debug_images.length > 0) {
    const urlsJson = escapeHtml(JSON.stringify(item.debug_images));
    debugBtn = `<button class="btn-ghost" style="padding: 4px 8px; font-size: 11px;" onclick="window.showDebugModal('${urlsJson}', '${escapeHtml(item.file_name)} - Page ${item.page_number}')">🔍 View Debug</button>`;
  }

  tr.innerHTML = `
    <td class="mono-cell">#${index}</td>
    <td class="doc-name">${escapeHtml(item.file_name)}</td>
    <td class="mono-cell">Page ${item.page_number}</td>
    <td class="mono-cell" style="color: ${item.status === 'error' ? '#DC2626' : '#3F3F46'};">${escapeHtml(item.status === 'error' ? item.error : (item.selected_summary || 'None'))}</td>
    <td class="mono-cell">${item.elapsed_seconds || '-'}s</td>
    <td>${statusBadge}</td>
    <td class="mono-cell" style="color: ${item.review_count > 0 ? '#D97706' : '#059669'}; font-weight: 700; display: flex; flex-direction: column; gap: 4px;">
      ${item.status === 'error' ? '-' : (item.review_count > 0 ? `${item.review_count} items` : '0')}
      ${debugBtn}
    </td>
  `;

  tableBody.appendChild(tr);
  const wrap = document.querySelector('.table-wrap');
  if (wrap) {
    wrap.scrollTop = wrap.scrollHeight;
  }
}

function renderFilteredRows() {
  tableBody.innerHTML = '';
  const filtered = processedRows.filter((r) => {
    if (currentFilter === 'all') return true;
    return r.status === currentFilter;
  });

  filtered.forEach((item, idx) => {
    appendTableRow(item, idx + 1);
  });
}

function handleComplete(msg) {
  liveDot.style.display = 'none';
  updateProgress(100);
  progressStatusText.textContent = `Completed extraction for ${msg.stats.total_pages} pages across ${msg.stats.total_files} files.`;

  if (currentEventSource) {
    currentEventSource.close();
    currentEventSource = null;
  }

  // Set download URL
  const downloadUrl = `${API_BASE}/api/download?file_path=${encodeURIComponent(msg.stats.excel_path)}`;
  btnDownloadExcel.href = downloadUrl;
  btnDownloadExcel.download = msg.stats.excel_file_name;
  footerExcelPath.textContent = msg.stats.excel_path;

  floatingBar.style.display = 'block';
}

function updateProgress(percent) {
  const p = Math.min(100, Math.max(0, percent || 0));
  progressPercent.textContent = `${p}%`;
  progressFill.style.width = `${p}%`;
}

function handleResetAll() {
  if (currentEventSource) {
    currentEventSource.close();
    currentEventSource = null;
  }

  selectedFiles = [];
  presetPath = null;
  folderFileInput.value = '';

  selectionStage.style.display = 'flex';
  readyStage.style.display = 'none';
  progressStage.style.display = 'none';
  floatingBar.style.display = 'none';
  btnNewBatch.style.display = 'none';

  tableBody.innerHTML = '';
  updateProgress(0);
}

function escapeHtml(str) {
  return String(str).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

window.showDebugModal = function(urlsJsonStr, title) {
  debugModalTitle.textContent = "Debug: " + title;
  debugModalContent.innerHTML = '';
  
  try {
    // Unescape quotes so JSON.parse can read it
    const unescaped = String(urlsJsonStr).replace(/&quot;/g, '"');
    const urls = JSON.parse(unescaped);
    const titles = ["1. Original Scan (Ảnh gốc căn lề)", "2. Ink Mask (Tách mực)", "3. AI Diagnostics (Lớp phủ phân tích)"];
    
    urls.forEach((url, index) => {
      // Prepend API_BASE
      const fullUrl = API_BASE + url;
      const stepTitle = titles[index] || `Layer ${index + 1}`;
      
      const block = document.createElement('div');
      block.style.cssText = "background: #fff; padding: 16px; border-radius: 8px; box-shadow: 0 1px 3px rgba(0,0,0,0.1); border: 1px solid #e4e4e7;";
      
      const header = document.createElement('h4');
      header.textContent = stepTitle;
      header.style.cssText = "margin: 0 0 12px 0; font-size: 14px; font-weight: 600; color: #18181b; text-align: left; padding-bottom: 8px; border-bottom: 1px solid #f4f4f5;";
      
      const img = document.createElement('img');
      img.src = fullUrl;
      img.style.cssText = "max-width: 100%; height: auto; display: block; margin: 0 auto; border: 1px solid #f4f4f5; background: #fafafa;";
      
      block.appendChild(header);
      block.appendChild(img);
      debugModalContent.appendChild(block);
    });
  } catch (e) {
    debugModalContent.innerHTML = '<p style="color: red;">Failed to load debug images.</p>';
    console.error(e);
  }
  
  debugModal.style.display = 'flex';
};
