// API Configuration
const API_URL = 'http://localhost:8000';

// ─── State ──────────────────────────────────────────────────────────────────
let currentFile = null;
let currentMode = 'image'; // 'image' or 'video'

// ─── DOM Elements ───────────────────────────────────────────────────────────
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const uploadTitle = document.getElementById('uploadTitle');
const fileInfo = document.getElementById('fileInfo');

// Mode buttons
const modeImage = document.getElementById('modeImage');
const modeVideo = document.getElementById('modeVideo');

// Image preview
const imagePreview = document.getElementById('imagePreview');
const previewSection = document.getElementById('previewSection');
const analyzeBtn = document.getElementById('analyzeBtn');

// Video preview
const videoPreview = document.getElementById('videoPreview');
const videoPreviewSection = document.getElementById('videoPreviewSection');
const analyzeVideoBtn = document.getElementById('analyzeVideoBtn');
const maxFramesSelect = document.getElementById('maxFramesSelect');

// Loading
const loadingSection = document.getElementById('loadingSection');
const loadingTitle = document.getElementById('loadingTitle');
const loadingDescription = document.getElementById('loadingDescription');

// Image results
const resultsSection = document.getElementById('resultsSection');
const resultBadge = document.getElementById('resultBadge');
const resultIcon = document.getElementById('resultIcon');
const resultLabel = document.getElementById('resultLabel');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceFill = document.getElementById('confidenceFill');
const realProb = document.getElementById('realProb');
const fakeProb = document.getElementById('fakeProb');
const resetBtn = document.getElementById('resetBtn');
const explainBtn = document.getElementById('explainBtn');
const gradcamBox = document.getElementById('gradcamBox');
const gradcamImg = document.getElementById('gradcamImg');

// Video results
const videoResultsSection = document.getElementById('videoResultsSection');
const videoResultBadge = document.getElementById('videoResultBadge');
const videoResultIcon = document.getElementById('videoResultIcon');
const videoResultLabel = document.getElementById('videoResultLabel');
const videoConfidenceValue = document.getElementById('videoConfidenceValue');
const videoConfidenceFill = document.getElementById('videoConfidenceFill');
const totalFrames = document.getElementById('totalFrames');
const fakeFrames = document.getElementById('fakeFrames');
const realFramesEl = document.getElementById('realFrames');
const fakeRatio = document.getElementById('fakeRatio');
const timelineBar = document.getElementById('timelineBar');
const videoResetBtn = document.getElementById('videoResetBtn');

// Error
const errorSection = document.getElementById('errorSection');
const errorMessage = document.getElementById('errorMessage');
const errorResetBtn = document.getElementById('errorResetBtn');

// Status
const apiStatus = document.getElementById('apiStatus');

// ─── Mode Switching ─────────────────────────────────────────────────────────

modeImage.addEventListener('click', () => switchMode('image'));
modeVideo.addEventListener('click', () => switchMode('video'));

function switchMode(mode) {
    currentMode = mode;
    currentFile = null;

    // Update active button
    modeImage.classList.toggle('active', mode === 'image');
    modeVideo.classList.toggle('active', mode === 'video');

    // Update file input
    if (mode === 'image') {
        fileInput.accept = 'image/*';
        uploadTitle.textContent = 'Drop your image here';
        fileInfo.textContent = 'Supports: JPG, PNG, JPEG, WEBP';
    } else {
        fileInput.accept = 'video/mp4,video/avi,video/quicktime,video/x-matroska,video/webm,.mp4,.avi,.mov,.mkv,.webm';
        uploadTitle.textContent = 'Drop your video here';
        fileInfo.textContent = 'Supports: MP4, AVI, MOV, MKV, WEBM';
    }

    // Reset everything
    resetApp();
}

// ─── Event Listeners ────────────────────────────────────────────────────────

browseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
analyzeBtn.addEventListener('click', analyzeImage);
analyzeVideoBtn.addEventListener('click', analyzeVideo);
explainBtn.addEventListener('click', explainDecision);
resetBtn.addEventListener('click', resetApp);
videoResetBtn.addEventListener('click', resetApp);
errorResetBtn.addEventListener('click', resetApp);

// Drag and Drop
dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');

    const files = e.dataTransfer.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
});

dropZone.addEventListener('click', (e) => {
    if (e.target === dropZone || e.target.closest('.upload-card')) {
        fileInput.click();
    }
});

// ─── File Handling ──────────────────────────────────────────────────────────

function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    const isImage = file.type.startsWith('image/');
    const isVideo = file.type.startsWith('video/') || /\.(mp4|avi|mov|mkv|webm)$/i.test(file.name);

    // Auto-switch mode if needed
    if (isImage && currentMode !== 'image') {
        switchMode('image');
    } else if (isVideo && currentMode !== 'video') {
        switchMode('video');
    }

    // Validate
    if (currentMode === 'image' && !isImage) {
        showError('Please select a valid image file (JPG, PNG, JPEG, WEBP)');
        return;
    }
    if (currentMode === 'video' && !isVideo) {
        showError('Please select a valid video file (MP4, AVI, MOV, MKV, WEBM)');
        return;
    }

    // File size limits
    const maxSize = currentMode === 'image' ? 10 * 1024 * 1024 : 500 * 1024 * 1024;
    const maxLabel = currentMode === 'image' ? '10MB' : '500MB';
    if (file.size > maxSize) {
        showError(`File size must be less than ${maxLabel}`);
        return;
    }

    currentFile = file;

    if (currentMode === 'image') {
        // Show image preview
        const reader = new FileReader();
        reader.onload = (e) => {
            imagePreview.src = e.target.result;
            showSection('preview');
        };
        reader.readAsDataURL(file);
    } else {
        // Show video preview
        const videoURL = URL.createObjectURL(file);
        videoPreview.src = videoURL;
        showSection('videoPreview');
    }
}

// ─── Image Analysis ─────────────────────────────────────────────────────────

async function analyzeImage() {
    if (!currentFile) {
        showError('Please select an image first');
        return;
    }

    loadingTitle.textContent = 'Analyzing Image...';
    loadingDescription.textContent = 'Our AI is examining the image for deepfake patterns';
    showSection('loading');

    try {
        const formData = new FormData();
        formData.append('file', currentFile);

        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to analyze image');
        }

        const data = await response.json();
        displayImageResults(data);

    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Failed to connect to the server. Please make sure the backend is running.');
    }
}

// ─── Grad-CAM Explanation ───────────────────────────────────────────────────

async function explainDecision() {
    if (!currentFile) return;

    explainBtn.disabled = true;
    const originalContent = explainBtn.innerHTML;
    explainBtn.textContent = 'Computing Heatmap...';

    try {
        const formData = new FormData();
        formData.append('file', currentFile);

        const response = await fetch(`${API_URL}/predict/explain`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to generate Grad-CAM explanation');
        }

        const data = await response.json();
        if (data.heatmap_base64) {
            gradcamImg.src = 'data:image/png;base64,' + data.heatmap_base64;
            gradcamBox.style.display = 'block';
        }
    } catch (error) {
        console.error('Grad-CAM Error:', error);
        alert(error.message || 'Could not compute Grad-CAM heatmap.');
    } finally {
        explainBtn.disabled = false;
        explainBtn.innerHTML = originalContent;
    }
}

// ─── Video Analysis ─────────────────────────────────────────────────────────

async function analyzeVideo() {
    if (!currentFile) {
        showError('Please select a video first');
        return;
    }

    const maxFrames = parseInt(maxFramesSelect.value);
    loadingTitle.textContent = 'Analyzing Video...';
    loadingDescription.textContent = `Extracting and analyzing ${maxFrames} frames for deepfake patterns`;
    showSection('loading');

    try {
        const formData = new FormData();
        formData.append('file', currentFile);

        const response = await fetch(`${API_URL}/predict/video?max_frames=${maxFrames}`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to analyze video');
        }

        const data = await response.json();
        displayVideoResults(data);

    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Failed to connect to the server. Please make sure the backend is running.');
    }
}

// ─── Display Image Results ──────────────────────────────────────────────────

function displayImageResults(data) {
    const isFake = data.prediction === 'Fake';
    const confidence = data.confidence;
    const realProbability = data.details.real_probability;
    const fakeProbability = data.details.fake_probability;

    // Update result badge
    resultBadge.className = 'result-badge ' + (isFake ? 'fake' : 'real');
    resultLabel.textContent = data.prediction;

    // Update icon
    if (isFake) {
        resultIcon.innerHTML = `
            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
            <path d="M15 9L9 15" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            <path d="M9 9L15 15" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        `;
    } else {
        resultIcon.innerHTML = `
            <path d="M9 11L12 14L22 4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M21 12V19C21 19.5304 20.7893 20.0391 20.4142 20.4142C20.0391 20.7893 19.5304 21 19 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V5C3 4.46957 3.21071 3.96086 3.58579 3.58579C3.96086 3.21071 4.46957 3 5 3H16" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        `;
    }

    // Update confidence
    confidenceValue.textContent = `${confidence.toFixed(1)}%`;
    confidenceFill.className = 'confidence-fill ' + (isFake ? 'fake' : 'real');

    // Animate confidence bar
    setTimeout(() => {
        confidenceFill.style.width = `${confidence}%`;
    }, 100);

    // Update probabilities
    realProb.textContent = `${realProbability.toFixed(1)}%`;
    fakeProb.textContent = `${fakeProbability.toFixed(1)}%`;

    showSection('results');
}

// ─── Display Video Results ──────────────────────────────────────────────────

function displayVideoResults(data) {
    const isFake = data.prediction === 'Fake';
    const confidence = data.confidence;
    const summary = data.summary;
    const frameResults = data.frame_results || [];

    // Update verdict badge
    videoResultBadge.className = 'result-badge ' + (isFake ? 'fake' : 'real');
    videoResultLabel.textContent = data.prediction;

    // Update icon
    if (isFake) {
        videoResultIcon.innerHTML = `
            <circle cx="12" cy="12" r="10" stroke="currentColor" stroke-width="2"/>
            <path d="M15 9L9 15" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
            <path d="M9 9L15 15" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
        `;
    } else {
        videoResultIcon.innerHTML = `
            <path d="M9 11L12 14L22 4" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            <path d="M21 12V19C21 19.5304 20.7893 20.0391 20.4142 20.4142C20.0391 20.7893 19.5304 21 19 21H5C4.46957 21 3.96086 20.7893 3.58579 20.4142C3.21071 20.0391 3 19.5304 3 19V5C3 4.46957 3.21071 3.96086 3.58579 3.58579C3.96086 3.21071 4.46957 3 5 3H16" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
        `;
    }

    // Update confidence
    videoConfidenceValue.textContent = `${confidence.toFixed(1)}%`;
    videoConfidenceFill.className = 'confidence-fill ' + (isFake ? 'fake' : 'real');
    setTimeout(() => {
        videoConfidenceFill.style.width = `${confidence}%`;
    }, 100);

    // Update summary stats
    totalFrames.textContent = summary.total_frames_analyzed;
    fakeFrames.textContent = summary.fake_frames;
    realFramesEl.textContent = summary.real_frames;
    fakeRatio.textContent = `${summary.fake_ratio}%`;

    // Build frame timeline
    timelineBar.innerHTML = '';
    frameResults.forEach((frame, i) => {
        const block = document.createElement('div');
        block.className = 'timeline-block ' + (frame.prediction === 'Fake' ? 'fake' : 'real');
        block.style.width = `${100 / frameResults.length}%`;
        block.title = `Frame ${frame.frame_index} (${frame.timestamp}s) — ${frame.prediction} (${frame.confidence}%)`;
        // Animate in with delay
        block.style.animationDelay = `${i * 30}ms`;
        timelineBar.appendChild(block);
    });

    showSection('videoResults');
}

// ─── Section Management ────────────────────────────────────────────────────

function showSection(section) {
    // Hide all dynamic sections
    previewSection.style.display = 'none';
    videoPreviewSection.style.display = 'none';
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    videoResultsSection.style.display = 'none';
    errorSection.style.display = 'none';

    switch (section) {
        case 'preview':
            previewSection.style.display = 'block';
            break;
        case 'videoPreview':
            videoPreviewSection.style.display = 'block';
            break;
        case 'loading':
            loadingSection.style.display = 'block';
            break;
        case 'results':
            resultsSection.style.display = 'block';
            break;
        case 'videoResults':
            videoResultsSection.style.display = 'block';
            break;
        case 'error':
            errorSection.style.display = 'block';
            break;
    }
}

// ─── Error & Reset ──────────────────────────────────────────────────────────

function showError(message) {
    errorMessage.textContent = message;
    showSection('error');
}

function resetApp() {
    currentFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    if (videoPreview.src) {
        URL.revokeObjectURL(videoPreview.src);
        videoPreview.src = '';
    }
    confidenceFill.style.width = '0%';
    videoConfidenceFill.style.width = '0%';
    timelineBar.innerHTML = '';

    // Reset Grad-CAM
    if (gradcamBox) gradcamBox.style.display = 'none';
    if (gradcamImg) gradcamImg.src = '';

    // Hide all dynamic sections
    previewSection.style.display = 'none';
    videoPreviewSection.style.display = 'none';
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    videoResultsSection.style.display = 'none';
    errorSection.style.display = 'none';
}

// ─── API Health Check ──────────────────────────────────────────────────────

async function checkAPIHealth() {
    const statusDot = apiStatus.querySelector('.status-dot');
    const statusText = apiStatus.querySelector('.status-text');

    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            const data = await response.json();
            statusDot.classList.add('online');
            statusDot.classList.remove('offline');
            const modelStatus = data.model_loaded ? 'Model loaded' : 'No model';
            const deviceInfo = data.device || 'cpu';
            statusText.textContent = `Online · ${modelStatus} · ${deviceInfo}`;
            console.log('✅ Backend API is running');
        } else {
            throw new Error('Not ok');
        }
    } catch (error) {
        statusDot.classList.add('offline');
        statusDot.classList.remove('online');
        statusText.textContent = 'Backend offline';
        console.warn('⚠️ Backend API is not running. Start with: uvicorn app:app --reload');
    }
}

// Initialize
checkAPIHealth();
// Re-check every 15 seconds
setInterval(checkAPIHealth, 15000);
