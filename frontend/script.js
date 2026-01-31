// API Configuration
const API_URL = 'http://localhost:8000';

// DOM Elements
const dropZone = document.getElementById('dropZone');
const fileInput = document.getElementById('fileInput');
const browseBtn = document.getElementById('browseBtn');
const imagePreview = document.getElementById('imagePreview');
const analyzeBtn = document.getElementById('analyzeBtn');
const resetBtn = document.getElementById('resetBtn');
const errorResetBtn = document.getElementById('errorResetBtn');

// Sections
const previewSection = document.getElementById('previewSection');
const loadingSection = document.getElementById('loadingSection');
const resultsSection = document.getElementById('resultsSection');
const errorSection = document.getElementById('errorSection');

// Result Elements
const resultBadge = document.getElementById('resultBadge');
const resultIcon = document.getElementById('resultIcon');
const resultLabel = document.getElementById('resultLabel');
const confidenceValue = document.getElementById('confidenceValue');
const confidenceFill = document.getElementById('confidenceFill');
const realProb = document.getElementById('realProb');
const fakeProb = document.getElementById('fakeProb');
const errorMessage = document.getElementById('errorMessage');

// State
let currentFile = null;

// Event Listeners
browseBtn.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFileSelect);
analyzeBtn.addEventListener('click', analyzeImage);
resetBtn.addEventListener('click', resetApp);
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

// File Handling
function handleFileSelect(e) {
    const files = e.target.files;
    if (files.length > 0) {
        handleFile(files[0]);
    }
}

function handleFile(file) {
    // Validate file type
    if (!file.type.startsWith('image/')) {
        showError('Please select a valid image file (JPG, PNG, JPEG)');
        return;
    }

    // Validate file size (max 10MB)
    if (file.size > 10 * 1024 * 1024) {
        showError('File size must be less than 10MB');
        return;
    }

    currentFile = file;

    // Show preview
    const reader = new FileReader();
    reader.onload = (e) => {
        imagePreview.src = e.target.result;
        showSection('preview');
    };
    reader.readAsDataURL(file);
}

// Analyze Image
async function analyzeImage() {
    if (!currentFile) {
        showError('Please select an image first');
        return;
    }

    showSection('loading');

    try {
        // Create form data
        const formData = new FormData();
        formData.append('file', currentFile);

        // Send request to API
        const response = await fetch(`${API_URL}/predict`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.detail || 'Failed to analyze image');
        }

        const data = await response.json();

        // Display results
        displayResults(data);

    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Failed to connect to the server. Please make sure the backend is running.');
    }
}

// Display Results
function displayResults(data) {
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

    // Show results section
    showSection('results');
}

// Show Error
function showError(message) {
    errorMessage.textContent = message;
    showSection('error');
}

// Show Section
function showSection(section) {
    // Hide all sections
    previewSection.style.display = 'none';
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';

    // Show requested section
    switch (section) {
        case 'preview':
            previewSection.style.display = 'block';
            break;
        case 'loading':
            loadingSection.style.display = 'block';
            break;
        case 'results':
            resultsSection.style.display = 'block';
            break;
        case 'error':
            errorSection.style.display = 'block';
            break;
    }
}

// Reset App
function resetApp() {
    currentFile = null;
    fileInput.value = '';
    imagePreview.src = '';
    confidenceFill.style.width = '0%';

    // Hide all sections
    previewSection.style.display = 'none';
    loadingSection.style.display = 'none';
    resultsSection.style.display = 'none';
    errorSection.style.display = 'none';
}

// Check API Health on Load
async function checkAPIHealth() {
    try {
        const response = await fetch(`${API_URL}/health`);
        if (response.ok) {
            console.log('✅ Backend API is running');
        }
    } catch (error) {
        console.warn('⚠️ Backend API is not running. Please start the server with: uvicorn app:app --reload');
    }
}

// Initialize
checkAPIHealth();
