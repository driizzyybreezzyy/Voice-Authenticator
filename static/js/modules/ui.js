// static/js/modules/ui.js
import {
    DOMElements,
    APP_CONFIG
} from './config.js';
// import { loadUsers } from './auth.js'; // Or a separate userManagement.js if you create one

// --- Tab Navigation ---
export function initializeTabs(onUserTabActivate) { // onUserTabActivate is a callback
    const tabBtns = DOMElements.tabBtns();
    const tabContents = DOMElements.tabContents();

    tabBtns.forEach(btn => {
        btn.addEventListener('click', function() {
            tabBtns.forEach(b => b.classList.remove('active'));
            tabContents.forEach(c => {
                c.classList.remove('active');
                c.style.display = 'none';
            });
            this.classList.add('active');
            const targetTabId = `${this.dataset.tab}-tab`;
            const targetTabContent = DOMElements.get(`#${targetTabId}`);
            if (targetTabContent) {
                targetTabContent.classList.add('active');
                targetTabContent.style.display = 'block';
                if (this.dataset.tab === 'users' && typeof onUserTabActivate === 'function') {
                    onUserTabActivate(); // Call the callback (e.g., loadUsers)
                }
                if (this.dataset.tab !== 'authenticate' && DOMElements.authBenchmarkTableContainer()) {
                    DOMElements.authBenchmarkTableContainer().innerHTML = '';
                    if (DOMElements.benchmarkTitle()) DOMElements.benchmarkTitle().style.display = 'none';
                }
            }
        });
    });
    tabContents.forEach(c => {
        if (!c.classList.contains('active')) c.style.display = 'none';
    });
}

// --- Feedback Messages ---
export function showResult(elementId, message, type) {
    const el = DOMElements.get(`#${elementId}`);
    if (el) {
        el.textContent = message;
        el.className = `result ${type}`;
        el.style.display = 'block';
        el.setAttribute('data-text', message);
        el.classList.add('active-feedback');
        setTimeout(() => {
            el.classList.remove('active-feedback');
        }, 300);
    } else {
        console.warn("showResult: Element not found with ID:", elementId);
    }
}

// --- Benchmark Table ---
export function displayBenchmarkTable(scores, containerId) {
    const container = DOMElements.get(`#${containerId}`);
    // ... (rest of the displayBenchmarkTable logic from original app.js) ...
    if (!container) return;
    container.innerHTML = '';
    if (!scores || Object.keys(scores).length === 0) {
        container.innerHTML = '<p style="text-align:center; color: var(--text-secondary);">No benchmark scores to display.</p>';
        return;
    }
    const table = document.createElement('table');
    table.innerHTML = `<thead><tr><th>Method</th><th>Similarity</th><th>Matches?</th></tr></thead><tbody></tbody>`;
    const tbody = table.querySelector('tbody');
    for (const method in scores) {
        const d = scores[method];
        const r = tbody.insertRow();
        r.insertCell().textContent = method;
        const simCell = r.insertCell();
        if (d.error) {
            simCell.textContent = d.error;
            simCell.style.color = 'var(--error-text)';
        } else {
            simCell.textContent = d.similarity !== null ? (d.similarity * 100).toFixed(2) + '%' : 'N/A';
        }
        const mc = r.insertCell();
        mc.textContent = d.similarity !== null ? (d.matches_threshold ? 'Yes' : 'No') : '-';
        mc.style.color = d.matches_threshold ? 'var(--success-text)' : (d.similarity !== null ? 'var(--error-text)' : 'var(--text-secondary)');
    }
    container.appendChild(table);
}


// --- Sample Quality Feedback ---
export function displaySampleQualityFeedback(sampleIndex0Based, report) {
    const prefix = `enroll_sample_${sampleIndex0Based}`;
    const qualityStatusEl = DOMElements.get(`#${prefix}_quality_status`);
    const recordSnrEl = DOMElements.get(`#${prefix}_record_snr_display`);
    const uploadSnrEl = DOMElements.get(`#${prefix}_upload_snr_display`);

    // ... (rest of the displaySampleQualityFeedback logic from original app.js, using APP_CONFIG.MIN_SNR_DB_THRESHOLD) ...
    if (qualityStatusEl) {
        qualityStatusEl.innerHTML = '';
        qualityStatusEl.style.display = 'none';
        qualityStatusEl.className = 'sample-quality-status';
    }
    if (recordSnrEl) {
        recordSnrEl.innerHTML = '';
        recordSnrEl.style.display = 'none';
        recordSnrEl.className = 'snr-display';
    }
    if (uploadSnrEl) {
        uploadSnrEl.innerHTML = '';
        uploadSnrEl.style.display = 'none';
        uploadSnrEl.className = 'snr-display';
    }

    if (!report) return;

    if (qualityStatusEl) {
        let statusText = `Sample ${sampleIndex0Based + 1}: `;
        if (report.is_usable) {
            statusText += `<strong style="color:var(--success-text);">OK</strong>`;
            qualityStatusEl.className = 'sample-quality-status good';
        } else {
            statusText += `<strong style="color:var(--error-text);">ISSUES DETECTED</strong>`;
            qualityStatusEl.className = 'sample-quality-status poor';
        }
        if (report.reasons && report.reasons.length > 0 && !(report.reasons.length === 1 && report.reasons[0] === "OK")) {
            statusText += ` (${report.reasons.join(', ')})`;
        }
        qualityStatusEl.innerHTML = statusText;
        qualityStatusEl.style.display = 'block';
    }

    const snrValue = report.snr_db;
    if (snrValue !== "N/A" && snrValue !== undefined && (recordSnrEl || uploadSnrEl)) {
        let targetSnrEl = null;
        const inputArea = DOMElements.get(`#${prefix}-input-area`);
        if (inputArea && inputArea.classList.contains('record-active') && recordSnrEl) {
            targetSnrEl = recordSnrEl;
        } else if (inputArea && inputArea.classList.contains('upload-active') && uploadSnrEl) {
            targetSnrEl = uploadSnrEl;
        } else if (recordSnrEl) {
            targetSnrEl = recordSnrEl;
        } else {
            targetSnrEl = uploadSnrEl;
        }

        if (targetSnrEl) {
            targetSnrEl.textContent = `SNR: ${snrValue} dB`;
            targetSnrEl.className = 'snr-display';

            if (typeof snrValue === 'number') {
                const minSnrThreshold = APP_CONFIG.MIN_SNR_DB_THRESHOLD;
                if (snrValue >= minSnrThreshold) {
                    targetSnrEl.classList.add('good');
                } else {
                    targetSnrEl.classList.add('poor');
                }
            }
            targetSnrEl.style.display = 'inline-block';
        }
    }
}

// --- Input Method Toggler ---
// This needs callbacks for enrollment and auth specific logic
let currentActiveMediaRecorder = null; // To be set by audio.js

export function setActiveMediaRecorderForToggler(recorder) {
    currentActiveMediaRecorder = recorder;
}

export function initializeInputToggles(callbacks) {
    // callbacks = {
    //  onEnrollmentAudioChange: function(targetPrefix, audioData),
    //  onAuthAudioChange: function(targetPrefix, audioData),
    //  getAuthName: function(),
    //  stopActiveRecorder: function() // from audio.js
    // }
    document.body.addEventListener('click', function(event) {
        const buttonElement = event.target.closest('.input-method-toggle .toggle-btn');
        if (buttonElement){
            handleToggleClick(buttonElement, false, callbacks);
        }
    });
    DOMElements.getAll('.input-method-toggle').forEach(toggleGroup => {
        const activeButton = toggleGroup.querySelector('.toggle-btn.active') || toggleGroup.firstElementChild;
        if (activeButton)
        {
            handleToggleClick(activeButton, true, callbacks);
        }
    });
}

async function handleToggleClick(buttonElement, isInitialSetup = false, callbacks) {
    const inputType = buttonElement.dataset.input;
    const targetPrefix = buttonElement.dataset.target;
    const {
        getElementsForGroup,
        getPrimaryButtonForGroup
    } = await import('./config.js'); // Dynamic import or pass them

    if (!isInitialSetup) {
        buttonElement.parentElement.querySelectorAll('.toggle-btn').forEach(btn => btn.classList.remove('active'));
        buttonElement.classList.add('active');
    }
    const inputArea = DOMElements.get(`#${targetPrefix}-input-area`);
    if (inputArea) inputArea.className = `voice-input-area ${inputType}-active`;

    const elGroup = getElementsForGroup(targetPrefix);
    if (!elGroup) return;

    if (inputType === 'record') {
        if (elGroup.fileNameDisplay) elGroup.fileNameDisplay.textContent = 'No file chosen';
        if (elGroup.uploadPlaybackAudio) {
            elGroup.uploadPlaybackAudio.src = '';
            if (elGroup.uploadPlaybackContainer) elGroup.uploadPlaybackContainer.style.display = 'none';
        }
        if (elGroup.fileInput) elGroup.fileInput.value = null;
        // Clear audio data
        if (elGroup.isEnrollSample) callbacks.onEnrollmentAudioChange(targetPrefix, null);
        else callbacks.onAuthAudioChange(targetPrefix, null);

    } else { // upload
        if (elGroup.playbackAudio) {
            elGroup.playbackAudio.src = '';
            if (elGroup.playbackContainer) elGroup.playbackContainer.style.display = 'none';
        }
        // Stop active recording IF it's for this group
        if (currentActiveMediaRecorder && currentActiveMediaRecorder.state === "recording" && currentActiveMediaRecorder.associatedPrefix === targetPrefix) {
            callbacks.stopActiveRecorder(); // Call the stop function from audio.js
        }
        // Clear audio data
        if (elGroup.isEnrollSample) callbacks.onEnrollmentAudioChange(targetPrefix, null);
        else callbacks.onAuthAudioChange(targetPrefix, null);
    }

    // Update primary button state
    if (elGroup.isEnrollSample) {
        // Enrollment primary button state is handled by updateSamplesCollectedCountAndButtons in enrollment.js
        // It will be called when onEnrollmentAudioChange results in an update.
    } else { // Auth
        const primaryBtn = getPrimaryButtonForGroup(targetPrefix);
        if (primaryBtn) {
            const nameFilled = callbacks.getAuthName() ? true : false;
            // Check current audio source state (managed by auth.js)
            // This part is tricky, auth.js needs to expose its audio source state or a check function
            // For now, let's assume auth.js handles its button state via onAuthAudioChange.
            // primaryBtn.disabled = !(/* audio exists for auth */ && nameFilled);
        }
    }
}