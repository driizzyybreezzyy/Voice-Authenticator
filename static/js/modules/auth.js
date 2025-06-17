// static/js/modules/auth.js
import { DOMElements, getElementsForGroup, getResultElIdForGroup } from './config.js';
import { fetchAPI } from './api.js';
import { showResult, displayBenchmarkTable, initializeInputToggles } from './ui.js';
import { startRecording, setupFileInput, stopActiveRecorderIfAny } from './audio.js';
import { resetEnrollmentStateForDBReset } from './enrollment.js';


let singleAuthAudioSource = null; // Blob or File

function updateAuthenticateButtonState() {
    const nameFilled = DOMElements.authNameInput() && DOMElements.authNameInput().value.trim();
    if (DOMElements.authenticateBtn()) {
        DOMElements.authenticateBtn().disabled = !(singleAuthAudioSource && nameFilled);
    }
}

async function processSingleAudioRequest(endpoint, name, audioSourceBlobOrFile, resultElId, primaryBtn, benchmarkDisplayCallbackFn) {
    showResult(resultElId, 'Processing audio...', 'pending');
    if (primaryBtn) primaryBtn.disabled = true;

    const reader = new FileReader();
    reader.readAsDataURL(audioSourceBlobOrFile);
    reader.onloadend = async () => {
        try {
            const data = await fetchAPI(endpoint, 'POST', { name: name, audio: reader.result });
            const backendSuccess = data.s === true || data.success === true;
            if (backendSuccess) {
                if (endpoint === '/authenticate') {
                    const simScore = data.user?.primary_method_similarity;
                    const simText = simScore !== undefined ? ` Sim: ${(simScore * 100).toFixed(1)}%` : '';
                    const msg = data.authenticated ? `Authenticated as ${data.user.name}! Method: ${data.user.primary_method_used},${simText}` : `Auth failed for ${name}. ${data.message || ''}${simText}`;
                    showResult(resultElId, msg, data.authenticated ? 'success' : 'error');
                    if (benchmarkDisplayCallbackFn && typeof benchmarkDisplayCallbackFn === 'function') {
                        benchmarkDisplayCallbackFn(data.benchmark_scores);
                    } else {
                        if (DOMElements.authBenchmarkTableContainer()) DOMElements.authBenchmarkTableContainer().innerHTML = '';
                        if (DOMElements.benchmarkTitle()) DOMElements.benchmarkTitle().style.display = 'none';
                    }
                }
            } else {
                showResult(resultElId, `Error: ${data.e || data.error || data.message || 'Operation failed.'}`, 'error');
                if (DOMElements.authBenchmarkTableContainer()) DOMElements.authBenchmarkTableContainer().innerHTML = '';
                if (DOMElements.benchmarkTitle()) DOMElements.benchmarkTitle().style.display = 'none';
            }
        } catch (error) {
            showResult(resultElId, `Client/Network Error: ${error.message}`, 'error');
            if (DOMElements.authBenchmarkTableContainer()) DOMElements.authBenchmarkTableContainer().innerHTML = '';
            if (DOMElements.benchmarkTitle()) DOMElements.benchmarkTitle().style.display = 'none';
        } finally {
            if (primaryBtn) {
                updateAuthenticateButtonState(); // Re-evaluates based on current state
            }
        }
    };
    reader.onerror = () => {
        showResult(resultElId, 'Error reading audio file.', 'error');
        if (primaryBtn) updateAuthenticateButtonState();
    };
}


export function initializeAuthenticationSystem() {
    const groupPrefix = 'auth_single';
    const elGroup = getElementsForGroup(groupPrefix);

    if (DOMElements.authNameInput()) {
        DOMElements.authNameInput().addEventListener('input', updateAuthenticateButtonState);
    }
    if (DOMElements.authenticateBtn()) {
        DOMElements.authenticateBtn().addEventListener('click', () => {
            const name = DOMElements.authNameInput().value.trim();
            if (!name) {
                showResult('auth-result', 'Name is required for authentication.', 'error');
                return;
            }
            if (!singleAuthAudioSource) {
                showResult('auth-result', 'Please provide voice audio.', 'error');
                return;
            }
            processSingleAudioRequest('/authenticate', name, singleAuthAudioSource, 'auth-result', DOMElements.authenticateBtn(), (benchmarkScores) => {
                if (DOMElements.benchmarkTitle()) DOMElements.benchmarkTitle().style.display = benchmarkScores && Object.keys(benchmarkScores).length > 0 ? 'block' : 'none';
                displayBenchmarkTable(benchmarkScores, 'auth-benchmark-table-container');
            });
        });
    }


    if (elGroup.recordBtn) {
        elGroup.recordBtn.addEventListener('click', () => {
            startRecording(groupPrefix, {
                onRecordingStartUI: (grp) => {
                    grp.recordBtn.classList.add('recording');
                    grp.recordBtn.innerHTML = '<i class="fas fa-stop"></i>';
                    if (grp.recordInstruction) grp.recordInstruction.textContent = 'Recording... Click Stop';
                    if (grp.recordingIndicator) grp.recordingIndicator.style.display = 'flex';
                    if (DOMElements.authenticateBtn()) DOMElements.authenticateBtn().disabled = true;
                },
                onRecordingStopSuccess: (blob, prefix, grp) => {
                    singleAuthAudioSource = blob;
                    updateAuthenticateButtonState();
                    if (grp.playbackAudio) {
                        grp.playbackAudio.src = URL.createObjectURL(blob);
                        if (grp.playbackContainer) grp.playbackContainer.style.display = 'block';
                    }
                },
                onRecordingStopEmpty: (prefix, grp) => {
                    showResult(getResultElIdForGroup(prefix), `Recording empty. Provide audio.`, 'error');
                    singleAuthAudioSource = null;
                    updateAuthenticateButtonState();
                },
                onRecordingError: (errorName, prefix, grp, isMicAccessError) => {
                    const resultElId = getResultElIdForGroup(prefix);
                    const msg = isMicAccessError ? `Mic Access Error: ${errorName}` : `Recording Error: ${errorName}`;
                    showResult(resultElId, msg, 'error');
                    singleAuthAudioSource = null; // Ensure it's cleared on error
                    updateAuthenticateButtonState();
                },
                onCleanupUI: (prefix, isEnroll) => { // isEnroll will be false
                    updateAuthenticateButtonState();
                }
            });
        });
    }

    if (elGroup.fileInput) {
        setupFileInput(groupPrefix, {
            onFileSelected: (file, prefix, grp) => {
                singleAuthAudioSource = file;
                updateAuthenticateButtonState();
            },
            onFileInvalid: (prefix, grp) => {
                singleAuthAudioSource = null;
                updateAuthenticateButtonState();
            }
        });
    }
    updateAuthenticateButtonState(); // Initial state
}


// --- User List ---
export async function loadUsers() {
    const div = DOMElements.usersListDiv();
    if (!div) return;
    div.innerHTML = '<p>Loading...</p>';
    try {
        const data = await fetchAPI('/users', 'GET'); // GET request
        if (data.success && data.users) {
            if (data.users.length === 0) div.innerHTML = '<p>No users registered.</p>';
            else {
                const ul = document.createElement('ul');
                data.users.forEach(u => {
                    const li = document.createElement('li');
                    li.classList.add('user-item');
                    li.innerHTML = `<strong>${u.name}</strong> (ID:${u.voiceId}) <small>Method:${u.agg_method || 'N/A'}</small><small>Enrolled:${new Date(u.enrolled).toLocaleString()}</small><small>Last Seen:${u.last_seen?new Date(u.last_seen).toLocaleString():'Never'}</small>`;
                    ul.appendChild(li);
                });
                div.innerHTML = '';
                div.appendChild(ul);
            }
        } else {
            div.innerHTML = `<p class="error-text">Error loading users: ${data.error || 'Unknown error'}</p>`;
        }
    } catch (error) {
        div.innerHTML = `<p class="error-text">Fetch error loading users: ${error.message}</p>`;
    }
}

export function initializeUserManagement() {
    if (DOMElements.refreshUsersBtn()) {
        DOMElements.refreshUsersBtn().addEventListener('click', loadUsers);
    }
}

// --- Database Reset ---
async function resetDatabase() {
    if (!confirm("Are you sure you want to reset the entire database? This cannot be undone.")) return;

    showResult('register-result', 'Resetting database...', 'pending');
    showResult('auth-result', '', 'pending'); // Clear auth result
    if (DOMElements.authBenchmarkTableContainer()) DOMElements.authBenchmarkTableContainer().innerHTML = '';
    if (DOMElements.benchmarkTitle()) DOMElements.benchmarkTitle().style.display = 'none';

    try {
        const data = await fetchAPI('/reset-db', 'POST');
        alert(data.message || (data.success ? "Database reset successfully." : "Failed to reset database."));
        if (data.success) {
            showResult('register-result', data.message, 'success');
            // Clear auth tab
            if (DOMElements.authNameInput()) DOMElements.authNameInput().value = '';
            singleAuthAudioSource = null;
            updateAuthenticateButtonState();
            const authAudioPlayback = DOMElements.get('#auth_single_audio-playback');
            if (authAudioPlayback) authAudioPlayback.src = '';
            const authPlaybackContainer = authAudioPlayback?.closest('.audio-player-container');
            if (authPlaybackContainer) authPlaybackContainer.style.display = 'none';
            const authFileInput = DOMElements.get('#auth_single_file-input');
            if (authFileInput) authFileInput.value = null;
            const authFileName = DOMElements.get('#auth_single_file-name');
            if (authFileName) authFileName.textContent = 'No file chosen';

            if (DOMElements.usersListDiv()) DOMElements.usersListDiv().innerHTML = '<p>Database reset. No users.</p>';
            resetEnrollmentStateForDBReset(); // Call reset from enrollment module
        } else {
            showResult('register-result', `Error: ${data.error || 'Failed to reset.'}`, 'error');
        }
    } catch (error) {
        alert(`Error resetting database: ${error.message}`);
        showResult('register-result', `Error: ${error.message}`, 'error');
    }
}


export function initializeDBReset() {
    if (DOMElements.resetDbBtn()) {
        DOMElements.resetDbBtn().addEventListener('click', resetDatabase);
    }
}

// For inputToggler
export function onAuthAudioChangeForToggler(targetPrefix, audioData) {
    if (targetPrefix === 'auth_single') { // Ensure it's for the auth section
        singleAuthAudioSource = audioData;
        updateAuthenticateButtonState();
    }
}
export function getAuthNameForToggler() {
    return DOMElements.authNameInput() ? DOMElements.authNameInput().value.trim() : null;
}

// ... (other exports and functions) ...
export function getAuthAudioStatus() {
    return !!singleAuthAudioSource;
}