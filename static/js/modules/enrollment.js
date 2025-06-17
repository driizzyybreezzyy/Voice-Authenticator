// static/js/modules/enrollment.js
import {
    MIN_REQUIRED_ENROLLMENT_SAMPLES,
    DOMElements,
    getElementsForGroup,
    getResultElIdForGroup
} from './config.js';
import {
    fetchAPI
} from './api.js';
import {
    showResult,
    displaySampleQualityFeedback
} from './ui.js';
import {
    startRecording,
    setupFileInput
} from './audio.js';

let collectedEnrollmentAudioDataURIs = [];
// currentEnrollmentSampleBlockCount is no longer needed as source of truth,
// collectedEnrollmentAudioDataURIs.length will be.
let loadUsersCallback;

export function setLoadUsersCallback(callback) {
    loadUsersCallback = callback;
}

function updateSamplesCollectedCountAndButtons() {
    const validAudiosCount = collectedEnrollmentAudioDataURIs.filter(Boolean).length;
    if (DOMElements.samplesCollectedCountEl()) DOMElements.samplesCollectedCountEl().textContent = validAudiosCount;
    if (DOMElements.registerBtn()) DOMElements.registerBtn().disabled = validAudiosCount < MIN_REQUIRED_ENROLLMENT_SAMPLES;

    const addBtn = DOMElements.addEnrollmentSampleBtn();
    if (addBtn) {
        // Enable 'Add Sample' if all existing samples have audio OR if there are no samples yet.
        // Or if max samples not yet reached (if you implement a max).
        const allFilledOrNone = collectedEnrollmentAudioDataURIs.every(audio => !!audio) || collectedEnrollmentAudioDataURIs.length === 0;
        addBtn.disabled = !allFilledOrNone && collectedEnrollmentAudioDataURIs.length > 0 && !collectedEnrollmentAudioDataURIs[collectedEnrollmentAudioDataURIs.length - 1];
    }
}

// Function to render all sample blocks based on collectedEnrollmentAudioDataURIs
// This will be called on init, after adding a new slot, and after deleting a slot.
async function renderAllSampleBlocks() {
    const samplesArea = DOMElements.enrollmentSamplesArea();
    if (!samplesArea) return;
    samplesArea.innerHTML = ''; // Clear existing blocks

    for (let i = 0; i < collectedEnrollmentAudioDataURIs.length; i++) {
        const audioDataURI = collectedEnrollmentAudioDataURIs[i];
        // Pass the current index 'i' which will be used for IDs and labels
        const sampleBlock = await _buildSingleSampleBlockDOM(i, audioDataURI);
        samplesArea.appendChild(sampleBlock);
        _attachListenersToSampleBlock(i); // Attach listeners using the correct index
    }
    updateSamplesCollectedCountAndButtons();
}

// Builds the DOM for a single sample block but does NOT append it or attach listeners
async function _buildSingleSampleBlockDOM(sampleIndex, audioDataURI = null) {
    const sampleIdPrefix = `enroll_sample_${sampleIndex}`;
    const sampleDiv = document.createElement('div');
    sampleDiv.className = 'enrollment-sample-group form-group';
    sampleDiv.id = `${sampleIdPrefix}_group`; // Group div still needs an ID for getElementsForGroup if used elsewhere

    const hasAudio = !!audioDataURI;

    sampleDiv.innerHTML = `
        <label>Voice Sample ${sampleIndex + 1}</label>
        
        <div class="input-method-toggle" id="${sampleIdPrefix}_input-method-toggle" style="display: ${hasAudio ? 'none' : 'flex'};">
            <button class="toggle-btn active" data-input="record" data-target="${sampleIdPrefix}">Record</button>
            <button class="toggle-btn" data-input="upload" data-target="${sampleIdPrefix}">Upload</button>
        </div>
        
        <div class="voice-input-area record-active" id="${sampleIdPrefix}-input-area" style="display: ${hasAudio ? 'none' : 'block'};">
            <div class="voice-recorder">
                <button id="${sampleIdPrefix}_record-btn" class="record-mic-btn" ${hasAudio ? 'disabled' : ''}><i class="fas fa-microphone"></i></button>
                <p id="${sampleIdPrefix}_record-instruction" class="record-instruction">Record Sample ${sampleIndex + 1}</p>
                <div id="${sampleIdPrefix}_recording-indicator" class="recording-indicator" style="display: none;">REC</div>
            </div>
            <div class="voice-uploader" style="display: none;">
                <input type="file" id="${sampleIdPrefix}_file-input" accept="audio/*" style="display:none;"  ${hasAudio ? 'disabled' : ''}>
                <label for="${sampleIdPrefix}_file-input" class="file-upload-label" style="${hasAudio ? 'pointer-events: none; opacity: 0.7;' : ''}"><i class="fas fa-upload"></i> Choose File</label>
                <span id="${sampleIdPrefix}_file-name" class="file-name-display">No file chosen</span>
            </div>
        </div>

        <div class="audio-player-wrapper" id="${sampleIdPrefix}_record_player_wrapper" style="display: ${hasAudio && !audioDataURI?.startsWith('data:application') ? 'block' : 'none'};">
            <div class="audio-player-container" id="${sampleIdPrefix}_record_player_container">
                <audio id="${sampleIdPrefix}_audio-playback" controls ${hasAudio ? `src="${audioDataURI}"` : ''}></audio>
            </div>
            <button id="${sampleIdPrefix}_delete_btn_record" class="btn-delete-sample-icon"><i class="fas fa-times"></i></button>
        </div>
        <div id="${sampleIdPrefix}_record_snr_display" class="snr-display" style="display: none;"></div> 
        
        <div class="audio-player-wrapper" id="${sampleIdPrefix}_upload_player_wrapper" style="display: ${hasAudio && audioDataURI?.startsWith('data:application') ? 'block' : 'none'};">
             <!-- Heuristic: uploaded files often don't have specific audio mime types like recorded webm/opus -->
             <!-- A better way would be to store sourceType with the audioDataURI -->
            <div class="audio-player-container upload-audio-player-container" id="${sampleIdPrefix}_upload_player_container">
                <audio id="${sampleIdPrefix}_upload-playback" controls ${hasAudio ? `src="${audioDataURI}"` : ''}></audio>
            </div>
            <button id="${sampleIdPrefix}_delete_btn_upload" class="btn-delete-sample-icon"><i class="fas fa-times"></i></button>
        </div>
        <div id="${sampleIdPrefix}_upload_snr_display" class="snr-display" style="display: none;"></div>

        <div id="${sampleIdPrefix}_quality_status" class="sample-quality-status"></div>
        <hr style="margin-top: 15px;">`;
    return sampleDiv;
}

// Attaches event listeners to a newly created/rendered sample block
async function _attachListenersToSampleBlock(sampleIndex) {
    const sampleIdPrefix = `enroll_sample_${sampleIndex}`;
    const elGroup = getElementsForGroup(sampleIdPrefix); // config.js provides these elements

    if (elGroup.recordBtn) {
        elGroup.recordBtn.addEventListener('click', () => {
            // The 'sampleIndex' from the closure is the key here
            if (collectedEnrollmentAudioDataURIs[sampleIndex]) {
                showResult('register-result', `Sample ${sampleIndex + 1} already recorded. Delete it first.`, 'error');
                return;
            }
            startRecording(sampleIdPrefix, { // startRecording still uses prefix for its internal UI updates
                onRecordingStartUI: (grp) => {
                    /* ... */
                    grp.recordBtn.classList.add('recording');
                    grp.recordBtn.innerHTML = '<i class="fas fa-stop"></i>';
                    if (grp.recordInstruction) grp.recordInstruction.textContent = 'Recording... Click Stop';
                    if (grp.recordingIndicator) grp.recordingIndicator.style.display = 'flex';
                    if (DOMElements.addEnrollmentSampleBtn()) DOMElements.addEnrollmentSampleBtn().disabled = true;
                    if (grp.recordPlayerWrapper) grp.recordPlayerWrapper.style.display = 'none';
                    if (grp.uploadPlayerWrapper) grp.uploadPlayerWrapper.style.display = 'none';
                },
                onRecordingStopSuccess: (blob, prefixForAudioJs, grpFromAudioJs) => { // prefixForAudioJs is sampleIdPrefix
                    updateCollectedEnrollmentAudio(sampleIndex, blob, 'record'); // Use sampleIndex
                    // Player src and visibility is now handled by renderAllSampleBlocks via updateCollected...
                },
                onRecordingStopEmpty: (prefixForAudioJs, grpFromAudioJs) => {
                    showResult(getResultElIdForGroup(prefixForAudioJs), `Recording empty for Sample ${sampleIndex + 1}.`, 'error');
                    updateCollectedEnrollmentAudio(sampleIndex, null);
                },
                onRecordingError: (errorName, prefixForAudioJs, grpFromAudioJs, isMicAccessError) => {
                    const msg = isMicAccessError ? `Mic Access Error (Sample ${sampleIndex + 1}): ${errorName}` : `Recording Error (Sample ${sampleIndex + 1}): ${errorName}`;
                    showResult(getResultElIdForGroup(prefixForAudioJs), msg, 'error');
                    updateCollectedEnrollmentAudio(sampleIndex, null);
                },
                onCleanupUI: (prefixForAudioJs, isEnroll) => {
                    updateSamplesCollectedCountAndButtons(); // Still useful for overall button state
                }
            });
        });
    }

    if (elGroup.fileInput) {
        setupFileInput(sampleIdPrefix, { // setupFileInput also uses prefix
            onFileSelected: (file, prefixForAudioJs, grpFromAudioJs) => {
                if (collectedEnrollmentAudioDataURIs[sampleIndex]) {
                    /* ... */
                    return;
                }
                updateCollectedEnrollmentAudio(sampleIndex, file, 'upload'); // Use sampleIndex
            },
            onFileInvalid: (prefixForAudioJs, grpFromAudioJs) => {
                updateCollectedEnrollmentAudio(sampleIndex, null);
            }
        });
    }

    if (elGroup.deleteBtnRecord) {
        elGroup.deleteBtnRecord.addEventListener('click', () => handleDeleteEnrollmentSample(sampleIndex));
    }
    if (elGroup.deleteBtnUpload) {
        elGroup.deleteBtnUpload.addEventListener('click', () => handleDeleteEnrollmentSample(sampleIndex));
    }

    // Initialize toggler for this specific block
    const toggleDiv = DOMElements.get(`#${sampleIdPrefix}_input-method-toggle`);
    const activeToggleBtn = toggleDiv?.querySelector('.toggle-btn.active');
    if (activeToggleBtn) {
        const {
            handleToggleClickForNewlyAddedBlock
        } = await import('../modules/ui.js');
        if (typeof handleToggleClickForNewlyAddedBlock === 'function') {
            handleToggleClickForNewlyAddedBlock(activeToggleBtn, true); // true for isInitialSetup
        }
    }
}


// Called when audio is successfully processed for a specific sampleIndex
function updateCollectedEnrollmentAudio(sampleIndex, audioBlobOrFile, audioSourceType) {
    if (audioBlobOrFile instanceof Blob || audioBlobOrFile instanceof File) {
        const reader = new FileReader();
        reader.readAsDataURL(audioBlobOrFile);
        reader.onloadend = () => {
            // Store a simple object if you need to remember the source type
            // For simplicity now, just storing the URI. The render function will try to guess.
            collectedEnrollmentAudioDataURIs[sampleIndex] = reader.result;
            renderAllSampleBlocks(); // Re-render to show the player and hide inputs
        };
        reader.onerror = () => {
            collectedEnrollmentAudioDataURIs[sampleIndex] = null;
            showResult('register-result', `Error reading audio for sample ${sampleIndex + 1}`, 'error');
            renderAllSampleBlocks(); // Re-render to show inputs again
        };
    } else { // audioBlobOrFile is null
        collectedEnrollmentAudioDataURIs[sampleIndex] = null;
        renderAllSampleBlocks(); // Re-render to reflect the cleared audio
    }
}

function handleDeleteEnrollmentSample(indexToDelete) {
    if (indexToDelete < 0 || indexToDelete >= collectedEnrollmentAudioDataURIs.length) return;

    collectedEnrollmentAudioDataURIs.splice(indexToDelete, 1); // Remove from array
    // No need to decrement currentEnrollmentSampleBlockCount, as length of array is the source of truth

    showResult('register-result', `Sample ${indexToDelete + 1} deleted.`, 'success');
    renderAllSampleBlocks(); // Re-render all blocks with new indices and count
}

function handleAddEnrollmentSampleClick() {
    // Check if the last sample (if any) is filled before adding a new one
    if (collectedEnrollmentAudioDataURIs.length > 0 && !collectedEnrollmentAudioDataURIs[collectedEnrollmentAudioDataURIs.length - 1]) {
        showResult('register-result', "Please complete the current sample before adding another.", "error");
        return;
    }
    collectedEnrollmentAudioDataURIs.push(null); // Add a slot for a new sample
    renderAllSampleBlocks(); // Re-render to include the new empty block at the end
}

// --- handleFullRegistration, initializeEnrollmentSystem, etc. remain largely the same ---
// --- but they will now rely on renderAllSampleBlocks to create the initial UI ---

async function handleFullRegistration() {
    if (DOMElements.registerResultEl()) DOMElements.registerResultEl().style.display = 'none';
    // Clear previous quality feedback by re-rendering (or clear manually if renderAllSampleBlocks doesn't do it)
    collectedEnrollmentAudioDataURIs.forEach((uri, index) => displaySampleQualityFeedback(index, null));


    const name = DOMElements.registerNameInput().value.trim();
    if (!name) {
        showResult('register-result', 'Please enter your name.', 'error');
        return;
    }
    const validAudioDataURIs = collectedEnrollmentAudioDataURIs.filter(Boolean); // Filter out nulls
    if (validAudioDataURIs.length < MIN_REQUIRED_ENROLLMENT_SAMPLES) {
        showResult('register-result', `Provide at least ${MIN_REQUIRED_ENROLLMENT_SAMPLES} valid audio samples.`, 'error');
        return;
    }

    showResult('register-result', 'Registering & checking samples...', 'pending');
    if (DOMElements.registerBtn()) DOMElements.registerBtn().disabled = true;
    if (DOMElements.addEnrollmentSampleBtn()) DOMElements.addEnrollmentSampleBtn().disabled = true;

    try {
        // Send only the valid audio URIs
        const data = await fetchAPI('/register', 'POST', {
            name: name,
            audios: validAudioDataURIs
        });

        if (data.sample_reports && Array.isArray(data.sample_reports)) {
            // The backend returns reports based on the order of audios it received.
            // We need to map this back to the original *overall* sample slots if some were empty.
            // This is tricky if empty samples were submitted.
            // For now, assuming backend reports correspond to the *validAudioDataURIs* sent.
            // A more robust way: backend returns original_index for each report.
            // Or, frontend clears quality for all, then iterates valid ones.
            let validSampleCounter = 0;
            collectedEnrollmentAudioDataURIs.forEach((uri, originalIndex) => {
                if (uri) { // If this slot had valid audio
                    const reportForThisSlot = data.sample_reports.find(r => r.sample_index === (validSampleCounter + 1));
                    if (reportForThisSlot) {
                        // Adjust report's sample_index to match the original slot's 0-based index for display
                        displaySampleQualityFeedback(originalIndex, {
                            ...reportForThisSlot,
                            sample_index: originalIndex + 1
                        });
                    }
                    validSampleCounter++;
                } else {
                    displaySampleQualityFeedback(originalIndex, null); // Clear for empty slots
                }
            });
        }

        if (data.s === true || data.success === true) {
            showResult('register-result', `User '${data.name}' registered. Check sample statuses.`, 'success');
            if (DOMElements.registerNameInput()) DOMElements.registerNameInput().value = '';
            if (typeof loadUsersCallback === 'function') loadUsersCallback();
            // Optionally, reset the enrollment form fully after success:
            // collectedEnrollmentAudioDataURIs = [];
            // renderAllSampleBlocks(); // This would clear all and add one new empty slot
        } else {
            let errorMsg = data.e || data.error || 'Registration failed. See sample details.';
            showResult('register-result', errorMsg, 'error');
        }
    } catch (error) {
        showResult('register-result', `Client/Network Error: ${error.message}`, 'error');
    } finally {
        updateSamplesCollectedCountAndButtons();
        if (DOMElements.registerBtn()) {
            const success = DOMElements.registerResultEl()?.classList.contains('success');
            const enoughSamples = collectedEnrollmentAudioDataURIs.filter(Boolean).length >= MIN_REQUIRED_ENROLLMENT_SAMPLES;
            DOMElements.registerBtn().disabled = success || !enoughSamples;
        }
        if (DOMElements.addEnrollmentSampleBtn()) {
            const allFilledOrNone = collectedEnrollmentAudioDataURIs.every(audio => !!audio) || collectedEnrollmentAudioDataURIs.length === 0;
            DOMElements.addEnrollmentSampleBtn().disabled = !allFilledOrNone && collectedEnrollmentAudioDataURIs.length > 0 && !collectedEnrollmentAudioDataURIs[collectedEnrollmentAudioDataURIs.length - 1];
        }
    }
}

export function initializeEnrollmentSystem() {
    if (DOMElements.enrollmentSamplesArea()) {
        collectedEnrollmentAudioDataURIs = []; // Start with an empty array
        collectedEnrollmentAudioDataURIs.push(null); // Add the first empty slot
        renderAllSampleBlocks(); // Render it
    }
    if (DOMElements.addEnrollmentSampleBtn()) DOMElements.addEnrollmentSampleBtn().addEventListener('click', handleAddEnrollmentSampleClick);
    if (DOMElements.registerBtn()) DOMElements.registerBtn().addEventListener('click', handleFullRegistration);
    updateSamplesCollectedCountAndButtons();
}

export function onEnrollmentAudioChangeForToggler(targetPrefix, audioData) {
    // The toggler typically calls this with audioData = null when switching input methods
    // to clear any uncommitted audio for THAT slot.
    // We need the actual index for this slot.
    const parts = targetPrefix.split('_');
    const sampleIndex = parseInt(parts[parts.length - 1]);

    if (!isNaN(sampleIndex) && sampleIndex >= 0 && sampleIndex < collectedEnrollmentAudioDataURIs.length) {
        if (audioData === null) {
            // Only clear if it wasn't already committed / has a full data URI
            if (collectedEnrollmentAudioDataURIs[sampleIndex] && typeof collectedEnrollmentAudioDataURIs[sampleIndex] === 'string') {
                // It means it was a proper audio URI, toggler should not clear it.
                // This function is more for clearing transient blobs from audio.js before they become URIs.
                // For now, let's assume if the toggler says clear, we clear, then re-render.
            }
            collectedEnrollmentAudioDataURIs[sampleIndex] = null;
            renderAllSampleBlocks(); // Re-render to show the input UI for this slot
        } else {
            console.warn("onEnrollmentAudioChangeForToggler received audio data, which is not its primary use.");
        }
    }
}

export function resetEnrollmentStateForDBReset() {
    if (DOMElements.registerNameInput()) DOMElements.registerNameInput().value = '';
    collectedEnrollmentAudioDataURIs = [];
    collectedEnrollmentAudioDataURIs.push(null); // Start with one empty slot
    renderAllSampleBlocks();
    updateSamplesCollectedCountAndButtons();
}

export function getEnrollmentAudioStatusForSample(sampleIndex) {
    // The sampleIndex here is the logical index from the UI (0-based)
    if (sampleIndex >= 0 && sampleIndex < collectedEnrollmentAudioDataURIs.length) {
        return !!collectedEnrollmentAudioDataURIs[sampleIndex];
    }
    return false; // Or throw error if index out of bounds
}