// static/js/modules/audio.js
import { DOMElements, getResultElIdForGroup, getElementsForGroup, ENROLLMENT_AUDIO_DURATION_S } from './config.js';
import { showResult, setActiveMediaRecorderForToggler } from './ui.js';

let activeMediaRecorder = null;
let activeAudioStream = null;
let currentRecordingChunks = [];

export function getActiveRecorderInstance() {
    return activeMediaRecorder;
}

export function stopActiveRecorderIfAny() {
    if (activeMediaRecorder && activeMediaRecorder.state === "recording") {
        activeMediaRecorder.stop();
    }
}

// Callbacks: onRecordingStartUI, onRecordingStopSuccess, onRecordingStopEmpty, onRecordingError, onCleanupUI
export function startRecording(targetPrefix, callbacks) {
    const elGroup = getElementsForGroup(targetPrefix);
    if (!elGroup || !elGroup.recordBtn) return;

    if (activeMediaRecorder && activeMediaRecorder.state === "recording" && activeMediaRecorder.associatedPrefix === targetPrefix) {
        activeMediaRecorder.stop(); // cleanupAfterRecording will be called by onstop
        return;
    }
    if (activeMediaRecorder && activeMediaRecorder.state === "recording") {
        activeMediaRecorder.stop(); // Stop other active recording
    }

    currentRecordingChunks = [];
    elGroup.recordBtn.disabled = true;

    if (elGroup.playbackContainer) elGroup.playbackContainer.style.display = 'none';
    if (elGroup.playbackAudio) elGroup.playbackAudio.src = '';

    navigator.mediaDevices.getUserMedia({ audio: true })
        .then(stream => {
            activeAudioStream = stream;
            let options = { mimeType: 'audio/webm;codecs=opus' };
            if (!MediaRecorder.isTypeSupported(options.mimeType) && MediaRecorder.isTypeSupported('audio/wav')) {
                options = { mimeType: 'audio/wav' };
            } else if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options = {};
            }

            activeMediaRecorder = new MediaRecorder(stream, options);
            activeMediaRecorder.associatedPrefix = targetPrefix;
            setActiveMediaRecorderForToggler(activeMediaRecorder); // Inform ui.js

            // UI update for recording start
            callbacks.onRecordingStartUI(elGroup, targetPrefix);

            activeMediaRecorder.ondataavailable = e => {
                if (e.data.size > 0) currentRecordingChunks.push(e.data);
            };

            activeMediaRecorder.onstop = () => {
                if (activeMediaRecorder && activeMediaRecorder.associatedPrefix !== targetPrefix && activeMediaRecorder.state !== "inactive") {
                    return; // Stale onstop
                }
                cleanupAfterRecording(elGroup, targetPrefix, callbacks.onCleanupUI); // Pass onCleanupUI

                if (currentRecordingChunks.length > 0) {
                    const blob = new Blob(currentRecordingChunks, { type: activeMediaRecorder.mimeType || 'application/octet-stream' });
                    callbacks.onRecordingStopSuccess(blob, targetPrefix, elGroup);
                } else {
                    callbacks.onRecordingStopEmpty(targetPrefix, elGroup);
                }
                if (activeMediaRecorder && activeMediaRecorder.associatedPrefix === targetPrefix) {
                    activeMediaRecorder = null;
                    setActiveMediaRecorderForToggler(null);
                }
            };

            activeMediaRecorder.onerror = (e) => {
                const errorName = e.error ? e.error.name : 'Unknown Error';
                cleanupAfterRecording(elGroup, targetPrefix, callbacks.onCleanupUI);
                callbacks.onRecordingError(errorName, targetPrefix, elGroup);
                if (activeMediaRecorder && activeMediaRecorder.associatedPrefix === targetPrefix) {
                    activeMediaRecorder = null;
                    setActiveMediaRecorderForToggler(null);
                }
            };

            activeMediaRecorder.start();
            elGroup.recordBtn.disabled = false;

        }).catch(err => {
            elGroup.recordBtn.disabled = false;
            cleanupAfterRecording(elGroup, targetPrefix, callbacks.onCleanupUI); // Ensure cleanup even on getUserMedia error
            callbacks.onRecordingError(err.message, targetPrefix, elGroup, true); // true for mic access error
        });
}

function cleanupAfterRecording(elGroup, targetPrefix, onCleanupUICallback) {
    if (activeAudioStream) {
        activeAudioStream.getTracks().forEach(t => t.stop());
        activeAudioStream = null;
    }
    // UI reset
    if (elGroup.recordBtn) {
        elGroup.recordBtn.disabled = false;
        elGroup.recordBtn.classList.remove('recording');
        elGroup.recordBtn.innerHTML = '<i class="fas fa-microphone"></i>';
    }
    if (elGroup.recordingIndicator) elGroup.recordingIndicator.style.display = 'none';
    if (elGroup.recordInstruction) {
        const sampleIndexText = elGroup.isEnrollSample ? `Sample ${elGroup.sampleIndex + 1}` : "Voice Sample";
        elGroup.recordInstruction.textContent = `Record ${sampleIndexText}`; // Removed SUGGESTED_MIN_AUDIO_S reference
    }
    if (typeof onCleanupUICallback === 'function') {
        onCleanupUICallback(targetPrefix, elGroup.isEnrollSample);
    }
}


// --- File Uploader ---
// Callbacks: onFileSelected, onFileInvalid
export function setupFileInput(targetPrefix, callbacks) {
    const elGroup = getElementsForGroup(targetPrefix);
    if (!elGroup || !elGroup.fileInput) return;

    elGroup.fileInput.addEventListener('change', function(event) {
        const file = event.target.files[0];
        if (file && file.type.startsWith('audio/')) {
            elGroup.fileNameDisplay.textContent = file.name;
            const reader = new FileReader();
            reader.onload = e => {
                if (elGroup.uploadPlaybackAudio) {
                    elGroup.uploadPlaybackAudio.src = e.target.result;
                    if (elGroup.uploadPlaybackContainer) elGroup.uploadPlaybackContainer.style.display = 'block';
                }
            };
            reader.readAsDataURL(file);
            callbacks.onFileSelected(file, targetPrefix, elGroup);
        } else {
            if (file) showResult(getResultElIdForGroup(targetPrefix), "Invalid file type.", "error");
            elGroup.fileNameDisplay.textContent = 'No file chosen';
            this.value = null;
            if (elGroup.uploadPlaybackAudio) {
                elGroup.uploadPlaybackAudio.src = '';
                if (elGroup.uploadPlaybackContainer) elGroup.uploadPlaybackContainer.style.display = 'none';
            }
            callbacks.onFileInvalid(targetPrefix, elGroup);
        }
    });
}