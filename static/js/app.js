// static/js/app.js (Main Entry Point)
import { MIN_REQUIRED_ENROLLMENT_SAMPLES, ENROLLMENT_AUDIO_DURATION_S, DOMElements } from './modules/config.js';
import { initializeTabs, initializeInputToggles } from './modules/ui.js';
import { 
    initializeEnrollmentSystem, 
    onEnrollmentAudioChangeForToggler, 
    setLoadUsersCallback,
    getEnrollmentAudioStatusForSample // NEW callback needed by UI
} from './modules/enrollment.js';
import { 
    initializeAuthenticationSystem, 
    initializeUserManagement, 
    loadUsers, 
    initializeDBReset, 
    onAuthAudioChangeForToggler, 
    getAuthNameForToggler,
    getAuthAudioStatus // NEW callback needed by UI
} from './modules/auth.js';
import { stopActiveRecorderIfAny } from './modules/audio.js';


document.addEventListener('DOMContentLoaded', function() {
    console.log("App JS Initializing. Min Samples:", MIN_REQUIRED_ENROLLMENT_SAMPLES, "Suggested Min Duration:", ENROLLMENT_AUDIO_DURATION_S);

    setLoadUsersCallback(loadUsers);
    initializeTabs(loadUsers);

    const inputTogglerCallbacks = {
        onEnrollmentAudioChange: onEnrollmentAudioChangeForToggler,
        onAuthAudioChange: onAuthAudioChangeForToggler,
        getAuthName: getAuthNameForToggler,
        stopActiveRecorder: stopActiveRecorderIfAny,
        getEnrollmentAudioStatus: getEnrollmentAudioStatusForSample, // Pass new function
        getAuthAudioStatus: getAuthAudioStatus // Pass new function
    };
    initializeInputToggles(inputTogglerCallbacks);

    initializeEnrollmentSystem();
    initializeAuthenticationSystem();
    initializeUserManagement();
    initializeDBReset();

    const initialActiveTabEl = DOMElements.get('.tab-btn.active');
    if (initialActiveTabEl && initialActiveTabEl.dataset.tab === 'users') {
        loadUsers();
    }
});