// static/js/modules/config.js
// (Content from previous correct version - no new changes for this specific delete feature)

export const MIN_REQUIRED_ENROLLMENT_SAMPLES = window.MIN_REQUIRED_ENROLLMENT_SAMPLES || 3;
export const ENROLLMENT_AUDIO_DURATION_S = window.ENROLLMENT_AUDIO_DURATION_S || 5; // SUGGESTED_MIN_AUDIO_S
export const APP_CONFIG = window.APP_CONFIG || { MIN_SNR_DB_THRESHOLD: 10.0 };

export const DOMElements = {
    get: (selector) => document.querySelector(selector),
    getAll: (selector) => document.querySelectorAll(selector),

    // General
    tabBtns: () => DOMElements.getAll('.tab-btn'),
    tabContents: () => DOMElements.getAll('.tab-content'),

    // Enrollment
    addEnrollmentSampleBtn: () => DOMElements.get('#add-enrollment-sample-btn'),
    enrollmentSamplesArea: () => DOMElements.get('#enrollment-samples-area'),
    samplesCollectedCountEl: () => DOMElements.get('#samples-collected-count'),
    registerNameInput: () => DOMElements.get('#register-name-input'),
    registerBtn: () => DOMElements.get('#register-btn'),
    registerResultEl: () => DOMElements.get('#register-result'),

    // Authentication
    authNameInput: () => DOMElements.get('#auth-name-input'),
    authenticateBtn: () => DOMElements.get('#authenticate-btn'),
    authResultEl: () => DOMElements.get('#auth-result'),
    authBenchmarkTableContainer: () => DOMElements.get('#auth-benchmark-table-container'),
    benchmarkTitle: () => DOMElements.get('#benchmark-title'),

    // Users
    refreshUsersBtn: () => DOMElements.get('#refresh-users-btn'),
    usersListDiv: () => DOMElements.get('#users-list'),

    // Footer
    resetDbBtn: () => DOMElements.get('#reset-db-btn'),
};

// Helper function structure
export function getElementsForGroup(prefix) {
    return {
        recordBtn: DOMElements.get(`#${prefix}_record-btn`),
        playbackAudio: DOMElements.get(`#${prefix}_audio-playback`), // Player for recorded
        recordPlayerContainer: DOMElements.get(`#${prefix}_record_player_container`),
        recordSnrDisplay: DOMElements.get(`#${prefix}_record_snr_display`),
        // NEW: Wrapper for recorded audio player and its delete X
        recordPlayerWrapper: DOMElements.get(`#${prefix}_record_player_wrapper`),
        // NEW: Delete X button for recorded audio
        deleteBtnRecord: DOMElements.get(`#${prefix}_delete_btn_record`),


        fileInput: DOMElements.get(`#${prefix}_file-input`),
        fileNameDisplay: DOMElements.get(`#${prefix}_file-name`),
        uploadPlaybackAudio: DOMElements.get(`#${prefix}_upload-playback`), // Player for uploaded
        uploadPlayerContainer: DOMElements.get(`#${prefix}_upload_player_container`),
        uploadSnrDisplay: DOMElements.get(`#${prefix}_upload_snr_display`),
        // NEW: Wrapper for uploaded audio player and its delete X
        uploadPlayerWrapper: DOMElements.get(`#${prefix}_upload_player_wrapper`),
        // NEW: Delete X button for uploaded audio
        deleteBtnUpload: DOMElements.get(`#${prefix}_delete_btn_upload`),


        inputMethodToggleDiv: DOMElements.get(`#${prefix}_input-method-toggle`),
        inputArea: DOMElements.get(`#${prefix}-input-area`),
        groupDiv: DOMElements.get(`#${prefix}_group`),
        qualityStatusEl: DOMElements.get(`#${prefix}_quality_status`),

        isEnrollSample: prefix.startsWith('enroll_sample_'),
        sampleIndex: prefix.startsWith('enroll_sample_') ? parseInt(prefix.split('_').pop()) : null
    };
}

export function getResultElIdForGroup(targetPrefix) {
    if (targetPrefix.startsWith('enroll_sample_')) return 'register-result';
    if (targetPrefix.startsWith('auth_single')) return 'auth-result';
    return null;
}

export function getPrimaryButtonForGroup(targetPrefix) {
    if (targetPrefix.startsWith('enroll_sample_')) return DOMElements.registerBtn();
    if (targetPrefix.startsWith('auth_single')) return DOMElements.authenticateBtn();
    return null;
}