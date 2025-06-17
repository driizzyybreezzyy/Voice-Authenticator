// static/js/modules/api.js
export async function handleFetchResponse(response) {
    const contentType = response.headers.get("content-type");
    let responseData;

    if (contentType && contentType.indexOf("application/json") !== -1) {
        responseData = await response.json();
    } else if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}. Response not JSON.`);
    }

    if (!response.ok) {
        if (response.status === 400 || response.status === 409) {
            return responseData;
        } else {
            const errorMsg = responseData?.e || responseData?.error || responseData?.message || response.statusText;
            throw new Error(errorMsg || `HTTP Error: ${response.status}`);
        }
    }
    return responseData;
}

// Optional generic fetch wrapper
export async function fetchAPI(endpoint, method = 'POST', bodyData = null) {
    const options = {
        method: method,
        headers: { 'Content-Type': 'application/json' },
    };
    if (bodyData) {
        options.body = JSON.stringify(bodyData);
    }

    try {
        const response = await fetch(endpoint, options);
        return handleFetchResponse(response);
    } catch (error) {
        console.error(`FetchAPI Error (${endpoint}):`, error);
        // Re-throw or return a standardized error object for the caller to handle
        throw error; // Or return { success: false, error: error.message };
    }
}