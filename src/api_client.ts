export const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8080';

export async function apiFetch(endpoint: string, options: RequestInit = {}) {
    // Use relative path for proxy during dev, or full URL for prod if env var is set
    // However, Vite proxy handles relative paths '/api/...', so we should prefer that if no VITE_API_URL is forced.
    // Actually, for a separate backend deployment, we need the full URL.

    let url = endpoint;
    if (!endpoint.startsWith('http')) {
        // If we have a Configured URL (Prod), prepend it.
        // If Dev, we use the proxy (which is usually relative '/api'), but if we are hardcoded to localhost:8080 in code, we should change that.

        // Logic:
        // If process.env.VITE_API_URL is set -> Use it + endpoint
        // Else -> Use '/api' + endpoint (assuming proxy or relative path)
        const base = import.meta.env.VITE_API_URL || '';
        url = `${base}${endpoint}`;
    }

    const res = await fetch(url, options);
    if (!res.ok) {
        throw new Error(`API Error: ${res.status}`);
    }
    return res.json();
}
