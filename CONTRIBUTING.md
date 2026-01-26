
# Contributing to StudentIO

We welcome contributions! Please follow these guidelines to keep the codebase clean and production-ready.

## Development Workflow

1.  **Modular Development**:
    *   If you are changing the UI, work strictly in `frontend/`.
    *   If you are changing the AI prompts, work in `core/ai_service/main.py`.
    *   **Do not mix logic.** Keep the API Gateway (`backend/`) dumb; it just passes messages.

2.  **Linting & Formatting**:
    *   Frontend: Run `npm run lint`.
    *   Python: Follow PEP-8.

3.  **Testing**:
    *   Before pushing, always run `.\run_all.ps1` to ensure you haven't broken the startup sequence.
    *   Verify the "Demo Triggers" still work.

## Pull Requests
*   Use conventional commits (e.g., `feat: add new physics module`).
*   Attach a screenshot if changing the UI.
