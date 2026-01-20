# Implementation Plan - UI Visibility and Interaction Fixes

## Goal
Fix the "all dark" UI issue where content is illegible, and ensure the `StudentCard` component is correctly implemented for `framer-motion` animations.

## User Review Required
> [!IMPORTANT]
> I will be adjusting the color palette to be more readable. The current "dark mode" is too aggressive. I'll move to a slightly lighter dark theme (slate/gray) with higher contrast text.

## Proposed Changes

### Frontend
#### [MODIFY] [dashboard.tsx](file:///c:/Users/nirav/OneDrive/Documents/GitHub/StudentIO-v2/dashboard.tsx)
- **Fix Component Ref**: Wrap `StudentCard` in `React.forwardRef` to resolve the console warning from `framer-motion`.
- **Improve Visibility**:
    - Change background from `#060609` (near pitch black) to a rich dark slate/gray (e.g., `bg-slate-950`).
    - Increase contrast of text elements (change `text-slate-500` to `text-slate-400` or `text-slate-300`).
    - Enhance `StudentCard` background opacity and border to make cards pop against the background.
- **Enhance Interactivity**:
    - Add hover effects to the `StudentCard`.
    - Make the "Belief" grid more visually distinct (brighter colors for active beliefs).

#### [MODIFY] [index.css](file:///c:/Users/nirav/OneDrive/Documents/GitHub/StudentIO-v2/index.css)
- Ensure base styles don't conflict, though the issue seems primarily in Tailwind utility usage in `dashboard.tsx`.

## Verification Plan

### Automated Tests
- **Browser Check**: Use the browser tool to navigate to `http://localhost:5173/` and capture a screenshot to verify:
    - Text is legible.
    - Cards are clearly distinguishable from the background.
    - No console warnings about `ref`.

### Manual Verification
- Visual check of the screenshot to ensure the "wow" factor is preserved but usability is restored.
