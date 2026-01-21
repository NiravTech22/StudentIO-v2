import React from 'react'
import ReactDOM from 'react-dom/client'
import StudentIODashboard from './dashboard'
import './index.css'

class ErrorBoundary extends React.Component<{ children: React.ReactNode }, { hasError: boolean, error: any }> {
    constructor(props) { super(props); this.state = { hasError: false, error: null }; }
    static getDerivedStateFromError(error) { return { hasError: true, error }; }
    render() {
        if (this.state.hasError) {
            return (
                <div style={{ color: 'red', padding: 20, background: '#1a1a1a', height: '100vh' }}>
                    <h1>CRASHED</h1>
                    <pre>{this.state.error?.toString()}</pre>
                </div>
            );
        }
        return this.props.children;
    }
}

ReactDOM.createRoot(document.getElementById('root')!).render(
    <React.StrictMode>
        <ErrorBoundary>
            <StudentIODashboard />
        </ErrorBoundary>
    </React.StrictMode>,
)
