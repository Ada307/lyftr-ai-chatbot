import { useState, useEffect, useRef } from 'react';
import './App.css';

const API_BASE_URL = (import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000').replace(/\/+$/, '');

function generateSessionId() {
  if (typeof crypto !== 'undefined' && crypto.randomUUID) {
    return 'session-' + crypto.randomUUID();
  }
  return 'session-' + Date.now() + '-' + Math.random().toString(36).substring(2, 9);
}

export default function App() {
  const [sessionId, setSessionId] = useState(generateSessionId);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('');
  const [isSystemPromptOpen, setIsSystemPromptOpen] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [backendStatus, setBackendStatus] = useState({
    state: 'checking', // 'checking' | 'online' | 'offline'
    keys: 0,
    cacheEnabled: false,
    detail: '',
  });
  const [copiedId, setCopiedId] = useState(null);

  const activeSessionIdRef = useRef(sessionId);
  const abortControllerRef = useRef(null);
  const messagesEndRef = useRef(null);
  const textareaRef = useRef(null);

  // Keep activeSessionIdRef synchronized
  useEffect(() => {
    activeSessionIdRef.current = sessionId;
  }, [sessionId]);

  // Auto-scroll on new message or loading state change
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isLoading]);

  // Check backend health status
  useEffect(() => {
    let isMounted = true;

    const fetchStatus = async () => {
      try {
        const res = await fetch(`${API_BASE_URL}/health`, { method: 'GET' });
        if (!isMounted) return;
        if (res.ok) {
          const data = await res.json();
          setBackendStatus({
            state: 'online',
            keys: data.keys_configured ?? 0,
            cacheEnabled: Boolean(data.cache_enabled),
            detail: '',
          });
        } else {
          const errData = await res.json().catch(() => ({}));
          setBackendStatus({
            state: 'offline',
            keys: errData.keys_configured ?? 0,
            cacheEnabled: Boolean(errData.cache_enabled),
            detail: errData.detail || `Server returned HTTP ${res.status}`,
          });
        }
      } catch {
        if (isMounted) {
          setBackendStatus({
            state: 'offline',
            keys: 0,
            cacheEnabled: false,
            detail: 'Cannot connect to backend',
          });
        }
      }
    };

    fetchStatus();
    const interval = setInterval(fetchStatus, 30000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  // Start a fresh new chat session
  const handleNewChat = () => {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    abortControllerRef.current = new AbortController();
    const freshId = generateSessionId();
    setSessionId(freshId);
    activeSessionIdRef.current = freshId;
    setMessages([]);
    setIsLoading(false);
    setInput('');
  };

  // Copy assistant response text to clipboard
  const handleCopy = (id, text) => {
    if (!navigator.clipboard) return;
    navigator.clipboard.writeText(text).then(() => {
      setCopiedId(id);
      setTimeout(() => setCopiedId(null), 2000);
    });
  };

  // Send message
  const handleSend = async (overrideMessage) => {
    const messageToSend = typeof overrideMessage === 'string' ? overrideMessage : input;
    const trimmed = messageToSend.trim();
    if (!trimmed || isLoading) return;

    const currentSession = activeSessionIdRef.current;
    setInput('');

    // Append user message to conversation list
    const userMessageItem = {
      id: 'usr-' + Date.now() + '-' + Math.random().toString(36).substring(2, 5),
      role: 'user',
      content: trimmed,
    };
    setMessages((prev) => [...prev, userMessageItem]);
    setIsLoading(true);

    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
    }
    const controller = new AbortController();
    abortControllerRef.current = controller;

    try {
      const payload = {
        session_id: currentSession,
        message: trimmed,
        ...(systemPrompt.trim() ? { system_prompt: systemPrompt.trim() } : {}),
      };

      const res = await fetch(`${API_BASE_URL}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(payload),
        signal: controller.signal,
      });

      // Discard response if user initiated a New Chat while awaiting
      if (activeSessionIdRef.current !== currentSession) {
        return;
      }

      if (res.ok) {
        const data = await res.json();
        const assistantMessageItem = {
          id: 'asst-' + Date.now() + '-' + Math.random().toString(36).substring(2, 5),
          role: 'assistant',
          content: data.reply,
          metadata: data.metadata,
        };
        setMessages((prev) => [...prev, assistantMessageItem]);
      } else {
        let errorMessage = `HTTP ${res.status}`;
        try {
          const errData = await res.json();
          if (res.status === 422) {
            errorMessage = `Validation Error (422): ${errData.detail?.[0]?.msg || errData.detail || 'Invalid request'}`;
          } else if (res.status === 503) {
            errorMessage = `Service Unavailable (503): ${errData.detail || 'All LLM providers are currently unavailable'}`;
          } else {
            errorMessage = `Error (${res.status}): ${errData.detail || res.statusText}`;
          }
        } catch {
          errorMessage = `Error (${res.status}): ${res.statusText || 'Request failed'}`;
        }

        const errorItem = {
          id: 'err-' + Date.now() + '-' + Math.random().toString(36).substring(2, 5),
          role: 'error',
          content: errorMessage,
        };
        setMessages((prev) => [...prev, errorItem]);
      }
    } catch (err) {
      if (err.name === 'AbortError') {
        return;
      }
      if (activeSessionIdRef.current === currentSession) {
        const networkErrorItem = {
          id: 'net-err-' + Date.now(),
          role: 'error',
          content: `Network Error: Unable to connect to backend at ${API_BASE_URL}. Ensure the server is running.`,
        };
        setMessages((prev) => [...prev, networkErrorItem]);
      }
    } finally {
      if (activeSessionIdRef.current === currentSession) {
        setIsLoading(false);
      }
    }
  };

  const handleKeyDown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="app-container">
      {/* Top Header */}
      <header className="app-header">
        <div className="brand-section">
          <div className="brand-logo" aria-hidden="true">R</div>
          <div>
            <h1 className="brand-title">RelayChat</h1>
            <p className="brand-subtitle">Resilient LLM Gateway with Caching & Failover</p>
          </div>
        </div>

        <div className="header-controls">
          <div
            className={`status-indicator ${backendStatus.state}`}
            title={backendStatus.detail || (backendStatus.state === 'online' ? `${backendStatus.keys} Keys active • Cache enabled` : 'Backend unavailable')}
          >
            <span className="status-dot" />
            <span>
              {backendStatus.state === 'online'
                ? `Online (${backendStatus.keys} Keys)`
                : backendStatus.state === 'checking'
                ? 'Connecting...'
                : 'Offline'}
            </span>
          </div>

          <button
            type="button"
            className="btn-new-chat"
            onClick={handleNewChat}
            title="Start a fresh chat session"
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <line x1="12" y1="5" x2="12" y2="19" />
              <line x1="5" y1="12" x2="19" y2="12" />
            </svg>
            <span>New Chat</span>
          </button>
        </div>
      </header>

      {/* Collapsible System Prompt Section */}
      <div className="system-prompt-bar">
        <button
          type="button"
          className="system-prompt-toggle"
          onClick={() => setIsSystemPromptOpen(!isSystemPromptOpen)}
        >
          <span>
            ⚙️ System Instructions {systemPrompt.trim() ? '(active)' : '(optional)'}
          </span>
          <span>{isSystemPromptOpen ? '▲ Hide' : '▼ Expand'}</span>
        </button>

        {isSystemPromptOpen && (
          <div className="system-prompt-content">
            <textarea
              data-testid="system-prompt-input"
              className="system-prompt-textarea"
              placeholder="Provide custom system instructions for this session (e.g. 'You are a concise technical assistant')..."
              value={systemPrompt}
              onChange={(e) => setSystemPrompt(e.target.value)}
            />
          </div>
        )}
      </div>

      {/* Messages Stream */}
      <main className="messages-container">
        {messages.length === 0 ? (
          <div className="welcome-screen">
            <div className="welcome-icon">
              <svg width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" />
              </svg>
            </div>
            <h2 className="welcome-title">Welcome to RelayChat</h2>
            <p className="welcome-description">
              Resilient AI conversations backed by automatic API key failover, response caching, and per-turn metadata.
            </p>
            <div className="welcome-suggestions">
              <button
                type="button"
                className="suggestion-btn"
                onClick={() => handleSend('What is the capital of France?')}
              >
                💡 &quot;What is the capital of France?&quot; (Fast verification for response caching)
              </button>
              <button
                type="button"
                className="suggestion-btn"
                onClick={() => handleSend('How does automatic API key failover work in RelayChat?')}
              >
                🛡️ &quot;How does automatic API key failover work in RelayChat?&quot;
              </button>
              <button
                type="button"
                className="suggestion-btn"
                onClick={() => handleSend('Give me 3 tips for efficient LLM response caching.')}
              >
                ⚡ &quot;Give me 3 tips for efficient LLM response caching.&quot;
              </button>
            </div>
          </div>
        ) : (
          messages.map((msg) => {
            if (msg.role === 'user') {
              return (
                <div key={msg.id} className="message-row user">
                  <div className="user-bubble">{msg.content}</div>
                </div>
              );
            }

            if (msg.role === 'error') {
              return (
                <div key={msg.id} className="message-row assistant">
                  <div className="error-bubble">
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                      <circle cx="12" cy="12" r="10" />
                      <line x1="12" y1="8" x2="12" y2="12" />
                      <line x1="12" y1="16" x2="12.01" y2="16" />
                    </svg>
                    <div>
                      <div className="error-title">Request Issue</div>
                      <div>{msg.content}</div>
                    </div>
                  </div>
                </div>
              );
            }

            // Assistant message: MUST contain all five meta-* elements inside data-testid="assistant-message"
            return (
              <div key={msg.id} className="message-row assistant">
                <div className="assistant-bubble" data-testid="assistant-message">
                  <div className="assistant-header">
                    <span className="assistant-label">
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                        <rect x="3" y="11" width="18" height="10" rx="2" />
                        <circle cx="12" cy="5" r="2" />
                        <path d="M12 7v4" />
                        <line x1="8" y1="16" x2="8" y2="16" />
                        <line x1="16" y1="16" x2="16" y2="16" />
                      </svg>
                      RelayChat Assistant
                    </span>
                    <button
                      type="button"
                      className={`btn-copy ${copiedId === msg.id ? 'copied' : ''}`}
                      onClick={() => handleCopy(msg.id, msg.content)}
                      title="Copy response text"
                    >
                      {copiedId === msg.id ? (
                        <>
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                            <polyline points="20 6 9 17 4 12" />
                          </svg>
                          <span>Copied!</span>
                        </>
                      ) : (
                        <>
                          <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                            <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                            <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                          </svg>
                          <span>Copy</span>
                        </>
                      )}
                    </button>
                  </div>

                  <div className="assistant-content">{msg.content}</div>

                  {msg.metadata && (
                    <div className="metadata-panel">
                      {/* 1. meta-source: must contain exactly LIVE_INFERENCE or LLM_CACHE */}
                      <span
                        data-testid="meta-source"
                        className={`badge-source ${msg.metadata.resolution_source === 'LLM_CACHE' ? 'cache' : 'live'}`}
                      >
                        {msg.metadata.resolution_source}
                      </span>

                      {/* 2. meta-model: backend model identifier */}
                      <span className="meta-group">
                        <span className="meta-label">Model:</span>
                        <span data-testid="meta-model" className="meta-value">
                          {msg.metadata.model}
                        </span>
                      </span>

                      {/* 3. meta-latency: value with ms */}
                      <span className="meta-group">
                        <span className="meta-label">Latency:</span>
                        <span data-testid="meta-latency" className="meta-value">
                          {msg.metadata.latency_ms}ms
                        </span>
                      </span>

                      {/* 4. meta-input-tokens: ONLY the integer value */}
                      <span className="meta-group">
                        <span className="meta-label">In:</span>
                        <span data-testid="meta-input-tokens" className="meta-value">
                          {msg.metadata.input_tokens}
                        </span>
                      </span>

                      {/* 5. meta-output-tokens: ONLY the integer value */}
                      <span className="meta-group">
                        <span className="meta-label">Out:</span>
                        <span data-testid="meta-output-tokens" className="meta-value">
                          {msg.metadata.output_tokens}
                        </span>
                      </span>

                      {/* Secondary metadata: key_id & fallback_attempts */}
                      <span className="meta-secondary">
                        Key: {msg.metadata.key_id ? msg.metadata.key_id : 'null'}
                      </span>
                      <span className="meta-secondary">
                        Failovers: {msg.metadata.fallback_attempts}
                      </span>
                    </div>
                  )}
                </div>
              </div>
            );
          })
        )}

        {isLoading && (
          <div className="message-row assistant">
            <div className="loading-bubble">
              <div className="typing-dots" aria-hidden="true">
                <span />
                <span />
                <span />
              </div>
              <span>Generating response...</span>
            </div>
          </div>
        )}

        <div ref={messagesEndRef} />
      </main>

      {/* Input Dock */}
      <footer className="input-dock">
        <form
          className="input-form"
          onSubmit={(e) => {
            e.preventDefault();
            handleSend();
          }}
        >
          <div className="input-row">
            <textarea
              ref={textareaRef}
              data-testid="chat-input"
              className="chat-textarea"
              placeholder="Ask anything... (Enter to send, Shift+Enter for newline)"
              rows={1}
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
            />
            <button
              type="submit"
              data-testid="send-button"
              className="btn-send"
              disabled={isLoading || !input.trim()}
              title="Send message"
            >
              <span>Send</span>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <line x1="22" y1="2" x2="11" y2="13" />
                <polygon points="22 2 15 22 11 13 2 9 22 2" />
              </svg>
            </button>
          </div>
          <div className="input-hints">
            <span>Press <strong>Enter</strong> to send • <strong>Shift+Enter</strong> for newline</span>
            <span className="session-badge" title="Active session ID (regenerates on New Chat)">
              Session: {sessionId.substring(0, 16)}...
            </span>
          </div>
        </form>
      </footer>
    </div>
  );
}
