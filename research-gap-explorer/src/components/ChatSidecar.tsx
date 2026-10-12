'use client';

import { useState, useRef, useEffect } from 'react';
import { Paper } from '../services/paperService';
import { Gap, ExtractedTool } from './Dashboard';
import styles from './ChatSidecar.module.css';

interface Message {
  role: 'user' | 'assistant';
  content: string;
}

interface ChatSidecarProps {
  topic: string;
  papers: Paper[];
  gaps: Gap[];
  extractedTools: ExtractedTool[];
}

export default function ChatSidecar({ topic, papers, gaps, extractedTools }: ChatSidecarProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState<Message[]>([
    {
      role: 'assistant',
      content: `Hello! I am your AI Co-Researcher. I have analyzed all ${papers.length} papers, ${gaps.length} gaps, and ${extractedTools.length} tools for "${topic}". How can I help with your thesis or literature synthesis?`
    }
  ]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  const samplePrompts = [
    'Which datasets were used across papers?',
    'Compare top model architectures',
    `Explain how to address Gap #1`,
    'List papers with open-access PDFs'
  ];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    if (isOpen) {
      scrollToBottom();
    }
  }, [messages, isOpen]);

  const handleSend = async (textToSend?: string) => {
    const query = textToSend || input;
    if (!query.trim() || loading) return;

    const userMsg: Message = { role: 'user', content: query };
    setMessages(prev => [...prev, userMsg]);
    if (!textToSend) setInput('');
    setLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic,
          message: query,
          papers,
          gaps,
          extractedTools,
          chatHistory: messages.slice(-6)
        })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Failed to fetch AI answer');

      const aiMsg: Message = { role: 'assistant', content: data.reply };
      setMessages(prev => [...prev, aiMsg]);
    } catch (err: any) {
      console.error(err);
      setMessages(prev => [
        ...prev,
        { role: 'assistant', content: '⚠️ Error reaching AI assistant: ' + (err.message || 'Try again') }
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <>
      {/* Floating Toggle Button */}
      <button onClick={() => setIsOpen(true)} className={styles.chatFloatingBtn}>
        💬 AI Co-Researcher
      </button>

      {/* Drawer Overlay & Panel */}
      {isOpen && (
        <>
          <div className={styles.drawerOverlay} onClick={() => setIsOpen(false)} />
          <div className={styles.drawerContainer}>
            {/* Header */}
            <div className={styles.drawerHeader}>
              <div className={styles.drawerTitle}>
                <span>💬 AI Academic Co-Researcher</span>
              </div>
              <button onClick={() => setIsOpen(false)} className={styles.closeBtn}>
                ✕
              </button>
            </div>

            {/* Quick Prompts */}
            <div className={styles.quickPrompts}>
              {samplePrompts.map((p, idx) => (
                <button key={idx} onClick={() => handleSend(p)} className={styles.promptChip}>
                  {p}
                </button>
              ))}
            </div>

            {/* Messages */}
            <div className={styles.chatMessages}>
              {messages.map((m, idx) => (
                <div
                  key={idx}
                  className={`${styles.message} ${m.role === 'user' ? styles.userMessage : styles.aiMessage}`}
                  dangerouslySetInnerHTML={{ __html: formatMarkdownToHtml(m.content) }}
                />
              ))}
              {loading && (
                <div className={`${styles.message} ${styles.aiMessage}`} style={{ fontStyle: 'italic', color: 'var(--text-muted)' }}>
                  Thinking & referencing literature...
                </div>
              )}
              <div ref={messagesEndRef} />
            </div>

            {/* Input Area */}
            <form
              onSubmit={e => {
                e.preventDefault();
                handleSend();
              }}
              className={styles.chatInputArea}
            >
              <input
                type="text"
                placeholder="Ask about papers, methods, or gaps..."
                value={input}
                onChange={e => setInput(e.target.value)}
                className={styles.chatInput}
              />
              <button type="submit" className={styles.sendBtn} disabled={loading || !input.trim()}>
                Send
              </button>
            </form>
          </div>
        </>
      )}
    </>
  );
}

function formatMarkdownToHtml(markdown: string): string {
  if (!markdown) return '';

  let html = markdown
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Code blocks
  html = html.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');

  // Inline code
  html = html.replace(/`([^`]+)`/g, '<code>$1</code>');

  // Headings
  html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

  // Bold (**text** or __text__)
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/__(.*?)__/g, '<strong>$1</strong>');

  // Markdown links [text](url)
  html = html.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noreferrer">$1</a>');

  // Bullet items starting with "* " or "- "
  html = html.replace(/(?:^|\n)[\*\-\+]\s+(.*)/g, '<li>$1</li>');

  // Wrap contiguous <li> tags in <ul>
  html = html.replace(/(<li>[\s\S]*?<\/li>)/gi, '<ul>$1</ul>');

  // Italics (*text* or _text_)
  html = html.replace(/(?<!\w)\*([^\*]+)\*(?!\w)/g, '<em>$1</em>');
  html = html.replace(/(?<!\w)_([^_]+)_(?!\w)/g, '<em>$1</em>');

  // Paragraph breaks
  const paragraphs = html
    .split(/\n\n+/)
    .map(p => p.trim())
    .filter(Boolean)
    .map(p => {
      if (p.startsWith('<h') || p.startsWith('<ul') || p.startsWith('<pre')) {
        return p;
      }
      return `<p>${p.replace(/\n/g, '<br />')}</p>`;
    });

  return paragraphs.join('');
}
