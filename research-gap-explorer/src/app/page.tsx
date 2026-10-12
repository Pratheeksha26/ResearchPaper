'use client';

import { useState } from 'react';
import styles from './page.module.css';
import { Paper, SearchMetrics } from '../services/paperService';
import Dashboard from '../components/Dashboard';

export default function Home() {
  const [topic, setTopic] = useState('');
  const [loading, setLoading] = useState(false);
  const [papers, setPapers] = useState<Paper[]>([]);
  const [metrics, setMetrics] = useState<SearchMetrics | undefined>(undefined);
  const [view, setView] = useState<'hero' | 'dashboard'>('hero');
  const [currentTopic, setCurrentTopic] = useState('');
  const [uploading, setUploading] = useState(false);

  const sampleTopics = [
    'Transformer models in healthcare',
    'Quantum computing encryption',
    'CRISPR gene editing ethics',
    'AI for climate modeling',
    'Graph neural networks drug discovery'
  ];

  const handleSearchTopic = async (searchQuery: string) => {
    if (!searchQuery.trim()) return;
    setLoading(true);

    try {
      const res = await fetch(`/api/search?q=${encodeURIComponent(searchQuery)}`);
      if (!res.ok) {
        throw new Error('Search failed');
      }

      const result = await res.json();
      setPapers(result.data || []);
      setMetrics(result.metrics);
      setCurrentTopic(searchQuery);
      setView('dashboard');
    } catch (err) {
      console.error(err);
      alert('Error fetching literature. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  const handleFormSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    handleSearchTopic(topic);
  };

  const handleFileUpload = async (file: File) => {
    setUploading(true);
    try {
      const formData = new FormData();
      formData.append('file', file);

      const res = await fetch('/api/upload', {
        method: 'POST',
        body: formData
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.error || 'Upload failed');
      }

      const data = await res.json();
      const uploadedPaper: Paper = data.paper;

      setPapers(prev => [uploadedPaper, ...prev]);

      if (view === 'hero') {
        const primaryTopic = uploadedPaper.extracted_topics?.[0] || uploadedPaper.title;
        setCurrentTopic(primaryTopic);
        setView('dashboard');
      }
    } catch (err: any) {
      console.error('File upload error:', err);
      alert(`Error processing uploaded file: ${err.message || 'Check file format'}`);
    } finally {
      setUploading(false);
    }
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileUpload(file);
    }
  };

  if (view === 'dashboard') {
    return (
      <Dashboard
        topic={currentTopic}
        papers={papers}
        metrics={metrics}
        onBack={() => setView('hero')}
        onUploadPaper={handleFileUpload}
        onSearchTopic={handleSearchTopic}
      />
    );
  }

  return (
    <div className={styles.container}>
      <div className={styles.glowSphere} />

      <header className={styles.heroCard + ' glass'}>
        <div className={styles.topBadge}>
          ✨ AI-Powered Academic Research Assistant
        </div>

        <h1 className={styles.title}>
          Discover <span className={styles.highlight}>Research Gaps</span> & Synthesize Literature
        </h1>

        <p className={styles.subtitle}>
          Skim thousands of research papers across ArXiv, CrossRef, and Semantic Scholar. Upload paywalled or custom papers to uncover unexplored research questions and draft publication-ready literature reviews.
        </p>

        {/* Search Form */}
        <form onSubmit={handleFormSubmit} className={styles.searchForm}>
          <div className={styles.searchInputWrapper}>
            <svg className={styles.searchIcon} xmlns="http://www.w3.org/2000/svg" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
              <circle cx="11" cy="11" r="8"></circle>
              <line x1="21" y1="21" x2="16.65" y2="16.65"></line>
            </svg>
            <input
              type="text"
              placeholder="Enter research topic (e.g., Transformer models in healthcare)..."
              value={topic}
              onChange={(e) => setTopic(e.target.value)}
              className={styles.searchInput}
            />
          </div>

          <button type="submit" className="btn" disabled={loading}>
            {loading ? 'Skimming Papers...' : 'Explore Gaps'}
          </button>
        </form>

        {/* Quick Topics */}
        <div className={styles.quickTopics}>
          <span>Popular topics:</span>
          {sampleTopics.map((t, idx) => (
            <button
              key={idx}
              type="button"
              className={styles.topicTag}
              onClick={() => {
                setTopic(t);
                handleSearchTopic(t);
              }}
            >
              {t}
            </button>
          ))}
        </div>

        <div className={styles.divider}>
          <span>OR UPLOAD CUSTOM RESEARCH PAPERS</span>
        </div>

        {/* Upload Section */}
        <div className={styles.uploadSection}>
          <label htmlFor="file-upload" className={styles.uploadDropzone}>
            <div className={styles.uploadIcon}>
              <svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path>
                <polyline points="17 8 12 3 7 8"></polyline>
                <line x1="12" y1="3" x2="12" y2="15"></line>
              </svg>
            </div>
            <div>
              <strong>{uploading ? 'Extracting Paper Metadata...' : 'Upload PDF / Text Research Papers'}</strong>
              <div style={{ fontSize: '0.85rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                Upload papers from non-scrappable or paywalled sites to analyze & find matching gaps
              </div>
            </div>
          </label>
          <input
            id="file-upload"
            type="file"
            accept="application/pdf,.txt,.md"
            onChange={handleFileChange}
            className={styles.fileInput}
          />
        </div>

        {/* Feature Highlights Grid */}
        <div className={styles.featuresGrid}>
          <div className={`glass ${styles.featureCard}`}>
            <div className={styles.featureIcon}>⚡</div>
            <h3 className={styles.featureTitle}>Multi-Database Skimmer</h3>
            <p className={styles.featureDesc}>
              Scans ArXiv, CrossRef, and Semantic Scholar simultaneously, displaying total scanned counts and top accessible papers.
            </p>
          </div>

          <div className={`glass ${styles.featureCard}`}>
            <div className={styles.featureIcon}>🎯</div>
            <h3 className={styles.featureTitle}>Categorized Gap Engine</h3>
            <p className={styles.featureDesc}>
              Identifies Methodological, Theoretical, and Empirical gaps with actionable PhD-level research questions.
            </p>
          </div>

          <div className={`glass ${styles.featureCard}`}>
            <div className={styles.featureIcon}>📝</div>
            <h3 className={styles.featureTitle}>Lit Review Synthesizer</h3>
            <p className={styles.featureDesc}>
              Generates structured literature review drafts with in-text citations, live preview, auto-save, and Markdown export.
            </p>
          </div>
        </div>
      </header>
    </div>
  );
}
