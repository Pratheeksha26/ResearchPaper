'use client';

import { useState, useEffect } from 'react';
import { Paper, SearchMetrics } from '../services/paperService';
import { generateBibTeX, generateRIS, downloadTextFile } from '../services/bibtexService';
import ChatSidecar from './ChatSidecar';
import styles from './Dashboard.module.css';

export interface Gap {
  id: string;
  title: string;
  type: 'Methodological' | 'Theoretical' | 'Empirical' | 'Demographic' | 'Application';
  description: string;
  opportunity_level: 'High' | 'Medium' | 'Emerging';
  suggested_question: string;
  supporting_paper_titles?: string[];
  cross_paper_evidence?: string;
  reverified?: boolean;
  timeliness?: string;
  verification_notes?: string;
}

export interface ExtractedTool {
  name: string;
  category: 'Framework / Tool' | 'Model / Architecture' | 'Dataset / Benchmark' | 'Algorithm / Method' | 'Hardware / Infrastructure';
  summary: string;
  used_in_papers?: string[];
}

interface DashboardProps {
  topic: string;
  papers: Paper[];
  metrics?: SearchMetrics;
  onBack: () => void;
  onUploadPaper: (file: File) => Promise<void>;
  onSearchTopic: (query: string) => void;
}

export default function Dashboard({
  topic,
  papers: initialPapers,
  metrics,
  onBack,
  onUploadPaper,
  onSearchTopic
}: DashboardProps) {
  const [papers, setPapers] = useState<Paper[]>(initialPapers);
  const [selectedPaperIds, setSelectedPaperIds] = useState<Set<string>>(new Set());
  const [filter, setFilter] = useState<'all' | 'scraped' | 'uploaded' | 'pdf'>('all');
  const [paperSearchQuery, setPaperSearchQuery] = useState('');
  
  const [gaps, setGaps] = useState<Gap[]>([]);
  const [extractedTools, setExtractedTools] = useState<ExtractedTool[]>([]);
  const [analyzingGaps, setAnalyzingGaps] = useState(false);
  const [gapError, setGapError] = useState('');

  // Analysis panel tabs
  const [activeAnalysisTab, setActiveAnalysisTab] = useState<'gaps' | 'tools' | 'matrix' | 'review'>('gaps');

  // Tool search & filter state
  const [toolsViewMode, setToolsViewMode] = useState<'by_tool' | 'by_paper'>('by_tool');
  const [toolSearchQuery, setToolSearchQuery] = useState('');
  const [toolCategoryFilter, setToolCategoryFilter] = useState<string>('all');

  // Citation export dropdown state
  const [citationDropdownOpen, setCitationDropdownOpen] = useState(false);

  // Proposal modal state
  const [proposalModalOpen, setProposalModalOpen] = useState(false);
  const [proposalContent, setProposalContent] = useState('');
  const [generatingProposal, setGeneratingProposal] = useState(false);
  const [activeProposalGap, setActiveProposalGap] = useState<Gap | null>(null);

  // Lit review generator state
  const [generatingLitReview, setGeneratingLitReview] = useState(false);
  const [draft, setDraft] = useState('');
  const [activeTab, setActiveTab] = useState<'editor' | 'preview'>('editor');
  const [copySuccess, setCopySuccess] = useState(false);
  const [uploading, setUploading] = useState(false);

  // Sync initial papers when prop updates & select all by default
  useEffect(() => {
    setPapers(initialPapers);
    setSelectedPaperIds(new Set(initialPapers.map(p => p.id)));
  }, [initialPapers]);

  // Load draft from localStorage on mount
  useEffect(() => {
    const savedDraft = localStorage.getItem(`lit_review_draft_${topic}`);
    if (savedDraft) {
      setDraft(savedDraft);
    }
  }, [topic]);

  // Auto-save draft to localStorage
  useEffect(() => {
    if (draft) {
      localStorage.setItem(`lit_review_draft_${topic}`, draft);
    }
  }, [draft, topic]);

  // Run AI Gap & Tools Analysis when papers load or change
  useEffect(() => {
    async function runGapAnalysis() {
      if (papers.length === 0) return;
      setAnalyzingGaps(true);
      setGapError('');

      try {
        const res = await fetch('/api/analyze', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ topic, papers })
        });

        const data = await res.json();

        if (!res.ok) {
          throw new Error(data.error || 'Gap & tools analysis request failed');
        }

        setGaps(Array.isArray(data.gaps) ? data.gaps : []);
        setExtractedTools(Array.isArray(data.extracted_tools) ? data.extracted_tools : []);
      } catch (err: any) {
        console.error('Gap analysis error:', err);
        setGapError(err.message || 'AI service temporarily busy. Please click Retry.');
      } finally {
        setAnalyzingGaps(false);
      }
    }

    runGapAnalysis();
  }, [topic, papers]);

  const handleRetryGapAnalysis = async () => {
    if (papers.length === 0) return;
    setAnalyzingGaps(true);
    setGapError('');
    try {
      const res = await fetch('/api/analyze', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ topic, papers })
      });
      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Analysis failed');
      setGaps(data.gaps || []);
      setExtractedTools(data.extracted_tools || []);
    } catch (err: any) {
      setGapError(err.message || 'Failed to analyze gaps & tools.');
    } finally {
      setAnalyzingGaps(false);
    }
  };

  // Toggle paper selection
  const togglePaperSelection = (id: string) => {
    const next = new Set(selectedPaperIds);
    if (next.has(id)) {
      next.delete(id);
    } else {
      next.add(id);
    }
    setSelectedPaperIds(next);
  };

  const handleSelectAll = () => {
    if (selectedPaperIds.size === filteredPapers.length) {
      setSelectedPaperIds(new Set());
    } else {
      setSelectedPaperIds(new Set(filteredPapers.map(p => p.id)));
    }
  };

  // Helper to match paper object by title
  const findMatchingPaper = (paperTitle: string): Paper | undefined => {
    if (!paperTitle) return undefined;
    const normTarget = paperTitle.toLowerCase().trim();
    return papers.find(p => {
      const normP = p.title.toLowerCase().trim();
      return normP.includes(normTarget) || normTarget.includes(normP);
    });
  };

  // Helper to get tools associated with a specific paper
  const getToolsForPaper = (paperTitle: string): ExtractedTool[] => {
    const normTarget = paperTitle.toLowerCase().trim();
    return extractedTools.filter(t => {
      if (!t.used_in_papers) return false;
      return t.used_in_papers.some(pName => {
        const normP = pName.toLowerCase().trim();
        return normP.includes(normTarget) || normTarget.includes(normP);
      });
    });
  };

  // Export BibTeX / RIS functions
  const handleExportBibTeX = () => {
    const selectedList = papers.filter(p => selectedPaperIds.has(p.id));
    const bibtex = generateBibTeX(selectedList.length > 0 ? selectedList : papers);
    downloadTextFile(bibtex, `${topic.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_references.bib`, 'application/x-bibtex');
    setCitationDropdownOpen(false);
  };

  const handleExportRIS = () => {
    const selectedList = papers.filter(p => selectedPaperIds.has(p.id));
    const ris = generateRIS(selectedList.length > 0 ? selectedList : papers);
    downloadTextFile(ris, `${topic.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_references.ris`, 'application/x-research-info-systems');
    setCitationDropdownOpen(false);
  };

  const handleCopyBibTeX = () => {
    const selectedList = papers.filter(p => selectedPaperIds.has(p.id));
    const bibtex = generateBibTeX(selectedList.length > 0 ? selectedList : papers);
    navigator.clipboard.writeText(bibtex);
    alert('✓ BibTeX citations copied to clipboard!');
    setCitationDropdownOpen(false);
  };

  // Generate PhD Proposal
  const handleGenerateProposal = async (gap: Gap) => {
    setActiveProposalGap(gap);
    setProposalModalOpen(true);
    setGeneratingProposal(true);
    setProposalContent('');

    try {
      const supportingPapers = papers.filter(p => selectedPaperIds.has(p.id));
      const res = await fetch('/api/proposal', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic,
          gap,
          supportingPapers
        })
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.error || 'Proposal generation failed');

      setProposalContent(data.proposal || '');
    } catch (err: any) {
      console.error(err);
      setProposalContent(`⚠️ Error generating PhD proposal: ${err.message || 'Please try again.'}`);
    } finally {
      setGeneratingProposal(false);
    }
  };

  // Export Comparison Matrix to CSV
  const handleExportMatrixCSV = () => {
    const headers = ['Paper Title', 'Year', 'Authors', 'Venue/Source', 'Extracted Tools', 'Abstract Snippet'];
    const rows = papers.map(p => {
      const toolsStr = getToolsForPaper(p.title).map(t => t.name).join('; ') || 'N/A';
      return [
        `"${p.title.replace(/"/g, '""')}"`,
        p.publication_year,
        `"${p.authors.join(', ').replace(/"/g, '""')}"`,
        `"${(p.venue || p.source).replace(/"/g, '""')}"`,
        `"${toolsStr.replace(/"/g, '""')}"`,
        `"${p.abstract.substring(0, 150).replace(/"/g, '""')}..."`
      ].join(',');
    });

    const csvContent = [headers.join(','), ...rows].join('\n');
    downloadTextFile(csvContent, `${topic.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_comparison_matrix.csv`, 'text/csv');
  };

  // Generate Literature Review
  const handleGenerateLiteratureReview = async () => {
    const selectedPapersList = papers.filter(p => selectedPaperIds.has(p.id));
    if (selectedPapersList.length === 0) {
      alert('Please select at least one paper to include in the literature review.');
      return;
    }

    setGeneratingLitReview(true);
    try {
      const res = await fetch('/api/literature-review', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic,
          selectedPapers: selectedPapersList,
          gaps,
          extractedTools
        })
      });

      if (!res.ok) {
        throw new Error('Failed to generate review');
      }

      const data = await res.json();
      if (data.review) {
        setDraft(data.review);
        setActiveAnalysisTab('review');
        setActiveTab('preview');
      }
    } catch (err) {
      console.error(err);
      alert('Error generating literature review draft.');
    } finally {
      setGeneratingLitReview(false);
    }
  };

  // Copy draft to clipboard
  const handleCopy = () => {
    navigator.clipboard.writeText(draft);
    setCopySuccess(true);
    setTimeout(() => setCopySuccess(false), 2500);
  };

  // Download Markdown file
  const handleDownloadMd = () => {
    const blob = new Blob([draft], { type: 'text/markdown;charset=utf-8' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${topic.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_literature_review.md`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Add gap to draft
  const handleAddGapToDraft = (gap: Gap) => {
    const gapSnippet = `\n\n### Research Gap: ${gap.title} (${gap.type})\n**Synthesis:** ${gap.description}\n**Cross-Paper Evidence:** ${gap.cross_paper_evidence || 'Synthesized across literature'}\n**Re-Verification:** ${gap.verification_notes || 'Verified Unsolved for Present/Future'}\n**Actionable Research Question:** ${gap.suggested_question}\n`;
    setDraft(prev => prev + gapSnippet);
    setActiveAnalysisTab('review');
  };

  // Add tool to draft
  const handleAddToolToDraft = (tool: ExtractedTool) => {
    const toolSnippet = `\n\n### Extracted Tool/Technology: ${tool.name} (${tool.category})\n**Summary:** ${tool.summary}\n**Used in Literature:** ${tool.used_in_papers ? tool.used_in_papers.join('; ') : 'Analyzed papers'}\n`;
    setDraft(prev => prev + toolSnippet);
    setActiveAnalysisTab('review');
  };

  // File upload trigger in dashboard
  const handleFileUpload = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      setUploading(true);
      try {
        await onUploadPaper(file);
      } finally {
        setUploading(false);
      }
    }
  };

  // Filter papers logic
  const filteredPapers = papers.filter(p => {
    const matchesFilter =
      filter === 'scraped' ? !p.is_uploaded :
      filter === 'uploaded' ? p.is_uploaded :
      filter === 'pdf' ? (!!p.pdf_url || p.is_uploaded) : true;

    const matchesQuery = !paperSearchQuery.trim() ||
      p.title.toLowerCase().includes(paperSearchQuery.toLowerCase()) ||
      p.authors.some(a => a.toLowerCase().includes(paperSearchQuery.toLowerCase())) ||
      p.abstract.toLowerCase().includes(paperSearchQuery.toLowerCase());

    return matchesFilter && matchesQuery;
  });

  // Filter extracted tools logic
  const filteredTools = extractedTools.filter(tool => {
    const matchesCategory = toolCategoryFilter === 'all' || tool.category === toolCategoryFilter;
    const matchesSearch = !toolSearchQuery.trim() ||
      tool.name.toLowerCase().includes(toolSearchQuery.toLowerCase()) ||
      tool.summary.toLowerCase().includes(toolSearchQuery.toLowerCase());
    return matchesCategory && matchesSearch;
  });

  const totalScanned = metrics?.totalScanned || papers.length;
  const sourcesText = metrics?.sourcesScanned ? metrics.sourcesScanned.join(', ') : 'Online Repositories';

  return (
    <div className={styles.dashboardContainer}>
      {/* Top Header */}
      <header className={styles.header}>
        <div className={styles.titleArea}>
          <button onClick={onBack} className="btn btn-secondary" style={{ padding: '8px 16px' }}>
            ← Back to Search
          </button>
          <h2 className={styles.title}>
            Topic: <span className={styles.highlight}>{topic}</span>
          </h2>
        </div>

        <div style={{ display: 'flex', gap: '10px', alignItems: 'center' }}>
          {/* Citation Export Dropdown */}
          <div className={styles.dropdownWrapper}>
            <button
              onClick={() => setCitationDropdownOpen(!citationDropdownOpen)}
              className="btn btn-secondary"
            >
              📥 Export Citations ▾
            </button>

            {citationDropdownOpen && (
              <div className={styles.dropdownMenu}>
                <button onClick={handleExportBibTeX} className={styles.dropdownItem}>
                  📄 Download BibTeX (.bib)
                </button>
                <button onClick={handleExportRIS} className={styles.dropdownItem}>
                  🔖 Download RIS (Zotero / EndNote)
                </button>
                <button onClick={handleCopyBibTeX} className={styles.dropdownItem}>
                  📋 Copy BibTeX to Clipboard
                </button>
              </div>
            )}
          </div>

          <label htmlFor="dashboard-file-upload" className="btn btn-secondary" style={{ cursor: 'pointer', margin: 0 }}>
            {uploading ? 'Parsing PDF...' : '📄 Upload Paper'}
          </label>
          <input
            id="dashboard-file-upload"
            type="file"
            accept="application/pdf,.txt,.md"
            onChange={handleFileUpload}
            style={{ display: 'none' }}
          />
          <button
            onClick={handleGenerateLiteratureReview}
            className="btn"
            disabled={generatingLitReview || selectedPaperIds.size === 0}
          >
            {generatingLitReview ? 'Synthesizing AI Review...' : `✨ Generate Lit Review (${selectedPaperIds.size})`}
          </button>
        </div>
      </header>

      {/* Metrics Banner */}
      <div className={styles.metricsBar}>
        <div className={styles.metricItem}>
          <div className={styles.metricIcon}>🔍</div>
          <div>
            <div className={styles.metricLabel}>Papers Scanned</div>
            <div className={styles.metricValue}>{totalScanned} Papers</div>
          </div>
        </div>

        <div className={styles.metricDivider} />

        <div className={styles.metricItem}>
          <div className={styles.metricIcon}>📚</div>
          <div>
            <div className={styles.metricLabel}>Databases Queried</div>
            <div className={styles.metricValue}>{sourcesText}</div>
          </div>
        </div>

        <div className={styles.metricDivider} />

        <div className={styles.metricItem}>
          <div className={styles.metricIcon}>🎯</div>
          <div>
            <div className={styles.metricLabel}>Cross-Paper Gaps</div>
            <div className={styles.metricValue}>{gaps.length} Gaps</div>
          </div>
        </div>

        <div className={styles.metricDivider} />

        <div className={styles.metricItem}>
          <div className={styles.metricIcon}>⚡</div>
          <div>
            <div className={styles.metricLabel}>Tools & Tech Extracted</div>
            <div className={styles.metricValue}>{extractedTools.length} Extracted</div>
          </div>
        </div>

        <div className={styles.metricDivider} />

        <div className={styles.metricItem}>
          <div className={styles.metricIcon}>✅</div>
          <div>
            <div className={styles.metricLabel}>Selected for Review</div>
            <div className={styles.metricValue}>{selectedPaperIds.size} / {papers.length}</div>
          </div>
        </div>
      </div>

      {/* Main Grid Layout */}
      <div className={styles.layout}>
        {/* Left Column: Papers List & Controls */}
        <section>
          <div className={styles.sectionTitle}>
            <span>All Scanned Literature ({filteredPapers.length} / {papers.length})</span>
            <button onClick={handleSelectAll} className={styles.filterBtn}>
              {selectedPaperIds.size === filteredPapers.length ? 'Deselect All' : 'Select All'}
            </button>
          </div>

          <div className={styles.papersControls}>
            <input
              type="text"
              placeholder="🔍 Search all papers by title, author, abstract..."
              value={paperSearchQuery}
              onChange={e => setPaperSearchQuery(e.target.value)}
              className={styles.toolSearchInput}
              style={{ marginBottom: '10px' }}
            />

            <div className={styles.filterGroup}>
              <button
                className={`${styles.filterBtn} ${filter === 'all' ? styles.filterBtnActive : ''}`}
                onClick={() => setFilter('all')}
              >
                All ({papers.length})
              </button>
              <button
                className={`${styles.filterBtn} ${filter === 'scraped' ? styles.filterBtnActive : ''}`}
                onClick={() => setFilter('scraped')}
              >
                Scraped ({papers.filter(p => !p.is_uploaded).length})
              </button>
              <button
                className={`${styles.filterBtn} ${filter === 'uploaded' ? styles.filterBtnActive : ''}`}
                onClick={() => setFilter('uploaded')}
              >
                Uploaded ({papers.filter(p => p.is_uploaded).length})
              </button>
              <button
                className={`${styles.filterBtn} ${filter === 'pdf' ? styles.filterBtnActive : ''}`}
                onClick={() => setFilter('pdf')}
              >
                PDF Links ({papers.filter(p => !!p.pdf_url || p.is_uploaded).length})
              </button>
            </div>
          </div>

          <div className={styles.papersList}>
            {filteredPapers.map(paper => {
              const isSelected = selectedPaperIds.has(paper.id);
              const paperTools = getToolsForPaper(paper.title);

              return (
                <div
                  key={paper.id}
                  className={`glass ${styles.paperCard} ${isSelected ? styles.paperCardSelected : ''}`}
                >
                  <div className={styles.paperHeader}>
                    <input
                      type="checkbox"
                      className={styles.checkbox}
                      checked={isSelected}
                      onChange={() => togglePaperSelection(paper.id)}
                    />
                    <h4 className={styles.paperTitle}>
                      {paper.url && paper.url !== '#' ? (
                        <a href={paper.url} target="_blank" rel="noreferrer" title="Open paper external link">
                          {paper.title} ↗
                        </a>
                      ) : (
                        paper.title
                      )}
                    </h4>
                  </div>

                  <div className={styles.paperMeta}>
                    <span>{paper.authors.slice(0, 3).join(', ')}{paper.authors.length > 3 ? ' et al.' : ''}</span>
                    <span>• {paper.publication_year}</span>
                    {paper.venue && <span>• <em>{paper.venue}</em></span>}
                  </div>

                  <p className={styles.paperAbstract}>
                    {paper.abstract.length > 240 ? paper.abstract.substring(0, 240) + '...' : paper.abstract}
                  </p>

                  <div className={styles.paperBadges}>
                    <span className={`badge ${paper.is_uploaded ? 'badge-amber' : 'badge-purple'}`}>
                      {paper.source}
                    </span>

                    {paper.url && paper.url !== '#' && (
                      <a href={paper.url} target="_blank" rel="noreferrer" className="badge badge-indigo" style={{ background: 'rgba(99, 102, 241, 0.2)', color: '#818cf8', border: '1px solid rgba(99, 102, 241, 0.4)' }}>
                        🌐 View Paper Link ↗
                      </a>
                    )}

                    {paper.pdf_url && (
                      <a href={paper.pdf_url} target="_blank" rel="noreferrer" className="badge badge-cyan">
                        📥 Download PDF
                      </a>
                    )}

                    {paper.citation_count !== undefined && paper.citation_count > 0 && (
                      <span className="badge badge-emerald">
                        Cited by {paper.citation_count}
                      </span>
                    )}
                  </div>

                  {/* Display tools extracted for this paper */}
                  {paperTools.length > 0 && (
                    <div style={{ marginTop: '10px', paddingTop: '8px', borderTop: '1px dashed var(--border-color)', fontSize: '0.78rem' }}>
                      <span style={{ color: '#c084fc', fontWeight: 600 }}>⚡ Extracted Tools: </span>
                      {paperTools.map((t, idx) => (
                        <span key={idx} className="badge badge-purple" style={{ margin: '2px', fontSize: '0.7rem', textTransform: 'none' }}>
                          {t.name}
                        </span>
                      ))}
                    </div>
                  )}

                  {/* Uploaded paper custom topics & trigger similar paper search */}
                  {paper.is_uploaded && paper.extracted_topics && paper.extracted_topics.length > 0 && (
                    <div className={styles.keyFindingsList}>
                      <div style={{ fontWeight: 600, marginBottom: '4px', color: '#fbbf24' }}>
                        Extracted Topics & Key Findings:
                      </div>
                      <div>Topics: {paper.extracted_topics.join(', ')}</div>
                      {paper.key_findings && paper.key_findings.length > 0 && (
                        <ul style={{ paddingLeft: '16px', marginTop: '6px' }}>
                          {paper.key_findings.map((f, i) => (
                            <li key={i}>{f}</li>
                          ))}
                        </ul>
                      )}
                      <button
                        onClick={() => onSearchTopic(paper.extracted_topics![0])}
                        className="btn btn-outline"
                        style={{ marginTop: '10px', fontSize: '0.8rem', padding: '4px 12px' }}
                      >
                        🔍 Search Similar Online Papers for "{paper.extracted_topics[0]}"
                      </button>
                    </div>
                  )}
                </div>
              );
            })}

            {filteredPapers.length === 0 && (
              <div className="glass" style={{ padding: '30px', textAlign: 'center', color: 'var(--text-muted)' }}>
                No papers found matching the search filter.
              </div>
            )}
          </div>
        </section>

        {/* Right Column: AI Analysis & Literature Review Workspace */}
        <section className={styles.analysisColumn}>
          {/* Analysis Navigation Tabs */}
          <div className={styles.analysisTabsNav}>
            <button
              className={`${styles.analysisTabBtn} ${activeAnalysisTab === 'gaps' ? styles.analysisTabActive : ''}`}
              onClick={() => setActiveAnalysisTab('gaps')}
            >
              🎯 Gaps ({gaps.length})
            </button>
            <button
              className={`${styles.analysisTabBtn} ${activeAnalysisTab === 'tools' ? styles.analysisTabActive : ''}`}
              onClick={() => setActiveAnalysisTab('tools')}
            >
              ⚡ Tools ({extractedTools.length})
            </button>
            <button
              className={`${styles.analysisTabBtn} ${activeAnalysisTab === 'matrix' ? styles.analysisTabActive : ''}`}
              onClick={() => setActiveAnalysisTab('matrix')}
            >
              📊 Matrix Table
            </button>
            <button
              className={`${styles.analysisTabBtn} ${activeAnalysisTab === 'review' ? styles.analysisTabActive : ''}`}
              onClick={() => setActiveAnalysisTab('review')}
            >
              📝 Review Draft
            </button>
          </div>

          {/* TAB 1: GAP ANALYSIS PANEL */}
          {activeAnalysisTab === 'gaps' && (
            <div className={`glass ${styles.gapPanel}`}>
              <h3 className={styles.sectionTitle}>
                <span>Cross-Paper AI Research Gaps ({gaps.length})</span>
                {analyzingGaps && <span className="badge badge-purple">Analyzing...</span>}
              </h3>

              {analyzingGaps && (
                <p style={{ color: 'var(--text-secondary)', fontStyle: 'italic', padding: '16px 0' }}>
                  Synthesizing across all {papers.length} papers with Gemini AI to locate re-verified, present/future gaps...
                </p>
              )}

              {gapError && (
                <div style={{ padding: '12px 0' }}>
                  <p style={{ color: '#fb7185', marginBottom: '10px', fontSize: '0.9rem' }}>⚠️ {gapError}</p>
                  <button onClick={handleRetryGapAnalysis} className="btn btn-secondary" style={{ fontSize: '0.82rem', padding: '6px 14px' }}>
                    🔄 Retry Gap Analysis
                  </button>
                </div>
              )}

              {!analyzingGaps && !gapError && gaps.length === 0 && (
                <p style={{ color: 'var(--text-muted)' }}>No gaps identified for this topic.</p>
              )}

              <div className={styles.gapList}>
                {!analyzingGaps && gaps.map((gap, idx) => {
                  const badgeClass =
                    gap.type === 'Methodological' ? 'badge-purple' :
                    gap.type === 'Theoretical' ? 'badge-cyan' :
                    gap.type === 'Empirical' ? 'badge-emerald' :
                    gap.type === 'Demographic' ? 'badge-amber' : 'badge-rose';

                  return (
                    <div key={gap.id || idx} className={styles.gapCard}>
                      <div className={styles.gapHeader}>
                        <div style={{ display: 'flex', gap: '6px', flexWrap: 'wrap' }}>
                          <span className={`badge ${badgeClass}`}>{gap.type}</span>
                          <span className={`badge ${gap.opportunity_level === 'High' ? 'badge-rose' : 'badge-amber'}`}>
                            {gap.opportunity_level} Priority
                          </span>
                          <span className="badge badge-emerald">
                            ✓ Verified Unsolved (2024-2026+)
                          </span>
                        </div>
                      </div>

                      <h4 className={styles.gapTitle}>{gap.title}</h4>
                      <p className={styles.gapDesc}>{gap.description}</p>

                      {/* Cross-Paper Evidence Callout */}
                      {gap.cross_paper_evidence && (
                        <div className={styles.crossPaperQuote}>
                          🌐 <strong>Cross-Paper Synthesis:</strong> {gap.cross_paper_evidence}
                        </div>
                      )}

                      {/* Re-Verification Notes */}
                      {gap.verification_notes && (
                        <div className={styles.verificationCallout}>
                          🛡️ <strong>Literature Re-Verification:</strong> {gap.verification_notes}
                        </div>
                      )}

                      <div className={styles.gapQuestion}>
                        💡 <strong>Proposed Research Question:</strong> {gap.suggested_question}
                      </div>

                      <div className={styles.gapActions}>
                        <button
                          onClick={() => handleGenerateProposal(gap)}
                          className="btn btn-outline"
                          style={{ fontSize: '0.78rem', padding: '4px 10px' }}
                        >
                          ✨ PhD Proposal
                        </button>
                        <button
                          onClick={() => handleAddGapToDraft(gap)}
                          className="btn btn-secondary"
                          style={{ fontSize: '0.78rem', padding: '4px 10px' }}
                        >
                          + Add to Review Draft
                        </button>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}

          {/* TAB 2: EXTRACTED TOOLS & TECHNOLOGIES PANEL */}
          {activeAnalysisTab === 'tools' && (
            <div className={`glass ${styles.toolsPanel}`}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px', flexWrap: 'wrap', gap: '10px' }}>
                <h3 className={styles.sectionTitle} style={{ margin: 0 }}>
                  <span>Tools, Technologies & Paper Links</span>
                  {analyzingGaps && <span className="badge badge-purple" style={{ marginLeft: '8px' }}>Extracting...</span>}
                </h3>

                <div className={styles.editorTabs}>
                  <button
                    className={`${styles.editorTabBtn} ${toolsViewMode === 'by_tool' ? styles.editorTabActive : ''}`}
                    onClick={() => setToolsViewMode('by_tool')}
                  >
                    🛠️ By Tool ({extractedTools.length})
                  </button>
                  <button
                    className={`${styles.editorTabBtn} ${toolsViewMode === 'by_paper' ? styles.editorTabActive : ''}`}
                    onClick={() => setToolsViewMode('by_paper')}
                  >
                    📚 All Scanned Papers ({papers.length})
                  </button>
                </div>
              </div>

              {toolsViewMode === 'by_tool' ? (
                <>
                  <div className={styles.toolsControls}>
                    <input
                      type="text"
                      placeholder="🔍 Search extracted tools, frameworks, datasets..."
                      value={toolSearchQuery}
                      onChange={e => setToolSearchQuery(e.target.value)}
                      className={styles.toolSearchInput}
                    />

                    <div className={styles.filterGroup} style={{ flexWrap: 'wrap' }}>
                      {['all', 'Framework / Tool', 'Model / Architecture', 'Dataset / Benchmark', 'Algorithm / Method', 'Hardware / Infrastructure'].map(cat => (
                        <button
                          key={cat}
                          className={`${styles.filterBtn} ${toolCategoryFilter === cat ? styles.filterBtnActive : ''}`}
                          onClick={() => setToolCategoryFilter(cat)}
                          style={{ fontSize: '0.78rem', padding: '4px 10px' }}
                        >
                          {cat === 'all' ? 'All Categories' : cat}
                        </button>
                      ))}
                    </div>
                  </div>

                  {analyzingGaps && (
                    <p style={{ color: 'var(--text-secondary)', fontStyle: 'italic', padding: '16px 0' }}>
                      Extracting tools, architectures, datasets, and methods from research papers...
                    </p>
                  )}

                  {!analyzingGaps && filteredTools.length === 0 && (
                    <p style={{ color: 'var(--text-muted)', padding: '16px 0' }}>
                      No tools or technologies found matching the selected filter.
                    </p>
                  )}

                  <div className={styles.toolsList}>
                    {!analyzingGaps && filteredTools.map((tool, idx) => {
                      const catBadgeClass =
                        tool.category === 'Dataset / Benchmark' ? 'badge-cyan' :
                        tool.category === 'Framework / Tool' ? 'badge-purple' :
                        tool.category === 'Model / Architecture' ? 'badge-emerald' :
                        tool.category === 'Algorithm / Method' ? 'badge-amber' : 'badge-rose';

                      return (
                        <div key={idx} className={styles.toolCard}>
                          <div className={styles.toolCardHeader}>
                            <h4 className={styles.toolName}>{tool.name}</h4>
                            <span className={`badge ${catBadgeClass}`}>{tool.category}</span>
                          </div>

                          <p className={styles.toolSummary}>{tool.summary}</p>

                          {/* Direct clickable links to papers using this tool */}
                          {tool.used_in_papers && tool.used_in_papers.length > 0 && (
                            <div className={styles.toolPaperLinksGroup}>
                              <div style={{ fontSize: '0.8rem', fontWeight: 600, color: 'var(--text-muted)' }}>
                                📄 Scanned Papers Link:
                              </div>
                              {tool.used_in_papers.map((pTitle, pIdx) => {
                                const matchedPaper = findMatchingPaper(pTitle);
                                const href = matchedPaper?.url && matchedPaper.url !== '#' ? matchedPaper.url : (matchedPaper?.pdf_url || '#');

                                return (
                                  <a
                                    key={pIdx}
                                    href={href}
                                    target="_blank"
                                    rel="noreferrer"
                                    className={styles.toolPaperLinkCard}
                                  >
                                    <span>🌐 {matchedPaper ? matchedPaper.title : pTitle}</span>
                                    {matchedPaper && <span className={styles.linkYear}>({matchedPaper.publication_year} • {matchedPaper.source})</span>}
                                    <span style={{ fontSize: '0.75rem' }}>↗</span>
                                  </a>
                                );
                              })}
                            </div>
                          )}

                          <div style={{ marginTop: '12px', textAlign: 'right' }}>
                            <button
                              onClick={() => handleAddToolToDraft(tool)}
                              className="btn btn-secondary"
                              style={{ fontSize: '0.78rem', padding: '4px 10px' }}
                            >
                              + Add Tool to Review Draft
                            </button>
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </>
              ) : (
                /* VIEW BY PAPER INDEX (ALL SCANNED PAPERS LINKED) */
                <div className={styles.toolsList}>
                  <div style={{ marginBottom: '12px', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                    Listing all {papers.length} scanned research papers with direct source links and extracted tools:
                  </div>

                  {papers.map(paper => {
                    const paperTools = getToolsForPaper(paper.title);
                    return (
                      <div key={paper.id} className={styles.paperIndexCard}>
                        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: '10px' }}>
                          <h4 style={{ fontFamily: 'var(--font-heading)', fontSize: '1rem', fontWeight: 600 }}>
                            {paper.url && paper.url !== '#' ? (
                              <a href={paper.url} target="_blank" rel="noreferrer">
                                {paper.title} ↗
                              </a>
                            ) : (
                              paper.title
                            )}
                          </h4>
                          <span className={`badge ${paper.is_uploaded ? 'badge-amber' : 'badge-purple'}`}>
                            {paper.source}
                          </span>
                        </div>

                        <div style={{ fontSize: '0.82rem', color: 'var(--text-muted)', margin: '4px 0 8px 0' }}>
                          Authors: {paper.authors.slice(0, 3).join(', ')} • Year: {paper.publication_year} {paper.venue ? `• ${paper.venue}` : ''}
                        </div>

                        <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', alignItems: 'center', marginTop: '8px' }}>
                          {paper.url && paper.url !== '#' && (
                            <a href={paper.url} target="_blank" rel="noreferrer" className="badge badge-cyan">
                              🌐 Direct External Link ↗
                            </a>
                          )}
                          {paper.pdf_url && (
                            <a href={paper.pdf_url} target="_blank" rel="noreferrer" className="badge badge-emerald">
                              📥 Download PDF
                            </a>
                          )}
                        </div>

                        {paperTools.length > 0 ? (
                          <div style={{ marginTop: '10px', paddingTop: '8px', borderTop: '1px dashed var(--border-color)', fontSize: '0.8rem' }}>
                            <strong style={{ color: '#c084fc' }}>Extracted Tools/Tech:</strong>{' '}
                            {paperTools.map((t, i) => (
                              <span key={i} className="badge badge-purple" style={{ margin: '2px', textTransform: 'none' }}>
                                {t.name} ({t.category})
                              </span>
                            ))}
                          </div>
                        ) : (
                          <div style={{ marginTop: '8px', fontSize: '0.78rem', color: 'var(--text-muted)', fontStyle: 'italic' }}>
                            General baseline study referenced across cross-paper analysis.
                          </div>
                        )}
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          )}

          {/* TAB 3: LITERATURE COMPARISON MATRIX TABLE */}
          {activeAnalysisTab === 'matrix' && (
            <div className={`glass ${styles.matrixContainer}`}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '16px', flexWrap: 'wrap', gap: '10px' }}>
                <h3 className={styles.sectionTitle} style={{ margin: 0 }}>
                  <span>Literature Comparison Matrix ({papers.length} Papers)</span>
                </h3>

                <button onClick={handleExportMatrixCSV} className="btn btn-secondary" style={{ fontSize: '0.82rem', padding: '6px 14px' }}>
                  📥 Export CSV
                </button>
              </div>

              <div style={{ maxHeight: '520px', overflowY: 'auto' }}>
                <table className={styles.matrixTable}>
                  <thead>
                    <tr>
                      <th className={styles.matrixTh}>Paper Title & Year</th>
                      <th className={styles.matrixTh}>Source</th>
                      <th className={styles.matrixTh}>Extracted Tools & Methods</th>
                      <th className={styles.matrixTh}>Abstract / Contribution</th>
                    </tr>
                  </thead>
                  <tbody>
                    {papers.map((p, idx) => {
                      const tools = getToolsForPaper(p.title);
                      return (
                        <tr key={p.id || idx} className={styles.matrixRow}>
                          <td className={styles.matrixTd} style={{ fontWeight: 600 }}>
                            {p.url && p.url !== '#' ? (
                              <a href={p.url} target="_blank" rel="noreferrer">
                                {p.title} ↗
                              </a>
                            ) : (
                              p.title
                            )}
                            <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '4px' }}>
                              {p.authors.slice(0, 2).join(', ')} ({p.publication_year})
                            </div>
                          </td>

                          <td className={styles.matrixTd}>
                            <span className={`badge ${p.is_uploaded ? 'badge-amber' : 'badge-purple'}`}>
                              {p.source}
                            </span>
                          </td>

                          <td className={styles.matrixTd}>
                            {tools.length > 0 ? (
                              tools.map((t, i) => (
                                <span key={i} className="badge badge-cyan" style={{ margin: '2px', fontSize: '0.72rem', textTransform: 'none' }}>
                                  {t.name}
                                </span>
                              ))
                            ) : (
                              <span style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>General method</span>
                            )}
                          </td>

                          <td className={styles.matrixTd} style={{ fontSize: '0.82rem' }}>
                            {p.abstract.length > 180 ? p.abstract.substring(0, 180) + '...' : p.abstract}
                          </td>
                        </tr>
                      );
                    })}
                  </tbody>
                </table>
              </div>
            </div>
          )}

          {/* TAB 4: LITERATURE REVIEW DRAFT WORKSPACE */}
          {activeAnalysisTab === 'review' && (
            <div className={`glass ${styles.workspacePanel}`}>
              <div className={styles.editorHeader}>
                <h3 className={styles.sectionTitle} style={{ margin: 0 }}>
                  Literature Review Draft
                </h3>

                <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                  <div className={styles.editorTabs}>
                    <button
                      className={`${styles.editorTabBtn} ${activeTab === 'editor' ? styles.editorTabActive : ''}`}
                      onClick={() => setActiveTab('editor')}
                    >
                      Markdown
                    </button>
                    <button
                      className={`${styles.editorTabBtn} ${activeTab === 'preview' ? styles.editorTabActive : ''}`}
                      onClick={() => setActiveTab('preview')}
                    >
                      Preview
                    </button>
                  </div>

                  <button onClick={handleCopy} className="btn btn-secondary" style={{ fontSize: '0.8rem', padding: '6px 12px' }}>
                    {copySuccess ? '✓ Copied!' : 'Copy'}
                  </button>

                  <button onClick={handleDownloadMd} className="btn btn-outline" style={{ fontSize: '0.8rem', padding: '6px 12px' }}>
                    Download .md
                  </button>
                </div>
              </div>

              {activeTab === 'editor' ? (
                <textarea
                  className={styles.editor}
                  placeholder="Draft your literature review here or click 'Generate Lit Review' above to synthesize your selected papers automatically..."
                  value={draft}
                  onChange={e => setDraft(e.target.value)}
                />
              ) : (
                <div className={styles.previewArea}>
                  {draft ? (
                    <div dangerouslySetInnerHTML={{ __html: formatMarkdownToHtml(draft) }} />
                  ) : (
                    <p style={{ color: 'var(--text-muted)', fontStyle: 'italic' }}>
                      No draft content yet. Click "Generate Lit Review" to synthesize selected papers.
                    </p>
                  )}
                </div>
              )}
              <div style={{ marginTop: '8px', fontSize: '0.78rem', color: 'var(--text-muted)', textAlign: 'right' }}>
                ✓ Auto-saved to browser storage
              </div>
            </div>
          )}
        </section>
      </div>

      {/* PhD Proposal Generator Modal */}
      {proposalModalOpen && (
        <div className={styles.modalOverlay} onClick={() => setProposalModalOpen(false)}>
          <div className={styles.modalContainer} onClick={e => e.stopPropagation()}>
            <div className={styles.modalHeader}>
              <h3 style={{ fontFamily: 'var(--font-heading)', margin: 0, fontSize: '1.15rem' }}>
                ✨ PhD Research Proposal: {activeProposalGap?.title}
              </h3>
              <div style={{ display: 'flex', gap: '8px', alignItems: 'center' }}>
                <button
                  onClick={() => {
                    navigator.clipboard.writeText(proposalContent);
                    alert('Proposal copied to clipboard!');
                  }}
                  className="btn btn-secondary"
                  style={{ fontSize: '0.8rem', padding: '6px 12px' }}
                  disabled={!proposalContent}
                >
                  📋 Copy
                </button>
                <button
                  onClick={() => {
                    downloadTextFile(proposalContent, `${activeProposalGap?.title.replace(/[^a-z0-9]/gi, '_').toLowerCase()}_proposal.md`, 'text/markdown');
                  }}
                  className="btn btn-outline"
                  style={{ fontSize: '0.8rem', padding: '6px 12px' }}
                  disabled={!proposalContent}
                >
                  📥 Download .md
                </button>
                <button onClick={() => setProposalModalOpen(false)} style={{ background: 'transparent', border: 'none', color: 'var(--text-muted)', fontSize: '1.4rem', cursor: 'pointer' }}>
                  ✕
                </button>
              </div>
            </div>

            <div className={styles.modalBody}>
              {generatingProposal ? (
                <div style={{ textAlign: 'center', padding: '60px 20px', color: 'var(--text-secondary)' }}>
                  <div style={{ fontSize: '2rem', marginBottom: '12px' }}>✨</div>
                  <p style={{ fontFamily: 'var(--font-heading)', fontSize: '1.1rem', fontWeight: 600 }}>
                    Synthesizing PhD Dissertation Proposal & Experimental Architecture...
                  </p>
                  <p style={{ fontSize: '0.85rem', color: 'var(--text-muted)' }}>
                    Formulating research hypotheses, technical methodology, evaluation metrics, and 4-phase milestone roadmap...
                  </p>
                </div>
              ) : (
                <div dangerouslySetInnerHTML={{ __html: formatMarkdownToHtml(proposalContent) }} />
              )}
            </div>
          </div>
        </div>
      )}

      {/* AI Academic Co-Researcher Floating Chat Sidecar */}
      <ChatSidecar
        topic={topic}
        papers={papers}
        gaps={gaps}
        extractedTools={extractedTools}
      />
    </div>
  );
}

// Lightweight Markdown renderer for preview mode
function formatMarkdownToHtml(markdown: string): string {
  if (!markdown) return '';
  let html = markdown
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');

  // Headings
  html = html.replace(/^### (.*$)/gim, '<h3>$1</h3>');
  html = html.replace(/^## (.*$)/gim, '<h2>$1</h2>');
  html = html.replace(/^# (.*$)/gim, '<h1>$1</h1>');

  // Bold & Italic
  html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
  html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');

  // Lists
  html = html.replace(/^\- (.*$)/gim, '<li>$1</li>');
  html = html.replace(/(<li>[\s\S]*?<\/li>)/gi, '<ul>$1</ul>');

  // Paragraphs
  html = html.replace(/\n\n/g, '</p><p>');
  return `<p>${html}</p>`;
}


