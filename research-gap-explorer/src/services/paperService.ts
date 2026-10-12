export interface Paper {
  id: string;
  title: string;
  publication_year: number;
  authors: string[];
  abstract: string;
  url: string;
  pdf_url?: string;
  venue?: string;
  citation_count?: number;
  source: 'Semantic Scholar' | 'ArXiv' | 'CrossRef' | 'Uploaded';
  extracted_topics?: string[];
  key_findings?: string[];
  is_uploaded?: boolean;
}

export interface SearchMetrics {
  totalScanned: number;
  sourcesScanned: string[];
  yearRange: { min: number; max: number };
  topicsFound: string[];
}

export interface SearchResponse {
  papers: Paper[];
  metrics: SearchMetrics;
}

// Clean abstract text from HTML / XML tags or extra whitespace
export function cleanText(input: string): string {
  if (!input) return '';
  return input
    .replace(/<[^>]*>?/gm, '')
    .replace(/\s+/g, ' ')
    .trim();
}

// Helper to parse ArXiv XML response using regex
function parseArxivXml(xmlText: string): Paper[] {
  const papers: Paper[] = [];
  const entries = xmlText.split('<entry>');
  
  for (let i = 1; i < entries.length; i++) {
    const entry = entries[i];
    
    const idMatch = entry.match(/<id>([\s\S]*?)<\/id>/);
    const titleMatch = entry.match(/<title>([\s\S]*?)<\/title>/);
    const summaryMatch = entry.match(/<summary>([\s\S]*?)<\/summary>/);
    const publishedMatch = entry.match(/<published>([\s\S]*?)<\/published>/);
    const pdfMatch = entry.match(/<link[^>]*title="pdf"[^>]*href="(.*?)"/);

    const authorMatches = [...entry.matchAll(/<author>\s*<name>(.*?)<\/name>/g)];
    const authors = authorMatches.map(m => cleanText(m[1])).filter(Boolean);

    const rawTitle = titleMatch ? cleanText(titleMatch[1]) : '';
    const rawSummary = summaryMatch ? cleanText(summaryMatch[1]) : '';
    const rawId = idMatch ? cleanText(idMatch[1]) : `arxiv-${i}`;
    const yearStr = publishedMatch ? publishedMatch[1].substring(0, 4) : '';
    const year = parseInt(yearStr, 10) || new Date().getFullYear();
    const pdfUrl = pdfMatch ? pdfMatch[1] : (rawId.includes('arxiv.org') ? rawId.replace('/abs/', '/pdf/') + '.pdf' : '');

    if (rawTitle && rawSummary) {
      papers.push({
        id: rawId,
        title: rawTitle,
        publication_year: year,
        authors: authors.length > 0 ? authors : ['ArXiv Contributor'],
        abstract: rawSummary,
        url: rawId,
        pdf_url: pdfUrl,
        venue: 'ArXiv Preprint',
        source: 'ArXiv'
      });
    }
  }
  
  return papers;
}

// Multi-source paper fetcher
export async function searchLiterature(query: string): Promise<SearchResponse> {
  const sourcesScanned: string[] = [];
  let allPapers: Paper[] = [];
  let scannedCount = 0;

  // 1. Try ArXiv API
  try {
    const arxivUrl = `https://export.arxiv.org/api/query?search_query=all:${encodeURIComponent(query)}&start=0&max_results=30`;
    const res = await fetch(arxivUrl);
    if (res.ok) {
      const xml = await res.text();
      const arxivPapers = parseArxivXml(xml);
      allPapers.push(...arxivPapers);
      scannedCount += arxivPapers.length;
      sourcesScanned.push('ArXiv');
    }
  } catch (err) {
    console.warn('ArXiv search failed:', err);
  }

  // 2. Try CrossRef API
  try {
    const crossRefUrl = `https://api.crossref.org/works?query=${encodeURIComponent(query)}&select=title,author,abstract,URL,published,container-title,is-referenced-by-count&rows=25`;
    const res = await fetch(crossRefUrl);
    if (res.ok) {
      const data = await res.json();
      const items = data.message?.items || [];
      scannedCount += items.length;
      sourcesScanned.push('CrossRef');

      const crossRefPapers: Paper[] = items.map((item: any, idx: number) => {
        const title = item.title?.[0] ? cleanText(item.title[0]) : 'Untitled Work';
        const abstract = item.abstract ? cleanText(item.abstract) : 'Abstract not indexed in CrossRef repository.';
        const year = item.published?.['date-parts']?.[0]?.[0] || new Date().getFullYear();
        const authors = item.author
          ? item.author.map((a: any) => `${a.given || ''} ${a.family || ''}`.trim()).filter(Boolean)
          : [];
        const venue = item['container-title']?.[0] || 'Academic Journal';

        return {
          id: item.URL || `crossref-${idx}-${Date.now()}`,
          title,
          publication_year: year,
          authors: authors.length > 0 ? authors : ['Unknown Author'],
          abstract,
          url: item.URL || '',
          venue,
          citation_count: item['is-referenced-by-count'] || 0,
          source: 'CrossRef'
        };
      });

      allPapers.push(...crossRefPapers);
    }
  } catch (err) {
    console.warn('CrossRef search failed:', err);
  }

  // 3. Try Semantic Scholar API
  try {
    const semUrl = `https://api.semanticscholar.org/graph/v1/paper/search?query=${encodeURIComponent(query)}&limit=25&fields=paperId,title,year,authors,abstract,url,openAccessPdf,venue,citationCount`;
    const res = await fetch(semUrl);
    if (res.ok) {
      const data = await res.json();
      const items = data.data || [];
      scannedCount += items.length;
      sourcesScanned.push('Semantic Scholar');

      const semPapers: Paper[] = items.map((item: any) => ({
        id: item.paperId,
        title: cleanText(item.title || 'Untitled'),
        publication_year: item.year || new Date().getFullYear(),
        authors: item.authors?.map((a: any) => a.name).filter(Boolean) || ['Unknown'],
        abstract: cleanText(item.abstract || 'Abstract unavailable on Semantic Scholar.'),
        url: item.url || `https://www.semanticscholar.org/paper/${item.paperId}`,
        pdf_url: item.openAccessPdf?.url,
        venue: item.venue || 'Semantic Scholar',
        citation_count: item.citationCount || 0,
        source: 'Semantic Scholar'
      }));

      allPapers.push(...semPapers);
    }
  } catch (err) {
    console.warn('Semantic Scholar search rate-limited or unavailable:', err);
  }

  // Deduplicate by normalized lower-case title
  const seenTitles = new Set<string>();
  const uniquePapers: Paper[] = [];

  for (const paper of allPapers) {
    const normTitle = paper.title.toLowerCase().replace(/[^a-z0-9]/g, '');
    if (normTitle && !seenTitles.has(normTitle)) {
      seenTitles.add(normTitle);
      uniquePapers.push(paper);
    }
  }

  // Sort papers: papers with detailed abstracts first, then by publication year descending
  uniquePapers.sort((a, b) => {
    const aHasAbstract = a.abstract && !a.abstract.includes('unavailable') && !a.abstract.includes('not indexed');
    const bHasAbstract = b.abstract && !b.abstract.includes('unavailable') && !b.abstract.includes('not indexed');
    if (aHasAbstract && !bHasAbstract) return -1;
    if (!aHasAbstract && bHasAbstract) return 1;
    return b.publication_year - a.publication_year;
  });

  // Calculate year range
  const years = uniquePapers.map(p => p.publication_year).filter(y => y > 1900);
  const minYear = years.length > 0 ? Math.min(...years) : new Date().getFullYear();
  const maxYear = years.length > 0 ? Math.max(...years) : new Date().getFullYear();

  // Return all unique scanned papers for complete user inspection
  return {
    papers: uniquePapers,
    metrics: {
      totalScanned: Math.max(scannedCount, uniquePapers.length),
      sourcesScanned: Array.from(new Set(sourcesScanned)),
      yearRange: { min: minYear, max: maxYear },
      topicsFound: [query]
    }
  };
}

