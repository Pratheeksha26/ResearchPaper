import { Paper } from './paperService';

// Generate clean citation key from author and year
function generateCiteKey(paper: Paper, index: number): string {
  const firstAuthor = paper.authors[0]
    ? paper.authors[0].split(' ').pop()?.replace(/[^a-zA-Z]/g, '').toLowerCase() || 'author'
    : 'author';
  const year = paper.publication_year || new Date().getFullYear();
  return `${firstAuthor}${year}_${index + 1}`;
}

// Convert paper list to BibTeX format
export function generateBibTeX(papers: Paper[]): string {
  if (!papers || papers.length === 0) return '% No papers selected for BibTeX export\n';

  return papers.map((paper, idx) => {
    const key = generateCiteKey(paper, idx);
    const authorsFormatted = paper.authors && paper.authors.length > 0
      ? paper.authors.join(' and ')
      : 'Unknown Author';
    const venue = paper.venue || paper.source || 'Academic Literature';
    const titleEscaped = paper.title.replace(/[{}]/g, '');

    return `@article{${key},
  title     = {${titleEscaped}},
  author    = {${authorsFormatted}},
  journal   = {${venue}},
  year      = {${paper.publication_year}},
  url       = {${paper.url || '#'}},
  publisher = {${paper.source}}
}`;
  }).join('\n\n');
}

// Convert paper list to RIS format (compatible with Zotero, Mendeley, EndNote)
export function generateRIS(papers: Paper[]): string {
  if (!papers || papers.length === 0) return 'TY  - JOUR\nER  -\n';

  return papers.map(paper => {
    const authorLines = paper.authors && paper.authors.length > 0
      ? paper.authors.map(a => `AU  - ${a}`).join('\n')
      : 'AU  - Unknown Author';

    const venue = paper.venue || paper.source || 'Academic Literature';

    return `TY  - JOUR
TI  - ${paper.title}
${authorLines}
PY  - ${paper.publication_year}
JO  - ${venue}
UR  - ${paper.url || '#'}
ER  -`;
  }).join('\n\n');
}

// Helper to trigger file download in browser
export function downloadTextFile(content: string, filename: string, mimeType: string = 'text/plain;charset=utf-8') {
  const blob = new Blob([content], { type: mimeType });
  const url = URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
}
