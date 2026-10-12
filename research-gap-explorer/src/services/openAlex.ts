export interface Paper {
  id: string;
  title: string;
  publication_year: number;
  authors: string[];
  abstract: string;
  url: string;
}

export async function fetchPapers(query: string): Promise<Paper[]> {
  try {
    const res = await fetch(`/api/search?q=${encodeURIComponent(query)}`);
    
    if (!res.ok) {
      const errText = await res.text();
      console.error("Proxy API Error:", res.status, errText);
      throw new Error('Failed to fetch from internal proxy');
    }
    
    const data = await res.json();
    return data.data || [];
  } catch (error) {
    console.error(error);
    return [];
  }
}
