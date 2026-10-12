import { NextResponse } from 'next/server';
import { callGeminiApi } from '@/services/geminiService';
// eslint-disable-next-line @typescript-eslint/no-require-imports
const pdfParse = require('pdf-parse/lib/pdf-parse.js');

export async function POST(request: Request) {
  try {
    const formData = await request.formData();
    const file = formData.get('file') as File | null;

    if (!file) {
      return NextResponse.json({ error: 'No file uploaded' }, { status: 400 });
    }

    const buffer = Buffer.from(await file.arrayBuffer());
    let rawText = '';

    if (file.name.toLowerCase().endsWith('.pdf')) {
      try {
        const parsedPdf = await pdfParse(buffer);
        rawText = parsedPdf.text || '';
      } catch (pdfErr) {
        console.error('PDF parsing error:', pdfErr);
        return NextResponse.json({ error: 'Failed to extract text from PDF' }, { status: 422 });
      }
    } else {
      rawText = buffer.toString('utf-8');
    }

    if (!rawText.trim()) {
      return NextResponse.json({ error: 'Uploaded file is empty or unreadable' }, { status: 422 });
    }

    const textSnippet = rawText.substring(0, 10000);

    const prompt = `
You are an expert scientific paper analyzer.
Analyze the following text extracted from an academic research paper file titled "${file.name}".

Extract and return a STRICT JSON object with the following fields:
- "title": The exact or best inferred title of the paper.
- "authors": An array of author names (string[]).
- "publication_year": The publication year as an integer (e.g. 2024).
- "abstract": A concise abstract or summary of the paper (150-250 words).
- "extracted_topics": An array of 3-5 core search topic strings/keywords that can be used to query literature databases for similar papers (e.g., ["Graph Neural Networks", "Drug Toxicity Prediction"]).
- "key_findings": An array of 3-4 key findings, contributions, or methodologies demonstrated in this paper (string[]).

Paper Text:
${textSnippet}
    `;

    const textResponse = await callGeminiApi(prompt, 'application/json');
    const parsedData = JSON.parse(textResponse);

    const uploadedPaper = {
      id: `uploaded-${Date.now()}-${Math.random().toString(36).substr(2, 5)}`,
      title: parsedData.title || file.name.replace(/\.[^/.]+$/, ''),
      publication_year: parsedData.publication_year || new Date().getFullYear(),
      authors: parsedData.authors || ['Uploaded Author'],
      abstract: parsedData.abstract || 'No abstract extracted.',
      url: '#',
      source: 'Uploaded' as const,
      extracted_topics: parsedData.extracted_topics || [file.name],
      key_findings: parsedData.key_findings || [],
      is_uploaded: true
    };

    return NextResponse.json({ paper: uploadedPaper });

  } catch (error: any) {
    console.error('Upload route error:', error);
    return NextResponse.json({ error: error.message || 'Internal server error during upload processing' }, { status: 500 });
  }
}
