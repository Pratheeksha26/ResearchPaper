import { NextResponse } from 'next/server';
import { callGeminiApi } from '@/services/geminiService';

export async function POST(request: Request) {
  try {
    const { topic, selectedPapers, gaps, extractedTools } = await request.json();

    if (!topic || !selectedPapers || !Array.isArray(selectedPapers) || selectedPapers.length === 0) {
      return NextResponse.json({ error: 'Valid topic and selected papers array are required.' }, { status: 400 });
    }

    const papersSummary = selectedPapers
      .map((p, i) => `[Paper ${i + 1}] Title: ${p.title}\nAuthors: ${p.authors ? p.authors.join(', ') : 'Unknown'}\nYear: ${p.publication_year}\nSource: ${p.source} (${p.venue || 'N/A'})\nAbstract/Summary: ${p.abstract}\n${p.key_findings ? 'Key Findings: ' + p.key_findings.join('; ') : ''}`)
      .join('\n\n');

    const gapsSummary = gaps && Array.isArray(gaps) && gaps.length > 0
      ? gaps.map((g: any, i: number) => `Gap ${i + 1} (${g.type} - Priority: ${g.opportunity_level}): ${g.title}\n- Description: ${g.description}\n- Cross-Paper Evidence: ${g.cross_paper_evidence || 'Synthesized across multiple papers'}\n- Re-verification Status: ${g.verification_notes || 'Verified Unsolved for Present/Future'}\n- Suggested Question: ${g.suggested_question}`).join('\n\n')
      : 'None provided.';

    const toolsSummary = extractedTools && Array.isArray(extractedTools) && extractedTools.length > 0
      ? extractedTools.map((t: any, i: number) => `[Tool/Tech ${i + 1}] ${t.name} (${t.category}): ${t.summary} (Referenced in: ${t.used_in_papers ? t.used_in_papers.join(', ') : 'Selected literature'})`).join('\n')
      : 'None provided.';

    const prompt = `
You are an expert academic professor and thesis advisor.
Synthesize the provided research papers into a high-quality, structured **Literature Review Draft** on the topic: "${topic}".

Format: Return pure Markdown text with clear headings, bullet points, synthesis paragraphs, and in-text academic citations (e.g. Smith et al., 2024).

Include the following sections in your Markdown output:
# Literature Review: ${topic}

## 1. Executive Summary & Research Context
A high-level synthesis of the state of research on "${topic}".

## 2. Synthesized Themes & Core Findings
Group the provided papers into 2-3 logical thematic clusters. Compare their methodologies, datasets, and conclusions. Use in-text citations.

## 3. Extracted Tools, Technologies & Methodological Stack
Synthesize the key tools, frameworks, model architectures, datasets, and algorithms utilized across the literature:
${toolsSummary}

## 4. Re-Verified Cross-Paper Research Gaps & Open Questions
Integrate the synthesized research gaps. Explain how current literature leaves these questions unresolved and why they represent present/future research opportunities:
${gapsSummary}

## 5. Proposed Research Roadmap & Future Directions
Concrete proposals on how future studies (or the user's research project) can directly address these gaps using the extracted tools and methods.

## 6. References (APA Style)
List formatted APA references for all selected papers provided below.

---
Provided Literature for Review:
${papersSummary.substring(0, 15000)}
`;

    const markdownReview = await callGeminiApi(prompt);
    return NextResponse.json({ review: markdownReview });

  } catch (error: any) {
    console.error('API Route Error in Lit Review:', error);
    return NextResponse.json({ error: error.message || 'Internal server error during literature review generation' }, { status: 503 });
  }
}

