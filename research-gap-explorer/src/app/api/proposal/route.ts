import { NextResponse } from 'next/server';
import { callGeminiApi } from '@/services/geminiService';

export async function POST(request: Request) {
  try {
    const { topic, gap, supportingPapers } = await request.json();

    if (!topic || !gap) {
      return NextResponse.json({ error: 'Topic and research gap are required.' }, { status: 400 });
    }

    const papersSummary = supportingPapers && Array.isArray(supportingPapers) && supportingPapers.length > 0
      ? supportingPapers.map((p: any, i: number) => `[Paper ${i + 1}] "${p.title}" (${p.publication_year})\nAuthors: ${p.authors ? p.authors.join(', ') : 'Unknown'}\nAbstract: ${p.abstract}`).join('\n\n')
      : 'Literature corpus provided.';

    const prompt = `
You are a distinguished university research chair and PhD dissertation committee advisor.
Generate a comprehensive, publication-grade **PhD Research Proposal** based on the following Research Gap:

Topic: "${topic}"
Gap Title: "${gap.title}" (${gap.type} Gap - Priority: ${gap.opportunity_level})
Gap Description: "${gap.description}"
Proposed Research Question: "${gap.suggested_question}"
Cross-Paper Evidence: "${gap.cross_paper_evidence || 'Synthesized from literature'}"

Format: Pure Markdown with clean headers, bullet points, formal mathematical hypothesis statements (\(H_1, H_2\)), and step-by-step methodologies.

Include the following sections in your output:
# PhD Research Proposal: ${gap.title}

## 1. Executive Summary & Problem Statement
Explain why current state of the art leaves this gap unresolved and why addressing it is critical for present/future (2024-2026+) research.

## 2. Formal Research Hypotheses
Formulate 2-3 precise testable hypotheses (\(H_1, H_2, H_3\)) addressing "${gap.suggested_question}".

## 3. Literature Baseline & Theoretical Framework
Synthesize how existing literature sets up this challenge and where current methods fail.

## 4. Proposed Technical Methodology & Experimental Architecture
Step-by-step technical plan:
- **Model Architecture / Algorithmic Innovations**
- **Datasets & Benchmarks to Evaluate**
- **Baseline Models to Compare Against**
- **Primary Metrics (e.g. Accuracy, Latency, F1, Loss)**

## 5. Expected Novelty, Key Contributions & Real-World Impact
What new capabilities will this research unlock?

## 6. Project Timeline & Key Milestones
A 4-phase milestone roadmap (Months 1–6, 7–12, 13–18, 19–24).

---
Supporting Literature:
${papersSummary.substring(0, 10000)}
`;

    const markdownProposal = await callGeminiApi(prompt);
    return NextResponse.json({ proposal: markdownProposal });

  } catch (error: any) {
    console.error('API Route Error in Proposal:', error);
    return NextResponse.json({ error: error.message || 'Internal server error during proposal generation' }, { status: 503 });
  }
}
