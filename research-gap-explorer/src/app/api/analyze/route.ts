import { NextResponse } from 'next/server';
import { callGeminiApi } from '@/services/geminiService';

export async function POST(request: Request) {
  try {
    const { topic, papers } = await request.json();

    if (!topic || !papers || !Array.isArray(papers) || papers.length === 0) {
      return NextResponse.json({ error: 'Valid topic and papers array required' }, { status: 400 });
    }

    const papersText = papers
      .map((p, i) => `[Paper ${i + 1}] "${p.title}" (${p.publication_year})\nAuthors: ${p.authors ? p.authors.join(', ') : 'Unknown'}\nSource: ${p.source} (${p.venue || 'N/A'})\nAbstract: ${p.abstract}\n${p.key_findings ? 'Key Findings: ' + p.key_findings.join('; ') : ''}`)
      .join('\n\n---\n\n');

    const prompt = `
You are a senior academic research strategist and peer review director.
Analyze the following entire set of ${papers.length} research papers on the topic: "${topic}".

Perform three main tasks:

TASK 1: CROSS-PAPER GAP ANALYSIS (SYNTHESIS ACROSS ALL PAPERS)
- Identify 4 to 6 distinct, high-value "Research Gaps".
- CRITICAL REQUIREMENT: Gaps MUST be synthesized across ALL provided papers as a collective body of literature. Do NOT create a gap based solely on one single paper in isolation.
- Each gap MUST cite 2 or more paper titles from the provided list in "supporting_paper_titles", explaining how these papers collectively leave this problem open.

TASK 2: RE-VERIFICATION & TIMELINESS FILTERING
- Re-verify each candidate gap against all papers in the corpus to confirm that NO paper in the dataset has already solved or invalidated this gap.
- Timeliness requirement: The gap MUST target present (2024-2026) or future-facing research horizons, NOT obsolete or historic questions that were already solved in past decades.
- Provide a clear "verification_notes" field explaining why the gap remains open despite the existing literature.

TASK 3: EXTRACTED TOOLS & TECHNOLOGIES SUMMARY
- Skim across all papers and extract 6 to 12 key tools, technologies, frameworks, datasets, model architectures, or specialized algorithms referenced or utilized across the papers.
- For each item, provide its name, category, a 1-2 sentence short summary of how it is used/relevant in these papers, and the list of paper titles using or mentioning it.

Return a STRICT JSON object with two top-level keys: "gaps" and "extracted_tools".

Schema:
{
  "gaps": [
    {
      "id": "gap-1",
      "title": "Short gap title (5-8 words)",
      "type": "Methodological" | "Theoretical" | "Empirical" | "Demographic" | "Application",
      "description": "2-3 sentence thorough explanation of the synthesized gap.",
      "opportunity_level": "High" | "Medium" | "Emerging",
      "suggested_question": "Formal PhD-level research question.",
      "supporting_paper_titles": ["Exact Title 1", "Exact Title 2"],
      "cross_paper_evidence": "1-2 sentence evidence showing how comparing these papers reveals the gap.",
      "reverified": true,
      "timeliness": "Present & Future Horizon (2024-2026+)",
      "verification_notes": "Explanation verifying why no existing paper in the set has solved this."
    }
  ],
  "extracted_tools": [
    {
      "name": "Name of tool/tech/dataset/algorithm",
      "category": "Framework / Tool" | "Model / Architecture" | "Dataset / Benchmark" | "Algorithm / Method" | "Hardware / Infrastructure",
      "summary": "Short 1-2 sentence description of its usage and importance across the papers.",
      "used_in_papers": ["Exact Title 1", "Exact Title 2"]
    }
  ]
}

Papers for Analysis:
${papersText.substring(0, 80000)}
`;

    const textResponse = await callGeminiApi(prompt, 'application/json');

    // Clean response if wrapped in markdown code blocks
    let cleanedText = textResponse.trim();
    if (cleanedText.startsWith('```')) {
      cleanedText = cleanedText.replace(/^```(json)?\n?/, '').replace(/\n?```$/, '').trim();
    }

    const parsedData = JSON.parse(cleanedText);

    const gaps = Array.isArray(parsedData) ? parsedData : (parsedData.gaps || []);
    const extracted_tools = Array.isArray(parsedData.extracted_tools) ? parsedData.extracted_tools : [];

    return NextResponse.json({ gaps, extracted_tools });

  } catch (error: any) {
    console.error('API Route Error in Analyze:', error);
    return NextResponse.json({ error: error.message || 'Internal server error during gap analysis' }, { status: 503 });
  }
}

