import { NextResponse } from 'next/server';
import { callGeminiApi } from '@/services/geminiService';

export async function POST(request: Request) {
  try {
    const { topic, message, papers, gaps, extractedTools, chatHistory } = await request.json();

    if (!message) {
      return NextResponse.json({ error: 'Message is required' }, { status: 400 });
    }

    const papersContext = papers && Array.isArray(papers)
      ? papers.slice(0, 30).map((p: any, i: number) => `[Paper ${i + 1}] "${p.title}" (${p.publication_year}) by ${p.authors ? p.authors.slice(0, 2).join(', ') : 'Unknown'} - Source: ${p.source} (${p.url || ''})\nAbstract: ${p.abstract.substring(0, 250)}`).join('\n\n')
      : 'No papers context.';

    const gapsContext = gaps && Array.isArray(gaps)
      ? gaps.map((g: any, i: number) => `[Gap ${i + 1}] ${g.title} (${g.type}): ${g.description} (Q: ${g.suggested_question})`).join('\n')
      : 'No gaps context.';

    const toolsContext = extractedTools && Array.isArray(extractedTools)
      ? extractedTools.map((t: any) => `- ${t.name} (${t.category}): ${t.summary}`).join('\n')
      : 'No tools context.';

    const historyContext = chatHistory && Array.isArray(chatHistory) && chatHistory.length > 0
      ? chatHistory.map((h: any) => `${h.role === 'user' ? 'User' : 'AI Advisor'}: ${h.content}`).join('\n')
      : '';

    const prompt = `
You are an expert AI Academic Co-Researcher & Peer Review Director assisting a researcher on the topic: "${topic}".
You have deep, complete context of all ${papers ? papers.length : 0} scanned papers in their literature corpus, identified research gaps, and extracted tools.

CONTEXT SUMMARY:
---
Identified Research Gaps:
${gapsContext}

Extracted Tools & Technologies:
${toolsContext}

Sample Scanned Literature Corpus (Top Papers):
${papersContext.substring(0, 15000)}
---

CONVERSATION HISTORY:
${historyContext}

USER QUESTION:
"${message}"

INSTRUCTIONS:
- Provide an insightful, crisp, academically rigorous answer.
- Reference exact paper titles, authors, tools, or gap IDs whenever appropriate.
- If asked about specific papers or comparisons, give clear bullet points and direct answers.
- Keep the tone professional, supportive, and thesis-advisor level.
`;

    const reply = await callGeminiApi(prompt);
    return NextResponse.json({ reply });

  } catch (error: any) {
    console.error('API Route Error in Co-Researcher Chat:', error);
    return NextResponse.json({ error: error.message || 'Internal server error during chat response' }, { status: 503 });
  }
}
