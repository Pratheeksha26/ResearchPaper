export async function callGeminiApi(
  prompt: string,
  responseMimeType?: string
): Promise<string> {
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    throw new Error('GEMINI_API_KEY is not configured in .env.local');
  }

  const model = process.env.GAPFINDER_MODEL;
  if (!model) {
    throw new Error('GAPFINDER_MODEL must select exactly one Gemini model');
  }

  const bodyObj: {
    contents: { parts: { text: string }[] }[];
    generationConfig?: { responseMimeType: string };
  } = {
    contents: [{ parts: [{ text: prompt }] }]
  };

  if (responseMimeType) {
    bodyObj.generationConfig = { responseMimeType };
  }

  const res = await fetch(
    `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent?key=${apiKey}`,
    {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(bodyObj)
    }
  );

  if (!res.ok) {
    const errText = await res.text();
    throw new Error(`Gemini API call to pinned model returned HTTP ${res.status}: ${errText}`);
  }

  const data = await res.json();
  const text = data.candidates?.[0]?.content?.parts?.[0]?.text;
  if (!text) {
    throw new Error('Gemini returned a successful response without text output');
  }
  return text;
}
