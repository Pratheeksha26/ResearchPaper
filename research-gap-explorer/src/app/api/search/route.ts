import { NextResponse } from 'next/server';
import { searchLiterature } from '@/services/paperService';

export async function GET(request: Request) {
  const { searchParams } = new URL(request.url);
  const query = searchParams.get('q');

  if (!query) {
    return NextResponse.json({ error: 'Query parameter is required' }, { status: 400 });
  }

  try {
    const result = await searchLiterature(query);
    return NextResponse.json({
      data: result.papers,
      metrics: result.metrics
    });
  } catch (error) {
    console.error('API Route Error:', error);
    return NextResponse.json({ error: 'Internal server error during search' }, { status: 500 });
  }
}

