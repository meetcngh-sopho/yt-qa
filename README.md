---
title: YouTube Q&A
emoji: 🎥
colorFrom: red
colorTo: gray
sdk: docker
pinned: false
---

# YouTube Q&A

Ask questions about any YouTube video using RAG + LLM.

## How it works
1. Paste a YouTube URL → app fetches transcript
2. Transcript is chunked and embedded into ChromaDB
3. Ask a question → RAG retrieves relevant chunks → LLM answers

## Tech stack
- FastAPI + ChromaDB + sentence-transformers
- Groq API (llama-3.3-70b) for chat
- yt-dlp + Whisper for audio transcription fallback
