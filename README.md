# 📓 Google NotebookLM RAG Clone

A premium, high-performance RAG-powered application that allows users to upload documents and seamlessly converse with them using AI.

Built to fulfill the requirements of Assignment 03, this project has been fully optimized for scale and features a stunning modern UI.

## ✨ Key Features

- **Blazing Fast Retrievals:** Uses **Qdrant** for vector storage, completely eliminating the need to re-embed documents on every question.
- **Premium UI/UX:** A beautifully designed frontend featuring dark mode, glassmorphism, responsive micro-animations, and clean typography.
- **Document Isolation:** Supports multiple documents simultaneously by intelligently tagging vector chunks with unique Document IDs.
- **Serverless Ready:** The backend is completely stateless, making it fully compatible with serverless deployment platforms.

## 🧠 Architecture & Chunking

We employ the `RecursiveCharacterTextSplitter` strategy to divide documents into manageable pieces. This approach respects semantic boundaries (like paragraphs and sentences) before breaking them apart.
- **Chunk Size:** 1000 characters
- **Chunk Overlap:** 200 characters

Embedded chunks are stored persistently in **Qdrant** and queried dynamically using Google's `gemini-1.5-flash` model to generate highly accurate, context-aware answers.

## 🚀 Setup & Local Development

### Prerequisites
- Node.js (v18+)
- A Google Gemini API Key
- A Qdrant Instance (Local Docker or Qdrant Cloud)

### Installation

1. **Install dependencies:**
   ```bash
   npm install
   ```

2. **Configure Environment Variables:**
   Copy `.env.example` to `.env` and fill in your keys:
   ```env
   GEMINI_API_KEY="your-gemini-api-key"
   QDRANT_URL="https://your-qdrant-cloud-url.io" # OR http://localhost:6333 for local Docker
   QDRANT_API_KEY="your-qdrant-api-key"          # Leave blank if using local Docker
   ```

3. **(Optional) Start Local Qdrant Database:**
   If you aren't using Qdrant Cloud, spin up the local database using Docker:
   ```bash
   docker-compose up -d
   ```

4. **Start the Application:**
   ```bash
   npm start
   ```
   Visit `http://localhost:3000` to interact with your documents!

## 💻 CLI Tools

You can also use the integrated CLI application to interact with your documents via the terminal.

**Index a Document:**
```bash
node index.js index sample.pdf
```

**Ask a Question:**
```bash
node index.js ask "What is this document about?"
```
*(Tip: You can use the `-d <documentId>` flag to query a specific document!)*

## ☁️ Deployment

For easy deployment (e.g., Render, Railway, Vercel):
1. Push this repository to GitHub.
2. Link it to your hosting provider.
3. Ensure `GEMINI_API_KEY`, `QDRANT_URL`, and `QDRANT_API_KEY` are set in your deployment environment variables.
4. The deployment service will automatically run `npm start`.
