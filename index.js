import "dotenv/config";
import { PDFLoader } from "@langchain/community/document_loaders/fs/pdf";
import { TextLoader } from "langchain/document_loaders/fs/text";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenAIEmbeddings, ChatGoogleGenAI } from "@langchain/google-genai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { program } from "commander";
import path from "path";
import crypto from "crypto";

program
  .name('notebook-lm-rag')
  .description('A RAG-powered CLI application to chat with your documents')
  .version('1.0.0');

const COLLECTION_NAME = "notebook_lm_rag_collection";

const getVectorStoreConfig = () => ({
  url: process.env.QDRANT_URL || "http://localhost:6333",
  collectionName: COLLECTION_NAME,
  ...(process.env.QDRANT_API_KEY && { apiKey: process.env.QDRANT_API_KEY })
});

program
  .command('index')
  .description('Index a PDF or text document into the vector database')
  .argument('<filePath>', 'Path to the document to index')
  .action(async (filePath) => {
    try {
      console.log(`Indexing document: ${filePath}`);
      const ext = path.extname(filePath).toLowerCase();
      let loader;

      if (ext === '.pdf') {
        loader = new PDFLoader(filePath);
      } else if (ext === '.txt') {
        loader = new TextLoader(filePath);
      } else {
        console.error("Unsupported file type. Please upload a .pdf or .txt file.");
        process.exit(1);
      }

      console.log("Loading document...");
      const docs = await loader.load();

      // Chunking strategy: RecursiveCharacterTextSplitter
      console.log("Chunking document...");
      const textSplitter = new RecursiveCharacterTextSplitter({
        chunkSize: 1000,
        chunkOverlap: 200,
      });

      const chunkedDocs = await textSplitter.splitDocuments(docs);
      console.log(`Created ${chunkedDocs.length} chunks from the document.`);

      const documentId = crypto.randomUUID();
      const docsWithMetadata = chunkedDocs.map(doc => {
          doc.metadata = { ...doc.metadata, documentId };
          return doc;
      });

      const embeddings = new GoogleGenAIEmbeddings({
        model: "text-embedding-004",
      });

      console.log("Generating embeddings and saving to Qdrant...");
      await QdrantVectorStore.fromDocuments(docsWithMetadata, embeddings, getVectorStoreConfig());

      console.log(`Indexing completed successfully. Document ID: ${documentId}`);
    } catch (error) {
      console.error("Error during indexing:", error.message);
    }
  });

program
  .command('ask')
  .description('Ask a question based on the indexed documents')
  .argument('<query>', 'The question to ask')
  .option('-d, --document-id <id>', 'Filter by specific document ID')
  .action(async (query, options) => {
    try {
      console.log(`Querying: "${query}"`);
      if (options.documentId) {
          console.log(`Filtering by Document ID: ${options.documentId}`);
      }
      
      const embeddings = new GoogleGenAIEmbeddings({
        model: "text-embedding-004",
      });

      const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, getVectorStoreConfig());

      const filter = options.documentId ? {
          must: [
              {
                  key: "metadata.documentId",
                  match: {
                      value: options.documentId
                  }
              }
          ]
      } : undefined;

      const retriever = vectorStore.asRetriever({
        k: 4,
        filter
      });

      console.log("Retrieving relevant context...");
      const searchedChunks = await retriever.invoke(query);

      if (searchedChunks.length === 0) {
         console.log("No relevant context found in the indexed documents.");
         return;
      }

      const model = new ChatGoogleGenAI({
        modelName: "gemini-1.5-flash",
        temperature: 0.1
      });

      const systemPrompt = `You are an AI Assistant that helps resolve user queries based strictly on the provided context from indexed documents.

Rule:
- Only answer based on the available context provided below.
- If the answer cannot be found in the context, clearly state that you do not know. Do not use outside knowledge.
- Keep your answers concise and accurate.

Context:
${searchedChunks.map(c => c.pageContent).join('\n\n')}
`;

      console.log("Generating answer...\n");
      const response = await model.invoke([
        ["system", systemPrompt],
        ["human", query]
      ]);

      console.log("=== Answer ===");
      console.log(response.content);
      console.log("==============");
      
      console.log("\nSources:");
      searchedChunks.forEach((chunk, index) => {
         const source = chunk.metadata.source || 'Unknown';
         const docId = chunk.metadata.documentId || 'Unknown';
         const loc = chunk.metadata.loc?.pageNumber ? `Page ${chunk.metadata.loc.pageNumber}` : `Lines ${chunk.metadata.loc?.lines?.from || '?'}-${chunk.metadata.loc?.lines?.to || '?'}`;
         console.log(`[${index + 1}] ${source} (DocID: ${docId}) - ${loc}`);
      });

    } catch (error) {
      console.error("Error during retrieval/generation:", error.message);
    }
  });

program.parse(process.argv);
