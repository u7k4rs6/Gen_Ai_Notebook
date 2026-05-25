import "dotenv/config";
import express from "express";
import multer from "multer";
import { getDocument } from "pdfjs-dist/legacy/build/pdf.mjs";
import { Document } from "@langchain/core/documents";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI } from "@langchain/google-genai";
import { QdrantVectorStore } from "@langchain/qdrant";
import { QdrantClient } from "@qdrant/js-client-rest";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import crypto from "crypto";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const app = express();
const port = process.env.PORT || 3000;

app.use(express.static(path.join(__dirname, "public")));
app.use(express.json());

const upload = multer({ dest: '/tmp/' });

const COLLECTION_NAME = "notebook_lm_rag_collection";

const getVectorStoreConfig = () => ({
    url: process.env.QDRANT_URL || "http://localhost:6333",
    collectionName: COLLECTION_NAME,
    ...(process.env.QDRANT_API_KEY && { apiKey: process.env.QDRANT_API_KEY })
});

app.post("/api/upload", upload.single("document"), async (req, res) => {
    try {
        if (!req.file) {
            return res.status(400).json({ error: "No file uploaded" });
        }
        
        console.log(`Uploaded file: ${req.file.path}`);
        const buffer = fs.readFileSync(req.file.path);
        const pdfDoc = await getDocument({ data: new Uint8Array(buffer) }).promise;
        const pageTexts = [];
        for (let i = 1; i <= pdfDoc.numPages; i++) {
            const page = await pdfDoc.getPage(i);
            const content = await page.getTextContent();
            pageTexts.push(content.items.map(item => item.str).join(" "));
        }
        const docs = [new Document({
            pageContent: pageTexts.join("\n\n"),
            metadata: { source: req.file.originalname || req.file.path }
        })];
        
        const textSplitter = new RecursiveCharacterTextSplitter({
            chunkSize: 1000,
            chunkOverlap: 200,
        });

        const documentId = crypto.randomUUID();
        const chunkedDocs = await textSplitter.splitDocuments(docs);
        
        // Tag chunks with documentId
        const docsWithMetadata = chunkedDocs.map(doc => {
            doc.metadata = { ...doc.metadata, documentId };
            return doc;
        });

        const embeddings = new GoogleGenerativeAIEmbeddings({
            model: "gemini-embedding-001",
        });

        console.log(`Storing ${docsWithMetadata.length} chunks in Qdrant...`);
        await QdrantVectorStore.fromDocuments(docsWithMetadata, embeddings, getVectorStoreConfig());

        const qdrantClient = new QdrantClient({
            url: process.env.QDRANT_URL || "http://localhost:6333",
            ...(process.env.QDRANT_API_KEY && { apiKey: process.env.QDRANT_API_KEY })
        });
        await qdrantClient.createPayloadIndex(COLLECTION_NAME, {
            field_name: "metadata.documentId",
            field_schema: "keyword"
        }).catch(() => {}); // ignore if index already exists

        console.log(`Successfully stored in Qdrant. Document ID: ${documentId}`);

        res.json({ 
            message: "Document loaded successfully", 
            documentId
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: error.message });
    } finally {
        if (req.file && fs.existsSync(req.file.path)) {
            fs.unlinkSync(req.file.path);
        }
    }
});

app.post("/api/ask", async (req, res) => {
    try {
        const { query, documentId } = req.body;
        if (!query || !documentId) {
            return res.status(400).json({ error: "Query and documentId are required" });
        }

        const embeddings = new GoogleGenerativeAIEmbeddings({
            model: "gemini-embedding-001",
        });

        const vectorStore = await QdrantVectorStore.fromExistingCollection(embeddings, getVectorStoreConfig());
        
        const retriever = vectorStore.asRetriever({
            k: 4,
            filter: {
                must: [
                    {
                        key: "metadata.documentId",
                        match: {
                            value: documentId
                        }
                    }
                ]
            }
        });

        const searchedChunks = await retriever.invoke(query);

        if (searchedChunks.length === 0) {
            return res.json({ answer: "No relevant context found in the indexed documents." });
        }

        const model = new ChatGoogleGenerativeAI({
            model: "gemini-2.5-flash",
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

        const response = await model.invoke([
            ["system", systemPrompt],
            ["human", query]
        ]);

        res.json({ 
            answer: response.content,
            sources: searchedChunks.map(c => c.metadata.source || "Unknown")
        });
    } catch (error) {
        console.error(error);
        res.status(500).json({ error: error.message });
    }
});

if (process.env.NODE_ENV !== 'production' && !process.env.VERCEL) {
    app.listen(port, () => {
        console.log(`Server running at http://localhost:${port}`);
    });
}

export default app;
