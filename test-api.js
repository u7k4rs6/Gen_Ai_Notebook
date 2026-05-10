import "dotenv/config";
import fs from "fs";
import { RecursiveCharacterTextSplitter } from "@langchain/textsplitters";
import { GoogleGenAIEmbeddings, ChatGoogleGenAI } from "@langchain/google-genai";
import { MemoryVectorStore } from "langchain/vectorstores/memory";

async function testPipeline() {
    try {
        console.log("1. Loading sample document...");
        const text = fs.readFileSync("sample.txt", "utf-8");
        const docs = [{ pageContent: text, metadata: { source: "sample.txt" } }];

        console.log("2. Chunking...");
        const textSplitter = new RecursiveCharacterTextSplitter({ chunkSize: 100, chunkOverlap: 20 });
        const chunkedDocs = await textSplitter.splitDocuments(docs);

        console.log("3. Generating Embeddings (Testing Gemini API)...");
        const embeddings = new GoogleGenAIEmbeddings({ model: "text-embedding-004" });
        const vectorStore = await MemoryVectorStore.fromDocuments(chunkedDocs, embeddings);

        console.log("4. Retrieval & Generation (Testing Gemini Flash)...");
        const query = "What is LangChain?";
        const retriever = vectorStore.asRetriever({ k: 2 });
        const searchedChunks = await retriever.invoke(query);

        const model = new ChatGoogleGenAI({ modelName: "gemini-1.5-flash", temperature: 0.1 });
        const systemPrompt = `Answer based only on this context:\n${searchedChunks.map(c => c.pageContent).join('\n')}`;

        const response = await model.invoke([
            ["system", systemPrompt],
            ["human", query]
        ]);

        console.log("\n✅ Test Successful!");
        console.log("Question:", query);
        console.log("Answer:", response.content);
    } catch (error) {
        console.error("❌ Test Failed:", error.message);
    }
}

testPipeline();
