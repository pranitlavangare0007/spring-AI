package com.example.springAi;



import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.client.advisor.vectorstore.QuestionAnswerAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.chat.model.ChatModel;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.document.Document;
import org.springframework.ai.embedding.EmbeddingModel;
import org.springframework.ai.google.genai.GoogleGenAiChatModel;
import org.springframework.ai.vectorstore.SearchRequest;
import org.springframework.ai.vectorstore.VectorStore;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.beans.factory.annotation.Qualifier;
import org.springframework.context.annotation.Primary;
import org.springframework.web.bind.annotation.*;


import java.util.List;
import java.util.Map;

@RestController
public class GenAiController {


    private final ChatClient chatClient;
    private final EmbeddingModel embeddingModel;

    @Autowired
    VectorStore vectorStore;

    public GenAiController(
            @Qualifier("googleGenAiChatModel") ChatModel chatModel,
            EmbeddingModel embeddingModel
    ) {

        this.chatClient = ChatClient.create(chatModel);

        this.embeddingModel = embeddingModel;
    }


    @GetMapping("/{message}")
    public String getAns(@PathVariable String message ,@RequestParam String conversationId){

        return this.chatClient.prompt()
                .user(message)
                .advisors(a -> a.param(
                        ChatMemory.CONVERSATION_ID,
                        conversationId
                ))
                .call()
                .content();
    }

    @PostMapping("/api/recom")
    public String recommendation(@RequestParam String type , @RequestParam  String year , @RequestParam String lang){

        String prom= """
                Recommend exactly ONE movie based on the following inputs:
                - Genre/Type: {type}
                - Release Year: {year}
                - Language: {lang}
                
                Requirements:
                - Return only ONE best movie recommendation.
                - Include:
                  1. Movie Name
                  2. Release Year
                  3. Language
                  4. Genre
                  5. IMDb Rating
                  6. Short 2-3 line description
                  7. Why this movie matches the user's preference
                
                """;

        PromptTemplate template = new PromptTemplate(prom);
        Prompt prompt = template.create(Map.of("type",type,"year",year,"lang",lang));

        String res = chatClient
                .prompt(prompt)

                .call()
                .content();

        return res;
    }

    @PostMapping("emb")
    public float[] getEmb(@RequestParam String text){

        return embeddingModel.embed(text);
    }
    @PostMapping("sim")
    public double sim(@RequestParam String text1,@RequestParam String text2){

        float[] vec1=embeddingModel.embed(text1);
        float[] vec2=embeddingModel.embed(text2);

        double dotProduct = 0.0;
        double normA = 0.0;
        double normB = 0.0;

        for (int i = 0; i < vec1.length; i++) {

            dotProduct += vec1[i] * vec2[i];

            normA += Math.pow(vec1[i], 2);

            normB += Math.pow(vec2[i], 2);
        }

        return dotProduct * 100 / (Math.sqrt(normA) * Math.sqrt(normB));
    }

    @PostMapping("product")
    public List<Document> getProduct(@RequestParam String text){

//     return  vectorStore.similaritySearch(text);
        return vectorStore.similaritySearch(SearchRequest.builder().query(text).topK(1).build());

    }

    @PostMapping("/ask")
    public String getAnsWithRag(@RequestParam String query){

        return chatClient.prompt(query)
                .advisors(QuestionAnswerAdvisor.builder(vectorStore).build())
                .call()
                .content();
    }
}