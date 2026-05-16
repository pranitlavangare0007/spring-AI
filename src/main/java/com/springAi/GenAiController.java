package com.springAi;

import org.springframework.ai.chat.client.ChatClient;
import org.springframework.ai.chat.client.advisor.MessageChatMemoryAdvisor;
import org.springframework.ai.chat.memory.ChatMemory;
import org.springframework.ai.chat.memory.MessageWindowChatMemory;
import org.springframework.ai.chat.prompt.Prompt;
import org.springframework.ai.chat.prompt.PromptTemplate;
import org.springframework.ai.google.genai.GoogleGenAiChatModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
public class GenAiController {


    private final ChatClient chatClient;


    ChatMemory chatMemory= MessageWindowChatMemory
            .builder()
            .maxMessages(10)
            .build();

    public GenAiController(ChatClient.Builder builder){

       this.chatClient=builder

               .build();
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
}
