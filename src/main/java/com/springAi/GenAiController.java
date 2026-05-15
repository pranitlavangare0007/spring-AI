package com.springAi;

import org.springframework.ai.google.genai.GoogleGenAiChatModel;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PathVariable;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class GenAiController {

    @Autowired
    private GoogleGenAiChatModel chatModel;


    @GetMapping("/{message}")
    public String getAns(@PathVariable String message){
        String res = chatModel.call(message);
        return res;
    }
}
