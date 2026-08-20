package com.credlens.backend.client;

import com.credlens.backend.dto.DecisionRequest;
import org.springframework.http.MediaType;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

import java.util.Map;

@Component
public class MlServiceClient {

    private final RestClient restClient;

    public MlServiceClient(RestClient mlServiceRestClient) {
        this.restClient = mlServiceRestClient;
    }

    public Map<String, Object> requestPrediction(DecisionRequest request) {
        String json = "{\"applicantId\": " + request.applicantId() + "}";

        return restClient.post()
                .uri("/api/v1/predictions")
                .contentType(MediaType.APPLICATION_JSON)
                .body(json)
                .retrieve()
                .body(Map.class);
    }
}