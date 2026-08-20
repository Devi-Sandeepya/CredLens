package com.credlens.backend.service;

import com.credlens.backend.client.MlServiceClient;
import com.credlens.backend.dto.DecisionRequest;
import org.springframework.stereotype.Service;

import java.util.Map;

@Service
public class DecisionService {

    private final MlServiceClient mlServiceClient;

    public DecisionService(MlServiceClient mlServiceClient) {
        this.mlServiceClient = mlServiceClient;
    }

    public Map<String, Object> makeDecision(DecisionRequest request) {
        // Persistence (decisions / audit_events tables) comes in the
        // next task, once this proxy path is confirmed working.
        return mlServiceClient.requestPrediction(request);
    }
}