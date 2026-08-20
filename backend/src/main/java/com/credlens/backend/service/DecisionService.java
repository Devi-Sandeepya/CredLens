package com.credlens.backend.service;

import com.credlens.backend.client.MlServiceClient;
import com.credlens.backend.dto.DecisionRequest;
import com.credlens.backend.entity.Decision;
import com.credlens.backend.repository.DecisionRepository;
import org.springframework.stereotype.Service;

import java.time.OffsetDateTime;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.UUID;

@Service
public class DecisionService {

    private final MlServiceClient mlServiceClient;
    private final DecisionRepository decisionRepository;

    public DecisionService(MlServiceClient mlServiceClient, DecisionRepository decisionRepository) {
        this.mlServiceClient = mlServiceClient;
        this.decisionRepository = decisionRepository;
    }

    public Map<String, Object> makeDecision(DecisionRequest request) {
        Map<String, Object> mlResponse = mlServiceClient.requestPrediction(request);

        Decision entity = new Decision();
        entity.setDecisionId(generateDecisionId());
        entity.setApplicantId(request.applicantId());
        entity.setRiskScore(((Number) mlResponse.get("riskScore")).doubleValue());
        entity.setConfidence(((Number) mlResponse.get("confidence")).doubleValue());
        entity.setIntegrityStatus((String) mlResponse.get("integrityStatus"));
        entity.setDecision((String) mlResponse.get("decision"));
        entity.setMode((String) mlResponse.get("mode"));
        entity.setModelVersion((String) mlResponse.get("modelVersion"));
        entity.setPolicyVersion((String) mlResponse.get("policyVersion"));
        entity.setCreatedAt(OffsetDateTime.now());

        Decision saved = decisionRepository.save(entity);

        Map<String, Object> enriched = new LinkedHashMap<>(mlResponse);
        enriched.put("decisionId", saved.getDecisionId());
        enriched.put("persistedAt", saved.getCreatedAt().toString());
        return enriched;
    }

    private String generateDecisionId() {
        return "DEC-" + UUID.randomUUID().toString().substring(0, 8).toUpperCase();
    }
}