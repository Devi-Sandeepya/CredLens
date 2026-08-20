package com.credlens.backend.controller;

import com.credlens.backend.dto.DecisionRequest;
import com.credlens.backend.service.DecisionService;
import jakarta.validation.Valid;
import org.springframework.web.bind.annotation.*;

import java.util.Map;

@RestController
@RequestMapping("/api/v1")
public class DecisionController {

    private final DecisionService decisionService;

    public DecisionController(DecisionService decisionService) {
        this.decisionService = decisionService;
    }

    @GetMapping("/health")
    public Map<String, String> health() {
        return Map.of("status", "ok", "service", "CredLens Backend");
    }

    @PostMapping("/decision")
    public Map<String, Object> decide(@Valid @RequestBody DecisionRequest request) {
        return decisionService.makeDecision(request);
    }
}
