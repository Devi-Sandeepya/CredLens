package com.credlens.backend.dto;

import jakarta.validation.constraints.NotNull;

public record DecisionRequest(@NotNull Long applicantId) {
}