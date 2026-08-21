package com.credlens.backend.dto;

import jakarta.validation.constraints.NotBlank;

public record RegisterRequest(
        @NotBlank String username,
        @NotBlank String password,
        @NotBlank String fullName,
        Integer age,
        @NotBlank String role
) {}