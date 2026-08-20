package com.credlens.backend.config;

import org.springframework.boot.context.properties.ConfigurationProperties;

@ConfigurationProperties(prefix = "ml.service")
public record MlServiceProperties(String baseUrl) {
}