package com.credlens.backend.controller;

import com.credlens.backend.dto.LoginRequest;
import com.credlens.backend.dto.RegisterRequest;
import com.credlens.backend.entity.User;
import com.credlens.backend.repository.UserRepository;
import jakarta.validation.Valid;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.web.bind.annotation.*;

import java.time.OffsetDateTime;
import java.util.LinkedHashMap;
import java.util.Map;

@RestController
@RequestMapping("/api/v1/auth")
public class AuthController {

    private final UserRepository userRepository;
    private final PasswordEncoder passwordEncoder;

    public AuthController(UserRepository userRepository, PasswordEncoder passwordEncoder) {
        this.userRepository = userRepository;
        this.passwordEncoder = passwordEncoder;
    }

    @PostMapping("/register")
    public ResponseEntity<?> register(@Valid @RequestBody RegisterRequest request) {
        if (userRepository.findByUsername(request.username()).isPresent()) {
            return ResponseEntity.status(HttpStatus.CONFLICT)
                    .body(Map.of("error", "Username already exists"));
        }

        User user = new User();
        user.setUsername(request.username());
        user.setPasswordHash(passwordEncoder.encode(request.password()));
        user.setFullName(request.fullName());
        user.setAge(request.age());
        user.setRole(request.role());
        user.setCreatedAt(OffsetDateTime.now());

        User saved = userRepository.save(user);
        return ResponseEntity.ok(toProfile(saved));
    }

    @PostMapping("/login")
    public ResponseEntity<?> login(@Valid @RequestBody LoginRequest request) {
        var userOpt = userRepository.findByUsername(request.username());

        if (userOpt.isEmpty() || !passwordEncoder.matches(request.password(), userOpt.get().getPasswordHash())) {
            return ResponseEntity.status(HttpStatus.UNAUTHORIZED)
                    .body(Map.of("error", "Invalid username or password"));
        }

        return ResponseEntity.ok(toProfile(userOpt.get()));
    }

    private Map<String, Object> toProfile(User user) {
        Map<String, Object> profile = new LinkedHashMap<>();
        profile.put("id", user.getId());
        profile.put("username", user.getUsername());
        profile.put("fullName", user.getFullName());
        profile.put("age", user.getAge());
        profile.put("role", user.getRole());
        return profile;
    }
}