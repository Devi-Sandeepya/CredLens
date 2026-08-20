package com.credlens.backend.repository;

import com.credlens.backend.entity.Decision;
import org.springframework.data.jpa.repository.JpaRepository;
import org.springframework.stereotype.Repository;

import java.util.List;

@Repository
public interface DecisionRepository extends JpaRepository<Decision, Long> {

    List<Decision> findByApplicantIdOrderByCreatedAtDesc(Long applicantId);
}