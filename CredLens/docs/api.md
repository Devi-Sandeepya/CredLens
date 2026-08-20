# API Contract

## Spring Boot

### POST /api/v1/decision

```json
{
  "applicantId": 100001
}
```

## FastAPI

### POST /api/v1/predictions

```json
{
  "applicantId": 100001
}
```

### POST /api/v1/applicants/{id}/behavior/update

```json
{
  "timestamp": "2026-08-19T16:40:00",
  "paymentAmount": 12500,
  "scheduledAmount": 12000,
  "balance": 84000,
  "daysPastDue": 3
}
```

The behavior update is intentionally lightweight for the prototype. It demonstrates the live-inference contract without introducing Kafka/Kinesis complexity.
