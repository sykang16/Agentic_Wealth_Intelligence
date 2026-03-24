# API Reference

Base URL: `http://localhost:8000`

## Health Check

```
GET /health
```

Response:
```json
{"status": "ok"}
```

---

## Chat (Orchestrator)

Unified endpoint that automatically routes messages to the appropriate module.

```
POST /api/v1/chat
```

**Request:**
```json
{
  "message": "What's my net worth?",
  "user_id": "user_001"
}
```

**Response:**
```json
{
  "response": "Your net worth is $327,250...",
  "intent": "portfolio_query",
  "module_source": "Portfolio",
  "success": true,
  "error": null
}
```

**Intent values:** `portfolio_query`, `profiling`, `recommendation`, `general`

**curl example:**
```bash
curl -X POST http://localhost:8000/api/v1/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "What is my net worth?", "user_id": "user_001"}'
```

---

## Portfolio

### Get Portfolio Summary

```
GET /api/v1/portfolio/{user_id}
```

**Response:**
```json
{
  "user_id": "user_001",
  "name": "John Doe",
  "metrics": {
    "net_worth": 327250,
    "total_assets": 627250,
    "total_liabilities": 300000,
    "liquidity_ratio": 0.1036
  },
  "allocation": {"cash": 10.36, "stocks": 5.59},
  "top_holdings": [
    {"symbol": "AAPL", "name": "Apple Inc.", "value": 8750.0, "gain_loss_pct": 16.67}
  ]
}
```

**Error:** `404` if user not found.

### Query Portfolio

```
POST /api/v1/portfolio/{user_id}/query
```

**Request:**
```json
{"query": "Show my asset allocation"}
```

**Response:**
```json
{
  "answer": "Here is your asset allocation...",
  "query_type": "asset_allocation",
  "success": true,
  "error": null,
  "data": {}
}
```

---

## Profiling

### Start Profiling Session

```
POST /api/v1/profiling/{user_id}/start
```

**Request (optional):**
```json
{"user_name": "John"}
```

**Response:**
```json
{
  "assistant_message": "Welcome! Let's build your investment profile...",
  "completion_percentage": 0.0,
  "is_complete": false,
  "filled_slots": {},
  "missing_required": ["Risk Tolerance", "Investment Horizon", ...]
}
```

### Respond to Profiling Question

```
POST /api/v1/profiling/{user_id}/respond
```

**Request:**
```json
{"message": "I have a moderate risk tolerance"}
```

**Response:** Same schema as start, with updated completion.

**Error:** `404` if no active session. Call `/start` first.

---

## Recommendations

### Generate Recommendations

```
POST /api/v1/recommendations/generate
```

**Request:**
```json
{
  "user_id": "user_001",
  "query": "How should I diversify?",
  "max_recommendations": 5,
  "include_live_data": false
}
```

**Response:**
```json
{
  "user_id": "user_001",
  "query": "How should I diversify?",
  "recommendations": [...],
  "summary": "Generated 3 recommendations...",
  "data_sources_used": ["portfolio", "rag"],
  "profile_completeness": 0.75,
  "warnings": ["..."],
  "success": true,
  "error": null
}
```

---

## Error Codes

| Code | Meaning |
|------|---------|
| `200` | Success |
| `404` | Resource not found (user, session) |
| `422` | Validation error (missing/invalid fields) |
| `500` | Internal server error |

Validation errors return detailed Pydantic error messages in the response body.
