# Module B: Agent-Guided Investment Profiling

#### 🔴 Priority: HIGH

## Objective
Conversational investment profiling using slot-filling

## LangGraph Workflow
```
START → User Input
  ↓
Extractor (fill slots)
  ↓
Validator (check validity)
  ↓
Completion Checker
  ├─ Complete? → END (return profile)
  └─ Incomplete? → Question Generator → Ask User → loop back
```

## Agent Architecture (LangGraph)
```python
# State definition
class ConversationState(TypedDict):
    messages: List[Message]
    profile: InvestmentProfile  # Partially filled
    missing_slots: List[str]
    current_question: str
```

## Pydantic Schema (Required Slots)
```python
class InvestmentProfile(BaseModel):
    # Risk Assessment
    risk_tolerance: Literal["conservative", "moderate", "aggressive"]
    loss_comfort: int = Field(ge=1, le=10)  # 1-10 scale
    
    # Time Horizon
    investment_period: Literal["short", "medium", "long"]  # <3yr, 3-10yr, >10yr
    liquidity_needs: str
    
    # Financial Situation
    income_stability: Literal["stable", "variable", "uncertain"]
    emergency_fund: bool
    debt_level: Literal["none", "low", "moderate", "high"]
    
    # Investment Experience
    experience_level: Literal["beginner", "intermediate", "advanced"]
    previous_investments: List[str]
    
    # Goals
    primary_goal: str
    target_return: Optional[float]
```

## Agent Nodes
1. Extractor: Parse user response to fill slots
2. Validator: Check if extracted data is valid
3. Question Generator: Generate next question for missing slots
4. Completion Checker: Determine if profile is complete

## Implementation Files
```
src/
├── profiling/
│   ├── __init__.py
│   ├── agent.py              # LangGraph agent definition
│   ├── nodes/
│   │   ├── extractor.py
│   │   ├── validator.py
│   │   ├── question_gen.py
│   │   └── checker.py
│   ├── schemas.py            # InvestmentProfile model
│   └── prompts/
│       ├── extractor_prompt.py
│       └── question_prompts.py
```




