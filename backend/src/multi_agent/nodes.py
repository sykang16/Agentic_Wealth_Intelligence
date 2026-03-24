"""Node functions for the orchestrator LangGraph."""

import logging
from decimal import Decimal
from typing import Any

from backend.src.agents.asset_agent import AssetAgent
from backend.src.asset_management.aggregator import PortfolioAggregator
from backend.src.common.llm_client import LLMClient
from backend.src.common.models import (
    ExperienceLevel,
    GoalType,
    InvestmentGoal,
    InvestmentHorizon,
    InvestmentProfile,
    Priority,
    RiskTolerance,
)
from backend.src.profiling import ProfilingAgent, ConversationState as ProfilingConversationState
from backend.src.recommendation.engine import RecommendationEngine, RecommendationRequest

from .routing import IntentRouter
from .state import OrchestratorState, UserIntent


_GOAL_TYPE_MAP: dict[str, GoalType] = {
    "retirement": GoalType.RETIREMENT,
    "house": GoalType.HOUSE,
    "education": GoalType.EDUCATION,
    "emergency": GoalType.EMERGENCY,
    "wealth": GoalType.WEALTH,
}


def conv_state_to_investment_profile(
    user_id: str, conv_state: ProfilingConversationState
) -> InvestmentProfile:
    """Convert a completed ConversationState into an InvestmentProfile.

    Only maps slots that have corresponding InvestmentProfile fields.
    """
    slots = conv_state.slots
    kwargs: dict[str, Any] = {"user_id": user_id}

    if slots.risk_tolerance.value is not None:
        kwargs["risk_tolerance"] = RiskTolerance(slots.risk_tolerance.value)

    if slots.investment_horizon.value is not None:
        kwargs["investment_horizon"] = InvestmentHorizon(slots.investment_horizon.value)

    if slots.experience_level.value is not None:
        kwargs["investment_experience"] = ExperienceLevel(slots.experience_level.value)

    if slots.liquidity_needs.value is not None:
        kwargs["liquidity_needs"] = Priority(slots.liquidity_needs.value)

    if slots.esg_preference.value is not None:
        kwargs["esg_preference"] = bool(slots.esg_preference.value)

    if slots.primary_goal.value is not None:
        raw_goal = str(slots.primary_goal.value).lower()
        goal_type = _GOAL_TYPE_MAP.get(raw_goal, GoalType.OTHER)
        kwargs["goals"] = [InvestmentGoal(goal_type=goal_type, target_amount=Decimal("0"))]

    # Profiling-only slots with no InvestmentProfile counterpart — persist them directly
    if slots.loss_comfort.value is not None:
        kwargs["loss_comfort"] = str(slots.loss_comfort.value)
    if slots.income_stability.value is not None:
        kwargs["income_stability"] = str(slots.income_stability.value)
    if slots.has_emergency_fund.value is not None:
        val = slots.has_emergency_fund.value
        if isinstance(val, bool):
            kwargs["has_emergency_fund"] = val
        else:
            kwargs["has_emergency_fund"] = str(val).lower() in ("yes", "true", "1")
    if slots.debt_level.value is not None:
        kwargs["debt_level"] = str(slots.debt_level.value)
    if slots.monthly_income.value is not None:
        kwargs["monthly_income"] = Decimal(str(slots.monthly_income.value))
    if slots.monthly_expenses.value is not None:
        kwargs["monthly_expenses"] = Decimal(str(slots.monthly_expenses.value))

    return InvestmentProfile(**kwargs)

logger = logging.getLogger(__name__)

GENERAL_SYSTEM_PROMPT = """You are a friendly wealth management AI assistant.
You can help users with:
- Portfolio analysis (net worth, holdings, allocation, etc.)
- Investment profiling (building a risk/preference profile)
- Personalized recommendations (what to buy, sell, rebalance)

Respond helpfully to the user's message. If they're asking about capabilities,
explain what you can do. Keep responses concise and friendly."""


def create_router_node(llm_client: LLMClient):
    """Create the router node function with injected LLM client."""
    router = IntentRouter(llm_client)

    def router_node(state: OrchestratorState) -> dict:
        """Classify user intent and route to appropriate module."""
        try:
            message = state["current_message"]
            messages = state.get("messages", [])
            intent = router.classify(message, history=messages)
            return {"intent": intent.value}
        except Exception as e:
            logger.error(f"Router failed: {e}", exc_info=True)
            return {"intent": UserIntent.GENERAL.value, "error": str(e)}

    return router_node


def create_portfolio_node(asset_agent: AssetAgent):
    """Create the portfolio node function with injected AssetAgent."""

    def portfolio_node(state: OrchestratorState) -> dict:
        """Handle portfolio queries using AssetAgent."""
        try:
            user_id = state["user_id"]
            message = state["current_message"]

            result = asset_agent.process(user_id, message)

            if result.success:
                return {
                    "response": result.answer,
                    "module_source": "Portfolio",
                    "visualization": result.visualization,
                }
            else:
                return {
                    "response": result.answer,
                    "module_source": "Portfolio",
                    "visualization": result.visualization,
                    "error": result.error,
                }
        except Exception as e:
            logger.error(f"Portfolio node failed: {e}", exc_info=True)
            return {
                "response": f"I encountered an error looking up your portfolio: {e}",
                "module_source": "Portfolio",
                "error": str(e),
            }

    return portfolio_node


def create_profiling_node(
    profiling_agent: ProfilingAgent, aggregator: PortfolioAggregator
):
    """Create the profiling node function with injected ProfilingAgent and aggregator."""

    def profiling_node(state: OrchestratorState) -> dict:
        """Handle profiling interactions using ProfilingAgent."""
        try:
            user_id = state["user_id"]
            message = state["current_message"]
            profiling_state = state.get("profiling_state")

            if profiling_state is None:
                # Start new profiling conversation
                conv_state = profiling_agent.start_conversation(user_id)
                assistant_msg = profiling_agent.get_last_assistant_message(conv_state)
                return {
                    "response": assistant_msg or "Let's build your investment profile!",
                    "module_source": "Profiling",
                    "profiling_state": conv_state,
                }
            else:
                # Continue profiling conversation
                conv_state = profiling_state
                conv_state = profiling_agent.process_response(conv_state, message)
                assistant_msg = profiling_agent.get_last_assistant_message(conv_state)

                response = assistant_msg or "Thank you for your response."
                if conv_state.is_complete:
                    # Persist the completed profile back to the aggregator and disk
                    try:
                        new_profile = conv_state_to_investment_profile(user_id, conv_state)
                        aggregator.update_investment_profile(user_id, new_profile)
                        aggregator.save_portfolios()
                        logger.info(
                            f"Investment profile saved for user {user_id} "
                            f"(completeness: {new_profile.profile_completeness:.0%})"
                        )
                    except Exception as save_err:
                        logger.error(
                            f"Failed to save investment profile for {user_id}: {save_err}",
                            exc_info=True,
                        )

                    summary = profiling_agent.get_profile_summary(conv_state)
                    response += (
                        f"\n\nYour profile is now {summary['completion_percentage']:.0f}% complete!"
                    )

                return {
                    "response": response,
                    "module_source": "Profiling",
                    "profiling_state": conv_state,
                }
        except Exception as e:
            logger.error(f"Profiling node failed: {e}", exc_info=True)
            return {
                "response": f"I encountered an error with the profiling system: {e}",
                "module_source": "Profiling",
                "error": str(e),
            }

    return profiling_node


def create_recommend_node(recommendation_engine: RecommendationEngine):
    """Create the recommendation node with injected RecommendationEngine."""

    def recommend_node(state: OrchestratorState) -> dict:
        """Handle recommendation requests using RecommendationEngine."""
        try:
            user_id = state["user_id"]
            message = state["current_message"]

            request = RecommendationRequest(
                user_id=user_id,
                query=message,
                max_recommendations=5,
                include_live_data=False,
            )
            result = recommendation_engine.generate_recommendations(request)

            if result.success:
                # Format recommendations into readable text (used as fallback/summary)
                parts = [result.summary]
                for i, rec in enumerate(result.recommendations, 1):
                    parts.append(
                        f"\n{i}. **{rec.title}** ({rec.category.value.replace('_', ' ').title()})\n"
                        f"   {rec.summary}\n"
                        f"   Action: {rec.suggested_action}"
                    )

                return {
                    "response": "\n".join(parts),
                    "module_source": "Recommendation",
                    "recommendation_response": result,
                }
            else:
                return {
                    "response": f"I couldn't generate recommendations: {result.error}",
                    "module_source": "Recommendation",
                    "error": result.error,
                }
        except Exception as e:
            logger.error(f"Recommendation node failed: {e}", exc_info=True)
            return {
                "response": f"I encountered an error generating recommendations: {e}",
                "module_source": "Recommendation",
                "error": str(e),
            }

    return recommend_node


def create_general_node(llm_client: LLMClient):
    """Create the general node with injected LLM client."""

    def general_node(state: OrchestratorState) -> dict:
        """Handle general messages with direct LLM response."""
        try:
            message = state["current_message"]
            history = state.get("messages", [])

            if history:
                # Use chat_with_history so the LLM sees recent conversation context
                msgs = history[-6:] + [{"role": "user", "content": message}]
                response = llm_client.chat_with_history(
                    messages=msgs,
                    system_prompt=GENERAL_SYSTEM_PROMPT,
                    max_tokens=512,
                    temperature=0.7,
                )
            else:
                response = llm_client.chat(
                    user_message=message,
                    system_prompt=GENERAL_SYSTEM_PROMPT,
                    max_tokens=512,
                    temperature=0.7,
                )

            return {
                "response": response.content,
                "module_source": "General",
            }
        except Exception as e:
            logger.error(f"General node failed: {e}", exc_info=True)
            return {
                "response": (
                    "Hello! I'm your wealth management assistant. I can help you with:\n"
                    "- **Portfolio Analysis**: Ask about your net worth, holdings, allocation\n"
                    "- **Investment Profiling**: Build your risk and preference profile\n"
                    "- **Recommendations**: Get personalized investment suggestions\n\n"
                    "How can I help you today?"
                ),
                "module_source": "General",
            }

    return general_node


def respond_node(state: OrchestratorState) -> dict:
    """Format the final response and update message history."""
    response = state.get("response", "I'm sorry, I couldn't process your request.")
    current_message = state.get("current_message", "")
    messages = list(state.get("messages", []))

    # Append the exchange to history, including visualization and recommendation data if present
    viz = state.get("visualization")
    rec_response = state.get("recommendation_response")
    messages.append({"role": "user", "content": current_message})
    msg = {"role": "assistant", "content": response}
    if viz is not None:
        msg["visualization"] = viz
    if rec_response is not None:
        msg["recommendation_response"] = rec_response
    messages.append(msg)

    return {"messages": messages, "visualization": None, "recommendation_response": None}
