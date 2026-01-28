# UI Design & Structure

## Overview

This document outlines the user interface architecture for the Agentic Wealth Intelligence System. The UI serves as the primary interaction layer between users and the multi-agent backend, supporting both single-user demos and multi-user deployments.

---

## UI Strategy

### Development Approach: Progressive Enhancement

We adopt a **progressive enhancement strategy** that balances speed with sophistication:

```
Phase 1 (Week 1)     →  Phase 2 (Weeks 2-3)  →  Phase 3 (Weeks 4-6)
─────────────────       ──────────────────       ──────────────────
CLI Interface           Streamlit Web App        Multi-User Support
(Validation)            (MVP/Demo)               (Authentication & Isolation)
```

**Recommendation**: Focus on **Streamlit** (Phase 2) for demo, then add **multi-user support** (Phase 3).

---

## Phase 1: CLI Interface (Validation)

### Purpose
- Quick validation of agent logic
- No UI overhead
- Developer-friendly testing

### Implementation

```python
# src/cli/main.py
from rich.console import Console
from rich.prompt import Prompt
from rich.panel import Panel
from rich.markdown import Markdown

console = Console()

def main():
    console.print(Panel.fit(
        "[bold blue]Agentic Wealth Intelligence[/bold blue]\n"
        "Multi-Agent Financial Advisory System",
        border_style="blue"
    ))
    
    while True:
        user_input = Prompt.ask("\n[bold green]You[/bold green]")
        
        if user_input.lower() in ['exit', 'quit']:
            break
        
        # Call orchestrator agent
        console.print("[dim]Thinking...[/dim]")
        response = await orchestrator_agent.run(user_input)
        
        # Display response
        console.print(f"\n[bold cyan]Assistant ({response.agent})[/bold cyan]")
        console.print(Markdown(response.content))
        
        # Show any visualizations
        if response.chart_data:
            console.print("[dim]Chart data available (view in UI)[/dim]")

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
```

### Features
✅ Simple text-based interaction
✅ Color-coded output (Rich library)
✅ Markdown rendering for responses
✅ Quick iteration and testing

---

## Phase 2: Streamlit Web App (Recommended MVP)


### Streamlit Application Structure

```
ui/
├── streamlit_app.py              # Main entry point
├── pages/
│   ├── 1_Dashboard.py         # Portfolio overview
│   ├── 2_Chat.py              # Agent conversation
│   ├── 3_Profile.py           # Investment profile
│   └── 4_Recommendations.py   # Product recommendations
├── components/
│   ├── __init__.py
│   ├── auth.py                   # Authentication UI
│   ├── charts.py                 # Visualization helpers
│   ├── chat_ui.py                # Chat components
│   ├── portfolio_cards.py        # Dashboard widgets
│   └── metrics.py                # Financial metrics display
├── utils/
│   ├── __init__.py
│   ├── api_client.py             # Backend API calls
│   ├── state_manager.py          # Session state
│   ├── auth_manager.py           # User authentication
│   └── formatters.py             # Data formatting
└── config.py                     # UI configuration
```

---

### Main Application Entry Point

```python
# ui/streamlit_app.py
import streamlit as st
from utils.state_manager import initialize_session_state
from utils.auth_manager import check_authentication
from utils.api_client import APIClient

# Page configuration
st.set_page_config(
    page_title="Agentic Wealth Intelligence",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Check authentication
if not check_authentication():
    st.stop()  # Will show login page

# Initialize session state
initialize_session_state()

# Get current user info
user_info = st.session_state.user_info

# Sidebar
with st.sidebar:
    st.title("💰 Wealth Intelligence")
    st.divider()
    
    # User info
    st.subheader(f"👤 {user_info['name']}")
    st.caption(f"ID: {user_info['user_id']}")
    
    # Logout button
    if st.button("🚪 Logout"):
        st.session_state.clear()
        st.rerun()
    
    st.divider()
    
    # Quick stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Net Worth", "$125,430", "+5.2%")
    with col2:
        st.metric("Portfolio", "$98,450", "+3.1%")
    
    st.divider()
    
    # Navigation info
    st.caption("📊 Dashboard - Overview")
    st.caption("💬 Chat - Talk to AI")
    st.caption("👤 Profile - Your preferences")
    st.caption("💡 Recommendations - Personalized advice")

# Main content
st.title(f"Welcome back, {user_info['name']}!")

st.markdown("""
This multi-agent AI system helps you understand your finances and make informed investment decisions.

### Getting Started
1. **📊 Dashboard**: View your complete financial picture
2. **💬 Chat**: Ask questions about your portfolio
3. **👤 Profile**: Complete your investment profile
4. **💡 Recommendations**: Get personalized advice
""")

# Quick actions
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("📊 View Dashboard", use_container_width=True):
        st.switch_page("pages/1_Dashboard.py")

with col2:
    if st.button("💬 Start Chat", use_container_width=True):
        st.switch_page("pages/2_Chat.py")

with col3:
    if st.button("💡 Get Advice", use_container_width=True):
        st.switch_page("pages/4_Recommendations.py")
```

---

### Page 1: Dashboard (Portfolio Overview)

```python
# ui/pages/1_Dashboard.py
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from components.charts import create_allocation_chart, create_performance_chart
from components.portfolio_cards import render_account_cards
from utils.api_client import APIClient
from utils.auth_manager import require_authentication

st.set_page_config(page_title="Dashboard", page_icon="📊", layout="wide")

# Require authentication
require_authentication()

st.title("📊 Portfolio Dashboard")

# Initialize API client with user context
api_client = APIClient(user_id=st.session_state.user_info['user_id'])

# Fetch portfolio data
with st.spinner("Loading portfolio data..."):
    portfolio = api_client.get_portfolio()

# Top metrics row
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Assets",
        f"${portfolio.total_assets:,.2f}",
        f"{portfolio.asset_change_pct:+.1%}"
    )

with col2:
    st.metric(
        "Net Worth",
        f"${portfolio.net_worth:,.2f}",
        f"{portfolio.net_worth_change_pct:+.1%}"
    )

with col3:
    st.metric(
        "Liquidity Ratio",
        f"{portfolio.liquidity_ratio:.2f}",
        "Healthy" if portfolio.liquidity_ratio > 0.5 else "Low",
        delta_color="normal" if portfolio.liquidity_ratio > 0.5 else "inverse"
    )

with col4:
    st.metric(
        "Monthly Cashflow",
        f"${portfolio.monthly_cashflow:,.2f}",
        f"{portfolio.cashflow_change:+.1%}"
    )

st.divider()

# Charts row
col1, col2 = st.columns(2)

with col1:
    st.subheader("Asset Allocation")
    fig = create_allocation_chart(portfolio.allocation)
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("Portfolio Performance (6 Months)")
    fig = create_performance_chart(portfolio.performance_history)
    st.plotly_chart(fig, use_container_width=True)

st.divider()

# Accounts section
st.subheader("Accounts Overview")
render_account_cards(portfolio.accounts)

st.divider()

# Interactive query section
st.subheader("💬 Ask About Your Portfolio")

col1, col2 = st.columns([3, 1])

with col1:
    query = st.text_input(
        "Natural language query",
        placeholder="e.g., What's my debt-to-income ratio?",
        label_visibility="collapsed"
    )

with col2:
    if st.button("Ask Asset Agent", use_container_width=True):
        if query:
            with st.spinner("Analyzing..."):
                response = api_client.query_asset_agent(query)
                st.info(response.answer)
                
                # Show any generated charts
                if response.chart:
                    st.plotly_chart(response.chart)
```

---

### Page 2: Chat Interface (Multi-Agent Conversation)

```python
# ui/pages/2_Chat.py
import streamlit as st
from utils.api_client import APIClient
from components.chat_ui import render_message, show_agent_indicator
from utils.auth_manager import require_authentication

st.set_page_config(page_title="Chat", page_icon="💬", layout="wide")

# Require authentication
require_authentication()

st.title("💬 Chat with AI Agents")

# Initialize chat history per user
chat_key = f"messages_{st.session_state.user_info['user_id']}"
if chat_key not in st.session_state:
    st.session_state[chat_key] = []
    st.session_state[chat_key].append({
        "role": "assistant",
        "content": f"Hello {st.session_state.user_info['name']}! I'm your wealth intelligence assistant. I can help you understand your finances, complete your investment profile, or provide personalized recommendations. What would you like to know?",
        "agent": "orchestrator"
    })

# API client
api_client = APIClient(user_id=st.session_state.user_info['user_id'])

# Display chat messages
for message in st.session_state[chat_key]:
    with st.chat_message(message["role"]):
        # Show which agent is responding
        if message["role"] == "assistant":
            show_agent_indicator(message.get("agent", "orchestrator"))
        
        st.markdown(message["content"])
        
        # Display any charts or data
        if "chart" in message:
            st.plotly_chart(message["chart"], use_container_width=True)
        
        if "data" in message:
            with st.expander("View data"):
                st.json(message["data"])

# Chat input
if prompt := st.chat_input("Type your message here..."):
    # Add user message
    st.session_state[chat_key].append({
        "role": "user",
        "content": prompt
    })
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get agent response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = api_client.chat(
                message=prompt,
                history=st.session_state[chat_key]
            )
            
            # Show active agent
            show_agent_indicator(response.agent)
            
            # Stream response (if supported)
            st.markdown(response.content)
            
            # Show any visualizations
            if response.chart:
                st.plotly_chart(response.chart, use_container_width=True)
    
    # Add assistant message to history
    st.session_state[chat_key].append({
        "role": "assistant",
        "content": response.content,
        "agent": response.agent,
        "chart": response.chart if hasattr(response, 'chart') else None
    })

# Sidebar: Conversation controls
with st.sidebar:
    st.subheader("Conversation")
    
    if st.button("🔄 New Conversation"):
        st.session_state[chat_key] = []
        st.rerun()
    
    if st.button("💾 Save Conversation"):
        # Save to file or database
        st.success("Conversation saved!")
    
    st.divider()
    
    # Show active agent context
    st.caption("**Active Context**")
    st.caption(f"Messages: {len(st.session_state[chat_key])}")
    
    # Agent activity
    if st.session_state[chat_key]:
        last_agent = st.session_state[chat_key][-1].get("agent", "unknown")
        st.caption(f"Last agent: {last_agent}")
```

---

### Page 3: Investment Profile

```python
# ui/pages/3_Profile.py
import streamlit as st
from utils.api_client import APIClient
from utils.auth_manager import require_authentication

st.set_page_config(page_title="Profile", page_icon="👤", layout="wide")

# Require authentication
require_authentication()

st.title("👤 Investment Profile")

api_client = APIClient(user_id=st.session_state.user_info['user_id'])

# Tabs for different modes
tab1, tab2 = st.tabs(["📝 Conversational Profiling", "📋 View Profile"])

with tab1:
    st.subheader("Complete Your Profile Through Conversation")
    st.info("💡 Our AI agent will ask you questions to understand your investment preferences. Just answer naturally!")
    
    # Profiling session key per user
    profiling_key = f"profiling_{st.session_state.user_info['user_id']}"
    
    # Start profiling conversation
    if st.button("🚀 Start Profiling Conversation", type="primary"):
        # Initialize profiling session
        st.session_state[profiling_key] = {
            "active": True,
            "messages": []
        }
        st.rerun()
    
    # Profiling conversation
    if st.session_state.get(profiling_key, {}).get("active", False):
        # Get profiling status
        status = api_client.get_profiling_status()
        
        # Progress indicator
        progress = status.filled_slots / status.total_slots
        st.progress(progress, text=f"Profile Completion: {progress*100:.0f}%")
        
        # Show which slots are filled
        with st.expander("Profile Status"):
            col1, col2 = st.columns(2)
            with col1:
                st.success(f"✅ Completed: {', '.join(status.filled_slots)}")
            with col2:
                st.warning(f"⏳ Remaining: {', '.join(status.missing_slots)}")
        
        st.divider()
        
        # Conversation interface (similar to chat)
        for msg in st.session_state[profiling_key]["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
        
        # User input
        if response := st.chat_input("Your answer..."):
            # Add to messages
            st.session_state[profiling_key]["messages"].append({
                "role": "user",
                "content": response
            })
            
            # Send to profiling agent
            with st.spinner("Processing..."):
                agent_response = api_client.profiling_agent_step(response)
                
                st.session_state[profiling_key]["messages"].append({
                    "role": "assistant",
                    "content": agent_response.question or agent_response.summary
                })
                
                # Check if complete
                if agent_response.is_complete:
                    st.session_state[profiling_key]["active"] = False
                    st.success("✅ Profile completed!")
                    st.balloons()
            
            st.rerun()

with tab2:
    st.subheader("Your Investment Profile")
    
    # Fetch current profile
    profile = api_client.get_profile()
    
    if profile:
        # Display profile in organized sections
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Risk Assessment")
            st.metric("Risk Tolerance", profile.risk_tolerance.title())
            st.metric("Loss Comfort (1-10)", profile.loss_comfort)
            
            st.markdown("### Time Horizon")
            st.metric("Investment Period", profile.investment_period.title())
            st.text_area("Liquidity Needs", profile.liquidity_needs, disabled=True)
        
        with col2:
            st.markdown("### Financial Situation")
            st.metric("Income Stability", profile.income_stability.title())
            st.metric("Emergency Fund", "Yes" if profile.emergency_fund else "No")
            st.metric("Debt Level", profile.debt_level.title())
            
            st.markdown("### Experience")
            st.metric("Level", profile.experience_level.title())
            st.text_area("Previous Investments", 
                        ", ".join(profile.previous_investments), 
                        disabled=True)
        
        st.divider()
        
        st.markdown("### Investment Goals")
        st.text_area("Primary Goal", profile.primary_goal, disabled=True)
        if profile.target_return:
            st.metric("Target Return", f"{profile.target_return}%")
        
        # Edit button
        if st.button("✏️ Update Profile"):
            st.session_state[profiling_key] = {"active": True, "messages": []}
            st.rerun()
    
    else:
        st.warning("No profile found. Please complete the conversational profiling.")
```

---

### Page 4: Recommendations

```python
# ui/pages/4_Recommendations.py
import streamlit as st
from utils.api_client import APIClient
from utils.auth_manager import require_authentication

st.set_page_config(page_title="Recommendations", page_icon="💡", layout="wide")

# Require authentication
require_authentication()

st.title("💡 Personalized Investment Recommendations")

api_client = APIClient(user_id=st.session_state.user_info['user_id'])

# Check if profile exists
profile = api_client.get_profile()
if not profile:
    st.warning("⚠️ Please complete your investment profile first.")
    if st.button("Go to Profile"):
        st.switch_page("pages/3_👤_Profile.py")
    st.stop()

# Get recommendations
with st.spinner("Generating personalized recommendations..."):
    recommendations = api_client.get_recommendations()

# Display recommendations
st.subheader(f"Based on your {profile.risk_tolerance} risk profile")

for idx, rec in enumerate(recommendations):
    with st.container():
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### {idx+1}. {rec.product_name}")
            st.caption(f"{rec.product_type} | Ticker: {rec.ticker}")
            st.markdown(rec.description)
            
            # Rationale
            with st.expander("Why this recommendation?"):
                st.markdown(rec.rationale)
                st.caption(f"**Sources**: {', '.join(rec.sources)}")
        
        with col2:
            # Metrics
            st.metric("Expected Return", f"{rec.expected_return}%")
            st.metric("Risk Level", rec.risk_level.title())
            st.metric("Allocation", f"{rec.suggested_allocation}%")
            
            # Action buttons
            if st.button("📊 View Details", key=f"details_{idx}"):
                st.session_state.selected_product = rec.ticker
                # Show detailed modal
            
            if st.button("💬 Ask About This", key=f"ask_{idx}"):
                st.session_state.prefill_chat = f"Tell me more about {rec.product_name}"
                st.switch_page("pages/2_💬_Chat.py")
        
        st.divider()

# Market context
with st.expander("📰 Current Market Context"):
    st.markdown("### Recent Market News")
    for news in recommendations.market_context:
        st.markdown(f"- **{news.headline}** ({news.source})")
        st.caption(news.summary)

# Refresh button
if st.button("🔄 Refresh Recommendations"):
    st.rerun()
```

---

### Component Examples

```python
# ui/components/charts.py
import plotly.express as px
import plotly.graph_objects as go

def create_allocation_chart(allocation_data):
    """Create pie chart for asset allocation"""
    fig = px.pie(
        values=allocation_data.values(),
        names=allocation_data.keys(),
        title="Asset Allocation",
        hole=0.4
    )
    fig.update_traces(textposition='inside', textinfo='percent+label')
    return fig

def create_performance_chart(performance_history):
    """Create line chart for portfolio performance"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=performance_history.dates,
        y=performance_history.values,
        mode='lines+markers',
        name='Portfolio Value',
        line=dict(color='#1f77b4', width=2)
    ))
    
    fig.update_layout(
        title="Portfolio Performance",
        xaxis_title="Date",
        yaxis_title="Value ($)",
        hovermode='x unified'
    )
    
    return fig
```

```python
# ui/components/chat_ui.py
import streamlit as st

def show_agent_indicator(agent_name):
    """Show which agent is responding"""
    agent_colors = {
        "orchestrator": "🎯",
        "asset_agent": "📊",
        "profiling_agent": "👤",
        "advisory_agent": "💡"
    }
    
    agent_labels = {
        "orchestrator": "Orchestrator",
        "asset_agent": "Asset Analyst",
        "profiling_agent": "Profile Guide",
        "advisory_agent": "Investment Advisor"
    }
    
    icon = agent_colors.get(agent_name, "🤖")
    label = agent_labels.get(agent_name, agent_name.title())
    
    st.caption(f"{icon} **{label}** is responding")
```

---

## Phase 3: Multi-User Support

### Overview

Phase 3 adds proper user management, authentication, and data isolation to support multiple users accessing the system simultaneously.

### Architecture Components

```
┌─────────────────────────────────────────────────┐
│              Frontend (Streamlit)               │
│  - Login/Registration UI                        │
│  - Session Management                           │
│  - User Context                                 │
└────────────────┬────────────────────────────────┘
                 │
                 │ HTTP/WebSocket
                 │
┌────────────────▼────────────────────────────────┐
│           Backend API (FastAPI)                 │
│  - User Authentication                          │
│  - JWT Token Management                         │
│  - User Context Middleware                      │
└────────────────┬────────────────────────────────┘
                 │
        ┌────────┴────────┐
        │                 │
┌───────▼──────┐  ┌──────▼────────┐
│  User DB     │  │  User Data DB │
│  (Auth)      │  │  (Portfolios, │
│              │  │   Profiles)   │
└──────────────┘  └───────────────┘
```

---

### User Authentication System

#### Database Schema

```python
# backend/src/auth/models.py
from sqlalchemy import Column, String, DateTime, Boolean
from sqlalchemy.orm import declarative_base
import uuid
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    
    user_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    email = Column(String, unique=True, nullable=False)
    username = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    last_login = Column(DateTime)
    is_active = Column(Boolean, default=True)
    is_verified = Column(Boolean, default=False)
    
    # Preferences
    theme = Column(String, default="light")
    language = Column(String, default="en")

class UserSession(Base):
    __tablename__ = "user_sessions"
    
    session_id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    user_id = Column(String, nullable=False)
    access_token = Column(String, nullable=False)
    refresh_token = Column(String)
    
    created_at = Column(DateTime, default=datetime.utcnow)
    expires_at = Column(DateTime, nullable=False)
    last_activity = Column(DateTime, default=datetime.utcnow)
    
    ip_address = Column(String)
    user_agent = Column(String)
```

#### Authentication Backend

```python
# backend/src/auth/auth_service.py
from passlib.context import CryptContext
from jose import JWTError, jwt
from datetime import datetime, timedelta
from typing import Optional

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

SECRET_KEY = "your-secret-key-here"  # Use environment variable
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

class AuthService:
    @staticmethod
    def hash_password(password: str) -> str:
        """Hash a password"""
        return pwd_context.hash(password)
    
    @staticmethod
    def verify_password(plain_password: str, hashed_password: str) -> bool:
        """Verify a password against hash"""
        return pwd_context.verify(plain_password, hashed_password)
    
    @staticmethod
    def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
        """Create JWT access token"""
        to_encode = data.copy()
        
        if expires_delta:
            expire = datetime.utcnow() + expires_delta
        else:
            expire = datetime.utcnow() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        
        to_encode.update({"exp": expire})
        encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
        return encoded_jwt
    
    @staticmethod
    def verify_token(token: str) -> Optional[dict]:
        """Verify and decode JWT token"""
        try:
            payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
            return payload
        except JWTError:
            return None
    
    async def register_user(
        self, 
        email: str, 
        username: str, 
        password: str,
        full_name: str
    ) -> User:
        """Register a new user"""
        # Check if user exists
        existing_user = await self.get_user_by_email(email)
        if existing_user:
            raise ValueError("User with this email already exists")
        
        # Create user
        hashed_password = self.hash_password(password)
        user = User(
            email=email,
            username=username,
            hashed_password=hashed_password,
            full_name=full_name
        )
        
        # Save to database
        await self.db.save(user)
        return user
    
    async def authenticate_user(self, email: str, password: str) -> Optional[User]:
        """Authenticate user with email and password"""
        user = await self.get_user_by_email(email)
        
        if not user:
            return None
        
        if not self.verify_password(password, user.hashed_password):
            return None
        
        # Update last login
        user.last_login = datetime.utcnow()
        await self.db.save(user)
        
        return user
```

#### Authentication Endpoints

```python
# backend/src/api/routes/auth.py
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel, EmailStr

router = APIRouter(prefix="/api/v1/auth", tags=["authentication"])

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

class UserRegister(BaseModel):
    email: EmailStr
    username: str
    password: str
    full_name: str

class UserLogin(BaseModel):
    email: EmailStr
    password: str

class Token(BaseModel):
    access_token: str
    token_type: str
    user_info: dict

@router.post("/register", response_model=Token)
async def register(user_data: UserRegister, auth_service: AuthService = Depends()):
    """Register a new user"""
    try:
        user = await auth_service.register_user(
            email=user_data.email,
            username=user_data.username,
            password=user_data.password,
            full_name=user_data.full_name
        )
        
        # Create access token
        access_token = auth_service.create_access_token(
            data={"sub": user.email, "user_id": user.user_id}
        )
        
        return Token(
            access_token=access_token,
            token_type="bearer",
            user_info={
                "user_id": user.user_id,
                "email": user.email,
                "username": user.username,
                "full_name": user.full_name
            }
        )
    
    except ValueError as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e)
        )

@router.post("/login", response_model=Token)
async def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    auth_service: AuthService = Depends()
):
    """Login user"""
    user = await auth_service.authenticate_user(
        email=form_data.username,  # Using username field for email
        password=form_data.password
    )
    
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Create access token
    access_token = auth_service.create_access_token(
        data={"sub": user.email, "user_id": user.user_id}
    )
    
    return Token(
        access_token=access_token,
        token_type="bearer",
        user_info={
            "user_id": user.user_id,
            "email": user.email,
            "username": user.username,
            "full_name": user.full_name
        }
    )

@router.get("/me")
async def get_current_user(token: str = Depends(oauth2_scheme), auth_service: AuthService = Depends()):
    """Get current user info"""
    payload = auth_service.verify_token(token)
    
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Could not validate credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    user = await auth_service.get_user_by_id(payload["user_id"])
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    return {
        "user_id": user.user_id,
        "email": user.email,
        "username": user.username,
        "full_name": user.full_name
    }

@router.post("/logout")
async def logout(token: str = Depends(oauth2_scheme)):
    """Logout user (invalidate token)"""
    # In production, add token to blacklist
    return {"message": "Successfully logged out"}
```

---

### Frontend Authentication Components

```python
# ui/components/auth.py
import streamlit as st
import requests
from typing import Optional

API_BASE_URL = "http://localhost:8000"

def show_login_page():
    """Display login page"""
    st.title("🔐 Login to Wealth Intelligence")
    
    tab1, tab2 = st.tabs(["Login", "Register"])
    
    with tab1:
        with st.form("login_form"):
            st.subheader("Welcome Back")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            
            col1, col2 = st.columns(2)
            with col1:
                submit = st.form_submit_button("Login", use_container_width=True)
            with col2:
                if st.form_submit_button("Forgot Password?", use_container_width=True):
                    st.info("Password reset feature coming soon!")
            
            if submit:
                if login_user(email, password):
                    st.success("Login successful!")
                    st.rerun()
                else:
                    st.error("Invalid email or password")
    
    with tab2:
        with st.form("register_form"):
            st.subheader("Create Account")
            full_name = st.text_input("Full Name")
            username = st.text_input("Username")
            email = st.text_input("Email")
            password = st.text_input("Password", type="password")
            password_confirm = st.text_input("Confirm Password", type="password")
            
            submit = st.form_submit_button("Create Account", use_container_width=True)
            
            if submit:
                if password != password_confirm:
                    st.error("Passwords don't match")
                elif len(password) < 8:
                    st.error("Password must be at least 8 characters")
                else:
                    if register_user(email, username, password, full_name):
                        st.success("Account created! You can now login.")
                    else:
                        st.error("Registration failed. Email may already exist.")

def login_user(email: str, password: str) -> bool:
    """Login user and store token"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/auth/login",
            data={"username": email, "password": password}  # OAuth2 uses 'username'
        )
        
        if response.status_code == 200:
            data = response.json()
            
            # Store in session state
            st.session_state.access_token = data["access_token"]
            st.session_state.user_info = data["user_info"]
            st.session_state.authenticated = True
            
            return True
        
        return False
    
    except Exception as e:
        st.error(f"Login error: {str(e)}")
        return False

def register_user(email: str, username: str, password: str, full_name: str) -> bool:
    """Register new user"""
    try:
        response = requests.post(
            f"{API_BASE_URL}/api/v1/auth/register",
            json={
                "email": email,
                "username": username,
                "password": password,
                "full_name": full_name
            }
        )
        
        if response.status_code == 200:
            return True
        
        return False
    
    except Exception as e:
        st.error(f"Registration error: {str(e)}")
        return False
```

```python
# ui/utils/auth_manager.py
import streamlit as st
from components.auth import show_login_page
import requests

API_BASE_URL = "http://localhost:8000"

def check_authentication() -> bool:
    """Check if user is authenticated"""
    if not st.session_state.get("authenticated", False):
        show_login_page()
        return False
    
    # Verify token is still valid
    if not verify_token():
        st.session_state.authenticated = False
        st.warning("Session expired. Please login again.")
        show_login_page()
        return False
    
    return True

def require_authentication():
    """Decorator-like function to require authentication"""
    if not check_authentication():
        st.stop()

def verify_token() -> bool:
    """Verify access token is still valid"""
    if "access_token" not in st.session_state:
        return False
    
    try:
        response = requests.get(
            f"{API_BASE_URL}/api/v1/auth/me",
            headers={"Authorization": f"Bearer {st.session_state.access_token}"}
        )
        
        return response.status_code == 200
    
    except:
        return False

def logout():
    """Logout current user"""
    if "access_token" in st.session_state:
        try:
            requests.post(
                f"{API_BASE_URL}/api/v1/auth/logout",
                headers={"Authorization": f"Bearer {st.session_state.access_token}"}
            )
        except:
            pass
    
    # Clear session
    st.session_state.clear()
```

---

### Data Isolation

#### User-Scoped Data Storage

```python
# backend/src/data/user_data.py
from typing import Optional
from datetime import datetime

class UserDataManager:
    """Manage user-specific data with proper isolation"""
    
    def __init__(self, db_session):
        self.db = db_session
    
    async def get_user_portfolio(self, user_id: str) -> Optional[AssetPortfolio]:
        """Get portfolio for specific user only"""
        portfolio = await self.db.query(AssetPortfolio).filter(
            AssetPortfolio.user_id == user_id
        ).first()
        
        return portfolio
    
    async def get_user_profile(self, user_id: str) -> Optional[InvestmentProfile]:
        """Get investment profile for specific user only"""
        profile = await self.db.query(InvestmentProfile).filter(
            InvestmentProfile.user_id == user_id
        ).first()
        
        return profile
    
    async def save_user_data(self, user_id: str, data: dict):
        """Save data for specific user"""
        # Always verify user_id matches
        if data.get("user_id") != user_id:
            raise ValueError("User ID mismatch")
        
        # Save with user_id constraint
        await self.db.save(data)
    
    async def get_user_chat_history(
        self, 
        user_id: str, 
        limit: int = 50
    ) -> List[dict]:
        """Get chat history for user"""
        messages = await self.db.query(ChatMessage).filter(
            ChatMessage.user_id == user_id
        ).order_by(ChatMessage.created_at.desc()).limit(limit).all()
        
        return [msg.to_dict() for msg in messages]
```

#### Middleware for User Context

```python
# backend/src/api/middleware.py
from fastapi import Request, HTTPException
from starlette.middleware.base import BaseHTTPMiddleware

class UserContextMiddleware(BaseHTTPMiddleware):
    """Add user context to all requests"""
    
    async def dispatch(self, request: Request, call_next):
        # Skip auth check for public endpoints
        if request.url.path in ["/api/v1/auth/login", "/api/v1/auth/register"]:
            return await call_next(request)
        
        # Extract token
        auth_header = request.headers.get("Authorization")
        if not auth_header or not auth_header.startswith("Bearer "):
            raise HTTPException(status_code=401, detail="Not authenticated")
        
        token = auth_header.split(" ")[1]
        
        # Verify token and get user
        auth_service = AuthService()
        payload = auth_service.verify_token(token)
        
        if not payload:
            raise HTTPException(status_code=401, detail="Invalid token")
        
        # Add user_id to request state
        request.state.user_id = payload["user_id"]
        request.state.user_email = payload["sub"]
        
        response = await call_next(request)
        return response
```

---

### Multi-User Agent Coordination

```python
# backend/src/agents/orchestrator.py
class OrchestratorAgent:
    async def run(self, message: str, user_id: str, history: List[dict]):
        """
        Run orchestrator with user context
        
        Args:
            message: User message
            user_id: Current user ID (for data isolation)
            history: Conversation history (user-specific)
        """
        # Create user-scoped state
        state = SystemState(
            user_id=user_id,  # CRITICAL: Always include user_id
            messages=history,
            user_intent=None,
            active_agent="orchestrator"
        )
        
        # Classify intent
        state["user_intent"] = await self.classify_intent(message)
        
        # Route to appropriate agent with user context
        if state["user_intent"] == "portfolio_query":
            # Asset agent will only access this user's portfolio
            result = await asset_agent.process(state)
        
        elif state["user_intent"] == "profile_update":
            # Profiling agent will only update this user's profile
            result = await profiling_agent.process(state)
        
        elif state["user_intent"] == "get_recommendation":
            # Advisory agent uses this user's profile + portfolio
            result = await advisory_agent.process(state)
        
        return result
```

```python
# backend/src/multi_agent/state.py
from typing import TypedDict

class SystemState(TypedDict):
    # ALWAYS include user_id for data isolation
    user_id: str  # CRITICAL: Identifies which user's data to access
    
    # User context
    session_id: str
    
    # Conversation
    messages: List[Message]
    user_intent: Optional[str]
    
    # Agent states
    active_agent: str
    agent_outputs: Dict[str, Any]
    
    # User-specific data (automatically filtered by user_id)
    portfolio: Optional[AssetPortfolio]
    profile: Optional[InvestmentProfile]
    recommendations: Optional[List[Recommendation]]
    
    # Control flow
    next_action: str
    is_complete: bool
```

---

### Database Schema for Multi-User

```python
# All tables must have user_id foreign key

class AssetPortfolio(Base):
    __tablename__ = "asset_portfolios"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)  # CRITICAL
    
    total_assets = Column(Numeric)
    total_liabilities = Column(Numeric)
    # ... other fields
    
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)
    
    # Index for fast user queries
    __table_args__ = (
        Index('idx_user_portfolio', 'user_id'),
    )

class InvestmentProfile(Base):
    __tablename__ = "investment_profiles"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)  # CRITICAL
    
    risk_tolerance = Column(String)
    time_horizon = Column(String)
    # ... other fields
    
    __table_args__ = (
        Index('idx_user_profile', 'user_id'),
    )

class ChatMessage(Base):
    __tablename__ = "chat_messages"
    
    id = Column(String, primary_key=True)
    user_id = Column(String, ForeignKey("users.user_id"), nullable=False)  # CRITICAL
    
    role = Column(String)  # user or assistant
    content = Column(Text)
    agent = Column(String)  # which agent responded
    
    created_at = Column(DateTime, default=datetime.utcnow)
    
    __table_args__ = (
        Index('idx_user_messages', 'user_id', 'created_at'),
    )
```

---

### Security Best Practices

#### 1. Password Security
```python
# Use strong hashing (bcrypt with high cost factor)
pwd_context = CryptContext(
    schemes=["bcrypt"],
    deprecated="auto",
    bcrypt__rounds=12  # Higher = more secure but slower
)
```

#### 2. Token Security
```python
# Short-lived access tokens
ACCESS_TOKEN_EXPIRE_MINUTES = 30

# Use refresh tokens for longer sessions
REFRESH_TOKEN_EXPIRE_DAYS = 7

# Rotate tokens periodically
```

#### 3. Data Access Control
```python
# ALWAYS verify user_id matches
async def get_portfolio(user_id: str, request_user_id: str):
    if user_id != request_user_id:
        raise HTTPException(403, "Access denied")
    
    return await db.get_portfolio(user_id)
```

#### 4. SQL Injection Prevention
```python
# Use parameterized queries (SQLAlchemy ORM does this automatically)
# NEVER use string concatenation for queries

# Good:
query = select(User).where(User.email == email)

# Bad (DON'T DO THIS):
query = f"SELECT * FROM users WHERE email = '{email}'"
```

#### 5. Rate Limiting
```python
from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

@app.post("/api/v1/auth/login")
@limiter.limit("5/minute")  # Max 5 login attempts per minute
async def login(...):
    ...
```

---

### Testing Multi-User System

```python
# tests/test_multi_user.py
import pytest
from httpx import AsyncClient

@pytest.mark.asyncio
async def test_user_data_isolation():
    """Test that users can only access their own data"""
    
    # Create two users
    user1 = await create_test_user("user1@example.com")
    user2 = await create_test_user("user2@example.com")
    
    # User1 creates portfolio
    async with AsyncClient() as client:
        response = await client.post(
            "/api/v1/portfolio",
            headers={"Authorization": f"Bearer {user1.token}"},
            json={"total_assets": 100000}
        )
        assert response.status_code == 200
    
    # User2 tries to access User1's portfolio (should fail)
    async with AsyncClient() as client:
        response = await client.get(
            f"/api/v1/portfolio/{user1.user_id}",
            headers={"Authorization": f"Bearer {user2.token}"}
        )
        assert response.status_code == 403  # Forbidden

@pytest.mark.asyncio
async def test_concurrent_users():
    """Test multiple users using system simultaneously"""
    
    users = [await create_test_user(f"user{i}@example.com") for i in range(10)]
    
    # All users chat simultaneously
    async def user_chat(user):
        async with AsyncClient() as client:
            response = await client.post(
                "/api/v1/chat",
                headers={"Authorization": f"Bearer {user.token}"},
                json={"message": "What's my net worth?"}
            )
            return response.status_code == 200
    
    # Run concurrently
    results = await asyncio.gather(*[user_chat(u) for u in users])
    
    assert all(results)  # All should succeed
```

---

### Deployment Considerations

#### Environment Variables
```bash
# .env for multi-user deployment
DATABASE_URL=postgresql://user:pass@localhost/wealthdb
REDIS_URL=redis://localhost:6379
SECRET_KEY=your-super-secret-key-change-in-production
ANTHROPIC_API_KEY=your-anthropic-key

# Security
CORS_ORIGINS=https://yourdomain.com
SESSION_TIMEOUT=1800  # 30 minutes

# Email (for verification, password reset)
SMTP_HOST=smtp.gmail.com
SMTP_PORT=587
SMTP_USER=your-email@gmail.com
SMTP_PASSWORD=your-app-password
```

#### Database Setup
```bash
# Use PostgreSQL for production (supports concurrent users)
# SQLite is OK for development/demo

# Migrations with Alembic
alembic init migrations
alembic revision --autogenerate -m "Add user tables"
alembic upgrade head
```

#### Scaling Considerations
- Use Redis for session storage (faster than DB)
- Implement connection pooling
- Add database read replicas for heavy load
- Use message queue (Celery + Redis) for long tasks

---

## Summary & Recommendations

### Recommended Implementation Path

**Phase 1 (Week 1)**: CLI
- Quick validation of agent logic

**Phase 2 (Weeks 2-3)**: Streamlit Single-User
- Full UI with all pages
- Demo-ready interface
- Basic session management

**Phase 3 (Weeks 4-6)**: Multi-User Support
- User registration & login
- JWT authentication
- Data isolation per user
- Concurrent user support

### Key Deliverables

**Must Have**:
1. ✅ Streamlit UI with 4 pages
2. ✅ Authentication system (login/register)
3. ✅ User data isolation
4. ✅ Session management

**Nice to Have**:
5. 🟡 Email verification
6. 🟡 Password reset
7. 🟡 User preferences/settings

**Optional**:
8. 🟢 Social login (Google, GitHub)
9. 🟢 Admin dashboard
10. 🟢 Usage analytics per user



