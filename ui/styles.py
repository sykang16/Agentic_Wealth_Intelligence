"""Custom styles and icons for the Wealth Intelligence UI."""

# Feather Icons as SVG (commonly used ones)
ICONS = {
    "trending-up": """<svg xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg>""",
    "layout": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="3" y="3" width="18" height="18" rx="2" ry="2"></rect><line x1="3" y1="9" x2="21" y2="9"></line><line x1="9" y1="21" x2="9" y2="9"></line></svg>""",
    "message-circle": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 11.5a8.38 8.38 0 0 1-.9 3.8 8.5 8.5 0 0 1-7.6 4.7 8.38 8.38 0 0 1-3.8-.9L3 21l1.9-5.7a8.38 8.38 0 0 1-.9-3.8 8.5 8.5 0 0 1 4.7-7.6 8.38 8.38 0 0 1 3.8-.9h.5a8.48 8.48 0 0 1 8 8v.5z"></path></svg>""",
    "target": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><circle cx="12" cy="12" r="6"></circle><circle cx="12" cy="12" r="2"></circle></svg>""",
    "book-open": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M2 3h6a4 4 0 0 1 4 4v14a3 3 0 0 0-3-3H2z"></path><path d="M22 3h-6a4 4 0 0 0-4 4v14a3 3 0 0 1 3-3h7z"></path></svg>""",
    "bar-chart-2": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>""",
    "dollar-sign": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="1" x2="12" y2="23"></line><path d="M17 5H9.5a3.5 3.5 0 0 0 0 7h5a3.5 3.5 0 0 1 0 7H6"></path></svg>""",
    "briefcase": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"></rect><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"></path></svg>""",
    "credit-card": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="1" y="4" width="22" height="16" rx="2" ry="2"></rect><line x1="1" y1="10" x2="23" y2="10"></line></svg>""",
    "percent": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="19" y1="5" x2="5" y2="19"></line><circle cx="6.5" cy="6.5" r="2.5"></circle><circle cx="17.5" cy="17.5" r="2.5"></circle></svg>""",
    "pie-chart": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21.21 15.89A10 10 0 1 1 8 2.83"></path><path d="M22 12A10 10 0 0 0 12 2v10z"></path></svg>""",
    "activity": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"></polyline></svg>""",
    "user": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M20 21v-2a4 4 0 0 0-4-4H8a4 4 0 0 0-4 4v2"></path><circle cx="12" cy="7" r="4"></circle></svg>""",
    "search": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="11" cy="11" r="8"></circle><line x1="21" y1="21" x2="16.65" y2="16.65"></line></svg>""",
    "database": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><ellipse cx="12" cy="5" rx="9" ry="3"></ellipse><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"></path><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"></path></svg>""",
    "refresh-cw": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 4 23 10 17 10"></polyline><polyline points="1 20 1 14 7 14"></polyline><path d="M3.51 9a9 9 0 0 1 14.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0 0 20.49 15"></path></svg>""",
    "file-text": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>""",
    "folder": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>""",
    "trash-2": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polyline points="3 6 5 6 21 6"></polyline><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"></path><line x1="10" y1="11" x2="10" y2="17"></line><line x1="14" y1="11" x2="14" y2="17"></line></svg>""",
    "check-circle": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"></path><polyline points="22 4 12 14.01 9 11.01"></polyline></svg>""",
    "alert-circle": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"></circle><line x1="12" y1="8" x2="12" y2="12"></line><line x1="12" y1="16" x2="12.01" y2="16"></line></svg>""",
    "cpu": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"></rect><rect x="9" y="9" width="6" height="6"></rect><line x1="9" y1="1" x2="9" y2="4"></line><line x1="15" y1="1" x2="15" y2="4"></line><line x1="9" y1="20" x2="9" y2="23"></line><line x1="15" y1="20" x2="15" y2="23"></line><line x1="20" y1="9" x2="23" y2="9"></line><line x1="20" y1="14" x2="23" y2="14"></line><line x1="1" y1="9" x2="4" y2="9"></line><line x1="1" y1="14" x2="4" y2="14"></line></svg>""",
    "send": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="22" y1="2" x2="11" y2="13"></line><polygon points="22 2 15 22 11 13 2 9 22 2"></polygon></svg>""",
    "arrow-up": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"></line><polyline points="5 12 12 5 19 12"></polyline></svg>""",
    "arrow-down": """<svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"></line><polyline points="19 12 12 19 5 12"></polyline></svg>""",
    "download": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"></path><polyline points="7 10 12 15 17 10"></polyline><line x1="12" y1="15" x2="12" y2="3"></line></svg>""",
    "zap": """<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg>""",
}


def get_icon(name: str, color: str = "currentColor", size: int = 18) -> str:
    """Get an SVG icon with custom color and size.

    Args:
        name: Icon name from ICONS dict.
        color: CSS color value.
        size: Icon size in pixels.

    Returns:
        SVG string with applied styles.
    """
    svg = ICONS.get(name, ICONS["activity"])
    svg = svg.replace('width="18"', f'width="{size}"')
    svg = svg.replace('height="18"', f'height="{size}"')
    svg = svg.replace('width="24"', f'width="{size}"')
    svg = svg.replace('height="24"', f'height="{size}"')
    svg = svg.replace('width="14"', f'width="{size}"')
    svg = svg.replace('height="14"', f'height="{size}"')
    svg = svg.replace('stroke="currentColor"', f'stroke="{color}"')
    return svg


def icon_html(name: str, color: str = "currentColor", size: int = 18) -> str:
    """Get an icon wrapped in a span for inline use.

    Args:
        name: Icon name.
        color: CSS color value.
        size: Icon size.

    Returns:
        HTML string with icon.
    """
    return f'<span style="display: inline-flex; vertical-align: middle;">{get_icon(name, color, size)}</span>'


# Main CSS styles for the app
MAIN_CSS = """
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    /* Root variables */
    :root {
        --primary: #0052cc;
        --primary-light: #e6f0ff;
        --secondary: #00875a;
        --background: #f4f5f7;
        --surface: #ffffff;
        --text-primary: #172b4d;
        --text-secondary: #64748b;
        --border: #dfe1e6;
        --gradient-start: #3b82f6;
        --gradient-end: #8b5cf6;
    }

    /* ============================================
       GLOBAL FONT SIZE OVERRIDE
       Streamlit 1.28 nests elements deeply so we
       must use broad selectors with !important.
       ============================================ */
    html, body,
    [class*="css"],
    .stApp, .stApp *,
    .main, .main *,
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* Set base font-size on root */
    html {
        font-size: 17px !important;
    }

    /* Override Streamlit text elements (exclude inline-styled header) */
    .stApp p,
    .stApp label,
    .stApp li,
    .stApp td,
    .stApp th,
    .stMarkdown,
    .stMarkdown p,
    .stMarkdown li,
    [data-testid="stMarkdownContainer"] p,
    [data-testid="stMarkdownContainer"] li {
        font-size: 17px !important;
        line-height: 1.6 !important;
    }

    /* Spans - only override those without inline style (preserves header) */
    .stMarkdown span:not([style]),
    [data-testid="stMarkdownContainer"] span:not([style]) {
        font-size: 17px !important;
        line-height: 1.6 !important;
    }

    /* Do NOT override font-size on divs with inline style (header, callouts, etc.) */
    .stApp div:not([style]) {
        line-height: 1.6;
    }

    /* Bold/strong text within markdown */
    [data-testid="stMarkdownContainer"] strong {
        font-size: inherit !important;
    }

    /* Headings - keep original sizes (already large) */
    .stApp h1, [data-testid="stMarkdownContainer"] h1 { font-size: 2rem !important; }
    .stApp h2, [data-testid="stMarkdownContainer"] h2 { font-size: 1.6rem !important; }
    .stApp h3, [data-testid="stMarkdownContainer"] h3 { font-size: 1.35rem !important; }
    .stApp h4, [data-testid="stMarkdownContainer"] h4 { font-size: 1.15rem !important; }

    /* Sidebar text */
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] .stMarkdown p {
        font-size: 16px !important;
        line-height: 1.6 !important;
    }

    /* Hide Streamlit default header */
    header[data-testid="stHeader"] {
        display: none;
    }

    /* Main container adjustments */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: var(--surface);
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 1rem;
    }

    /* ============================================
       TAB STYLING
       ============================================ */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.35rem;
        background: #f1f5f9;
        padding: 0.5rem;
        border-radius: 12px;
    }

    .stTabs [data-baseweb="tab"],
    .stTabs [data-baseweb="tab"] * {
        border-radius: 8px;
        padding: 0.5rem 0.9rem;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        white-space: nowrap;
    }

    .stTabs [aria-selected="true"] {
        background: #0052cc !important;
        color: white !important;
    }

    /* ============================================
       METRIC STYLING - keep values at original
       size (already large), only increase labels
       ============================================ */
    [data-testid="stMetricValue"],
    [data-testid="stMetricValue"] * {
        font-size: 1.75rem !important;
        font-weight: 700 !important;
        line-height: 1.3 !important;
    }

    [data-testid="stMetricLabel"],
    [data-testid="stMetricLabel"] * {
        font-size: 1.05rem !important;
        color: #475569 !important;
    }

    [data-testid="stMetricDelta"],
    [data-testid="stMetricDelta"] * {
        font-size: 0.95rem !important;
    }

    /* ============================================
       EXPANDER STYLING
       ============================================ */
    .streamlit-expanderHeader,
    [data-testid="stExpander"] summary,
    [data-testid="stExpander"] summary span,
    [data-testid="stExpander"] summary p {
        font-weight: 600 !important;
        font-size: 1.05rem !important;
        background: #f8fafc;
        border-radius: 10px;
        min-height: 48px;
    }

    /* Streamlit 1.55: hide Material icon text in expander (keyboard_arrow_down leaks as text) */
    [data-testid="stExpander"] [data-testid="stIconMaterial"] {
        display: none !important;
    }

    /* Replace with CSS-only arrow */
    [data-testid="stExpander"] details > summary::before {
        content: '▶';
        font-size: 0.75rem;
        color: #64748b;
        margin-right: 0.4rem;
        display: inline-block;
        transition: transform 0.2s ease;
    }
    [data-testid="stExpander"] details[open] > summary::before {
        transform: rotate(90deg);
    }

    /* ============================================
       BUTTON STYLING
       ============================================ */
    .stButton > button,
    .stButton > button * {
        border-radius: 10px !important;
        font-weight: 500 !important;
        padding: 0.4rem 1rem !important;
        font-size: 0.95rem !important;
        transition: all 0.2s ease;
    }

    /* Form submit button container — prevent flex-stretch, align with input */
    div[data-testid="stFormSubmitButton"] {
        display: flex !important;
        align-items: flex-end !important;
        padding-bottom: 0 !important;
    }

    /* Form submit buttons — height aligned with text_input (min-height: 48px) */
    div[data-testid="stFormSubmitButton"] > button,
    .stFormSubmitButton > button {
        border-radius: 10px !important;
        font-weight: 600 !important;
        padding: 0 1rem !important;
        height: 48px !important;
        min-height: 48px !important;
        max-height: 48px !important;
        line-height: 48px !important;
        font-size: 1rem !important;
        width: 100% !important;
        box-sizing: border-box !important;
    }

    .stButton > button:hover {
        border-color: var(--primary) !important;
        box-shadow: 0 2px 8px rgba(0, 82, 204, 0.15) !important;
    }

    .stButton > button[kind="primary"],
    .stFormSubmitButton > button[kind="primary"],
    div[data-testid="stFormSubmitButton"] > button[kind="primary"] {
        background: linear-gradient(135deg, #0052cc 0%, #0747a6 100%) !important;
    }

    /* ============================================
       FOCUS INDICATORS (WCAG AA)
       ============================================ */
    .stButton > button:focus-visible,
    .stFormSubmitButton > button:focus-visible,
    .stTextInput input:focus-visible,
    .stSelectbox > div > div:focus-within,
    .stTabs [data-baseweb="tab"]:focus-visible {
        outline: none !important;
        box-shadow: 0 0 0 3px rgba(59, 130, 246, 0.35) !important;
    }

    /* ============================================
       INPUT STYLING
       ============================================ */
    .stTextInput input {
        border-radius: 10px !important;
        min-height: 48px !important;
        font-size: 1rem !important;
        padding: 0.75rem 1rem !important;
    }

    .stSelectbox > div > div {
        border-radius: 10px !important;
        min-height: 48px !important;
        font-size: 1rem !important;
    }

    .stTextInput label,
    .stSelectbox label,
    .stCheckbox label,
    .stRadio label {
        font-size: 1rem !important;
        font-weight: 500 !important;
    }

    /* Sidebar selectbox border */
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        border: 2px solid #94a3b8 !important;
        border-radius: 10px !important;
        transition: border-color 0.2s ease;
    }

    section[data-testid="stSidebar"] .stSelectbox > div > div:hover {
        border-color: var(--primary) !important;
    }

    /* Sidebar buttons - keep compact */
    section[data-testid="stSidebar"] .stButton > button,
    section[data-testid="stSidebar"] .stButton > button * {
        min-height: 36px !important;
        font-size: 0.9rem !important;
        padding: 0.35rem 0.75rem !important;
    }

    /* ============================================
       CAPTION / HELP TEXT (AA contrast)
       ============================================ */
    .stCaption,
    .stCaption *,
    [data-testid="stCaptionContainer"],
    [data-testid="stCaptionContainer"] * {
        color: #64748b !important;
        font-size: 15px !important;
    }

    /* Streamlit info/warning/error boxes */
    [data-testid="stAlert"] p,
    [data-testid="stAlert"] span,
    .stAlert p,
    .stAlert span {
        font-size: 16px !important;
    }

    /* ============================================
       SECTION TITLE (inline HTML)
       ============================================ */
    .section-header {
        display: flex;
        align-items: center;
        gap: 0.75rem;
        font-size: 1.375rem;
        font-weight: 600;
        color: #0f172a;
        margin-bottom: 1.25rem;
        padding-bottom: 0.625rem;
        border-bottom: 2px solid #e2e8f0;
    }

    .section-header svg {
        color: #0052cc;
    }

    /* ============================================
       STATUS INDICATORS (border + color)
       ============================================ */
    .status-indicator {
        display: inline-flex;
        align-items: center;
        gap: 0.375rem;
        padding: 0.375rem 1rem;
        border-radius: 9999px;
        font-size: 0.95rem;
        font-weight: 600;
    }

    .status-success {
        background: #dcfce7;
        color: #15803d;
        border: 2px solid #86efac;
    }

    .status-warning {
        background: #fef3c7;
        color: #a16207;
        border: 2px solid #fcd34d;
    }

    .status-error {
        background: #fee2e2;
        color: #b91c1c;
        border: 2px solid #fca5a5;
    }

    /* ============================================
       CARD STYLING
       ============================================ */
    div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
        background: white;
        border-radius: 14px;
        padding: 1.25rem;
        border: 1px solid #cbd5e1;
    }

    /* ============================================
       DATAFRAME / TABLE
       ============================================ */
    .stDataFrame td,
    .stDataFrame th {
        font-size: 15px !important;
    }

    /* ============================================
       DISMISS BUTTON (welcome banner only)
       Targets the button column that follows the
       welcome banner markdown container.
       ============================================ */
    [data-testid="stMarkdownContainer"]:has(.welcome-banner-container)
    ~ [data-testid="stHorizontalBlock"]
    [data-testid="stButton"] > button,
    [data-testid="stMarkdownContainer"]:has(.welcome-banner-container)
    ~ [data-testid="stHorizontalBlock"]
    [data-testid="stButton"] > button * {
        padding: 0.1rem 0.5rem !important;
        height: auto !important;
        min-height: unset !important;
        line-height: 1.4 !important;
    }
</style>
"""


def get_header_html():
    """Get the fancy header as HTML that works in Streamlit."""
    return '<div style="background: linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #0f172a 100%); padding: 1.75rem 2rem; border-radius: 16px; margin-bottom: 1.5rem; position: relative; overflow: hidden; box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);"><div style="position: absolute; top: 0; left: 0; right: 0; bottom: 0; background: radial-gradient(ellipse at 20% 50%, rgba(59, 130, 246, 0.2) 0%, transparent 50%), radial-gradient(ellipse at 80% 50%, rgba(147, 51, 234, 0.15) 0%, transparent 50%); pointer-events: none;"></div><div style="position: relative; z-index: 1;"><div style="display: flex; align-items: center; gap: 1rem; margin-bottom: 0.5rem;"><div style="width: 52px; height: 52px; background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%); border-radius: 14px; display: flex; align-items: center; justify-content: center; box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4);"><svg xmlns="http://www.w3.org/2000/svg" width="26" height="26" viewBox="0 0 24 24" fill="none" stroke="white" stroke-width="2.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"></polyline><polyline points="17 6 23 6 23 12"></polyline></svg></div><span style="font-size: 2.25rem; font-weight: 800; color: white; letter-spacing: -0.5px; font-family: Inter, sans-serif;">Wealth</span><span style="font-size: 2.25rem; font-weight: 800; background: linear-gradient(135deg, #60a5fa 0%, #a78bfa 100%); -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text; letter-spacing: -0.5px; font-family: Inter, sans-serif;">Nexus</span><span style="display: inline-flex; align-items: center; gap: 6px; background: rgba(59, 130, 246, 0.25); border: 1px solid rgba(96, 165, 250, 0.4); padding: 8px 14px; border-radius: 20px; font-size: 13px; font-weight: 600; color: #93c5fd; margin-left: 0.75rem; font-family: Inter, sans-serif;"><svg xmlns="http://www.w3.org/2000/svg" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon></svg> AI Powered</span></div><div style="font-size: 1rem; color: rgba(255, 255, 255, 0.75); margin-left: 68px; font-weight: 400; letter-spacing: 0.2px; font-family: Inter, sans-serif;">Smart Portfolio Analysis & Investment Insights</div></div></div>'


def render_header():
    """Render the fancy app header. Alias for get_header_html."""
    return get_header_html()


def render_metric_card(label: str, value: str, change: str = None, change_type: str = "positive", icon: str = "dollar-sign"):
    """Render a styled metric card.

    Args:
        label: Metric label.
        value: Metric value.
        change: Change text (optional).
        change_type: 'positive' or 'negative'.
        icon: Icon name.

    Returns:
        HTML string for the metric card.
    """
    change_color = "#15803d" if change_type == "positive" else "#b91c1c"
    change_html = ""
    if change:
        arrow_icon = "arrow-up" if change_type == "positive" else "arrow-down"
        change_html = f'<div style="display: flex; align-items: center; gap: 6px; font-size: 0.95rem; font-weight: 600; color: {change_color}; margin-top: 0.625rem;">{get_icon(arrow_icon, change_color, 16)} {change}</div>'

    return f'<div style="background: white; border-radius: 14px; padding: 1.5rem; border: 1px solid #cbd5e1; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06);"><div style="display: flex; align-items: center; gap: 8px; font-size: 1.05rem; font-weight: 600; color: #475569; margin-bottom: 0.625rem;">{get_icon(icon, "#475569", 18)} {label}</div><div style="font-size: 1.75rem; font-weight: 700; color: #0f172a;">{value}</div>{change_html}</div>'


def render_section_title(title: str, icon: str = "layout"):
    """Render a section title with icon.

    Args:
        title: Section title text.
        icon: Icon name.

    Returns:
        HTML string for the section title.
    """
    return f'<div style="display: flex; align-items: center; gap: 0.75rem; font-size: 1.375rem; font-weight: 600; color: #0f172a; margin-bottom: 1.25rem; padding-bottom: 0.625rem; border-bottom: 2px solid #e2e8f0;"><span style="color: #0052cc;">{get_icon(icon, "#0052cc", 22)}</span> {title}</div>'


def render_card(title: str, content: str, icon: str = "activity"):
    """Render a card with header and content.

    Args:
        title: Card title.
        content: Card body content (HTML).
        icon: Icon name for title.

    Returns:
        HTML string for the card.
    """
    return f'<div style="background: white; border-radius: 14px; border: 1px solid #cbd5e1; box-shadow: 0 2px 6px rgba(0, 0, 0, 0.06); overflow: hidden;"><div style="padding: 1.125rem 1.5rem; border-bottom: 1px solid #e2e8f0; background: #f8fafc;"><div style="display: flex; align-items: center; gap: 0.625rem; font-size: 1.1rem; font-weight: 600; color: #0f172a;">{get_icon(icon, "#0f172a", 20)} {title}</div></div><div style="padding: 1.5rem;">{content}</div></div>'


def render_status_badge(text: str, status: str = "success"):
    """Render a status badge.

    Args:
        text: Badge text.
        status: 'success', 'warning', or 'error'.

    Returns:
        HTML string for the badge.
    """
    icon_map = {
        "success": "check-circle",
        "warning": "alert-circle",
        "error": "alert-circle",
    }
    color_map = {
        "success": {"bg": "#dcfce7", "text": "#15803d"},
        "warning": {"bg": "#fef3c7", "text": "#a16207"},
        "error": {"bg": "#fee2e2", "text": "#b91c1c"},
    }
    icon = icon_map.get(status, "check-circle")
    colors = color_map.get(status, color_map["success"])
    border_map = {
        "success": "#86efac",
        "warning": "#fcd34d",
        "error": "#fca5a5",
    }
    border_color = border_map.get(status, "#86efac")

    return f'<span style="display: inline-flex; align-items: center; gap: 0.375rem; padding: 0.375rem 1rem; border-radius: 9999px; font-size: 0.95rem; font-weight: 600; background: {colors["bg"]}; color: {colors["text"]}; border: 2px solid {border_color};">{get_icon(icon, colors["text"], 16)} {text}</span>'


def render_guidance_callout(text: str, style: str = "info") -> str:
    """Render a contextual guidance callout box.

    Args:
        text: Guidance message (supports HTML).
        style: 'info' (blue), 'tip' (purple), or 'warning' (amber).

    Returns:
        HTML string for the callout.
    """
    styles = {
        "info": {"bg": "#eff6ff", "border": "#3b82f6", "color": "#1e40af", "icon": "&#128161;"},
        "tip": {"bg": "#faf5ff", "border": "#8b5cf6", "color": "#6d28d9", "icon": "&#127919;"},
        "warning": {"bg": "#fffbeb", "border": "#f59e0b", "color": "#92400e", "icon": "&#9888;"},
    }
    s = styles.get(style, styles["info"])
    return (
        f'<div style="background: {s["bg"]}; border-left: 4px solid {s["border"]}; '
        f'border-radius: 0 10px 10px 0; padding: 16px 20px; margin-bottom: 16px;">'
        f'<div style="font-size: 1rem; color: {s["color"]}; line-height: 1.6;">'
        f'{s["icon"]} {text}</div></div>'
    )


def render_welcome_banner() -> str:
    """Render a welcome/getting-started banner for first-time users.

    Returns:
        HTML string for the welcome banner.
    """
    return (
        '<div class="welcome-banner-container" style="background: linear-gradient(135deg, #eff6ff 0%, #f5f3ff 100%); '
        'border: 1px solid #c7d2fe; border-radius: 14px; padding: 24px 28px; margin-bottom: 20px;">'
        '<div style="display: flex; align-items: flex-start; gap: 16px;">'
        '<div style="width: 44px; height: 44px; background: linear-gradient(135deg, #3b82f6, #8b5cf6); '
        'border-radius: 12px; display: flex; align-items: center; justify-content: center; flex-shrink: 0;">'
        '<span style="font-size: 22px; color: white;">&#9733;</span></div>'
        '<div style="flex: 1;">'
        '<div style="font-size: 1.2rem; font-weight: 700; color: #1e293b; margin-bottom: 6px;">'
        'Welcome to WealthNexus</div>'
        '<div style="font-size: 0.95rem; color: #475569; margin-bottom: 14px;">'
        'Get started with your personalized financial insights:</div>'
        '<div style="display: flex; gap: 16px; flex-wrap: wrap;">'
        '<div style="flex: 1; min-width: 180px; background: white; border-radius: 10px; '
        'padding: 12px 14px; border: 1px solid #e2e8f0;">'
        '<strong style="font-size: 0.95rem; color: #1e293b;">1. Select a user</strong>'
        '<div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">Choose a portfolio from the sidebar</div></div>'
        '<div style="flex: 1; min-width: 180px; background: white; border-radius: 10px; '
        'padding: 12px 14px; border: 1px solid #e2e8f0;">'
        '<strong style="font-size: 0.95rem; color: #1e293b;">2. Explore the dashboard</strong>'
        '<div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">View net worth, allocation, and holdings</div></div>'
        '<div style="flex: 1; min-width: 180px; background: white; border-radius: 10px; '
        'padding: 12px 14px; border: 1px solid #e2e8f0;">'
        '<strong style="font-size: 0.95rem; color: #1e293b;">3. Ask your AI advisor</strong>'
        '<div style="font-size: 0.85rem; color: #64748b; margin-top: 4px;">Chat about your portfolio and get recommendations</div></div>'
        '</div></div></div></div>'
    )
