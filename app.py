import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from rag_pipeline import RAG, ingest_pdf_to_user, delete_user_pdf, clear_user_data

# Load env variables from .env.local and .env
load_dotenv(".env.local")
load_dotenv(".env")

st.set_page_config(page_title="Personal Scientific RAG", layout="wide")

# Custom CSS for high-quality dark glassmorphic aesthetics
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');

/* Apply modern typography */
html, body, [class*="css"], .stMarkdown {
    font-family: 'Outfit', sans-serif;
}

/* Title gradient animation */
.main-title {
    background: linear-gradient(90deg, #3b82f6 0%, #8b5cf6 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
    font-size: 2.8rem;
    margin-bottom: 0.2rem;
    letter-spacing: -0.05rem;
}

.subtitle {
    font-size: 1.1rem;
    color: #9ca3af;
    margin-bottom: 2rem;
}

/* Glassmorphic card styling for outputs */
.glass-card {
    background: rgba(30, 41, 59, 0.4);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 12px;
    padding: 1.5rem;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.2);
    backdrop-filter: blur(8px);
    -webkit-backdrop-filter: blur(8px);
    margin-bottom: 1.5rem;
}

.workspace-badge {
    background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
    color: white;
    padding: 0.3rem 0.8rem;
    border-radius: 20px;
    font-weight: 600;
    font-size: 0.85rem;
    display: inline-block;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 12px rgba(59, 130, 246, 0.3);
}

/* File list border and styles */
.file-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.05);
    padding: 0.5rem 0.8rem;
    border-radius: 8px;
    margin-bottom: 0.4rem;
}

/* Remove button borders on trash icon buttons */
div[data-testid="column"] button {
    border: none !important;
    background: transparent !important;
    color: #ef4444 !important;
}

div[data-testid="column"] button:hover {
    color: #f87171 !important;
    transform: scale(1.1);
}
</style>
""", unsafe_allow_html=True)

# Main Title & Subtitle
st.markdown('<div class="main-title">🧬 Personal Multi-User RAG App</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload your own papers and query your isolated personal knowledge base securely.</div>', unsafe_allow_html=True)

# Check for Clerk Publishable Key in environment variables
clerk_publishable_key = os.getenv("CLERK_PUBLISHABLE_KEY") or os.getenv("NEXT_PUBLIC_CLERK_PUBLISHABLE_KEY")

if not clerk_publishable_key:
    st.error("🔑 **Clerk Publishable Key is missing!**")
    st.markdown("""
    Please configure your Clerk Publishable Key inside your project's `.env.local` or `.env` file to enable secure user registration and login.
    ```env
    CLERK_PUBLISHABLE_KEY=pk_test_...
    ```
    """)
    st.stop()

# Handle active sign-out flow
if "clerk_signout" in st.session_state:
    st.markdown("### Signing out...")
    import streamlit.components.v1 as components
    html_code = f"""
    <html>
    <head>
        <script 
          async 
          crossorigin="anonymous" 
          data-clerk-publishable-key="{clerk_publishable_key}" 
          src="https://cdn.clerk.com/clerk.js" 
          type="text/javascript"
        ></script>
    </head>
    <body>
        <script>
            function initSignout() {{
                if (window.Clerk) {{
                    if (window.Clerk.isReady && window.Clerk.isReady()) {{
                        runSignout();
                    }} else {{
                        window.Clerk.load().then(runSignout);
                    }}
                }} else {{
                    setTimeout(initSignout, 100);
                }}
            }}

            async function runSignout() {{
                if (window.Clerk.user) {{
                    await window.Clerk.signOut();
                }}
                window.parent.location.href = window.parent.location.origin + window.parent.location.pathname;
            }}

            window.addEventListener('load', initSignout);
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=100)
    st.stop()

# Check URL query parameters for authenticated Clerk user
print("DEBUG: st.query_params keys:", list(st.query_params.keys()), "username:", st.query_params.get("username"))
clerk_user = st.query_params.get("username")

# If user is not signed in, display embedded Clerk login panel
if not clerk_user:
    st.subheader("🔑 Workspace Authentication Required")
    st.info("Please sign up or sign in using Clerk to unlock your secure personal workspace.")
    
    import streamlit.components.v1 as components
    html_code = f"""
    <html>
    <head>
        <script 
          async 
          crossorigin="anonymous" 
          data-clerk-publishable-key="{clerk_publishable_key}" 
          src="https://cdn.clerk.com/clerk.js" 
          type="text/javascript"
        ></script>
        <style>
            body {{
                background-color: transparent;
                display: flex;
                justify-content: center;
                align-items: center;
                height: 100vh;
                margin: 0;
                font-family: 'Outfit', sans-serif;
            }}
            #sign-in-container {{
                background: rgba(30, 41, 59, 0.4);
                border: 1px solid rgba(255, 255, 255, 0.08);
                border-radius: 16px;
                padding: 1rem;
                box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.27);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
            }}
        </style>
    </head>
    <body>
        <div id="sign-in-container">
            <div id="sign-in"></div>
        </div>
        <script>
            function initClerk() {{
                if (window.Clerk) {{
                    if (window.Clerk.isReady && window.Clerk.isReady()) {{
                        startClerk();
                    }} else {{
                        window.Clerk.load().then(startClerk);
                    }}
                }} else {{
                    setTimeout(initClerk, 100);
                }}
            }}

            function startClerk() {{
                window.Clerk.addListener((state) => {{
                    if (state.user) {{
                        const username = state.user.username || state.user.id;
                        window.parent.location.href = window.parent.location.origin + window.parent.location.pathname + '?username=' + encodeURIComponent(username);
                    }}
                }});

                if (window.Clerk.user) {{
                    const username = window.Clerk.user.username || window.Clerk.user.id;
                    window.parent.location.href = window.parent.location.origin + window.parent.location.pathname + '?username=' + encodeURIComponent(username);
                }} else {{
                    window.Clerk.mountSignIn(document.getElementById('sign-in'));
                }}
            }}

            window.addEventListener('load', initClerk);
        </script>
    </body>
    </html>
    """
    components.html(html_code, height=650)
    st.stop()

username = clerk_user

# ---------------- Sidebar Configuration ---------------- #
with st.sidebar:
    st.header("Workspace Account")
    st.write(f"Logged in as: **{username}**")
    
    if st.button("Sign Out", type="secondary", use_container_width=True):
        st.session_state["clerk_signout"] = True
        st.rerun()

    st.markdown("---")
    st.header("Settings")
    
    top_k = st.slider("Top-K passages", 3, 10, 5)
    model_choice = st.selectbox(
        "Groq Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
        index=0,
        help="Llama 3.3 = high capability. Llama 3.1 = extremely fast."
    )

    st.markdown("---")
    st.header("Upload Documents")
    
    # Initialize dynamic user directories
    user_dir = Path("data") / "users" / username / "papers"
    user_dir.mkdir(parents=True, exist_ok=True)
    
    # File Uploader
    uploaded_files = st.file_uploader(
        "Upload PDF papers to this workspace", 
        type="pdf", 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        for uploaded_file in uploaded_files:
            dest_path = user_dir / uploaded_file.name
            if not dest_path.exists():
                with open(dest_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                with st.spinner(f"Ingesting {uploaded_file.name}..."):
                    chunks_added = ingest_pdf_to_user(dest_path, username)
                st.success(f"Indexed {chunks_added} chunks from {uploaded_file.name}!")
                st.rerun()

# Initialize/Switch RAG session state dynamically
if (
    "rag" not in st.session_state
    or st.session_state.get("username") != username
    or st.session_state.get("top_k") != top_k
    or st.session_state.get("model_choice") != model_choice
):
    st.session_state["rag"] = RAG(user_id=username, top_k=top_k, model_name=model_choice)
    st.session_state["username"] = username
    st.session_state["top_k"] = top_k
    st.session_state["model_choice"] = model_choice

# ---------------- Main Page Content ---------------- #

# Show Workspace Badge
st.markdown(f'<span class="workspace-badge">Active Workspace: <b>{username}</b></span>', unsafe_allow_html=True)

# Workspace contents listing & management
col_left, col_right = st.columns([2, 1])

with col_right:
    st.markdown("### 📚 Workspace Documents")
    existing_files = list(user_dir.glob("*.pdf"))
    
    if existing_files:
        for file_path in existing_files:
            c1, c2 = st.columns([5, 1])
            with c1:
                st.markdown(f"📄 **{file_path.name}**")
            with c2:
                # Use key based on username and filename to prevent duplicate stream ids
                if st.button("🗑️", key=f"del_{username}_{file_path.name}"):
                    with st.spinner(f"Deleting {file_path.name}..."):
                        delete_user_pdf(file_path, username)
                    st.warning(f"Deleted {file_path.name}")
                    st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Clear Workspace Data", type="primary", use_container_width=True):
            with st.spinner("Clearing all workspace files and indexes..."):
                clear_user_data(username)
            if "rag" in st.session_state:
                del st.session_state["rag"]
            st.success("Workspace cleared successfully!")
            st.rerun()
    else:
        st.info("No documents uploaded yet. Drag & drop papers in the sidebar to populate your workspace.")

with col_left:
    st.markdown("### 🔍 Query Workspace")
    q = st.text_input("Ask a question about the uploaded papers in your workspace:")

    if st.button("Submit Question", type="primary") and q.strip():
        with st.spinner("Generating answer..."):
            answer, citations, hits = st.session_state["rag"].answer(q)

        st.markdown('<div class="glass-card">', unsafe_allow_html=True)
        st.markdown("#### 💡 Answer")
        st.write(answer)
        st.markdown('</div>', unsafe_allow_html=True)

        if citations.strip():
            with st.expander("Sources"):
                st.code(citations)

        if hits:
            with st.expander("Retrieved Context Chunks"):
                for doc, meta, dist in hits:
                    st.markdown(
                        f"📁 **{meta['doc_id']}** • chunk#{meta['chunk_index']} • distance={dist:.3f}"
                    )
                    st.write(doc)
                    st.markdown("---")
        else:
            st.info("No relevant context found in this workspace for your query.")
