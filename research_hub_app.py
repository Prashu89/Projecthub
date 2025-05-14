import streamlit as st
import pandas as pd
import google.generativeai as genai
from datetime import datetime
import uuid # To generate unique keys for notes

# --- Page Configuration (Set this at the very top) ---
st.set_page_config(
    page_title="Dissertation Research Hub",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Initial Data (from your React app) ---
initial_dataset_data_list = [
    {"id": 1, "officialArchive": "Landsat-5 TM SR", "datasetName": "Landsat-5 TM SR (1984 – 2012)", "subset": "1997 – 1999 (dry-season Nov–Apr composites)", "yearsUsed": "3 yr", "notes": "Earliest cloud-tolerable, Tier-1/2 SR coverage for Kerala; forms historical baseline.", "status": "Downloaded", "userEditableNotes": "Initial thoughts: This will be crucial for historical comparison."},
    {"id": 2, "officialArchive": "Landsat-9 OLI-2 SR", "datasetName": "Landsat-9 OLI-2 SR (Oct 2021 – present)", "subset": "May 2024 → Apr 2025 rolling 12-month composite", "yearsUsed": "1 yr", "notes": "Gives “now” imagery with a full seasonal cycle; May-2025 scenes are still ingesting, so end date fixed at 30 Apr 2025.", "status": "Downloaded", "userEditableNotes": ""},
    {"id": 3, "officialArchive": "Sentinel-2 MSI L2A", "datasetName": "Sentinel-2 MSI L2A (Jun 2015 – present)", "subset": "2019 – Apr 2025 (all clear scenes; red-edge indices)", "yearsUsed": "7 yr", "notes": "Overlaps GEDI (2019-23) and extends to 2025 for freshest fine-detail NDVI/EVI.", "status": "Not Downloaded", "userEditableNotes": "Consider cloud cover impact for Sentinel-2."},
    {"id": 4, "officialArchive": "Sentinel-1 C-SAR GRD", "datasetName": "Sentinel-1 C-SAR GRD (Oct 2014 – present)", "subset": "Oct 2014 – Apr 2025", "yearsUsed": "11 yr", "notes": "Full radar record; last processed frame is 22 Apr 2025 in the ESA archive.", "status": "Not Downloaded", "userEditableNotes": ""},
    {"id": 5, "officialArchive": "GEDI L4B AGB", "datasetName": "GEDI L4B AGB (Apr 2019 – Mar 2023)", "subset": "2019 – 2023", "yearsUsed": "5 yr", "notes": "LiDAR mission ended Mar 2023; v2.1 footprints complete, no newer data.", "status": "Downloaded", "userEditableNotes": "GEDI data is key for biomass estimates."},
    {"id": 6, "officialArchive": "ESA Biomass-CCI", "datasetName": "ESA Biomass-CCI (epochs 2010, 2017; annual 2018 → 2022 v6)", "subset": "2018 – 2022", "yearsUsed": "5 yr", "notes": "2023 global map still in production; 2018-22 used for biomass cross-checks.", "status": "Not Downloaded", "userEditableNotes": ""},
    {"id": 7, "officialArchive": "CHIRPS v2 precip", "datasetName": "CHIRPS v2 precip (1981 – present)", "subset": "1997 – 2023", "yearsUsed": "27 yr", "notes": "Covers entire Landsat change window + climate variability up to latest QC year (2024 monthly provisional—excluded).", "status": "Downloaded", "userEditableNotes": "Precipitation data will be vital for contextualizing changes."},
    {"id": 8, "officialArchive": "MOD17A3HGF NPP", "datasetName": "MOD17A3HGF NPP (2001 – present)", "subset": "2001 – 2023", "yearsUsed": "23 yr", "notes": "2024 tiles released but flagged “beta” as of May 2025; we keep QC-passed 2001-23 record.", "status": "Not Downloaded", "userEditableNotes": ""},
    {"id": 9, "officialArchive": "ESA WorldCover 10 m", "datasetName": "ESA WorldCover 10 m (2020 & 2021)", "subset": "2021", "yearsUsed": "1 yr", "notes": "Highest-quality global 10 m LULC; 2022-23 update not yet published.", "status": "Downloaded", "userEditableNotes": "WorldCover for LULC baseline."},
    {"id": 10, "officialArchive": "WorldClim v2 baseline", "datasetName": "WorldClim v2 baseline (1970-2000 normals)", "subset": "1970 – 2000 mean", "yearsUsed": "31 yr avg", "notes": "Supplies mean temp & radiation scalars in CASA; static climatology.", "status": "Downloaded", "userEditableNotes": ""}
]
status_options_list = ["Downloaded", "Not Downloaded", "Pending", "Error"]

# --- Session State Initialization ---
if 'datasets_df' not in st.session_state:
    st.session_state.datasets_df = pd.DataFrame(initial_dataset_data_list)
if 'gemini_api_key' not in st.session_state:
    st.session_state.gemini_api_key = ""
if "messages" not in st.session_state:
    st.session_state.messages = [{"role": "assistant", "content": "Hello! I am your AI Research Assistant. How can I help you?"}]
if "editing_dataset_id" not in st.session_state:
    st.session_state.editing_dataset_id = None
if "current_edit_notes" not in st.session_state:
    st.session_state.current_edit_notes = ""

# Session state for Google Keep style notes
if "general_notes_list" not in st.session_state:
    st.session_state.general_notes_list = [
        {"id": str(uuid.uuid4()), "title": "My First Note", "content": "This is a sample note to get started!", "timestamp": datetime.now()},
        {"id": str(uuid.uuid4()), "title": "Research Idea", "content": "Explore the impact of X on Y.", "timestamp": datetime.now()}
    ]
if "new_note_title" not in st.session_state:
    st.session_state.new_note_title = ""
if "new_note_content" not in st.session_state:
    st.session_state.new_note_content = ""


# --- Gemini API Configuration (Sidebar) ---
st.sidebar.header("API Configuration")
st.session_state.gemini_api_key = st.sidebar.text_input(
    "Enter your Gemini API Key:",
    type="password",
    value=st.session_state.gemini_api_key,
    help="Get your API key from Google AI Studio."
)

# --- Helper function to call Gemini API ---
def get_gemini_response(prompt_text, history_chat=None):
    if not st.session_state.gemini_api_key:
        return "Error: Gemini API Key not provided. Please enter it in the sidebar."
    try:
        genai.configure(api_key=st.session_state.gemini_api_key)
        model = genai.GenerativeModel(
            model_name="gemini-1.5-flash",
            safety_settings=[
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_MEDIUM_AND_ABOVE"},
            ]
        )
        if history_chat:
            chat_session = model.start_chat(history=history_chat)
            response = chat_session.send_message(prompt_text)
        else:
            response = model.generate_content(prompt_text)
        return response.text
    except Exception as e:
        return f"Error communicating with AI: {str(e)}"

# --- Navigation (Sidebar) ---
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio(
    "Choose a section:",
    ("📚 Dataset Dashboard", "💬 AI Chatbot", "📝 General Notes", "ℹ️ About"),
    key="navigation_radio"
)

# --- Main App Sections ---

# =========================
# 1. DATASET DASHBOARD
# =========================
def dataset_dashboard_page():
    st.header("📚 Dataset Dashboard")
    st.markdown("Track, update, and annotate your project datasets.")
    search_term = st.text_input("Search datasets (by name, notes, subset):", placeholder="Type to search...", key="dataset_search")
    df = st.session_state.datasets_df.copy()
    if search_term:
        search_term_lower = search_term.lower()
        df = df[
            df['datasetName'].str.lower().str.contains(search_term_lower, na=False) |
            df['notes'].str.lower().str.contains(search_term_lower, na=False) |
            df['userEditableNotes'].str.lower().str.contains(search_term_lower, na=False) |
            df['subset'].str.lower().str.contains(search_term_lower, na=False)
        ]
    st.info(f"Displaying {len(df)} of {len(st.session_state.datasets_df)} datasets.")
    for index, row_series in df.iterrows():
        row = row_series.to_dict()
        st.subheader(f"{row['id']}. {row['datasetName']}")
        cols_main, cols_status_edit = st.columns([3, 1])
        with cols_main:
            st.markdown(f"**Archive:** {row['officialArchive']}")
            st.markdown(f"**Subset Ingested:** {row['subset']}")
            st.markdown(f"**Years Used:** {row['yearsUsed']}")
        with cols_status_edit:
            current_status_index = status_options_list.index(row['status']) if row['status'] in status_options_list else 0
            new_status = st.selectbox("Status", options=status_options_list, index=current_status_index, key=f"status_select_{row['id']}")
            if new_status != row['status']:
                st.session_state.datasets_df.loc[st.session_state.datasets_df['id'] == row['id'], 'status'] = new_status
                st.rerun()
            if st.button(f"📝 Edit/View Notes", key=f"edit_btn_{row['id']}"):
                st.session_state.editing_dataset_id = row['id']
                st.session_state.current_edit_notes = row['userEditableNotes']
                st.rerun()
        with st.expander("Original Notes & Rationale (Read-only)"):
            st.caption(row['notes'])
        with st.expander("User Editable Notes (Click 'Edit/View Notes' to modify)"):
            st.caption(row['userEditableNotes'] if row['userEditableNotes'] else "_No user notes yet._")
        st.divider()

    if st.session_state.editing_dataset_id is not None:
        mask = st.session_state.datasets_df['id'] == st.session_state.editing_dataset_id
        if mask.any():
            selected_dataset_series = st.session_state.datasets_df[mask].iloc[0]
            selected_dataset = selected_dataset_series.to_dict()
            st.sidebar.subheader(f"✏️ Editing Notes for: ID {selected_dataset['id']}")
            st.sidebar.markdown(f"**{selected_dataset['datasetName']}**")
            st.session_state.current_edit_notes = st.sidebar.text_area(
                "Your Notes:", value=st.session_state.current_edit_notes, height=200, key=f"notes_text_area_{st.session_state.editing_dataset_id}"
            )
            ai_summary_prompt = f"""Summarize the key aspects and potential uses of the following dataset for research notes:
Dataset Name: {selected_dataset['datasetName']}
Subset/Coverage: {selected_dataset['subset']}
Original Rationale: {selected_dataset['notes']}
Focus on generating a concise summary that could be useful for a researcher's personal notes."""
            if st.sidebar.button("✨ AI: Generate Summary", key=f"ai_summary_btn_{st.session_state.editing_dataset_id}"):
                if st.session_state.gemini_api_key:
                    with st.spinner("🤖 AI is thinking..."):
                        summary = get_gemini_response(ai_summary_prompt)
                    if not summary.startswith("Error:"):
                        st.session_state.current_edit_notes += f"\n\n[AI Summary - {datetime.now().strftime('%Y-%m-%d %H:%M')}]:\n{summary}"
                    else:
                        st.sidebar.error(summary)
                    st.rerun()
                else:
                    st.sidebar.warning("Please enter your Gemini API Key to generate summaries.")
            col1_sidebar, col2_sidebar = st.sidebar.columns(2)
            with col1_sidebar:
                if st.button("💾 Save Notes", key=f"save_notes_btn_{st.session_state.editing_dataset_id}"):
                    st.session_state.datasets_df.loc[st.session_state.datasets_df['id'] == st.session_state.editing_dataset_id, 'userEditableNotes'] = st.session_state.current_edit_notes
                    st.session_state.editing_dataset_id = None
                    st.session_state.current_edit_notes = ""
                    st.sidebar.success("Notes saved!")
                    st.rerun()
            with col2_sidebar:
                if st.button("✖️ Cancel Edit", key=f"cancel_edit_btn_{st.session_state.editing_dataset_id}"):
                    st.session_state.editing_dataset_id = None
                    st.session_state.current_edit_notes = ""
                    st.rerun()
        else:
            st.sidebar.error("Could not find the dataset to edit. Please refresh.")
            st.session_state.editing_dataset_id = None

# =========================
# 2. AI CHATBOT
# =========================
def ai_chatbot_page():
    st.header("💬 AI Research Chatbot")
    st.markdown("Your personal AI assistant. Ask anything about your research!")
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    if prompt := st.chat_input("What can I help you with today?"):
        if not st.session_state.gemini_api_key:
            st.warning("Please enter your Gemini API Key in the sidebar to use the chatbot.")
            st.stop()
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)
        gemini_history = []
        for msg in st.session_state.messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            gemini_history.append({"role": role, "parts": [{"text": msg["content"]}]})
        with st.chat_message("assistant"):
            with st.spinner("🤖 Thinking..."):
                response_text = get_gemini_response(prompt, history_chat=gemini_history)
            st.markdown(response_text)
        st.session_state.messages.append({"role": "assistant", "content": response_text})

# =========================
# 3. GENERAL NOTES PAGE (Google Keep Style)
# =========================
def general_notes_page():
    st.header("📝 General Research Notes")
    st.markdown("Jot down your thoughts, ideas, and quick notes. Click 'Add Note' to save.")

    # --- Input area for new note ---
    with st.expander("➕ Take a new note...", expanded=True): # Keep it expanded by default
        st.session_state.new_note_title = st.text_input(
            "Note Title (Optional):",
            value=st.session_state.new_note_title,
            key="new_note_title_input",
            placeholder="Enter a title for your note..."
        )
        st.session_state.new_note_content = st.text_area(
            "Note Content:",
            value=st.session_state.new_note_content,
            key="new_note_content_input",
            placeholder="Type your note here...",
            height=150
        )
        if st.button("➕ Add Note", key="add_new_note_btn", type="primary"):
            if st.session_state.new_note_content: # Only add if there's content
                new_note = {
                    "id": str(uuid.uuid4()), # Unique ID for each note
                    "title": st.session_state.new_note_title if st.session_state.new_note_title else "Untitled Note",
                    "content": st.session_state.new_note_content,
                    "timestamp": datetime.now()
                }
                st.session_state.general_notes_list.insert(0, new_note) # Add to the beginning of the list
                # Clear input fields after adding
                st.session_state.new_note_title = ""
                st.session_state.new_note_content = ""
                st.toast("Note added!", icon="✏️")
                st.rerun() # Rerun to update the display immediately
            else:
                st.warning("Note content cannot be empty.")
    
    st.markdown("---") # Divider

    # --- Display existing notes in a grid-like layout ---
    if not st.session_state.general_notes_list:
        st.info("No notes yet. Add your first note above!")
    else:
        # Define number of columns for the grid
        num_columns = 3 # You can adjust this
        
        # Create columns
        cols = st.columns(num_columns)
        
        # Distribute notes into columns
        for i, note in enumerate(st.session_state.general_notes_list):
            col_index = i % num_columns
            with cols[col_index]:
                with st.container(): # Use st.container for each card
                    # Apply some basic styling for a card-like appearance
                    st.markdown(
                        f"""
                        <div style="border: 1px solid #4A4A4A; border-radius: 7px; padding: 15px; margin-bottom: 15px; background-color: #222226;">
                            <h5 style="margin-top: 0; margin-bottom: 5px; color: #FAFAFA;">{note['title']}</h5>
                            <p style="font-size: 0.9em; color: #D0D0D0; white-space: pre-wrap; word-wrap: break-word;">{note['content']}</p>
                            <p style="font-size: 0.75em; color: #888888; text-align: right; margin-bottom:0;">
                                {note['timestamp'].strftime('%Y-%m-%d %H:%M')}
                            </p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
                    # Delete button for each note
                    if st.button(f"🗑️ Delete", key=f"delete_note_{note['id']}", help="Delete this note"):
                        st.session_state.general_notes_list = [n for n in st.session_state.general_notes_list if n['id'] != note['id']]
                        st.toast(f"Note '{note['title']}' deleted.", icon="🗑️")
                        st.rerun()
    
    st.markdown("---")
    st.caption("Note: Currently, these general notes are stored in the session and will be cleared if you close the browser tab or the app restarts. For persistent storage, integration with a database or local file saving would be needed.")


# =========================
# 4. ABOUT PAGE
# =========================
def about_page():
    st.header("ℹ️ About This Hub")
    st.markdown("""
        This **Dissertation Research Hub** is a Streamlit application designed to help you manage datasets
        and interact with an AI research assistant powered by Google's Gemini models.

        **Features:**
        - **Dataset Dashboard:** View, search, and update the status of your research datasets. You can also add and edit personal notes, with an option to generate AI-powered summaries for them.
        - **AI Chatbot:** Engage in a conversation with an AI assistant to get help with your research questions, brainstorming, or drafting content.
        - **General Notes:** A dedicated space for your free-form research notes and ideas, displayed in a card-style format.

        **How to Use:**
        1.  **Enter Your Gemini API Key:** Go to the "API Configuration" section in the sidebar and enter your Gemini API key. You can obtain a key from [Google AI Studio](https://aistudio.google.com/app/apikey).
        2.  **Navigate Sections:** Use the sidebar navigation to switch between the different sections.
        3.  **Interact:** Manage datasets, chat with the AI, or jot down your thoughts in the General Notes section.

        This application is a demonstration and can be extended with more features.
    """)
    st.markdown("---")
    st.markdown("Built with [Streamlit](https://streamlit.io) and [Google Generative AI](https://ai.google.dev/).")

# --- Main Application Logic ---
if app_mode == "📚 Dataset Dashboard":
    dataset_dashboard_page()
elif app_mode == "💬 AI Chatbot":
    ai_chatbot_page()
elif app_mode == "📝 General Notes":
    general_notes_page()
elif app_mode == "ℹ️ About":
    about_page()
