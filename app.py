import streamlit as st
import joblib
import re  # <-- Make sure this is imported

# --- 1. SET PAGE CONFIG (MUST be the first Streamlit command) ---
st.set_page_config(
    page_title="Email Spam Detector",
    page_icon="📧",
    layout="wide",                  # <-- NEW: Use the full page width
    initial_sidebar_state="expanded"  # <-- NEW: Keep sidebar open
)

# --- 2. CUSTOM CSS ---
st.markdown("""
<style>
/* This targets all buttons */
.stButton > button {
    border: 2px solid #9c4be1;
    border-radius: 12px;
    background-color: transparent;
    color: #9c4be1;
    font-weight: bold;
}

/* This makes them "pop" when you hover */
.stButton > button:hover {
    border-color: #FAFAFA;
    background-color: #9c4be1;
    color: #FAFAFA;
}

/* --- REPLACE your old info box rules with THIS --- */
div[data-testid="stAlert"] {
    background-color: #262730;  /* A dark background (matches sidebar) */
    color: #9c4be1;            /* Makes ALL text inside purple */
}
</style>
""", unsafe_allow_html=True)


# --- Load Your Saved Pipeline ---
# Use the filename of your latest, best model (v9 or v10)
try:
    model = joblib.load('spam_model_v5-5.joblib') 
except FileNotFoundError:
    st.error("Model file not found! Make sure 'spam_model_v5-5.joblib' is in the same folder as this app.")
    st.stop() # Don't run the rest of the app

# --- Helper Function for Keyword Counter (Fixed Version) ---
SPAM_KEYWORDS = [
    'free', 'win', 'winner', 'prize', 'claim', 'urgent', 'congratulations', 
    'click', 'limited', 'offer', 'viagra', 'money', 'cash', '100%', '$$$',
    'act now', 'apply now', 'buy now', 'no cost', 'no fees', 'risk-free'
]

def count_spam_keywords(text):
    text = text.lower()
    count = 0
    found_words = []
    for word in SPAM_KEYWORDS:
        # Use re.search with \b (word boundary) to find whole words
        if re.search(r'\b' + re.escape(word) + r'\b', text):
            count += 1
            found_words.append(word)
    return count, found_words

# --- Sidebar ---
st.sidebar.title("About This App")
st.sidebar.markdown(
    """
    This app uses a **Logistic Regression** classifier to detect email spam.
    
    The model was trained on a combination of 4 datasets from Kaggle to be
    robust against both common spam and legitimate corporate emails.
    
    Model Version: **v5**  
    Date Updated: **November 22, 2025**
    """
)
st.sidebar.title("Created By")
st.sidebar.write("Ryan Kristoffer Suganob")
st.sidebar.write("Jhon Nicholson Manalang")
st.sidebar.write("Aaron James Jared Papa")


# --- Main App UI ---
st.title("📧 Email Spam Detector")
st.write("Check if an email is spam by typing it in or uploading a .txt file.")

tab1, tab2 = st.tabs(["Check by Text", "Check by File Upload"])

# --- Tab 1: Check by Text ---
with tab1:
    st.subheader("Compose an email to check")
    
    # --- 3. USE COLUMNS for a cleaner layout ---
    col1, col2 = st.columns(2)
    with col1:
        st.text_input("From:", "your_email@example.com", key="text_from")
    with col2:
        st.text_input("To:", "recipient@example.com", key="text_to")
    
    # --- Back to full width ---
    subject_input = st.text_input("Subject:", key="text_subject")
    body_input = st.text_area("Email Body:", height=200, key="text_body")

    if st.button("Check Text", key="text_check_button"):
        
        full_email_text = subject_input + " " + body_input

        if subject_input or body_input:
            
            probabilities = model.predict_proba([full_email_text])
            prediction = probabilities.argmax() # 0 for ham, 1 for spam
            confidence = probabilities.max() * 100 
            keyword_count, found_words = count_spam_keywords(full_email_text)

            st.subheader("Result:")
            if prediction == 1:
                st.error(f"This looks like SPAM! (Confidence: {confidence:.2f}%)", icon="🚨")
            else:
                st.success(f"This looks like HAM (Not Spam). (Confidence: {confidence:.2f}%)", icon="✅")
                
                # --- 4. ADD BALLOONS on success! ---
                st.balloons()
            
            if keyword_count > 0:
                st.warning(f"Found {keyword_count} potential spam keywords: {', '.join(found_words)}")
        else:
            st.warning("Please enter a subject or body text to check.")

# --- Tab 2: Check by File Upload ---
with tab2:
    st.subheader("Upload an email file to check")
    
    uploaded_file = st.file_uploader("Choose a .txt file", type=["txt"])

    if st.button("Check File", key="file_check_button"):
        if uploaded_file is not None:
            try:
                bytes_data = uploaded_file.getvalue()
                file_text = bytes_data.decode("utf-8")
                
                probabilities = model.predict_proba([file_text])
                prediction = probabilities.argmax() 
                confidence = probabilities.max() * 100 
                keyword_count, found_words = count_spam_keywords(file_text)

                st.subheader("Result:")
                if prediction == 1:
                    st.error(f"This file content looks like SPAM! (Confidence: {confidence:.2f}%)", icon="🚨")
                else:
                    st.success(f"This file content looks like HAM (Not Spam). (Confidence: {confidence:.2f}%)", icon="✅")
                    
                    # --- 4. ADD BALLOONS on success! ---
                    st.balloons()
                
                if keyword_count > 0:
                    st.warning(f"Found {keyword_count} potential spam keywords: {', '.join(found_words)}")

                with st.expander("See file content"):
                    st.text(file_text)
                    
            except UnicodeDecodeError:
                st.error("Error: Could not read the file. Please make sure it is a plain .txt file encoded in UTF-8.")
        else:
            st.warning("Please upload a .txt file first.")