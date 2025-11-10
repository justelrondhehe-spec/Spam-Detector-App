import re
import streamlit as st
import joblib

# --- Load Your Saved Pipeline ---
try:
    model = joblib.load('spam_model_v2.joblib')
except FileNotFoundError:
    st.error("Model file not found! Make sure 'spam_model_v2.joblib' is in the same folder as this app.")
    st.stop() # Don't run the rest of the app

# --- Helper Function for Keyword Counter ---
# Define a list of common spam keywords
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
        # NEW: We use re.search with \b (word boundary)
        # This stops it from finding "win" inside "expiring" or "following"
        if re.search(r'\b' + re.escape(word) + r'\b', text):
            count += 1
            found_words.append(word)
    return count, found_words

# --- Sidebar ---
st.sidebar.title("About This App")
st.sidebar.info(
    """
    This app uses a **Multinomial Naive Bayes** classifier to detect email spam.
    
    The model was trained on the "Email Spam Detection" dataset from Kaggle.
    """
)
st.sidebar.title("**Created By:**")
st.sidebar.write("Jhon Nicholson Manalang")
st.sidebar.write("Aaron James Jared Papa")
st.sidebar.write("Ryan Kristoffer Suganob")

# --- Main App UI ---
st.set_page_config(page_title="Email Spam Detector", page_icon="📧")
st.title("📧 Email Spam Detector")
st.write("Check if an email is spam by typing it in or uploading a .txt file.")

tab1, tab2 = st.tabs(["Check by Text", "Check by File Upload"])

# --- Tab 1: Check by Text ---
with tab1:
    st.subheader("Compose an email to check")
    
    st.text_input("From:", "your_email@example.com", key="text_from")
    st.text_input("To:", "recipient@example.com", key="text_to")
    subject_input = st.text_input("Subject:", key="text_subject")
    body_input = st.text_area("Email Body:", height=200, key="text_body")

    if st.button("Check Text", key="text_check_button"):
        
        full_email_text = subject_input + " " + body_input

        if subject_input or body_input:
            
            # --- NEW: Get probability scores ---
            probabilities = model.predict_proba([full_email_text])
            prediction = probabilities.argmax() # 0 for ham, 1 for spam
            confidence = probabilities.max() * 100 # Get the highest score
            
            # --- NEW: Keyword counter ---
            keyword_count, found_words = count_spam_keywords(full_email_text)

            st.subheader("Result:")
            if prediction == 1:
                st.error(f"This looks like SPAM! (Confidence: {confidence:.2f}%)", icon="🚨")
            else:
                st.success(f"This looks like HAM (Not Spam). (Confidence: {confidence:.2f}%)", icon="✅")
            
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
                
                # --- NEW: Get probability scores ---
                probabilities = model.predict_proba([file_text])
                prediction = probabilities.argmax() # 0 for ham, 1 for spam
                confidence = probabilities.max() * 100 # Get the highest score
                
                # --- NEW: Keyword counter ---
                keyword_count, found_words = count_spam_keywords(file_text)

                st.subheader("Result:")
                if prediction == 1:
                    st.error(f"This file content looks like SPAM! (Confidence: {confidence:.2f}%)", icon="🚨")
                else:
                    st.success(f"This file content looks like HAM (Not Spam). (Confidence: {confidence:.2f}%)", icon="✅")
                
                if keyword_count > 0:
                    st.warning(f"Found {keyword_count} potential spam keywords: {', '.join(found_words)}")

                with st.expander("See file content"):
                    st.text(file_text)
                    
            except UnicodeDecodeError:
                st.error("Error: Could not read the file. Please make sure it is a plain .txt file encoded in UTF-8.")
        else:
            st.warning("Please upload a .txt file first.")