import os
import concurrent.futures
import pymupdf as fitz  # PyMuPDF
import google.generativeai as genai  # google-generativeai
import pickle
import requests
import streamlit as st

# Configure Gemini API
genai.configure(api_key='AIzaSyAjvIVCHQFrha7E1hRkvFk7e4X4si5wXPM')

# Function to fetch list of PDF files from a GitHub repository
def fetch_pdf_files_from_github(repo_url):
    """
    Fetches the list of PDF files from a GitHub repository.
    """
    api_url = repo_url.replace("https://github.com/", "https://api.github.com/repos/") + "/contents/"
    response = requests.get(api_url)
    if response.status_code == 200:
        files = response.json()
        pdf_files = [file["download_url"] for file in files if file["name"].endswith(".pdf")]
        return pdf_files
    else:
        st.error(f"Failed to fetch files from GitHub repository: {response.status_code}")
        return []

# Function to download PDFs from GitHub
def download_pdf_from_github(url, save_path):
    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        return True
    else:
        st.error(f"Failed to download PDF from {url}")
        return False

# Function to read PDF content
def read_pdf(file_path):
    try:
        doc = fitz.open(file_path)
        document_content = ""
        for page_num in range(doc.page_count):
            page = doc.load_page(page_num)
            document_content += page.get_text("text")
        return document_content
    except Exception as e:
        st.error(f"Error processing {file_path}: {e}")
        return None

# Function to save content to pickle
def save_content_to_pickle(content, file_name):
    with open(file_name, 'wb') as file:
        pickle.dump(content, file)

# Function to load content from pickle
def load_content_from_pickle(file_name):
    if os.path.exists(file_name):
        with open(file_name, 'rb') as file:
            return pickle.load(file)
    return None

# Gemini model configuration
generation_config = {
    "temperature": 1,
    "top_p": 0.95,
    "top_k": 40,
    "max_output_tokens": 8192,
    "response_mime_type": "text/plain",
}

model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config=generation_config,
)

chat_session = model.start_chat(history=[])

# Streamlit app
st.title("PDF Chatbot")
st.write("This chatbot answers questions based on the content of PDF documents from a GitHub repository.")

# GitHub repository URL
github_repo_url = "https://github.com/Luthra-Digital/rbi_docs"

# Fetch list of PDF files from the GitHub repository
pdf_urls = fetch_pdf_files_from_github(github_repo_url)

if not pdf_urls:
    st.error("No PDF files found in the repository.")
    st.stop()

# Download and process PDFs
pdf_folder_path = "downloaded_pdfs"
output_folder_path = "extracted_data"

if not os.path.exists(pdf_folder_path):
    os.makedirs(pdf_folder_path)

if not os.path.exists(output_folder_path):
    os.makedirs(output_folder_path)

combined_document_content = ""

def process_pdf(pdf_url):
    file_name = pdf_url.split("/")[-1]
    file_path = os.path.join(pdf_folder_path, file_name)
    output_file_name = os.path.join(output_folder_path, file_name.replace('.pdf', '.pkl'))

    # Download PDF if not already downloaded
    if not os.path.exists(file_path):
        download_pdf_from_github(pdf_url, file_path)

    document_content = load_content_from_pickle(output_file_name)

    if not document_content:
        document_content = read_pdf(file_path)
        if document_content:
            save_content_to_pickle(document_content, output_file_name)
    return f"\n\n--- End of {file_name} ---\n\n{document_content}"

# Process all PDFs
with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(executor.map(process_pdf, pdf_urls))

combined_document_content = ''.join(results)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat interface
if combined_document_content:
    st.write("All documents processed. You can start asking questions.")

    # Display chat history
    st.write("### Chat History")
    for chat in st.session_state.chat_history:
        st.write(f"**You:** {chat['user']}")
        st.write(f"**AI:** {chat['ai']}")
        st.write("---")

    # User input
    user_question = st.text_input("You: ", "")

    if user_question:
        # Prepare the prompt
        user_question = f"Please answer based only on the content of the document provided above. " \
                        f"Do not include any external or general knowledge. Question: {user_question}"

        prompt = f"Document content: {combined_document_content}\n\n{user_question}"

        # Send the combined document content and the refined question to the model
        response = chat_session.send_message(prompt)

        # Add user question and AI response to chat history
        st.session_state.chat_history.append({"user": user_question, "ai": response.text})

        # Clear the input box after processing
        st.experimental_rerun()
else:
    st.write("No document content was read successfully.")
