import streamlit as st
import os
import pickle
import base64
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore, EncoderBackedStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Site Engineer AI", 
    page_icon="🏗️", 
    layout="wide"
)
load_dotenv()

# ==========================================
# 2. CACHED RESOURCE LOADING
# ==========================================
@st.cache_resource
def get_rag_chain():
    """
    Loads the saved Database and initializes the RAG chain.
    Cached so it doesn't reload on every user interaction.
    """
    print("🔄 Loading Database...")
    
    embedding_function = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-base",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

    vectorstore = Chroma(
        collection_name="multi_modal_rag", 
        embedding_function=embedding_function,
        persist_directory="./chroma_db"
    )

    store = EncoderBackedStore(
        store=LocalFileStore("./docstore"),
        key_encoder=lambda x: x,
        value_serializer=pickle.dumps,
        value_deserializer=pickle.loads 
    )

    retriever = MultiVectorRetriever(
        vectorstore=vectorstore, 
        docstore=store, 
        id_key="doc_id", 
        search_kwargs={"k": 6} 
    )


    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro", 
        temperature=0
    )

    return retriever, llm

# Load resources once
retriever, llm = get_rag_chain()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================
def crop_and_encode_image(image_path, bbox, orig_size=(1000, 1000)):
    """Crops the image based on bbox and returns base64 string."""
    # Fix path separators for Windows/Linux compatibility
    image_path = image_path.replace("\\", "/")
    
    if not os.path.exists(image_path):
        return None
        
    try:
        img = Image.open(image_path)
        new_w, new_h = img.size
        
        # Calculate scaling
        scale_x = new_w / orig_size[0]
        scale_y = new_h / orig_size[1]
        
        # Apply scaling to bbox
        scaled_coords = [
            int(bbox[0] * scale_x), int(bbox[1] * scale_y),
            int(bbox[2] * scale_x), int(bbox[3] * scale_y),
        ]
        
        # Crop
        crop = img.crop(tuple(scaled_coords))
        buffered = BytesIO()
        crop.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")
        
    except Exception as e:
        print(f"Error cropping image: {e}")
        return None

def parse_docs(retrieved_docs):
    """Separates text chunks and crops images from the retrieved documents."""
    texts = []
    images_b64 = []
    
    for doc in retrieved_docs:

        if isinstance(doc, dict):
            content = doc.get("cleaned_text", "")
            metadata = doc
        else:
            content = doc.page_content
            metadata = doc.metadata
            
        if content:
            texts.append(content)
            
        if "bboxes" in metadata and "pages_path" in metadata:
            bboxes = metadata["bboxes"]
            img_path = metadata["pages_path"]
            
            if isinstance(bboxes, str):
                import ast
                try:
                    bboxes = ast.literal_eval(bboxes)
                except:
                    bboxes = []

            for bbox in bboxes:
                img_str = crop_and_encode_image(img_path, bbox)
                if img_str:
                    images_b64.append(img_str)
                    
    return {"texts": texts, "images": images_b64}

def build_prompt(kwargs):
    """Constructs the prompt with text and images."""
    docs_by_type = kwargs["context"]
    user_question = kwargs["question"]

    context_text = "\n\n".join(docs_by_type["texts"])

    system_instruction = """
    You are a technical RAG assistant responsible for analyzing construction documents (text, tables, and site photos).

    ### CORE OBJECTIVE:
    Your goal is to answer the user's question by extracting information from the provided text chunks and images. 
    You must convert any visual data (like numbers in a table or dimensions in a diagram) into clear, natural text.

    ### STRICT RULES:

    1.  **Silent Visual Extraction:**
        * You must read and analyze tables and diagrams within the images to find answers.
        * **DO NOT** cite the image. Do not write `[Image N]`.
        * **DO NOT** mention "According to the image..." or "As seen in the figure...".
        * Just state the fact directly. 
        * *Example:* If an image shows a table with "Depth: 5m", your output should simply be: "The excavation depth is 5.0 meters."

    2.  **Direct Output Only:**
        * Provide the answer directly and professionally.
        * Do not use conversational fillers.

    3.  **Handling Missing Info:**
        * If the answer is not in the text or images, return EXACTLY this phrase:
        ".لا تتوفر لدي معلومات كافية للإجابة على هذا السؤال ضمن السياق المتاح"

    ### INPUT PROCESSING:
    * **Step 1:** Read text and look at all images.
    * **Step 2:** Find the specific value or fact requested.
    * **Step 3:** Return **only** the extracted text/data as the final answer.
    """

    prompt_content = []
    
    prompt_content.append({
        "type": "text", 
        "text": f"### Context Content:\n{context_text}\n\n### User Question:\n{user_question}"
    })

    for i, img_b64 in enumerate(docs_by_type["images"][:5]):
        prompt_content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })

    return ChatPromptTemplate.from_messages([
        SystemMessage(content=system_instruction),
        HumanMessage(content=prompt_content)
    ])

# ==========================================
# 4. STREAMLIT UI LOGIC
# ==========================================
st.title("👷الدليل الذكي لأعمال رخص تمديدات البنية التحتية")

st.markdown(
    """
    <style>
        body {
            direction: rtl;
            text-align: right;
        }
    </style>
    """,
    unsafe_allow_html=True
)

# A. Initialize Chat History
if "messages" not in st.session_state:
    st.session_state.messages = []

# B. Display Previous Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# C. Chat Input
if prompt := st.chat_input("مثال: ما هي انوع الحفريات ؟"):
    
    # 1. Show User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Generate Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        
        with st.spinner("Analyzing documents & drawings..."):
            try:
                # Define the Chain
                chain = (
                    {
                        "context": retriever | RunnableLambda(parse_docs),
                        "question": RunnablePassthrough(),
                    }
                    | RunnableLambda(build_prompt)
                    | llm 
                    | StrOutputParser()
                )
                
                # Run Chain
                full_response = chain.invoke(prompt)
                
                # Display Result
                message_placeholder.markdown(full_response)
                
            except Exception as e:
                full_response = f"⚠️ **Error:** {str(e)}"
                message_placeholder.error(full_response)
        
        # 3. Save Assistant Message
        st.session_state.messages.append({"role": "assistant", "content": full_response})