import streamlit as st
import os
import pickle
import base64
from base64 import b64encode
import re
import ast
from io import BytesIO
from PIL import Image
from dotenv import load_dotenv

# LangChain Imports
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.retrievers.multi_vector import MultiVectorRetriever
from langchain.storage import LocalFileStore, EncoderBackedStore
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage

# ==========================================
# 1. APP CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="المساعد الذكي للمهندس", 
    page_icon="🏗️", 
    layout="wide"
)
load_dotenv()

# --- CSS FOR RTL (ARABIC) ---
st.markdown("""
<style>
    /* Force Right-to-Left for the entire app */
    .stApp {
        direction: rtl;
        text-align: right;
    }
    
    /* Adjust chat input to align right */
    .stChatInput textarea {
        direction: rtl;
        text-align: right;
    }
    
    /* Ensure markdown lists (bullets) align right */
    .stMarkdown ul, .stMarkdown ol {
        direction: rtl;
        text-align: right;
        padding-right: 20px; 
    }
    
    /* Fix Image alignment */
    img {
        display: block;
        margin-right: 0;
        margin-left: auto;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 2. CACHED RESOURCE LOADING
# ==========================================
@st.cache_resource
def get_rag_chain():
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
        search_kwargs={"k": 8} 
    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.0-flash", 
        temperature=0.0
    )

    return retriever, llm

retriever, llm = get_rag_chain()

# ==========================================
# 3. HELPER FUNCTIONS
# ==========================================

def crop_and_encode_image(image_path, bbox, orig_size=(1000, 1000)):
    """
    Crops the image based on scaled bbox (1000x1000) and returns base64 string.
    """
    # Fix path separators
    image_path = str(image_path).replace("\\", "/")
    
    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        return None

    try:
        img = Image.open(image_path)
        new_w, new_h = img.size
        
        # Calculate scaling factor based on the original size assumption (marker default is 1000x1000)
        scale_x = new_w / orig_size[0]
        scale_y = new_h / orig_size[1]

        scaled_coords = [
            int(bbox[0] * scale_x),
            int(bbox[1] * scale_y),
            int(bbox[2] * scale_x),
            int(bbox[3] * scale_y),
        ]

        crop = img.crop(tuple(scaled_coords))
        buffered = BytesIO()
        crop.save(buffered, format="JPEG")
        return b64encode(buffered.getvalue()).decode("utf-8")
        
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return None

def prepare_context_and_images(retrieved_docs):
    """
    Organizes retrieved docs into a format Gemini can understand.
    """
    message_content = []
    image_map = {}
    img_counter = 0

    # Add header
    message_content.append({
        "type": "text", 
        "text": "### RETRIEVED CONTEXT (Text & Images):"
    })

    for doc in retrieved_docs:
        # Determine Metadata source
        meta = doc if isinstance(doc, dict) else doc.metadata
        content = doc.get("cleaned_text", "") if isinstance(doc, dict) else doc.page_content
        
        page_num = meta.get("page", "Unknown")
        
        # 1. Process Text Content
        if content:
            message_content.append({
                "type": "text",
                "text": f"\n[TEXT BLOCK] (Page {page_num}):\n{content}\n"
            })

        # 2. Process Images/Tables (Visuals)
        raw_bboxes = meta.get("bboxes", [])
        if isinstance(raw_bboxes, str):
            try:
                raw_bboxes = ast.literal_eval(raw_bboxes)
            except:
                raw_bboxes = []

        img_path = meta.get("pages_path", "")

        # Iterate through visual elements in this doc
        for item in raw_bboxes:
            # Item structure: [bbox, caption]
            if len(item) == 2:
                bbox, caption = item
                
                # USE NEW FUNCTION HERE
                b64_str = crop_and_encode_image(img_path, bbox)
                
                if b64_str:
                    img_id = f"img_{img_counter}"
                    image_map[img_id] = b64_str
                    
                    # Add to prompt interleaved
                    message_content.append({
                        "type": "text",
                        "text": f"\n[IMAGE ID: {img_id}] (Page {page_num})\nCaption: {caption}\nImage Content:"
                    })
                    message_content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_str}"}
                    })
                    
                    img_counter += 1

    return message_content, image_map

def render_final_response(text, image_map):
    """
    Replaces {{img_ID}} placeholders with HTML <img> tags.
    """
    def replace_match(match):
        img_id = match.group(1)
        if img_id in image_map:
            b64 = image_map[img_id]
            # HTML for RTL display
            return (
                f'<div style="width:100%; margin: 10px 0;">'
                f'<img src="data:image/jpeg;base64,{b64}" '
                f'style="border: 1px solid #ddd; border-radius: 8px; max-width: 100%; height: auto; display: block;">'
                f'</div>'
            )
        return ""

    processed_text = re.sub(r"\{\{(img_\d+)\}\}", replace_match, text)
    return processed_text

# ==========================================
# 4. STREAMLIT UI LOGIC
# ==========================================
st.title("👷 الدليل الذكي لأعمال البنية التحتية")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Display history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"], unsafe_allow_html=True)

# Chat Input
if prompt := st.chat_input("اطرح سؤالك هنا..."):
    
    # Show User Message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Show Assistant Response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        
        with st.spinner("🔍 جاري البحث في المخططات والوثائق..."):
            try:
                # 1. Retrieve Documents
                docs = retriever.invoke(prompt)
                
                # 2. Parse into Multimodal Message parts
                context_blocks, image_map = prepare_context_and_images(docs)
                
                # 3. Define System Prompt (Strict Arabic Instructions)
                system_prompt = """
                أنت مساعد هندسي متخصص في أعمال البنية التحتية.
                
                ### التعليمات:
                1. أجب على سؤال المستخدم بناءً فقط على السياق المرفق (نصوص وصور).
                2. **استخدام الصور:** - لقد زودتك بصور لها معرفات مثل `img_0`, `img_1`.
                   - إذا كانت الصورة توضح الإجابة أو تحتوي على جدول بيانات مطلوب، يجب عليك عرضها.
                   - لعرض الصورة، ضع معرفها بين قوسين مزدوجين في المكان المناسب في الجملة. مثال: "تفاصيل الحفر موضحة في الجدول أدناه: {{img_0}}".
                
                3. **اللغة والأسلوب:**
                   - أجب باللغة العربية فقط.
                   - لا تستخدم مقدمات ترحيبية أو حشو (مثل "بناءً على المستندات..."). ادخل في الإجابة مباشرة.
                   - إذا لم تكن الإجابة موجودة في السياق، قل: "لا تتوفر معلومات كافية في المستندات للإجابة على هذا السؤال".

                4. **المراجع:**
                   - في نهاية الإجابة، ضع سطرًا للمصادر بالصيغة التالية تماماً:
                   `المراجع: صفحة X، صفحة Y`
                """

                # 4. Build Final Message Payload
                messages = [
                    SystemMessage(content=system_prompt),
                    HumanMessage(content=context_blocks + [{"type": "text", "text": f"\n### سؤال المستخدم:\n{prompt}"}])
                ]

                # 5. Generate Response
                ai_msg = llm.invoke(messages)
                raw_response = ai_msg.content

                # 6. Post-Process (Insert Images into HTML)
                final_html = render_final_response(raw_response, image_map)

                # 7. Display
                message_placeholder.markdown(final_html, unsafe_allow_html=True)
                
                # Save to history
                st.session_state.messages.append({"role": "assistant", "content": final_html})

            except Exception as e:
                st.error(f"حدث خطأ: {e}")