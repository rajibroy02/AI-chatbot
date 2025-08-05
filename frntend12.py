import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from pymongo import MongoClient
import time
import openai

# Page configuration
st.set_page_config(page_title="AI Multi-task Chatbot", page_icon="🤖")

# Title and caption
st.title("🤖 AI Multi-task Chatbot")
st.caption("A streamlined interface for various AI tasks powered by TinyLlama 1.1B Chatbot")

# Access the MongoDB URI from secrets.toml
# MONGO_URI = st.secrets["MONGO_URI"]

# Initialize session state for messages
if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "Welcome! I can help you with several tasks. Please select a task from the sidebar and enter your text."}]

if "current_task" not in st.session_state:
    st.session_state["current_task"] = "Chat"

if "model" not in st.session_state:
    st.session_state["model"], st.session_state["tokenizer"] = None, None

# Sidebar for task selection
with st.sidebar:
    st.title("Options")
    task = st.selectbox("Choose a task", ["Chat", "Summarization", "Sentiment Analysis", "Translation", "Text-to-Image"])
    st.session_state["current_task"] = task
    use_gpu = st.checkbox("Use GPU (if available)", value=True)
    use_8bit = st.checkbox("Use 8-bit quantization", value=True)
    # show_history = st.button("Show Past Queries")
    # clear_chat = st.button("Clear Chat History")  # Button to clear chat history

# MongoDB connection
# try:
#     client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
#     db = client["ai_chatbot"]
#     collection = db["user_queries"]
#     client.server_info()  # Test connection
#     db_connected = True
# except Exception:
#     db_connected = False

# Function to load the model only once
def load_model():
    # Use TinyLlama 1.1B Chatbot
    model_name = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model with GPU/8-bit support
    load_options = {
        "low_cpu_mem_usage": True,
        "device_map": "auto",
        "load_in_8bit": use_8bit and torch.cuda.is_available(),
        "torch_dtype": torch.float16 if use_gpu and torch.cuda.is_available() else torch.float32
    }
    
    model = AutoModelForCausalLM.from_pretrained(model_name, **load_options)
    return model, tokenizer

# Load model once when app runs
if st.session_state["model"] is None:
    with st.spinner("Loading TinyLlama 1.1B Chatbot..."):
        st.session_state["model"], st.session_state["tokenizer"] = load_model()

# Function to generate response
def generate_with_tinyllama(prompt, max_length=512, temperature=0.3):
    model = st.session_state["model"]
    tokenizer = st.session_state["tokenizer"]
    
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            inputs.input_ids,
            max_new_tokens=max_length,
            temperature=temperature,  # Lower temperature for less randomness
            top_p=0.9,
            repetition_penalty=1.1,
            do_sample=True,
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id  # Use the EOS token to stop generation
        )
    
    # Decode the output and stop at the first occurrence of "<|user|>" or "<|assistant|>"
    response_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
    
    # Manually stop at "<|user|>" or "<|assistant|>"
    for stop_sequence in ["<|user|>", "<|assistant|>"]:
        if stop_sequence in response_text:
            response_text = response_text.split(stop_sequence)[0].strip()
    
    return response_text.strip()

# Save user query to MongoDB
# def save_to_db(input_text, task_type, result):
#     if not db_connected:
#         return
#     record = {
#         "input": input_text,
#         "task": task_type,
#         "result": result,
#         "timestamp": time.time()
#     }
#     collection.insert_one(record)

# Clear chat history
# if clear_chat:
    # st.session_state["messages"] = [{"role": "assistant", "content": "Welcome! I can help you with several tasks. Please select a task from the sidebar and enter your text."}]

# Display conversation history
for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

# Handle user input
if prompt := st.chat_input("Enter your message here..."):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    # Generate response based on the selected task
    with st.spinner("Thinking..."):
        result = ""
        current_task = st.session_state["current_task"]
        
        if current_task == "Chat":
            # Format the prompt with user and assistant roles
            conversation = "\n".join([f"<|{msg['role']}|> {msg['content']}" for msg in st.session_state.messages[-5:]])
            full_prompt = f"{conversation}\n<|assistant|>"
            result = generate_with_tinyllama(full_prompt, max_length=50)  # Limit response length
        
        elif current_task == "Summarization":
            task_prompt = f"<|user|> Summarize the following text: {prompt}\n<|assistant|>"
            result = generate_with_tinyllama(task_prompt, max_length=100)
        
        elif current_task == "Sentiment Analysis":
            task_prompt = f"<|user|> Analyze the sentiment of the following text: {prompt}\n<|assistant|>"
            result = generate_with_tinyllama(task_prompt, max_length=50)
        
        elif current_task == "Translation":
            task_prompt = f"<|user|> Translate the following English text to French: {prompt}\n<|assistant|>"
            result = generate_with_tinyllama(task_prompt, max_length=100)
        
        elif current_task == "Text-to-Image":
            try:
                response = openai.Image.create(model="dall-e-3", prompt=prompt, n=1, size="1024x1024")
                result = response["data"][0]["url"]
                st.session_state.messages.append({"role": "assistant", "content": "Here's the generated image:"})
                st.chat_message("assistant").write("Here's the generated image:")
                st.image(result)
            except Exception as e:
                result = f"Error generating image: {str(e)}"
        
        # Add assistant's response to chat history
        st.session_state.messages.append({"role": "assistant", "content": result})
        st.chat_message("assistant").write(result)

        # Save to MongoDB
        # save_to_db(prompt, current_task, result)