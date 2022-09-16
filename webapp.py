import streamlit as st
from model import MODEL_FILE_PATH
from image_path import IMAGE_PATH
from transformers import GPT2Tokenizer
import torch

import base64
def add_bg_from_local(image_file):
    with open(image_file, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read())
    st.markdown(
    f"""
    <style>
    .stApp {{
        background-image: url(data:image/{"png"};base64,{encoded_string.decode()});
        background-size: cover
    }}
    </style>
    """,
    unsafe_allow_html=True
    )
add_bg_from_local(IMAGE_PATH)

st.title("HAIKU GENERATION")

label = 'Enter prompt here:'
prompt_input = st.text_input(label, value="")

MAX_LEN = 25
#tokenizer
tokenizer = GPT2Tokenizer.from_pretrained('fabianmmueller/deep-haiku-gpt-2')
#model
poem_stanza_model_1 = torch.load(MODEL_FILE_PATH, map_location='cpu')
#eval
poem_stanza_model_1.eval()

generated = torch.tensor(tokenizer.encode("(" + prompt_input + "=")).unsqueeze(0)
sample_outputs = poem_stanza_model_1.generate(
                                generated,
                                do_sample=True,
                                top_k=50,
                                max_length=MAX_LEN,
                                top_p=0.95,
                                num_return_sequences=1
                                )


#output the hiku text
def preprocess_output(text):
    text = text.split(")")[0]
    text = text.split("=")[1]
    return text

output_text = tokenizer.decode(sample_outputs[0], skip_special_tokens=True)
st.title(prompt_input)
text = preprocess_output(output_text)

col1, col2, col3 = st.columns(3)
with col1:
    st.empty()
with col2:
    for part in text.split("/"):
        st.write(part)
with col3:
    st.empty()
