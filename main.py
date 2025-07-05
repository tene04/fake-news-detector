import streamlit as st
import torch
from transformers import BertTokenizer, BertForSequenceClassification

# set the model and tokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "./data/model/best_bert_model.pt"
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model.load_state_dict(torch.load(model_path, map_location='cpu'))
model.to(device)
model.eval()
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


# make the interface
st.set_page_config(page_title="News classifier", layout="centered")
st.title("News Authenticity Detector")
st.write("This tool uses BERT to analyze whether news content appears to be true or fake.")

title = st.text_input("News title")
text = st.text_area("Full text of the new", height=200)
subject = st.text_input("News subject")

if st.button("Classify"):
    if not title.strip() or not text.strip() or not subject.strip():
        st.warning("Please fill in all fields.")
    else:
        with st.spinner("Analyzing news content..."):
            input = " ".join([title, text, subject])
            input = tokenizer(input, return_tensors="pt", truncation=True, padding=True, max_length=512)
            input = {k: v.to(device) for k, v in input.items()}

            with torch.no_grad():
                outputs = model(**input)
                probs = torch.nn.functional.softmax(outputs.logits, dim=1).numpy()[0]


            st.subheader("Analysis Results")
                
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Fake Probability", f"{probs[0]:.3%}", delta=f"{(probs[0]-0.5):.3%}" if probs[0] > 0.5 else None, delta_color="inverse")
            with col2:
                st.metric("True Probability", f"{probs[1]:.3%}", delta=f"{(probs[1]-0.5):.3%}" if probs[1] > 0.5 else None)
                
            if probs[1] > probs[0]:
                st.success(f"This news appears to be TRUE (confidence: {probs[1]:.3%})")
            else:
                st.error(f"This news appears to be FAKE (confidence: {probs[0]:.3%})")
                
