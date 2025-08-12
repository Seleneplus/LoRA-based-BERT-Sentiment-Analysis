import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import gradio as gr

# Load model
model_path = './lora_bert_imdb'
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

label_map = {0: 'negative', 1: 'positive'}

# Prediction function
def predict(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.softmax(outputs.logits, dim=-1).squeeze().tolist()
    return {label_map[i]: float(probs[i]) for i in range(2)}

# Gradio interface
interface = gr.Interface(
    fn=predict,
    inputs=gr.Textbox(lines=3, placeholder='Enter movie review...'),
    outputs=gr.Label(num_top_classes=2),
    examples=[["This movie was fantastic!"], ["Terrible acting and plot."]],
    title='IMDb Sentiment Analysis (BERT + LoRA)'
)

if __name__ == '__main__':
    interface.launch(server_name='0.0.0.0', server_port=7860)
