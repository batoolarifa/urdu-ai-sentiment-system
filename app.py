import gradio as gr
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

# ====================== MODEL ======================
model_path = "./best_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

def predict_sentiment(text):
    if not text or text.strip() == "":
        return "ENTER TEXT", ""

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.softmax(outputs.logits, dim=1)

    pred_id = torch.argmax(probs, dim=1).item()
    confidence = torch.max(probs).item()

    label = model.config.id2label[pred_id]
    label = label.upper()
    confidence_str = f"Confidence: {confidence * 100:.2f}%"

    return label, confidence_str


custom_css = """
/* Fullscreen Dark Theme */
html, body, .gradio-container {
    background: radial-gradient(circle at top, #0B1020, #05060A) !important;
    background-attachment: fixed !important;
    min-height: 100vh !important;
    margin: 0 !important;
}
/* Main Heading */
h1 {
    font-size: 55px !important;
    font-weight: 800 !important;
    background: linear-gradient(90deg, #7C3AED, #22D3EE);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
}
textarea {
    background: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(124, 58, 237, 0.25) !important;
    border-radius: 20px !important;
    color: #ffffff !important;
    font-size: 20px !important;
    padding: 18px !important;
    width: 100% !important;
    transition: all 0.3s ease;
}
/* FOCUS EFFECT */
textarea:focus {
    outline: none !important;
    border: 1px solid rgba(34, 211, 238, 0.9) !important;
    box-shadow: 0 0 25px rgba(124, 58, 237, 0.35) !important;
}
/* PLACEHOLDER STYLE */
textarea::placeholder {
    color: rgba(255, 255, 255, 0.4) !important;
}
/* THE GRADIENT RESULT BOX */
#sentiment_display {
    background: linear-gradient(135deg, #7C3AED, #22D3EE) !important;
    border-radius: 24px !important;
    padding: 40px !important;
    text-align: center !important;
    border: none !important;
    box-shadow: 0 20px 50px rgba(124, 58, 237, 0.3);
}
/* Result Text - Bold White */
#sentiment_display textarea {
    background: transparent !important;
    border: none !important;
    color: white !important;
    font-size: 45px !important;
    font-weight: 900 !important;
    text-align: center !important;
    pointer-events: none;
}
/* Confidence Text */
#confidence_display textarea {
    background: transparent !important;
    border: none !important;
    color: rgba(255, 255, 255, 0.8) !important;
    font-size: 20px !important;
    text-align: center !important;
    margin-top: -20px !important;
    pointer-events: none;
}
/* Button */
button.primary {
    background: linear-gradient(90deg, #7C3AED, #22D3EE) !important;
    border-radius: 16px !important;
    font-weight: 800 !important;
    height: 70px !important;
    font-size: 20px !important;
    border: none !important;
}
"""

with gr.Blocks(css=custom_css) as interface:
    gr.Markdown("# 🇵🇰 Urdu Sentiment Analyzer")
    
    with gr.Row():
        with gr.Column(scale=7):
            text_input = gr.Textbox(
                label=None,
                placeholder="اپنا اردو جملہ یہاں لکھیں...",
                lines=10,
                max_lines=25
            )
            analyze_btn = gr.Button("ANALYZE NOW", variant="primary")
            
            gr.Examples(
                examples=[
                    ["یہ بہت ہی بہترین اور معیاری پروڈکٹ ہے"],
                    ["مجھے آپ کی سروس بالکل بھی پسند نہیں آئی"],
                    ["استاد کا پڑھانے کا انداز بہت اچھا ہے"],
                    ["انتہائی ناقص اور بیکار سروس"],
                    ["وہ بازار گیا اور سامان خریدا"],
                    ["ہم نے میٹنگ میں مختلف موضوعات پر بات کی۔"]
                ],
                inputs=text_input
            )

        with gr.Column(scale=5):
            # Combined result area that feels like one big gradient card
            with gr.Group(elem_id="sentiment_display"):
                sentiment_output = gr.Textbox(
                    show_label=False,
                    interactive=False,
                    elem_id="sentiment_text"
                )
                confidence_output = gr.Textbox(
                    show_label=False,
                    interactive=False,
                    elem_id="confidence_display"
                )

    analyze_btn.click(
        fn=predict_sentiment,
        inputs=text_input,
        outputs=[sentiment_output, confidence_output]
    )

interface.launch()