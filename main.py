import gradio as gr
import os
from dotenv import load_dotenv
from src.models.chatbot import AviationChatbot

def create_interface():
    # Load environment variables
    load_dotenv()
    chatbot = AviationChatbot(os.getenv('GOOGLE_API_KEY'))
    
    # Use a custom theme for a polished look
    theme = gr.themes.Soft(primary_hue="blue", font="Roboto")
    
    with gr.Blocks(theme=theme) as interface:
        # Header section with a title
        gr.Markdown(
            """
            <h1 style="text-align:center; font-family: Roboto, sans-serif; color:#2c3e50;">
                Aviation Chatbot 🚀
            </h1>
            """,
            elem_id="header"
        )
        
        # Main chat area that occupies most of the screen space.
        # Here we set render_mode="html" so that HTML snippets (e.g., Plotly visualizations) are displayed correctly.
        with gr.Row():
            chatbot_ui = gr.Chatbot(render_mode="html", elem_id="chat_window", height=600)
        
        # Input area at bottom: text input and file upload.
        with gr.Row():
            with gr.Column(scale=10):
                msg = gr.Textbox(
                    placeholder="Type your message here...",
                    elem_id="text_input"
                )
            with gr.Column(scale=2):
                file_input = gr.File(
                    file_types=[".tsv", ".csv"],
                    elem_id="file_upload"
                )
                
        # A small and less obtrusive Clear button.
        clear_btn = gr.Button("Clear", size="sm", elem_id="clear_button")
        
        # Event handler for file processing.
        def process_file_wrapper(file_obj):
            try:
                return chatbot.process_file(file_obj)
            except Exception as e:
                return f"Error processing file: {str(e)}"
                
        # Event handler for sending messages.
        def respond(message: str, history: list):
            try:
                # generate_response now returns a tuple (cleared input, updated history)
                return chatbot.generate_response(message, history)
            except Exception as e:
                error_msg = f"⚠️ Error generating response: {str(e)}"
                return "", history + [(message, error_msg)]
        
        # Event handler for clearing the chat history.
        def clear_history():
            chatbot.reset_chat()
            return []
        
        # Link event handlers to components.
        file_input.change(
            process_file_wrapper, 
            inputs=file_input, 
            outputs=[]
        )
        msg.submit(
            respond, 
            inputs=[msg, chatbot_ui], 
            outputs=[msg, chatbot_ui]
        )
        clear_btn.click(
            clear_history, 
            inputs=None, 
            outputs=chatbot_ui
        )
        
    return interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(share=True)
