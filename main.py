# main.py
import gradio as gr
import os
from dotenv import load_dotenv
from src.models.chatbot import AviationChatbot

def create_interface():
    # Load environment variables
    load_dotenv()
    api_key = os.getenv('GOOGLE_API_KEY')
    
    # Initialize chatbot
    chatbot = AviationChatbot(api_key)
    
    # Create Gradio interface
    with gr.Blocks() as interface:
        gr.Markdown("# Aviation Analysis Chatbot")
        
        with gr.Row():
            file_input = gr.File(label="Upload Aviation Dataset")
            process_button = gr.Button("Process Data")
            status_text = gr.Textbox(label="Processing Status")  # Added status textbox
        
        with gr.Row():
            with gr.Column():
                chatbot_interface = gr.Chatbot()
                msg = gr.Textbox(label="Ask a question")
                clear = gr.Button("Clear Conversation")
            
            with gr.Column():
                viz_dropdown = gr.Dropdown(
                    choices=[
                        "fuel_by_aircraft",
                        "co2_by_route",
                        "daily_flights",
                        "fuel_phases"
                    ],
                    label="Select Visualization",
                    value="fuel_by_aircraft"
                )
                viz_output = gr.Plot()
        
        # Event handlers
        def process_uploaded_file(file_obj):
            if file_obj is None:
                return "No file uploaded"
            return chatbot.process_file(file_obj)
        
        def respond(message, history):
            bot_message = chatbot.generate_response(message, history)
            history.append((message, bot_message))
            return "", history
        
        def clear_conversation():
            chatbot.reset_chat()
            return None
        
        # Set up event handlers
        process_button.click(
            fn=process_uploaded_file,
            inputs=[file_input],
            outputs=[status_text]
        )
        
        msg.submit(respond, [msg, chatbot_interface], [msg, chatbot_interface])
        clear.click(clear_conversation, None, chatbot_interface)
        
        viz_dropdown.change(
            fn=lambda x: chatbot.create_visualization(x),
            inputs=[viz_dropdown],
            outputs=[viz_output]
        )
    
    return interface

if __name__ == "__main__":
    demo = create_interface()
    demo.launch()
