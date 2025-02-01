import google.generativeai as genai
from typing import List, Tuple
from src.processors.data_processor import AutoDataProcessor
from src.utils.rag_system import ConversationalRAG
from src.processors.visualization import AviationVisualizer
class AviationChatbot:
    def __init__(self, api_key: str):
        self.data_processor = AutoDataProcessor()
        self.rag = ConversationalRAG()
        self.visualizer = None  # This is correct - initialize as None
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.chat = None
        self.reset_chat()
    
    def reset_chat(self):
        """Reset chat session"""
        self.chat = self.model.start_chat(history=[])
        self.rag.conversation_history = []
    
    def process_file(self, file_obj) -> str:
        """Process uploaded file"""
        try:
            if file_obj is None:
                return "No file provided"
            
            # Load and process data
            result = self.data_processor.load_data(file_obj.name)
            
            if "successfully" in result:
                # Get processed dataframe
                processed_df = self.data_processor.get_dataframe()
                
                # Initialize RAG system
                self.rag.create_vector_store(processed_df)
                
                # Initialize visualizer
                self.visualizer = AviationVisualizer(processed_df)
                
                return "File processed successfully"
            return result
            
        except Exception as e:
            print(f"Detailed error in process_file: {str(e)}")
            return f"Error in process_file: {str(e)}"
    def create_visualization(self, viz_type: str):
        """Create visualization based on type"""
        if self.visualizer is None:
            return None
        return self.visualizer.create_visualization(viz_type)
    def generate_response(self, query: str, chat_history: List[Tuple[str, str]]) -> str:
        """Generate contextual response"""
        try:
            context = self.rag.get_relevant_context(query, chat_history)
            prompt = self._create_prompt(query, context, chat_history)
            response = self.chat.send_message(prompt)
            
            self.rag.conversation_history.append({
                "query": query,
                "response": response.text,
                "context": context
            })
            
            return response.text
            
        except Exception as e:
            return f"Error generating response: {str(e)}"
    
    def _create_prompt(self, query: str, context: str, chat_history: List) -> str:
        """Create contextual prompt"""
        return f"""Context: {context}
                  Chat History: {chat_history[-3:] if chat_history else 'No previous conversation'}
                  Current Query: {query}
                  Please provide a detailed response considering the conversation history and context."""
