import os
import google.generativeai as genai
from typing import List, Tuple
import pandas as pd
import io, base64
import plotly.express as px
from src.utils.hybrid_search import CacheAugmentedSystem

class AviationChatbot:
    def __init__(self, api_key: str):
        # Initialize the Cache-Augmented System for the uploaded dataset.
        self.search_system = CacheAugmentedSystem()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-2.0-flash')
        self.chat = self.model.start_chat(history=[])
        # To store the uploaded dataset.
        self.uploaded_df = None

    def process_file(self, file_obj) -> str:
        """
        Process a user-uploaded file: save it locally, load it into a DataFrame,
        build the vector store, and cache summary statistics.
        """
        try:
            # Check if file_obj is file-like (has a read() method)
            if hasattr(file_obj, "read"):
                file_content = file_obj.read()
                if hasattr(file_obj, "seek"):
                    file_obj.seek(0)
                local_filename = os.path.basename(file_obj.name)
            else:
                local_filename = os.path.basename(file_obj)
                with open(file_obj, "rb") as f:
                    file_content = f.read()

            local_path = os.path.join(".", local_filename)
            with open(local_path, "wb") as f:
                f.write(file_content)

            # Attempt to read the file into a DataFrame (first as TSV, then CSV).
            try:
                df = pd.read_csv(local_path, sep='\t', header=0)
            except Exception:
                df = pd.read_csv(local_path, header=0)
            if df.empty:
                return "Uploaded file is empty."
            self.uploaded_df = df
            self.search_system.create_vector_store(df)
            print(f"File '{local_filename}' processed successfully.")
            return f"File '{local_filename}' processed successfully."
        except Exception as e:
            print(f"Error processing file: {str(e)}")
            return f"Error processing file: {str(e)}"
    
    def generate_visualization(self, query: str) -> str:
        """
        Generate an interactive visualization using Plotly based on the uploaded dataset.
        If the query mentions 'fuel', create a histogram of 'fuel_total';
        otherwise, create a bar chart of flights by carrier.
        Returns an HTML snippet.
        """
        if self.uploaded_df is None or self.uploaded_df.empty:
            return "No dataset available for visualization."
        if "fuel" in query.lower():
            fig = px.histogram(self.uploaded_df, x="fuel_total", nbins=20,
                               title="Distribution of Fuel Consumption")
        else:
            counts = self.uploaded_df['carrier_code'].value_counts().reset_index()
            counts.columns = ["carrier", "flight_count"]
            fig = px.bar(counts, x="carrier", y="flight_count",
                         title="Number of Flights by Carrier")
        # Convert the Plotly figure to an HTML snippet and wrap in a <div>.
        html_str = "<div>" + fig.to_html(full_html=False, include_plotlyjs="cdn") + "</div>"
        return html_str

    def generate_response(self, query: str, history: List[Tuple[str, str]]) -> Tuple[str, List[Tuple[str, str]]]:
        """
        Generate a response using the uploaded dataset as context.
        If the query indicates visualization (contains "visualize" or "plot"),
        generate a Plotly visualization and return it as HTML.
        Otherwise, construct a text prompt using summary statistics and
        additional context (via similarity search) and then query Gemini.
        Returns a tuple: (cleared input, updated chat history).
        """
        try:
            if "visualize" in query.lower() or "plot" in query.lower():
                vis_html = self.generate_visualization(query)
                history.append((query, vis_html))
                return "", history
            
            uploaded_summary = self.search_system.get_uploaded_summary()
            context = self.search_system.get_relevant_context(query, k=5)
            prompt = f"""
Dataset Summary:
- Total Flights: {uploaded_summary['total_flights']}
- Unique Carriers: {uploaded_summary['unique_carriers']}
- Total Fuel Consumption: {uploaded_summary['total_fuel']:.2f} tonnes
- Total CO2 Emissions: {uploaded_summary['total_co2']:.2f} tonnes

Additional Relevant Flights:
{context}

Question: {query}
"""
            response = self.model.generate_content([{"text": prompt}])
            response_text = response.text
            history.append((query, response_text))
            return "", history
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            history.append((query, error_msg))
            return "", history

    def reset_chat(self):
        self.chat = self.model.start_chat(history=[])
