import google.generativeai as genai
from typing import List, Tuple, Dict, Any
import pandas as pd
from src.utils.hybrid_search import HybridSearchSystem

class AviationChatbot:
    def __init__(self, api_key: str):
        self.search_system = HybridSearchSystem()
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
        self.chat = self.model.start_chat(history=[])

    def process_file(self, file_obj) -> str:
        try:
            df = pd.read_csv(file_obj.name, sep='\t')
            self.search_system.create_vector_store(df)
            return "Data processed successfully"
        except Exception as e:
            return f"Error: {str(e)}"

    def generate_response(self, query: str, _) -> str:
        try:
            if "summary" in query.lower() or "overview" in query.lower():
                summary = self.search_system.get_dataset_summary()
                return self._format_summary_response(summary)
            
            if "carrier" in query.lower() and any(code in query.upper() for code in ["XH", "YH"]):
                carrier_code = "XH" if "XH" in query.upper() else "YH"
                stats = self.search_system.get_carrier_statistics(carrier_code)
                return self._format_carrier_response(carrier_code, stats)
            
            context = self.search_system.get_relevant_context(query, k=5)
            response = self.chat.send_message(f"Context: {context}\n\nQuestion: {query}")
            return response.text
        
        except Exception as e:
            return f"Error: {str(e)}"

    def _format_summary_response(self, summary: Dict[str, Any]) -> str:
        return f"""
Dataset Summary:
- Total Flights: {summary['total_flights']}
- Unique Carriers: {summary['unique_carriers']}
- Total Fuel Consumption: {summary['total_fuel']:.2f} tonnes
- Total CO2 Emissions: {summary['total_co2']:.2f} tonnes
- Most Common Aircraft: {summary['most_common_aircraft']}
- Date Range: {summary['date_range'][0]} to {summary['date_range'][1]}
- Top 5 Busiest Routes:
  {self._format_routes(summary['top_routes'])}
"""

    def _format_carrier_response(self, carrier_code: str, stats: Dict[str, Any]) -> str:
        return f"""
Carrier {carrier_code} Statistics:
- Total Flights: {stats['total_flights']}
- Total Fuel Consumption: {stats['total_fuel']:.2f} tonnes
- Total CO2 Emissions: {stats['total_co2']:.2f} tonnes
- Most Used Aircraft: {stats['most_used_aircraft']}
- Top 3 Routes:
  {self._format_routes(stats['top_routes'])}
"""

    def _format_routes(self, routes: List[Dict[str, Any]]) -> str:
        return "\n  ".join([f"{route['departure_airport']} to {route['arrival_airport']}: {route['count']} flights" for route in routes])

    def reset_chat(self):
        self.chat = self.model.start_chat(history=[])
