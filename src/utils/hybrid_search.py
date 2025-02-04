import sqlite3
import pandas as pd
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from threading import Lock
import numpy as np
from tqdm import tqdm

def normalize_column_name(col: str) -> str:
    """Normalize user columns to expected format"""
    col_lower = col.lower().replace(' ', '_')
    column_mapping = {
        "carrier_code": "carrier_code",
        "flight_number": "flight_number",
        "departure_airport": "departure_airport",
        "arrival_airport": "arrival_airport",
        "scheduled_departure_date": "scheduled_departure",
        "aircraft_type": "aircraft_type",
        "estimated_fuelburn(t)": "fuel_total",
        "estimated_co2_(t)": "co2_total"
    }
    return column_mapping.get(col_lower, col_lower)

class HybridSearchSystem:
    def __init__(self):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.db_lock = Lock()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.vector_store = None
        self._create_tables()

    def _create_tables(self):
        with self.db_lock:
            self.conn.execute('''
                CREATE TABLE IF NOT EXISTS flights (
                    id INTEGER PRIMARY KEY,
                    carrier_code TEXT,
                    flight_number TEXT,
                    departure_airport TEXT,
                    arrival_airport TEXT,
                    scheduled_departure DATE,
                    aircraft_type TEXT,
                    fuel_total REAL,
                    co2_total REAL
                )
            ''')
            self.conn.commit()

    def create_vector_store(self, df: pd.DataFrame) -> None:
        try:
            df.columns = [normalize_column_name(col) for col in df.columns]
            
            required_columns = {'carrier_code', 'fuel_total'}
            missing = required_columns - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            with self.db_lock:
                df.to_sql('flights', self.conn, if_exists='replace', index=False)

            batch_size = 1000
            texts = []
            embeddings = []

            for i in tqdm(range(0, len(df), batch_size), desc="Processing batches"):
                batch = df.iloc[i:i+batch_size]
                batch_texts = batch.apply(self._format_row, axis=1).tolist()
                batch_embeddings = self.embedding_model.encode(batch_texts)
                
                texts.extend(batch_texts)
                embeddings.extend(batch_embeddings)

            embeddings_array = np.array(embeddings)

            self.vector_store = FAISS.from_embeddings(
                zip(texts, embeddings_array), 
                self.embedding_model.encode
            )
            print(f"Vector store created successfully with {len(texts)} entries")
        except Exception as e:
            raise RuntimeError(f"Processing failed: {str(e)}")

    def _format_row(self, row: pd.Series) -> str:
        return (
            f"Carrier: {row['carrier_code']} | "
            f"Flight: {row['flight_number']} | "
            f"Route: {row['departure_airport']}-{row['arrival_airport']} | "
            f"Date: {row['scheduled_departure']} | "
            f"Aircraft: {row['aircraft_type']} | "
            f"Fuel: {row['fuel_total']}t | "
            f"CO2: {row['co2_total']}t"
        )

    def get_dataset_summary(self) -> Dict[str, Any]:
        """Provide a comprehensive summary of the entire dataset."""
        with self.db_lock:
            summary = {}
            
            summary['total_flights'] = pd.read_sql("SELECT COUNT(*) FROM flights", self.conn).iloc[0, 0]
            summary['unique_carriers'] = pd.read_sql("SELECT COUNT(DISTINCT carrier_code) FROM flights", self.conn).iloc[0, 0]
            summary['total_fuel'] = pd.read_sql("SELECT SUM(fuel_total) FROM flights", self.conn).iloc[0, 0]
            summary['total_co2'] = pd.read_sql("SELECT SUM(co2_total) FROM flights", self.conn).iloc[0, 0]
            summary['most_common_aircraft'] = pd.read_sql("SELECT aircraft_type, COUNT(*) as count FROM flights GROUP BY aircraft_type ORDER BY count DESC LIMIT 1", self.conn).iloc[0, 0]
            
            date_range = pd.read_sql("SELECT MIN(scheduled_departure) as min_date, MAX(scheduled_departure) as max_date FROM flights", self.conn)
            summary['date_range'] = (date_range['min_date'].iloc[0], date_range['max_date'].iloc[0])
            
            top_routes = pd.read_sql("SELECT departure_airport, arrival_airport, COUNT(*) as count FROM flights GROUP BY departure_airport, arrival_airport ORDER BY count DESC LIMIT 5", self.conn)
            summary['top_routes'] = top_routes.to_dict('records')

            return summary

    def get_carrier_statistics(self, carrier_code: str) -> Dict[str, Any]:
        """Provide detailed statistics for a specific carrier."""
        with self.db_lock:
            stats = {}
            
            stats['total_flights'] = pd.read_sql(f"SELECT COUNT(*) FROM flights WHERE carrier_code = '{carrier_code}'", self.conn).iloc[0, 0]
            stats['total_fuel'] = pd.read_sql(f"SELECT SUM(fuel_total) FROM flights WHERE carrier_code = '{carrier_code}'", self.conn).iloc[0, 0]
            stats['total_co2'] = pd.read_sql(f"SELECT SUM(co2_total) FROM flights WHERE carrier_code = '{carrier_code}'", self.conn).iloc[0, 0]
            stats['most_used_aircraft'] = pd.read_sql(f"SELECT aircraft_type, COUNT(*) as count FROM flights WHERE carrier_code = '{carrier_code}' GROUP BY aircraft_type ORDER BY count DESC LIMIT 1", self.conn).iloc[0, 0]
            
            top_routes = pd.read_sql(f"SELECT departure_airport, arrival_airport, COUNT(*) as count FROM flights WHERE carrier_code = '{carrier_code}' GROUP BY departure_airport, arrival_airport ORDER BY count DESC LIMIT 3", self.conn)
            stats['top_routes'] = top_routes.to_dict('records')

            return stats

    def get_relevant_context(self, query: str, k: int = 5) -> str:
        if not self.vector_store:
            return "Dataset not loaded"
        
        total_flights = pd.read_sql("SELECT COUNT(*) FROM flights", self.conn).iloc[0, 0]
        
        relevant_docs = self.vector_store.similarity_search(query, k=k)
        context = "\n".join(f"{i+1}. {doc.page_content}" for i, doc in enumerate(relevant_docs))
        
        return f"Total flights in dataset: {total_flights}\n\nTop {k} relevant flights:\n{context}"
