import sqlite3
import pandas as pd
from typing import Dict, Any
from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from threading import Lock
import numpy as np
from tqdm import tqdm

def normalize_column_name(col: str) -> str:
    """Normalize column names to a standard format."""
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

class CacheAugmentedSystem:
    def __init__(self):
        # Create an in-memory SQLite database.
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.db_lock = Lock()
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v1')
        self.vector_store = None
        self.cache = {}  # Will store summary statistics for the uploaded file.
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
        """
        Process the uploaded DataFrame in batches to build a FAISS vector index and
        compute summary statistics. The summary is cached under 'uploaded_summary'.
        """
        try:
            # Normalize column names.
            df.columns = [normalize_column_name(col) for col in df.columns]
            required_columns = {'carrier_code', 'fuel_total'}
            missing = required_columns - set(df.columns)
            if missing:
                raise ValueError(f"Missing required columns: {missing}")

            # Load DataFrame into SQLite.
            with self.db_lock:
                df.to_sql('flights', self.conn, if_exists='replace', index=False)
            
            # Build the vector index in batches.
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
            self._preload_cache(df)
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

    def _preload_cache(self, df: pd.DataFrame) -> None:
        """
        Compute summary statistics from the uploaded dataset and store them in the cache.
        """
        try:
            summary = {}
            summary['total_flights'] = len(df)
            summary['unique_carriers'] = df['carrier_code'].nunique()
            summary['total_fuel'] = df['fuel_total'].sum()
            summary['total_co2'] = df['co2_total'].sum()
            summary['most_common_aircraft'] = df['aircraft_type'].mode()[0]
            summary['date_range'] = (df['scheduled_departure'].min(), df['scheduled_departure'].max())
            top_routes = df.groupby(['departure_airport', 'arrival_airport']).size().nlargest(5).to_dict()
            summary['top_routes'] = top_routes
            self.cache['uploaded_summary'] = summary
            print("Cache preloaded successfully.")
        except Exception as e:
            raise RuntimeError(f"Cache preloading failed: {str(e)}")

    def get_uploaded_summary(self) -> Dict[str, Any]:
        """
        Retrieve the summary of the uploaded dataset from the cache.
        """
        if 'uploaded_summary' not in self.cache:
            raise ValueError("No dataset has been uploaded yet.")
        return self.cache['uploaded_summary']

    def get_relevant_context(self, query: str, k: int = 5) -> str:
        """
        Retrieve relevant context using similarity search on the vector store.
        """
        if not self.vector_store:
            return "Dataset not loaded"
        total_flights = pd.read_sql("SELECT COUNT(*) FROM flights", self.conn).iloc[0, 0]
        relevant_docs = self.vector_store.similarity_search(query, k=k)
        context = "\n".join(f"{i+1}. {doc.page_content}" for i, doc in enumerate(relevant_docs))
        return f"Total flights in dataset: {total_flights}\n\nTop {k} relevant flights:\n{context}"
