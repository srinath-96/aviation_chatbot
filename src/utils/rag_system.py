# src/utils/rag_system.py
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
import torch

class ConversationalRAG:
    def __init__(self):
        self.embeddings = HuggingFaceEmbeddings(
            model_name="jinaai/jina-embeddings-v2-base-en",
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': True, 'batch_size': 128}
        )
        self.vector_store = None
        self.conversation_history = []
        self.current_context = None
        self.original_df = None

    def _categorize_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """Automatically categorize columns based on content and names"""
        categories = {
            'id_columns': [],
            'location_columns': [],
            'date_columns': [],
            'measurement_columns': [],
            'type_columns': []
        }
        for col in df.columns:
            col_lower = col.lower()
            if any(word in col_lower for word in ['code', 'number', 'num', 'id']):
                categories['id_columns'].append(col)
            elif any(word in col_lower for word in ['airport', 'location', 'departure', 'arrival']):
                categories['location_columns'].append(col)
            elif any(word in col_lower for word in ['date', 'time']):
                categories['date_columns'].append(col)
            elif any(word in col_lower for word in ['fuel', 'co2', 'tonnes']):
                categories['measurement_columns'].append(col)
            elif any(word in col_lower for word in ['type', 'aircraft']):
                categories['type_columns'].append(col)
        return categories

    def create_vector_store(self, df: pd.DataFrame) -> None:
        try:
            print("Creating vector store...")
            self.original_df = df
            self.column_categories = self._categorize_columns(df)
            # Use DataFrame.apply for vectorized text chunk creation
            texts = df.apply(self._format_flight_info, axis=1).tolist()
            print(f"Creating vector store with {len(texts)} documents...")
            self.vector_store = FAISS.from_texts(texts, self.embeddings)
            print("Vector store created successfully")
        except Exception as e:
            print(f"Error creating vector store: {str(e)}")
            raise e

    def _format_flight_info(self, row: pd.Series) -> str:
        """Dynamically format flight information based on column categories"""
        info_parts = []
        # Add identifiers
        id_info = ' '.join(f"{row[col]}" for col in self.column_categories['id_columns'])
        info_parts.append(f"Flight {id_info}")
        # Add location information
        locations = ' '.join(f"{col.split('_')[-1]}: {row[col]}" 
                             for col in self.column_categories['location_columns'])
        info_parts.append(locations)
        # Add date information
        for col in self.column_categories['date_columns']:
            info_parts.append(f"on {row[col]}")
        # Add type information
        for col in self.column_categories['type_columns']:
            info_parts.append(f"using {row[col]}")
        # Add measurements
        measurements = []
        for col in self.column_categories['measurement_columns']:
            clean_name = col.split('_')[-2].lower() if '_' in col else col
            measurements.append(f"{clean_name}: {row[col]}")
        info_parts.append("Measurements: " + ", ".join(measurements))
        return " | ".join(filter(None, info_parts))

    def get_relevant_context(self, query: str, chat_history: List) -> str:
        try:
            if not self.vector_store:
                return "No data available. Please upload a dataset first."
            docs = self.vector_store.similarity_search(query, k=5)
            context = "\n\n".join(doc.page_content for doc in docs)
            self.current_context = context
            return context
        except Exception as e:
            print(f"Error getting context: {str(e)}")
            return f"Error retrieving context: {str(e)}"
