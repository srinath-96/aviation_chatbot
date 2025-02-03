# src/utils/rag_system.py
from langchain_community.vectorstores import FAISS
from fastembed.embedding import DefaultEmbedding
import pandas as pd
from typing import List, Dict
from tqdm import tqdm
import numpy as np

class ConversationalRAG:
    def __init__(self):
        """
        Initialize RAG system with FastEmbed for optimized embeddings
        Components:
        - embedding_model: FastEmbed's lightweight embedding model
        - vector_store: FAISS vector database for similarity search
        - conversation_history: Stores chat context
        """
        self.embedding_model = DefaultEmbedding()
        self.vector_store = None
        self.conversation_history = []
        self.column_categories = {}
        self.original_df = None

    def _categorize_columns(self, df: pd.DataFrame) -> Dict[str, List[str]]:
        """
        Automatically group columns by semantic type
        Returns dictionary with:
        - id_columns: Unique identifiers (codes, numbers)
        - location_columns: Airports and routes
        - date_columns: Temporal information
        - measurement_columns: Fuel/CO2 metrics
        - type_columns: Aircraft details
        """
        categories = {
            'id_columns': [], 
            'location_columns': [],
            'date_columns': [],
            'measurement_columns': [],
            'type_columns': []
        }
        
        for col in df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ['code', 'num', 'id']):
                categories['id_columns'].append(col)
            elif any(kw in col_lower for kw in ['airport', 'departure', 'arrival']):
                categories['location_columns'].append(col)
            elif 'date' in col_lower or 'time' in col_lower:
                categories['date_columns'].append(col)
            elif any(kw in col_lower for kw in ['fuel', 'co2', 'tonnes']):
                categories['measurement_columns'].append(col)
            elif 'type' in col_lower or 'aircraft' in col_lower:
                categories['type_columns'].append(col)
                
        return categories

    def create_vector_store(self, df: pd.DataFrame) -> None:
        """
        Create FAISS vector store using FastEmbed embeddings
        Process:
        1. Categorize columns
        2. Generate text chunks
        3. Create embeddings in batches
        4. Build FAISS index
        """
        try:
            print("⚡ Starting vector store creation with FastEmbed")
            self.original_df = df
            self.column_categories = self._categorize_columns(df)
            
            # Generate text chunks
            texts = df.apply(self._format_flight_info, axis=1).tolist()
            
            # Create embeddings in optimized batches
            embeddings = list(self.embedding_model.embed(
                texts, 
                batch_size=512,  # Process 512 documents at once
                show_progress_bar=True
            ))
            
            # Create FAISS index from embeddings
            self.vector_store = FAISS.from_embeddings(
                text_embeddings=zip(texts, embeddings),
                embedding=self.embedding_model
            )
            
            print(f"✅ Vector store created with {len(texts)} documents")

        except Exception as e:
            print(f"❌ Error creating vector store: {str(e)}")
            raise

    def _format_flight_info(self, row: pd.Series) -> str:
        """
        Convert row data to optimized text format
        Example output:
        "FLT_XH9779 | LAS-SAN | 2024-01-15 | B772 | FuelTotal:45.2, CO2:137.8"
        """
        parts = []
        
        # Add identifiers
        if self.column_categories['id_columns']:
            ids = "_".join(str(row[col]) for col in self.column_categories['id_columns'])
            parts.append(f"FLT_{ids}")
            
        # Add locations
        if self.column_categories['location_columns']:
            locs = "-".join(str(row[col]) for col in self.column_categories['location_columns'])
            parts.append(locs)
            
        # Add dates
        if self.column_categories['date_columns']:
            dates = "|".join(str(row[col])[:10] for col in self.column_categories['date_columns'])
            parts.append(dates)
            
        # Add measurements
        if self.column_categories['measurement_columns']:
            measures = ", ".join(
                f"{col.split('_')[-1]}:{row[col]:.1f}" 
                for col in self.column_categories['measurement_columns']
            )
            parts.append(measures)
            
        return " | ".join(parts)

    def get_relevant_context(self, query: str, chat_history: List) -> str:
        """
        Retrieve context using similarity search
        - query: Current user question
        - chat_history: Previous conversation context
        Returns concatenated relevant documents
        """
        try:
            if not self.vector_store:
                return "Please load data first"
                
            # Combine query with last 2 history entries
            full_query = " ".join([h[0] for h in chat_history[-2:]] + [query])
            
            docs = self.vector_store.similarity_search(
                full_query, 
                k=5,  # Return top 5 matches
                score_threshold=0.3  # Filter low-quality matches
            )
            
            return "\n\n".join(doc.page_content for doc in docs)
            
        except Exception as e:
            print(f"Search error: {str(e)}")
            return "Error retrieving information"

    def save_vector_store(self, path: str):
        """Persist FAISS index for future use"""
        self.vector_store.save_local(path)

    def load_vector_store(self, path: str):
        """Load pre-computed FAISS index"""
        self.vector_store = FAISS.load_local(path, self.embedding_model)
