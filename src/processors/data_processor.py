# src/utils/data_processor.py
import pandas as pd
from typing import Dict, Any

class AutoDataProcessor:
    def __init__(self):
        self.df = None
        self.column_categories = {}

    def load_data(self, file_path: str) -> str:
        """Load and process aviation data with dynamic column handling"""
        try:
            self.df = pd.read_csv(file_path)
            self._categorize_columns()
            self._process_columns()
            return "Data loaded successfully"
        except Exception as e:
            return f"Error loading data: {str(e)}"

    def _categorize_columns(self):
        """Categorize columns based on names and content"""
        self.column_categories = {
            'date': [],
            'numeric': [],
            'categorical': []
        }
        for col in self.df.columns:
            if 'DATE' in col.upper():
                self.column_categories['date'].append(col)
            elif self.df[col].dtype in ['int64', 'float64'] or any(keyword in col.upper() for keyword in ['FUEL', 'CO2']):
                self.column_categories['numeric'].append(col)
            else:
                self.column_categories['categorical'].append(col)

    def _process_columns(self):
        """Process columns based on their categories"""
        for date_col in self.column_categories['date']:
            self.df[date_col] = pd.to_datetime(self.df[date_col], format='mixed', errors='coerce')
        
        for num_col in self.column_categories['numeric']:
            self.df[num_col] = pd.to_numeric(self.df[num_col], errors='coerce')
        
        for cat_col in self.column_categories['categorical']:
            self.df[cat_col] = self.df[cat_col].astype(str).str.strip()

    def get_dataframe(self) -> pd.DataFrame:
        """Get the processed dataframe"""
        return self.df

    def get_statistics(self) -> Dict[str, Any]:
        """Calculate basic statistics dynamically"""
        if self.df is None:
            return {}
        
        stats = {
            'total_rows': len(self.df),
            'column_types': {col: str(dtype) for col, dtype in self.df.dtypes.items()}
        }
        
        # Add specific statistics based on column categories
        if self.column_categories['date']:
            date_col = self.column_categories['date'][0]  # Use the first date column
            stats['date_range'] = [self.df[date_col].min(), self.df[date_col].max()]
        
        for col in self.column_categories['numeric']:
            stats[f'sum_{col}'] = self.df[col].sum()
            stats[f'mean_{col}'] = self.df[col].mean()
        
        for col in self.column_categories['categorical']:
            stats[f'unique_{col}'] = self.df[col].nunique()
        
        return stats
    def get_column_summary(self) -> Dict[str, list]:
        """Get summary of column types"""
        return self.column_categories
