# src/utils/data_processor.py
import pandas as pd
from typing import Dict, Any, Optional

class AutoDataProcessor:
    def __init__(self):
        self.df: Optional[pd.DataFrame] = None
        self.column_categories: Dict[str, list] = {
            'date': [],
            'numeric': [],
            'categorical': []
        }

    def load_data(self, file_path: str) -> str:
        """Load and process data from TSV/CSV file"""
        try:
            self.df = pd.read_csv(file_path, sep='\t')
            self._validate_dataframe()
            self._categorize_columns()
            self._process_columns()
            print("✅ Successfully loaded data with columns:", self.df.columns.tolist())
            return "Data loaded successfully"
        except Exception as e:
            self.df = None
            return f"Error loading data: {str(e)}"

    def _validate_dataframe(self):
        """Ensure dataframe meets minimum requirements"""
        if self.df.empty:
            raise ValueError("Loaded dataframe is empty")
        if len(self.df.columns) < 1:
            raise ValueError("No columns detected in the dataset")

    def _categorize_columns(self):
        """Improved column categorization with case insensitivity"""
        for col in self.df.columns:
            col_lower = col.lower()
            if any(kw in col_lower for kw in ['date', 'time']):
                self.column_categories['date'].append(col)
            elif self._is_numeric_column(col):
                self.column_categories['numeric'].append(col)
            else:
                self.column_categories['categorical'].append(col)

    def _is_numeric_column(self, col: str) -> bool:
        """Check if column should be treated as numeric"""
        col_lower = col.lower()
        return (
            self.df[col].dtype in ['int64', 'float64'] or
            any(kw in col_lower for kw in ['fuel', 'co2', 'tonnes'])
        )

    def _process_columns(self):
        """Safer column processing with error handling"""
        # Process date columns
        for date_col in self.column_categories['date']:
            self.df[date_col] = pd.to_datetime(
                self.df[date_col], 
                errors='coerce',
                format='mixed'
            )

        # Process numeric columns
        for num_col in self.column_categories['numeric']:
            self.df[num_col] = pd.to_numeric(
                self.df[num_col], 
                errors='coerce'
            )

        # Process categorical columns
        for cat_col in self.column_categories['categorical']:
            self.df[cat_col] = (
                self.df[cat_col]
                .astype(str)
                .str.strip()
                .replace({'nan': pd.NA, 'None': pd.NA})
            )

    def get_dataframe(self) -> pd.DataFrame:
        """Get processed dataframe with validation"""
        if self.df is None:
            raise ValueError("No data loaded. Call load_data() first")
        return self.df

    def get_column_summary(self) -> Dict[str, list]:
        """Get categorized column summary"""
        return self.column_categories
