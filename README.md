# Aviation Chatbot with Interactive Visualizations

## Project Overview

This project implements an intelligent chatbot for aviation data analysis. It uses a Cache-Augmented Generation (CAG) approach, combining a vector store for efficient similarity search with pre-computed summaries for rapid responses. The chatbot can process uploaded datasets (CSV/TSV), answer queries using Google's Gemini API, and generate interactive visualizations using Plotly.

## Features

- File Upload: Support for CSV and TSV files containing aviation data.
- Natural Language Queries: Powered by Google's Gemini API for intelligent responses.
- Interactive Visualizations: Generate and display Plotly charts based on user queries.
- Efficient Data Retrieval: Utilizes FAISS for fast similarity search on large datasets.
- Caching: Pre-computes and caches dataset summaries for quick access to common statistics.

## Project Structure


aviation_chatbot/
│
├── src/
│ ├── utils/
│ │ └── hybrid_search.py
│ └── models/
│ └── chatbot.py
│
├── main.py
├── requirements.txt
├── .env
└── README.md


## Key Components

1. **hybrid_search.py**: Implements the CacheAugmentedSystem class for data processing, vector store creation, and caching.
2. **chatbot.py**: Contains the AviationChatbot class, handling file processing, query responses, and visualization generation.
3. **main.py**: Sets up the Gradio interface for user interaction.

## Setup and Installation

1. Clone the repository:

git clone https://github.com/yourusername/aviation_chatbot.git
cd aviation_chatbot


2. Create and activate a virtual environment:

python -m venv .venv
source .venv/bin/activate # On Windows, use


3. Install required packages:

pip install -r requirements.txt


4. Set up your .env file with your Google API key:

GOOGLE_API_KEY=your_api_key_here

5. 
## Usage

1. Run the application:

python main.py



2. Open the provided URL in your web browser.

3. Upload a CSV or TSV file containing aviation data.

4. Start chatting! Ask questions about the dataset or request visualizations.

## Example Queries

- "What is the total fuel consumption across all flights?"
- "Show me a visualization of flights by carrier."
- "Which carrier has the highest CO2 emissions?"
- "Visualize the relationship between fuel consumption and CO2 emissions."

## Visualization Types

The chatbot can generate various types of visualizations based on user queries:

1. Bar charts (e.g., flights per carrier)
2. Scatter plots (e.g., fuel consumption vs. CO2 emissions)
3. Line charts (e.g., time series of fuel usage)
4. Tables (e.g., top 5 flights by fuel consumption)

## Technologies Used

- Python 3.8+
- Gradio: For the web interface
- Google Generative AI (Gemini): For natural language processing
- Pandas: For data manipulation
- FAISS: For efficient similarity search
- Sentence Transformers: For text embedding
- Plotly: For interactive visualizations
- SQLite: For structured data storage and querying

## Future Improvements

- Implement more advanced visualization types
- Add support for real-time data streaming
- Enhance error handling and user feedback
- Optimize performance for larger datasets


