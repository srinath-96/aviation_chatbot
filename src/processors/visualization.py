import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from typing import Dict, Any

class AviationVisualizer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def create_visualization(self, viz_type: str) -> Any:
        """Create different types of visualizations"""
        if viz_type == "fuel_by_aircraft":
            return self.fuel_consumption_by_aircraft()
        elif viz_type == "co2_by_route":
            return self.co2_emissions_by_route()
        elif viz_type == "daily_flights":
            return self.daily_flight_distribution()
        elif viz_type == "fuel_phases":
            return self.fuel_consumption_phases()
        return None

    def fuel_consumption_by_aircraft(self) -> go.Figure:
        """Create bar chart of fuel consumption by aircraft type"""
        avg_fuel = self.df.groupby('AIRCRAFT_TYPE')['ESTIMATED_FUEL_BURN_TOTAL_TONNES'].mean().reset_index()
        fig = px.bar(
            avg_fuel,
            x='AIRCRAFT_TYPE',
            y='ESTIMATED_FUEL_BURN_TOTAL_TONNES',
            title='Average Fuel Consumption by Aircraft Type',
            labels={'ESTIMATED_FUEL_BURN_TOTAL_TONNES': 'Average Fuel Burn (Tonnes)'}
        )
        return fig

    def co2_emissions_by_route(self) -> go.Figure:
        """Create scatter plot of CO2 emissions by route"""
        route_emissions = self.df.groupby(['DEPARTURE_AIRPORT', 'ARRIVAL_AIRPORT'])['ESTIMATED_CO2_TOTAL_TONNES'].mean().reset_index()
        fig = px.scatter(
            route_emissions,
            x='DEPARTURE_AIRPORT',
            y='ESTIMATED_CO2_TOTAL_TONNES',
            color='ARRIVAL_AIRPORT',
            title='CO2 Emissions by Route',
            labels={'ESTIMATED_CO2_TOTAL_TONNES': 'Average CO2 Emissions (Tonnes)'}
        )
        return fig

    def daily_flight_distribution(self) -> go.Figure:
        """Create line chart of daily flight distribution"""
        daily_flights = self.df.groupby('SCHEDULED_DEPARTURE_DATE').size().reset_index(name='count')
        fig = px.line(
            daily_flights,
            x='SCHEDULED_DEPARTURE_DATE',
            y='count',
            title='Daily Flight Distribution',
            labels={'count': 'Number of Flights'}
        )
        return fig

    def fuel_consumption_phases(self) -> go.Figure:
        """Create stacked bar chart of fuel consumption phases"""
        phase_cols = [
            'ESTIMATED_FUEL_BURN_TAXI_OUT_TONNES',
            'ESTIMATED_FUEL_BURN_TAKEOFF_TONNES',
            'ESTIMATED_FUEL_BURN_CLIMBOUT_TONNES',
            'ESTIMATED_FUEL_BURN_CRUISE_TONNES',
            'ESTIMATED_FUEL_BURN_APPROACH_TONNES',
            'ESTIMATED_FUEL_BURN_TAXI_IN_TONNES'
        ]
        
        phase_means = self.df[phase_cols].mean()
        fig = px.bar(
            x=phase_means.index,
            y=phase_means.values,
            title='Average Fuel Consumption by Flight Phase',
            labels={'x': 'Flight Phase', 'y': 'Average Fuel Burn (Tonnes)'}
        )
        return fig
