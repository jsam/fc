"""
Helper functions for saving notebook outputs to the docs folder.
This module provides utilities to save plots, text, and tables from Jupyter notebooks.
"""

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import json
import numpy as np
from typing import Union, Any, Optional
import plotly.graph_objects as go
import io
import re

class OutputManager:
    """Manages saving outputs from Jupyter notebooks to organized folders."""
    
    def __init__(self, notebook_name: str, base_dir: str = "../docs"):
        """
        Initialize the OutputManager.
        
        Args:
            notebook_name: Name of the notebook (used for folder creation)
            base_dir: Base directory for saving outputs
        """
        self.notebook_name = notebook_name
        self.base_dir = Path(base_dir)
        self.output_dir = self.base_dir / notebook_name
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Track figure and output counters
        self.figure_counter = 0
        self.text_counter = 0
        self.table_counter = 0
        
        # Create a markdown file for all text outputs
        self.markdown_file = self.output_dir / f"{notebook_name}_outputs.md"
        
        # Initialize the markdown file with a header
        with open(self.markdown_file, 'w') as f:
            f.write(f"# {notebook_name.replace('_', ' ').title()} - Outputs\n\n")
            f.write("This file contains all text and table outputs from the notebook.\n\n")
            f.write("---\n\n")
    
    def save_figure(self, fig=None, name: Optional[str] = None, dpi: int = 150, 
                   bbox_inches: str = 'tight', transparent: bool = False):
        """
        Save a matplotlib or plotly figure as PNG.
        
        Args:
            fig: Figure object (matplotlib or plotly). If None, uses current matplotlib figure
            name: Custom name for the figure file
            dpi: Resolution for PNG output
            bbox_inches: How to handle the bounding box
            transparent: Whether to save with transparent background
        """
        if name is None:
            self.figure_counter += 1
            name = f"figure_{self.figure_counter:03d}"
        
        filepath = self.output_dir / f"{name}.png"
        
        # Handle different figure types
        if fig is None:
            # Use current matplotlib figure
            plt.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, 
                       transparent=transparent)
            print(f"✅ Saved matplotlib figure: {filepath}")
        elif hasattr(fig, 'write_image'):
            # Plotly figure
            fig.write_image(filepath)
            print(f"✅ Saved plotly figure: {filepath}")
        elif hasattr(fig, 'savefig'):
            # Matplotlib figure object
            fig.savefig(filepath, dpi=dpi, bbox_inches=bbox_inches, 
                       transparent=transparent)
            print(f"✅ Saved matplotlib figure: {filepath}")
        else:
            print(f"⚠️ Unknown figure type, could not save")
    
    def save_text(self, text: str, title: Optional[str] = None, 
                  as_code: bool = False):
        """
        Save text output to the markdown file.
        
        Args:
            text: Text to save
            title: Optional title for the text section
            as_code: Whether to format as code block
        """
        self.text_counter += 1
        
        with open(self.markdown_file, 'a') as f:
            if title:
                f.write(f"## {title}\n\n")
            else:
                f.write(f"## Output {self.text_counter}\n\n")
            
            if as_code:
                f.write("```\n")
                f.write(str(text))
                f.write("\n```\n\n")
            else:
                f.write(str(text))
                f.write("\n\n")
            
            f.write("---\n\n")
        
        print(f"✅ Saved text output to {self.markdown_file}")
    
    def save_dataframe(self, df: pd.DataFrame, name: Optional[str] = None,
                      max_rows: Optional[int] = None, float_format: str = ".3f"):
        """
        Save a pandas DataFrame as a markdown table.
        
        Args:
            df: DataFrame to save
            name: Optional name for the table
            max_rows: Maximum number of rows to include (None for all)
            float_format: Format string for floating point numbers
        """
        self.table_counter += 1
        
        if name is None:
            name = f"Table {self.table_counter}"
        
        # Limit rows if specified
        if max_rows and len(df) > max_rows:
            df_to_save = df.head(max_rows)
            truncated = True
        else:
            df_to_save = df
            truncated = False
        
        # Convert to markdown
        markdown_table = df_to_save.to_markdown(floatfmt=float_format)
        
        # Save to markdown file
        with open(self.markdown_file, 'a') as f:
            f.write(f"## {name}\n\n")
            f.write(markdown_table)
            if truncated:
                f.write(f"\n\n*Note: Table truncated to {max_rows} rows*")
            f.write("\n\n---\n\n")
        
        print(f"✅ Saved table '{name}' to {self.markdown_file}")
    
    def save_dict(self, data: dict, name: Optional[str] = None):
        """
        Save a dictionary as formatted markdown.
        
        Args:
            data: Dictionary to save
            name: Optional name for the output
        """
        if name is None:
            self.text_counter += 1
            name = f"Data {self.text_counter}"
        
        with open(self.markdown_file, 'a') as f:
            f.write(f"## {name}\n\n")
            
            for key, value in data.items():
                # Format key
                formatted_key = key.replace('_', ' ').title()
                
                # Format value based on type
                if isinstance(value, float):
                    f.write(f"- **{formatted_key}**: {value:.4f}\n")
                elif isinstance(value, (list, tuple)) and len(value) <= 5:
                    f.write(f"- **{formatted_key}**: {', '.join(map(str, value))}\n")
                else:
                    f.write(f"- **{formatted_key}**: {value}\n")
            
            f.write("\n---\n\n")
        
        print(f"✅ Saved dictionary '{name}' to {self.markdown_file}")
    
    def save_statistics(self, stats_dict: dict, name: str = "Statistical Summary"):
        """
        Save statistical results in a formatted way.
        
        Args:
            stats_dict: Dictionary containing statistical results
            name: Name for the statistics section
        """
        with open(self.markdown_file, 'a') as f:
            f.write(f"## {name}\n\n")
            
            # Create a table for statistics
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            
            for key, value in stats_dict.items():
                formatted_key = key.replace('_', ' ').title()
                
                if isinstance(value, float):
                    f.write(f"| {formatted_key} | {value:.4f} |\n")
                elif isinstance(value, tuple) and len(value) == 2:
                    # Assume it's (statistic, p-value)
                    f.write(f"| {formatted_key} (Statistic) | {value[0]:.4f} |\n")
                    f.write(f"| {formatted_key} (p-value) | {value[1]:.4f} |\n")
                else:
                    f.write(f"| {formatted_key} | {value} |\n")
            
            f.write("\n---\n\n")
        
        print(f"✅ Saved statistics '{name}' to {self.markdown_file}")
    
    def save_correlation_matrix(self, corr_matrix: pd.DataFrame, name: str = "Correlation Matrix"):
        """
        Save a correlation matrix as a formatted markdown table.
        
        Args:
            corr_matrix: Correlation matrix DataFrame
            name: Name for the correlation matrix
        """
        # Format the correlation matrix for better readability
        formatted_corr = corr_matrix.round(3)
        
        with open(self.markdown_file, 'a') as f:
            f.write(f"## {name}\n\n")
            f.write(formatted_corr.to_markdown())
            f.write("\n\n---\n\n")
        
        print(f"✅ Saved correlation matrix '{name}' to {self.markdown_file}")
    
    def save_model_results(self, model_name: str, metrics: dict, 
                          coefficients: Optional[pd.DataFrame] = None):
        """
        Save machine learning model results.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of performance metrics
            coefficients: Optional DataFrame of model coefficients
        """
        with open(self.markdown_file, 'a') as f:
            f.write(f"## Model Results: {model_name}\n\n")
            
            # Save metrics
            f.write("### Performance Metrics\n\n")
            f.write("| Metric | Value |\n")
            f.write("|--------|-------|\n")
            
            for metric, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"| {metric} | {value:.4f} |\n")
                else:
                    f.write(f"| {metric} | {value} |\n")
            
            # Save coefficients if provided
            if coefficients is not None:
                f.write("\n### Model Coefficients\n\n")
                f.write(coefficients.to_markdown(floatfmt=".4f"))
            
            f.write("\n\n---\n\n")
        
        print(f"✅ Saved model results for '{model_name}' to {self.markdown_file}")
    
    def create_summary(self):
        """Create a summary file listing all saved outputs."""
        summary_file = self.output_dir / "README.md"
        
        with open(summary_file, 'w') as f:
            f.write(f"# {self.notebook_name.replace('_', ' ').title()} - Output Summary\n\n")
            f.write(f"This folder contains all outputs from the notebook analysis.\n\n")
            
            f.write("## Contents\n\n")
            f.write(f"- **Figures saved**: {self.figure_counter}\n")
            f.write(f"- **Tables saved**: {self.table_counter}\n")
            f.write(f"- **Text outputs saved**: {self.text_counter}\n\n")
            
            f.write("## Files\n\n")
            f.write(f"- `{self.markdown_file.name}`: All text and table outputs\n")
            
            # List all PNG files
            png_files = list(self.output_dir.glob("*.png"))
            if png_files:
                f.write("\n### Figures\n\n")
                for png in sorted(png_files):
                    f.write(f"- `{png.name}`\n")
        
        print(f"✅ Created summary file: {summary_file}")


def format_p_value(p_value: float) -> str:
    """
    Format p-value for display.
    
    Args:
        p_value: The p-value to format
    
    Returns:
        Formatted string representation
    """
    if p_value < 0.001:
        return "< 0.001"
    elif p_value < 0.01:
        return f"{p_value:.3f}"
    elif p_value < 0.05:
        return f"{p_value:.3f}"
    else:
        return f"{p_value:.3f}"


def interpret_correlation(r: float) -> str:
    """
    Interpret correlation coefficient strength.
    
    Args:
        r: Correlation coefficient
    
    Returns:
        String interpretation
    """
    abs_r = abs(r)
    if abs_r < 0.1:
        strength = "negligible"
    elif abs_r < 0.3:
        strength = "weak"
    elif abs_r < 0.5:
        strength = "moderate"
    elif abs_r < 0.7:
        strength = "strong"
    else:
        strength = "very strong"
    
    direction = "positive" if r > 0 else "negative"
    return f"{strength} {direction}"