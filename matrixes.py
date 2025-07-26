import sys
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QComboBox, QPushButton, QTextEdit, QFrame,
                            QMessageBox, QSizePolicy)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont, QIcon
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns
import numpy as np
from app import analyze_all_models

class ConfusionMatrixViewer(QWidget):
    def __init__(self, parent=None):
        super().__init__()
        # Make this a completely independent window
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint | Qt.WindowMinMaxButtonsHint)
        self.setWindowTitle("Confusion Matrix Analysis - Anomaly Detection System")
        self.setGeometry(250, 250, 950, 750)
        
        # Set window icon (optional, uses default if no icon available)
        self.setWindowIcon(self.style().standardIcon(self.style().SP_ComputerIcon))
        
        # Store reference to parent but don't make this window dependent on it
        self.parent_app = parent
        
        self.setStyleSheet("""
            QWidget {
                background-color: #f0f0f0;
                font-family: Arial;
            }
            QLabel {
                color: #333;
            }
            QComboBox {
                padding: 8px;
                border: 2px solid #ccc;
                border-radius: 5px;
                background: white;
                min-height: 20px;
            }
            QComboBox:hover {
                border-color: #4CAF50;
            }
            QPushButton {
                background-color: #4CAF50;
                color: white;
                border: none;
                padding: 10px 20px;
                border-radius: 5px;
                font-weight: bold;
                min-height: 15px;
            }
            QPushButton:hover {
                background-color: #45a049;
            }
            QPushButton:pressed {
                background-color: #3d8b40;
            }
            QTextEdit {
                border: 2px solid #ccc;
                border-radius: 5px;
                background: white;
                font-family: 'Courier New';
                padding: 5px;
            }
            QFrame {
                border: 2px solid #ccc;
                border-radius: 5px;
                background: white;
                margin: 5px;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #ccc;
                border-radius: 5px;
                margin: 5px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
        """)
        
        # Store analysis results
        self.results = None
        self.current_model = None
        
        # Create GUI elements
        self.init_ui()
        
        # Load data and populate dropdown
        self.load_analysis_data()
        
    def init_ui(self):
        # Main layout
        main_layout = QVBoxLayout()
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(15, 15, 15, 15)
        self.setLayout(main_layout)
        
        # Title
        title_label = QLabel("🔍 Confusion Matrix Analysis")
        title_label.setFont(QFont("Arial", 20, QFont.Bold))
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("color: #2c3e50; margin: 15px; padding: 10px;")
        main_layout.addWidget(title_label)
        
        # Control panel frame
        control_frame = QFrame()
        control_frame.setStyleSheet("QFrame { background: #ffffff; border: 2px solid #3498db; }")
        control_layout = QHBoxLayout(control_frame)
        control_layout.setSpacing(15)
        control_layout.setContentsMargins(15, 10, 15, 10)
        
        # Model selection label
        selection_label = QLabel("📊 Select Model:")
        selection_label.setFont(QFont("Arial", 12, QFont.Bold))
        control_layout.addWidget(selection_label)
        
        # Model dropdown
        self.model_dropdown = QComboBox()
        self.model_dropdown.setFont(QFont("Arial", 11))
        self.model_dropdown.setMinimumWidth(280)
        self.model_dropdown.currentTextChanged.connect(self.on_model_selected)
        control_layout.addWidget(self.model_dropdown)
        
        # Refresh button
        refresh_btn = QPushButton("🔄 Refresh Data")
        refresh_btn.setToolTip("Reload analysis data from the main application")
        refresh_btn.clicked.connect(self.load_analysis_data)
        control_layout.addWidget(refresh_btn)
        
        # Close button
        close_btn = QPushButton("❌ Close Window")
        close_btn.setToolTip("Close this confusion matrix viewer")
        close_btn.setStyleSheet("QPushButton { background-color: #e74c3c; } QPushButton:hover { background-color: #c0392b; }")
        close_btn.clicked.connect(self.close)
        control_layout.addWidget(close_btn)
        
        # Add stretch to center the controls
        control_layout.addStretch()
        
        main_layout.addWidget(control_frame)
        
        # Matrix display frame
        self.matrix_frame = QFrame()
        self.matrix_frame.setMinimumHeight(450)
        self.matrix_frame.setStyleSheet("QFrame { background: white; }")
        matrix_layout = QVBoxLayout(self.matrix_frame)
        matrix_layout.setContentsMargins(10, 10, 10, 10)
        
        # Create matplotlib figure
        self.figure = Figure(figsize=(10, 7))
        self.figure.patch.set_facecolor('white')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        matrix_layout.addWidget(self.canvas)
        
        main_layout.addWidget(self.matrix_frame)
        
        # Metrics display frame
        metrics_frame = QFrame()
        metrics_frame.setStyleSheet("QFrame { background: white; }")
        metrics_layout = QVBoxLayout(metrics_frame)
        metrics_layout.setContentsMargins(15, 10, 15, 10)
        
        metrics_label = QLabel("📈 Model Performance Metrics")
        metrics_label.setFont(QFont("Arial", 14, QFont.Bold))
        metrics_layout.addWidget(metrics_label)
        
        self.metrics_text = QTextEdit()
        self.metrics_text.setMaximumHeight(220)
        self.metrics_text.setFont(QFont("Courier New", 10))
        self.metrics_text.setReadOnly(True)
        metrics_layout.addWidget(self.metrics_text)
        
        main_layout.addWidget(metrics_frame)
        
        # Initial placeholder
        self.show_placeholder()
        
    def closeEvent(self, event):
        """Handle window close event"""
        reply = QMessageBox.question(self, 'Close Confirmation', 
                                   'Are you sure you want to close the Confusion Matrix viewer?',
                                   QMessageBox.Yes | QMessageBox.No, 
                                   QMessageBox.No)
        
        if reply == QMessageBox.Yes:
            # Clear matplotlib figures to free memory
            if hasattr(self, 'figure'):
                self.figure.clear()
                plt.close(self.figure)
            
            # Update parent status if available
            if self.parent_app and hasattr(self.parent_app, 'status'):
                self.parent_app.status.setText('Confusion Matrix viewer closed.')
            
            event.accept()
        else:
            event.ignore()
            
    def show_placeholder(self):
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        ax.text(0.5, 0.5, '🔍 Select a model to view its confusion matrix\n\n📊 Available models will appear in the dropdown above', 
                horizontalalignment='center', verticalalignment='center',
                transform=ax.transAxes, fontsize=16, color='#7f8c8d',
                bbox=dict(boxstyle="round,pad=0.5", facecolor="#ecf0f1", alpha=0.8))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)
        self.canvas.draw()
        self.metrics_text.clear()
        
    def load_analysis_data(self):
        try:
            # Get analysis results from app.py
            self.results = analyze_all_models()
            
            # Clear and populate dropdown with model names
            self.model_dropdown.clear()
            model_names = list(self.results.keys())
            
            if model_names:
                self.model_dropdown.addItems(model_names)
                # Set default selection and display
                self.display_confusion_matrix(model_names[0])
                
                # Update status in parent if available
                if self.parent_app and hasattr(self.parent_app, 'status'):
                    self.parent_app.status.setText(f'Confusion Matrix data loaded. {len(model_names)} models available.')
            else:
                self.show_placeholder()
                QMessageBox.information(self, "No Models", "No trained models found. Please train models first in the main application.")
                
        except Exception as e:
            error_msg = f"Failed to load analysis data:\n{str(e)}"
            QMessageBox.critical(self, "Error", error_msg)
            self.model_dropdown.clear()
            self.show_placeholder()
            
    def on_model_selected(self, model_name):
        if model_name:
            self.display_confusion_matrix(model_name)
            
    def display_confusion_matrix(self, model_name):
        if not self.results or model_name not in self.results:
            self.show_placeholder()
            return
            
        try:
            # Get confusion matrix and metrics
            model_data = self.results[model_name]
            cm = model_data['confusion_matrix']
            report = model_data['report']
            
            # Clear the figure
            self.figure.clear()
            
            # Create subplot with padding
            ax = self.figure.add_subplot(111)
            
            # Create enhanced heatmap
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['Normal', 'Attack'], 
                       yticklabels=['Normal', 'Attack'],
                       cbar_kws={'label': 'Count', 'shrink': 0.8},
                       square=True, linewidths=1, linecolor='white',
                       annot_kws={'size': 14, 'weight': 'bold'})
            
            ax.set_title(f'Confusion Matrix - {model_name}', fontsize=18, fontweight='bold', pad=25)
            ax.set_xlabel('Predicted Label', fontsize=14, fontweight='bold')
            ax.set_ylabel('True Label', fontsize=14, fontweight='bold')
            
            # Rotate labels for better readability
            ax.set_xticklabels(ax.get_xticklabels(), rotation=0, fontsize=12)
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
            
            # Adjust layout and draw
            self.figure.tight_layout(pad=3.0)
            self.canvas.draw()
            
            # Display metrics
            self.display_metrics(report, model_name)
            
            # Update window title to include current model
            self.setWindowTitle(f"Confusion Matrix Analysis - {model_name} - ADS")
            
        except Exception as e:
            self.show_placeholder()
            QMessageBox.critical(self, "Error", f"Error displaying matrix:\n{str(e)}")
            
    def display_metrics(self, report, model_name):
        # Format and display metrics with enhanced formatting
        metrics_content = f"🤖 Model: {model_name}\n"
        metrics_content += "=" * 70 + "\n\n"
        
        # Overall metrics
        if 'accuracy' in report:
            accuracy = report['accuracy']
            metrics_content += f"🎯 Overall Accuracy: {accuracy:.4f} ({accuracy:.1%})\n\n"
        
        # Class-wise metrics with emojis
        class_labels = {'0': ('Normal', '✅'), '1': ('Attack', '🚨')}
        for class_id, (class_name, emoji) in class_labels.items():
            if class_id in report:
                class_metrics = report[class_id]
                metrics_content += f"{emoji} {class_name} Class Metrics:\n"
                metrics_content += f"   Precision: {class_metrics['precision']:.4f} ({class_metrics['precision']:.1%})\n"
                metrics_content += f"   Recall:    {class_metrics['recall']:.4f} ({class_metrics['recall']:.1%})\n"
                metrics_content += f"   F1-Score:  {class_metrics['f1-score']:.4f} ({class_metrics['f1-score']:.1%})\n"
                metrics_content += f"   Support:   {class_metrics['support']} samples\n\n"
        
        # Macro and weighted averages
        avg_labels = {'macro avg': '📊', 'weighted avg': '⚖️'}
        for avg_type, emoji in avg_labels.items():
            if avg_type in report:
                avg_metrics = report[avg_type]
                metrics_content += f"{emoji} {avg_type.title()}:\n"
                metrics_content += f"   Precision: {avg_metrics['precision']:.4f} ({avg_metrics['precision']:.1%})\n"
                metrics_content += f"   Recall:    {avg_metrics['recall']:.4f} ({avg_metrics['recall']:.1%})\n"
                metrics_content += f"   F1-Score:  {avg_metrics['f1-score']:.4f} ({avg_metrics['f1-score']:.1%})\n\n"
        
        # Confusion Matrix Values with interpretation
        if 'confusion_matrix' in self.results[model_name]:
            cm = self.results[model_name]['confusion_matrix']
            metrics_content += "📋 Confusion Matrix Breakdown:\n"
            metrics_content += f"   True Negatives (TN):   {cm[0][0]} - Correctly identified Normal\n"
            metrics_content += f"   False Positives (FP):  {cm[0][1]} - Normal classified as Attack\n"
            metrics_content += f"   False Negatives (FN):  {cm[1][0]} - Attack classified as Normal\n"
            metrics_content += f"   True Positives (TP):   {cm[1][1]} - Correctly identified Attack\n\n"
            
            # Calculate additional metrics
            total = np.sum(cm)
            tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            metrics_content += "🔍 Additional Metrics:\n"
            metrics_content += f"   Specificity (TNR): {specificity:.4f} ({specificity:.1%})\n"
            metrics_content += f"   Sensitivity (TPR): {sensitivity:.4f} ({sensitivity:.1%})\n"
            metrics_content += f"   Total Samples:     {total}\n"
        
        self.metrics_text.setText(metrics_content)

def show_confusion_matrices(parent=None):
    """Function to be called from the main GUI"""
    try:
        viewer = ConfusionMatrixViewer(parent)
        viewer.show()
        viewer.raise_()  # Bring window to front
        viewer.activateWindow()  # Activate the window
        return viewer
    except Exception as e:
        error_msg = f"Failed to open confusion matrix viewer:\n{str(e)}"
        if parent:
            QMessageBox.critical(parent, "Error", error_msg)
        else:
            print(f"Error: {error_msg}")
        return None

if __name__ == "__main__":
    # For testing purposes
    app = QApplication(sys.argv)
    viewer = show_confusion_matrices()
    if viewer:
        sys.exit(app.exec_()) 