import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, 
                            QLabel, QPushButton, QFileDialog, QTextEdit, 
                            QGroupBox, QMessageBox, QProgressBar)
from PyQt5.QtCore import QTimer
from PyQt5.QtGui import QFont
import traceback
import app
from matrixes import show_confusion_matrices

class ADSApp(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.streaming = False
        self.stream_timer = QTimer()
        self.stream_timer.setInterval(2000)  # 2 seconds
        self.stream_timer.timeout.connect(self.stream_next_auto)
        self.streamed_indices = []
        self.streamed_labels = []
        self.anomaly_rates = []

    def initUI(self):
        self.setWindowTitle('Anomaly Detection System')
        self.setGeometry(100, 100, 1100, 800)
        main_layout = QVBoxLayout()

        # --- File Selection Card ---
        file_group = QGroupBox('Dataset Selection')
        file_group.setStyleSheet('QGroupBox { font-weight: bold; font-size: 16px; }')
        file_layout = QHBoxLayout()
        self.train_label = QLabel('No training file selected')
        self.test_label = QLabel('No testing file selected')
        btn_train = QPushButton('Select Training CSV')
        btn_test = QPushButton('Select Testing CSV')
        btn_train.clicked.connect(self.select_train)
        btn_test.clicked.connect(self.select_test)
        file_layout.addWidget(btn_train)
        file_layout.addWidget(self.train_label)
        file_layout.addWidget(btn_test)
        file_layout.addWidget(self.test_label)
        file_group.setLayout(file_layout)

        # --- Actions Card ---
        action_group = QGroupBox('Actions')
        action_group.setStyleSheet('QGroupBox { font-weight: bold; font-size: 16px; }')
        action_layout = QHBoxLayout()
        self.btn_load = QPushButton('Load & Preprocess Data')
        self.btn_train_models = QPushButton('Train Models')
        self.btn_stream_toggle = QPushButton('Start Streaming')
        self.btn_analyze = QPushButton('Full Analysis')
        self.btn_confusion_matrix = QPushButton('View Confusion Matrices')
        self.btn_train_models.setEnabled(False)
        self.btn_stream_toggle.setEnabled(False)
        self.btn_analyze.setEnabled(False)
        self.btn_confusion_matrix.setEnabled(False)
        self.btn_load.clicked.connect(self.load_and_preprocess)
        self.btn_train_models.clicked.connect(self.train_models)
        self.btn_stream_toggle.clicked.connect(self.toggle_streaming)
        self.btn_analyze.clicked.connect(self.analyze_all)
        self.btn_confusion_matrix.clicked.connect(self.show_confusion_matrices)
        action_layout.addWidget(self.btn_load)
        action_layout.addWidget(self.btn_train_models)
        action_layout.addWidget(self.btn_stream_toggle)
        action_layout.addWidget(self.btn_analyze)
        action_layout.addWidget(self.btn_confusion_matrix)
        action_group.setLayout(action_layout)

        # --- Status Card ---
        status_group = QGroupBox('Status')
        status_layout = QVBoxLayout()
        self.status = QLabel('Ready')
        self.status.setStyleSheet('font-size: 14px;')
        status_layout.addWidget(self.status)
        status_group.setLayout(status_layout)

        # --- Results Card ---
        results_group = QGroupBox('Results')
        results_layout = QVBoxLayout()
        self.results = QTextEdit()
        self.results.setReadOnly(True)
        self.results.setStyleSheet('background: #f8f9fa; font-size: 13px;')
        results_layout.addWidget(self.results)
        results_group.setLayout(results_layout)

        # --- Live Plots Card ---
        plot_group = QGroupBox('Live Analytics')
        plot_layout = QHBoxLayout()
        # Bar plot for cumulative counts
        self.figure_bar, self.ax_bar = plt.subplots(figsize=(4, 3))
        self.canvas_bar = FigureCanvas(self.figure_bar)
        plot_layout.addWidget(self.canvas_bar)
        # Line plot for anomaly rate
        self.figure_line, self.ax_line = plt.subplots(figsize=(4, 3))
        self.canvas_line = FigureCanvas(self.figure_line)
        plot_layout.addWidget(self.canvas_line)
        # Orange line+scatter plot with horizontal divider at 0.5 for normal/anomaly
        self.figure_spike_line, self.ax_spike_line = plt.subplots(figsize=(4, 2))
        self.canvas_spike_line = FigureCanvas(self.figure_spike_line)
        plot_layout.addWidget(self.canvas_spike_line)
        plot_group.setLayout(plot_layout)

        # Add all cards to main layout
        main_layout.addWidget(file_group)
        main_layout.addWidget(action_group)
        main_layout.addWidget(status_group)
        main_layout.addWidget(results_group)
        main_layout.addWidget(plot_group)
        self.setLayout(main_layout)

        self.train_path = None
        self.test_path = None

    def select_train(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Training CSV', '', 'CSV Files (*.csv)')
        if path:
            self.train_path = path
            self.train_label.setText(f'Training: {path}')

    def select_test(self):
        path, _ = QFileDialog.getOpenFileName(self, 'Select Testing CSV', '', 'CSV Files (*.csv)')
        if path:
            self.test_path = path
            self.test_label.setText(f'Testing: {path}')

    def load_and_preprocess(self):
        if not self.train_path or not self.test_path:
            self.show_error('Please select both training and testing files.')
            return
        self.status.setText('Loading and preprocessing data...')
        QApplication.processEvents()
        try:
            n_train, n_test = app.load_and_preprocess(self.train_path, self.test_path)
            self.status.setText(f'Data loaded. Training samples: {n_train}, Testing samples: {n_test}')
            self.btn_train_models.setEnabled(True)
            self.btn_stream_toggle.setEnabled(False)
            self.btn_analyze.setEnabled(False)
            self.btn_confusion_matrix.setEnabled(False)
            self.results.clear()
            self.streamed_indices = []
            self.streamed_labels = []
            self.anomaly_rates = []
            self.update_plots()
        except Exception as e:
            self.show_error(f'Error: {str(e)}\n{traceback.format_exc()}')

    def train_models(self):
        self.status.setText('Training models...')
        QApplication.processEvents()
        try:
            models = app.train_models()
            self.status.setText(f'Models trained: {", ".join(models)}')
            self.btn_stream_toggle.setEnabled(True)
            self.btn_analyze.setEnabled(True)
            self.btn_confusion_matrix.setEnabled(True)
            self.results.append('Models trained successfully.')
        except Exception as e:
            self.show_error(f'Error: {str(e)}\n{traceback.format_exc()}')

    def toggle_streaming(self):
        if not self.streaming:
            self.streaming = True
            self.btn_stream_toggle.setText('Stop Streaming')
            self.status.setText('Streaming started...')
            self.stream_timer.start()
        else:
            self.streaming = False
            self.btn_stream_toggle.setText('Start Streaming')
            self.status.setText('Streaming stopped.')
            self.stream_timer.stop()

    def stream_next_auto(self):
        try:
            # Use ensemble method for streaming
            result = app.stream_next_ensemble(threshold=0.7)
            idx = len(self.streamed_indices)
            is_anomaly = result['prediction'] == 1
            status_str = f"Index: {idx} | Status: {'Anomaly' if is_anomaly else 'Normal'}"
            # Optionally show votes/probabilities for transparency
            votes_str = ', '.join([f"{name}: {'Anomaly' if v else 'Normal'} (p={result['probabilities'][name]:.2f})" for name, v in zip(app.models.keys(), result['votes'])])
            self.results.append(status_str)
            self.results.append(f"Votes: {votes_str}")
            self.status.setText(f"Streamed one row. Remaining: {result['remaining_samples']}")
            self.streamed_indices.append(idx)
            self.streamed_labels.append(result['prediction'])
            anomaly_count = sum(self.streamed_labels)
            total = len(self.streamed_labels)
            self.anomaly_rates.append(anomaly_count / total if total > 0 else 0)
            self.update_plots()
            if result['remaining_samples'] == 0:
                self.toggle_streaming()
                self.status.setText('Streaming finished. No more data.')
        except Exception as e:
            self.show_error(f'Error: {str(e)}\n{traceback.format_exc()}')
            self.toggle_streaming()

    def update_plots(self):
        # Bar plot: cumulative counts
        self.ax_bar.clear()
        normal_count = self.streamed_labels.count(0)
        anomaly_count = self.streamed_labels.count(1)
        self.ax_bar.bar(['Normal', 'Anomaly'], [normal_count, anomaly_count], color=['green', 'red'])
        self.ax_bar.set_title('Cumulative Count')
        self.ax_bar.set_ylabel('Count')
        self.canvas_bar.draw()
        
        # Line plot: anomaly rate with anomaly markers
        self.ax_line.clear()
        if self.anomaly_rates:
            x = np.arange(1, len(self.anomaly_rates) + 1)
            y = np.array(self.anomaly_rates)
            self.ax_line.plot(x, y, color='blue', marker='o', label='Normal')

            # Overlay blue triangle markers for anomalies
            anomaly_indices = np.where(np.array(self.streamed_labels) == 1)[0]
            if len(anomaly_indices) > 0:
                self.ax_line.plot(
                    x[anomaly_indices], y[anomaly_indices],
                    'v', color='red', markersize=10, label='Anomaly'
                )

            self.ax_line.set_ylim(0, 1)
            self.ax_line.legend()

        self.ax_line.set_title('Cumulative Anomaly Rate')
        self.ax_line.set_xlabel('Stream Index')
        self.ax_line.set_ylabel('Anomaly Rate')
        self.canvas_line.draw()

        # Orange line+scatter plot with horizontal divider at 0.5 for normal/anomaly
        self.ax_spike_line.clear()
        if self.streamed_labels:
            x = np.arange(1, len(self.streamed_labels) + 1)
            y = np.full_like(x, 0.5, dtype=float)
            labels = np.array(self.streamed_labels)
            y[labels == 1] = 1.0  # Anomaly
            y[labels == 0] = 0.0  # Normal
            # Draw smooth line (interpolated for visual effect)
            self.ax_spike_line.plot(x, y, color='orange', linewidth=2, marker='o', markersize=7, markerfacecolor='orange', markeredgecolor='orange', label='Normal/Anomaly')
            # Draw horizontal divider at 0.5
            self.ax_spike_line.axhline(0.5, color='black', linewidth=2, linestyle='--', alpha=0.7, label='Divider (0.5)')
            self.ax_spike_line.set_ylim(-0.1, 1.1)
            self.ax_spike_line.set_title('Normal(0) - Anomaly(1) with Divider')
            self.ax_spike_line.set_xlabel('Stream Index')
            self.ax_spike_line.set_ylabel('Value')
            self.ax_spike_line.legend()
        self.canvas_spike_line.draw()

    def analyze_all(self):
        self.status.setText('Running full analysis...')
        QApplication.processEvents()
        try:
            results = app.analyze_all_models()
            self.results.clear()  # Clear previous results for clean display
            
            # Header
            self.results.append('='*70)
            self.results.append('              ANOMALY DETECTION SYSTEM - ANALYSIS REPORT')
            self.results.append('='*70)
            self.results.append('')
            
            for i, (model, metrics) in enumerate(results.items(), 1):
                # Model header
                self.results.append(f"[{i}] {model.upper().replace('_', ' ')}")
                self.results.append('-'*50)
                
                # Extract key metrics from classification report
                if isinstance(metrics, dict) and 'report' in metrics:
                    report = metrics['report']
                    if isinstance(report, dict):
                        # Overall accuracy
                        accuracy = report.get('accuracy', 'N/A')
                        if isinstance(accuracy, float):
                            self.results.append(f"  Overall Accuracy............ {accuracy:.1%}")
                        
                        # Anomaly class metrics (class '1')
                        anomaly_metrics = report.get('1', {})
                        if anomaly_metrics:
                            precision = anomaly_metrics.get('precision', 'N/A')
                            recall = anomaly_metrics.get('recall', 'N/A')
                            f1_score = anomaly_metrics.get('f1-score', 'N/A')
                            
                            if isinstance(precision, float):
                                self.results.append(f"  Anomaly Precision........... {precision:.1%}")
                            if isinstance(recall, float):
                                self.results.append(f"  Anomaly Recall.............. {recall:.1%}")
                            if isinstance(f1_score, float):
                                self.results.append(f"  Anomaly F1-Score............ {f1_score:.1%}")
                        
                        # Normal class metrics (class '0')
                        normal_metrics = report.get('0', {})
                        if normal_metrics:
                            precision = normal_metrics.get('precision', 'N/A')
                            recall = normal_metrics.get('recall', 'N/A')
                            
                            if isinstance(precision, float):
                                self.results.append(f"  Normal Precision............ {precision:.1%}")
                            if isinstance(recall, float):
                                self.results.append(f"  Normal Recall............... {recall:.1%}")
                
                else:
                    # Fallback for other metric formats
                    self.results.append(f"  Results: {str(metrics)}")
                
                self.results.append('')  # Empty line between models
            
            # Footer
            self.results.append('='*70)
            self.results.append('                         ANALYSIS COMPLETED')
            self.results.append('='*70)
            
            self.status.setText('Analysis complete.')
        except Exception as e:
            self.show_error(f'Error: {str(e)}\n{traceback.format_exc()}')

    def show_confusion_matrices(self):
        """Open the confusion matrix viewer window"""
        self.matrix_viewer = show_confusion_matrices(self)
        if self.matrix_viewer:
            self.status.setText('Confusion Matrix viewer opened.')

    def show_error(self, msg):
        self.status.setText('Error')
        QMessageBox.critical(self, 'Error', msg)
        self.results.append(msg)

if __name__ == '__main__':
    app_qt = QApplication(sys.argv)
    ex = ADSApp()
    ex.show()
    sys.exit(app_qt.exec_()) 