import os
import json
import sqlite3
from datetime import datetime
import logging
from pathlib import Path


class DocumentLogger:
    def __init__(self, db_path='logs/document_processing.db', log_file='logs/document_processing.log'):
        self.db_path = db_path
        self.log_file = log_file
        
        # Ensure log directory exists
        log_dir = os.path.dirname(db_path)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Setup file logger
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('document_processor')
        
        # Initialize database
        self._init_db()
    
    def _init_db(self):
        """Initialize SQLite database with required tables"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create documents table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    filename TEXT NOT NULL,
                    predicted_type TEXT NOT NULL,
                    confidence REAL,
                    processing_time REAL,
                    quality_score REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            # Create extracted_fields table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS extracted_fields (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER,
                    field_name TEXT NOT NULL,
                    field_value TEXT,
                    confidence REAL,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            ''')
            
            # Create quality_metrics table
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS quality_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    document_id INTEGER,
                    metric_name TEXT NOT NULL,
                    metric_value REAL,
                    FOREIGN KEY (document_id) REFERENCES documents (id)
                )
            ''')
            
            conn.commit()
            conn.close()
            
        except sqlite3.Error as e:
            self.logger.error(f"Database initialization error: {str(e)}")
            raise
    
    def log_prediction(self, filename, result, processing_time):
        """Log document processing result"""
        timestamp = datetime.now().isoformat()
        
        # Format result for logging
        log_entry = {
            'timestamp': timestamp,
            'filename': filename,
            'predicted_type': result.get('document_type', 'unknown'),
            'processing_time': processing_time,
            'extracted_fields': result.get('extracted_fields', {}),
            'quality_metrics': result.get('quality_metrics', {})
        }
        
        # Log to file
        self.logger.info(json.dumps(log_entry, indent=2))
        
        try:
            # Log to database
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert document record
            cursor.execute('''
                INSERT INTO documents
                (filename, predicted_type, confidence, processing_time, quality_score, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                filename,
                result.get('document_type', 'unknown'),
                result.get('classification_confidence', 0.0),
                processing_time,
                result.get('quality_metrics', {}).get('overall_score', 0.0),
                timestamp
            ))
            
            document_id = cursor.lastrowid
            
            # Insert extracted fields
            for field_name, field_data in result.get('extracted_fields', {}).items():
                if isinstance(field_data, dict):
                    # Handle case where field_data contains value and confidence
                    value = field_data.get('value', '')
                    confidence = field_data.get('confidence', 0.0)
                else:
                    # Handle case where field_data is the value directly
                    value = field_data
                    confidence = 0.0
                
                cursor.execute('''
                    INSERT INTO extracted_fields
                    (document_id, field_name, field_value, confidence)
                    VALUES (?, ?, ?, ?)
                ''', (document_id, field_name, str(value), confidence))
            
            # Insert quality metrics
            for metric_name, metric_value in result.get('quality_metrics', {}).items():
                if isinstance(metric_value, (int, float)):
                    cursor.execute('''
                        INSERT INTO quality_metrics
                        (document_id, metric_name, metric_value)
                        VALUES (?, ?, ?)
                    ''', (document_id, metric_name, metric_value))
            
            conn.commit()
            conn.close()
            
        except sqlite3.Error as e:
            self.logger.error(f"Database logging error: {str(e)}")
            # Continue execution even if database logging fails
            pass
    
    def get_recent_predictions(self, limit=10):
        """Retrieve recent predictions from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT d.*, GROUP_CONCAT(ef.field_name || ': ' || ef.field_value)
                FROM documents d
                LEFT JOIN extracted_fields ef ON d.id = ef.document_id
                GROUP BY d.id
                ORDER BY d.timestamp DESC
                LIMIT ?
            ''', (limit,))
            
            results = cursor.fetchall()
            conn.close()
            
            return results
            
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving predictions: {str(e)}")
            return []
    
    def get_document_stats(self, days=7):
        """Get processing statistics for the last N days"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                SELECT
                    predicted_type,
                    COUNT(*) as count,
                    AVG(processing_time) as avg_time,
                    AVG(quality_score) as avg_quality
                FROM documents
                WHERE timestamp >= datetime('now', '-? days')
                GROUP BY predicted_type
            ''', (days,))
            
            stats = cursor.fetchall()
            conn.close()
            
            return stats
            
        except sqlite3.Error as e:
            self.logger.error(f"Error retrieving stats: {str(e)}")
            return []


# Example usage in document_pipeline.py:
"""
from app.utils.logging_utils import DocumentLogger

class DocumentPipeline:
    def __init__(self, debug=True):
        self.debug = debug
        self.logger = DocumentLogger()
        ...
    
    def process_document(self, pdf_path):
        start_time = time.time()
        try:
            result = self._process_document_internal(pdf_path)
            processing_time = time.time() - start_time
            
            # Log the prediction
            self.logger.log_prediction(
                filename=os.path.basename(pdf_path),
                result=result,
                processing_time=processing_time
            )
            
            return result
            
        except Exception as e:
            self.logger.logger.error(f"Error processing {pdf_path}: {str(e)}")
            return {"error": str(e)}
"""