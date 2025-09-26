#!/usr/bin/env python3

"""
Enhanced Progress tracking module for CHO calculations
This module provides a singleton ProgressTracker class to track and report
calculation progress that can be accessed from API endpoints.
"""

import time
import threading
from collections import deque

class ProgressTracker:
    """Singleton class to track calculation progress"""
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(ProgressTracker, cls).__new__(cls)
                cls._instance._init_tracker()
            return cls._instance
    
    def _init_tracker(self):
        """Initialize the tracking data structures"""
        self._calculations = {}
        self._lock = threading.Lock()
        # Store the last 100 completed calculations
        self._history = deque(maxlen=100)

        print("Progress tracker initialized")
        
    def start_calculation(self, series_id, metadata=None):
        """Start tracking a new calculation"""
        with self._lock:
            if metadata is None:
                metadata = {}
                
            calculation_data = {
                'series_id': series_id,
                'status': 'running',
                'progress': 0,
                'message': 'Initializing calculation...',
                'details': [],
                'current_stage': 'initialization',
                'start_time': time.time(),
                'last_update': time.time(),
                'metadata': metadata,
                'results': None,
                'error': None
            }
            
            self._calculations[series_id] = calculation_data
            
            print(f"Started tracking calculation for series {series_id}")
            
            return self._calculations[series_id]
    
    def update_progress(self, series_id, progress, message=None, stage=None, details=None):
        """Update the progress of a calculation"""
        with self._lock:
            if series_id not in self._calculations:
                print(f"Attempted to update non-existent calculation: {series_id}")
                return False
                
            calc = self._calculations[series_id]
            old_progress = calc['progress']
            calc['progress'] = progress
            calc['last_update'] = time.time()
            
            if message:
                calc['message'] = message
                # Add to details log
                detail = {
                    'time': time.time(),
                    'message': message,
                    'progress': progress
                }
                calc['details'].append(detail)
                
            if stage:
                calc['current_stage'] = stage
                
            if details:
                if isinstance(details, dict):
                    # Add any additional details
                    for key, value in details.items():
                        if key not in calc:
                            calc[key] = value
            
            print(f"Updated progress for {series_id}: {progress}%, {message}")
            
            return True
    
    def complete_calculation(self, series_id, results=None):
        """Mark a calculation as complete and broadcast completion"""
        with self._lock:
            if series_id not in self._calculations:
                print(f"Attempted to complete non-existent calculation: {series_id}")
                return False
                
            calc = self._calculations[series_id]
            calc['status'] = 'completed'
            calc['progress'] = 100
            calc['message'] = 'Calculation completed successfully'
            calc['current_stage'] = 'completed'
            calc['end_time'] = time.time()
            calc['last_update'] = time.time()
            
            if results:
                calc['results'] = results
            
            # Add to details log
            detail = {
                'time': time.time(),
                'message': 'Calculation completed successfully',
                'progress': 100
            }
            calc['details'].append(detail)
            
            # Add to history for completed calculations
            self._history.append(calc.copy())
            
            print(f"Completed calculation for series {series_id}")
            # Delete the series DICOMS
            # orthanc.RestApiDelete(f'/series/{series_id}')
            
            return True
    
    def fail_calculation(self, series_id, error_message):
        """Mark a calculation as failed and broadcast failure"""
        with self._lock:
            if series_id not in self._calculations:
                print(f"Attempted to fail non-existent calculation: {series_id}")
                return False
                
            calc = self._calculations[series_id]
            calc['status'] = 'failed'
            calc['message'] = f'Calculation failed: {error_message}'
            calc['error'] = error_message
            calc['end_time'] = time.time()
            calc['last_update'] = time.time()
            
            # Add to details log
            detail = {
                'time': time.time(),
                'message': f'Calculation failed: {error_message}',
                'progress': calc['progress']
            }
            calc['details'].append(detail)
            
            # Add to history for completed calculations
            self._history.append(calc.copy())
            
            print(f"Failed calculation for series {series_id}: {error_message}")
            
            return True
    
    def get_calculation_status(self, series_id):
        """Get the current status of a calculation"""
        with self._lock:
            if series_id in self._calculations:
                # Return a copy to prevent modification
                return self._calculations[series_id].copy()
            
            # Check history if not in active calculations
            for calc in self._history:
                if calc['series_id'] == series_id:
                    return calc.copy()
                    
            return None
    
    def get_all_active_calculations(self):
        """Get all active calculations"""
        with self._lock:
            return [calc.copy() for calc in self._calculations.values()]
    
    def get_calculation_history(self):
        """Get calculation history"""
        with self._lock:
            return list(self._history)

    def cleanup_history(self, series_id):
        """Remove old entries from the calculation history"""
        with self._lock:
            self._history = [calc for calc in self._history if calc['series_id'] != series_id]


    def cleanup_calculation(self, series_id):
        """Remove a calculation from tracking (called after a delay)"""
        with self._lock:
            if series_id in self._calculations:
                del self._calculations[series_id]
                print(f"Cleaned up calculation for series {series_id}")
                
                return True
            return False
    
    def cancel_calculation(self, series_id, reason="User cancelled"):
        """Cancel a running calculation"""
        with self._lock:
            if series_id not in self._calculations:
                print(f"Attempted to cancel non-existent calculation: {series_id}")
                return False
                
            calc = self._calculations[series_id]
            if calc['status'] not in ['running']:
                print(f"Attempted to cancel non-running calculation: {series_id}")
                return False
                
            calc['status'] = 'cancelled'
            calc['message'] = f'Calculation cancelled: {reason}'
            calc['end_time'] = time.time()
            calc['last_update'] = time.time()
            
            # Add to details log
            detail = {
                'time': time.time(),
                'message': f'Calculation cancelled: {reason}',
                'progress': calc['progress']
            }
            calc['details'].append(detail)
            
            # Add to history
            self._history.append(calc.copy())
            
            print(f"Cancelled calculation for series {series_id}: {reason}")
            
            return True
    
    def get_statistics(self):
        """Get tracker statistics"""
        with self._lock:
            active_count = len(self._calculations)
            history_count = len(self._history)
            
            # Count statuses in history
            status_counts = {}
            for calc in self._history:
                status = calc.get('status', 'unknown')
                status_counts[status] = status_counts.get(status, 0) + 1
            
            return {
                'active_calculations': active_count,
                'completed_calculations': history_count,
                'status_breakdown': status_counts,
                'total_tracked': active_count + history_count
            }

# Create the singleton instance
progress_tracker = ProgressTracker()