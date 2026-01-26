#!/usr/bin/env python3
"""
DICOM Stress Test Generator for Orthanc

Generates fake DICOM series from a template for stress testing the Orthanc server.
Features:
- Creates unique series by modifying DICOM identifiers
- Optional noise addition to pixel data
- Concurrent sending to test server performance
- Automatic cleanup to manage disk space
- Progress tracking and statistics
"""

import os
import sys
import json
import time
import uuid
import random
import argparse
import threading
import requests
import tempfile
import shutil
import numpy as np
from datetime import datetime, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import pydicom
from pydicom.uid import generate_uid
from pydicom.dataset import Dataset
from pydicom.sequence import Sequence

class DicomStressTest:
    def __init__(self, orthanc_url="http://localhost:8042", username="demo", password="demo"):
        """Initialize the stress test generator"""
        self.orthanc_url = orthanc_url.rstrip('/')
        self.auth = (username, password) if username and password else None
        self.temp_dir = None
        self.stats = {
            'generated': 0,
            'sent': 0,
            'failed': 0,
            'start_time': None,
            'series_times': []
        }
        self.lock = threading.Lock()
        
    def __enter__(self):
        """Context manager entry"""
        # self.temp_dir = tempfile.mkdtemp(prefix="dicom_stress_", dir="D:/temp")
        self.temp_dir = tempfile.mkdtemp(prefix="dicom_stress_")
        print(f"Using temporary directory: {self.temp_dir}")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - cleanup"""
        if self.temp_dir and os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)
            print(f"Cleaned up temporary directory: {self.temp_dir}")

    def load_template_series(self, template_path):
        """Load template DICOM series from directory or files"""
        template_files = []
        
        if os.path.isdir(template_path):
            # Load all DICOM files from directory
            for file_path in Path(template_path).rglob("*"):
                if file_path.is_file():
                    try:
                        ds = pydicom.dcmread(file_path, force=True)
                        if hasattr(ds, 'SOPInstanceUID'):
                            template_files.append(str(file_path))
                    except Exception:
                        continue
        else:
            # Single file
            template_files = [template_path]
        
        if not template_files:
            raise ValueError(f"No valid DICOM files found in {template_path}")
        
        # Sort by instance number or slice location if available
        def sort_key(filepath):
            try:
                ds = pydicom.dcmread(filepath, stop_before_pixels=True)
                return (
                    getattr(ds, 'InstanceNumber', 0),
                    getattr(ds, 'SliceLocation', 0)
                )
            except:
                return (0, 0)
        
        template_files.sort(key=sort_key)
        print(f"Loaded template series with {len(template_files)} files")
        return template_files

    def generate_fake_identifiers(self):
        """Generate new DICOM identifiers for a fake series"""
        return {
            'PatientID': f"STRESS_{random.randint(100000, 999999)}",
            'PatientName': f"StressTest^Patient{random.randint(1, 9999)}",
            'PatientBirthDate': (datetime.now() - timedelta(days=random.randint(18*365, 90*365))).strftime('%Y%m%d'),
            'PatientSex': random.choice(['M', 'F']),
            'StudyInstanceUID': generate_uid(),
            'StudyID': f"ST{random.randint(1000, 9999)}",
            'StudyDate': (datetime.now() - timedelta(days=random.randint(0, 365))).strftime('%Y%m%d'),
            'StudyTime': f"{random.randint(8, 17):02d}{random.randint(0, 59):02d}{random.randint(0, 59):02d}",
            'StudyDescription': f"StressTest_{random.choice(['CHEST', 'ABDOMEN', 'PELVIS', 'HEAD'])}",
            'SeriesInstanceUID': generate_uid(),
            'SeriesNumber': random.randint(1, 99),
            'SeriesDescription': f"StressTest_Series_{random.randint(1, 999)}",
            'AccessionNumber': f"ACC{random.randint(100000, 999999)}",
            'InstitutionName': f"StressTest_Hospital_{random.randint(1, 10)}"
        }

    def add_noise_to_pixels(self, pixel_array, noise_level=0.1):
        """Add noise to DICOM pixel data"""
        if noise_level <= 0:
            return pixel_array
        
        # Add Gaussian noise
        noise = np.random.normal(0, noise_level * np.std(pixel_array), pixel_array.shape)
        noisy_pixels = pixel_array + noise
        
        # Ensure values stay within valid range
        noisy_pixels = np.clip(noisy_pixels, pixel_array.min(), pixel_array.max())
        return noisy_pixels.astype(pixel_array.dtype)

    def modify_dicom_file(self, input_path, output_path, identifiers, noise_level=0):
        """Modify a DICOM file with new identifiers and optional noise"""
        try:
            # Read DICOM file
            ds = pydicom.dcmread(input_path)
            
            # Update identifiers
            for tag, value in identifiers.items():
                if hasattr(ds, tag):
                    setattr(ds, tag, value)
            
            # Generate new SOP Instance UID for each instance
            ds.SOPInstanceUID = generate_uid()
            
            # Add noise to pixel data if requested
            if noise_level > 0 and hasattr(ds, 'pixel_array'):
                try:
                    original_pixels = ds.pixel_array
                    noisy_pixels = self.add_noise_to_pixels(original_pixels, noise_level)
                    ds.PixelData = noisy_pixels.tobytes()
                except Exception as e:
                    print(f"Warning: Could not add noise to {input_path}: {e}")
            
            # Update timestamps
            now = datetime.now()
            if hasattr(ds, 'ContentDate'):
                ds.ContentDate = now.strftime('%Y%m%d')
            if hasattr(ds, 'ContentTime'):
                ds.ContentTime = now.strftime('%H%M%S')
            
            # Save modified DICOM
            ds.save_as(output_path)
            return True
            
        except Exception as e:
            print(f"Error modifying DICOM file {input_path}: {e}")
            return False

    def generate_fake_series(self, template_files, output_dir, noise_level=0):
        """Generate a fake DICOM series from template"""
        identifiers = self.generate_fake_identifiers()
        series_dir = os.path.join(output_dir, f"series_{identifiers['PatientID']}")
        os.makedirs(series_dir, exist_ok=True)
        
        success_count = 0
        for i, template_file in enumerate(template_files):
            output_file = os.path.join(series_dir, f"IMG_{i+1:04d}.dcm")
            if self.modify_dicom_file(template_file, output_file, identifiers, noise_level):
                success_count += 1
        
        if success_count > 0:
            with self.lock:
                self.stats['generated'] += 1
            return series_dir, identifiers
        else:
            # Clean up failed series
            if os.path.exists(series_dir):
                shutil.rmtree(series_dir)
            return None, None

    def send_series_to_orthanc(self, series_dir):
        """Send a DICOM series to Orthanc"""
        if not series_dir or not os.path.exists(series_dir):
            return False, "Series directory not found"
        
        start_time = time.time()
        try:
            dicom_files = [f for f in os.listdir(series_dir) if f.endswith('.dcm')]
            
            for dicom_file in dicom_files:
                file_path = os.path.join(series_dir, dicom_file)
                
                with open(file_path, 'rb') as f:
                    response = requests.post(
                        f"{self.orthanc_url}/instances",
                        data=f.read(),
                        headers={'Content-Type': 'application/dicom'},
                        auth=self.auth,
                        timeout=30
                    )
                
                if response.status_code not in [200, 409]:  # 409 = already exists
                    return False, f"HTTP {response.status_code}: {response.text}"
            
            send_time = time.time() - start_time
            with self.lock:
                self.stats['sent'] += 1
                self.stats['series_times'].append(send_time)
            
            return True, f"Sent {len(dicom_files)} files in {send_time:.2f}s"
            
        except Exception as e:
            with self.lock:
                self.stats['failed'] += 1
            return False, str(e)

    def cleanup_old_series(self, max_series=50):
        """Clean up old series to manage disk space"""
        if not self.temp_dir:
            return
        
        series_dirs = [d for d in os.listdir(self.temp_dir) 
                      if os.path.isdir(os.path.join(self.temp_dir, d)) and d.startswith('series_')]
        
        if len(series_dirs) > max_series:
            # Sort by modification time (oldest first)
            series_dirs.sort(key=lambda d: os.path.getmtime(os.path.join(self.temp_dir, d)))
            
            # Remove oldest series
            to_remove = series_dirs[:-max_series]
            for series_dir in to_remove:
                series_path = os.path.join(self.temp_dir, series_dir)
                try:
                    shutil.rmtree(series_path)
                    print(f"Cleaned up old series: {series_dir}")
                except Exception as e:
                    print(f"Error cleaning up {series_dir}: {e}")

    def run_stress_test(self, template_files, num_series=10, noise_level=0, 
                       max_workers=4, max_series_on_disk=50, delay_between=0):
        """Run the stress test"""
        print(f"Starting stress test:")
        print(f"  Series to generate: {num_series}")
        print(f"  Noise level: {noise_level}")
        print(f"  Max workers: {max_workers}")
        print(f"  Max series on disk: {max_series_on_disk}")
        print(f"  Delay between sends: {delay_between}s")
        print()
        
        self.stats['start_time'] = time.time()
        
        def generate_and_send():
            """Generate and send a single series"""
            try:
                # Generate fake series
                series_dir, identifiers = self.generate_fake_series(
                    template_files, self.temp_dir, noise_level
                )
                
                if not series_dir:
                    return False, "Failed to generate series"
                
                # Add delay if specified
                if delay_between > 0:
                    time.sleep(delay_between)
                
                # Send to Orthanc
                success, message = self.send_series_to_orthanc(series_dir)
                
                # Cleanup this series directory after sending
                try:
                    shutil.rmtree(series_dir)
                except:
                    pass
                
                return success, message
                
            except Exception as e:
                return False, str(e)
        
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            futures = [executor.submit(generate_and_send) for _ in range(num_series)]
            
            # Process results as they complete
            for i, future in enumerate(as_completed(futures), 1):
                try:
                    success, message = future.result()
                    status = "[SUCCESS]" if success else "[ERROR]"
                    print(f"[{i:3d}/{num_series}] {status} {message}")
                    
                    # Periodic cleanup
                    if i % 10 == 0:
                        self.cleanup_old_series(max_series_on_disk)
                        
                except Exception as e:
                    print(f"[{i:3d}/{num_series}] [ERROR] Exception: {e}")
                    with self.lock:
                        self.stats['failed'] += 1

    def print_statistics(self):
        """Print test statistics"""
        if not self.stats['start_time']:
            return
        
        elapsed = time.time() - self.stats['start_time']
        series_times = self.stats['series_times']
        
        print("\n" + "="*50)
        print("STRESS TEST RESULTS")
        print("="*50)
        print(f"Total time: {elapsed:.2f} seconds")
        print(f"Series generated: {self.stats['generated']}")
        print(f"Series sent successfully: {self.stats['sent']}")
        print(f"Failed sends: {self.stats['failed']}")
        print(f"Success rate: {(self.stats['sent']/(self.stats['sent']+self.stats['failed'])*100):.1f}%" 
              if (self.stats['sent'] + self.stats['failed']) > 0 else "N/A")
        
        if series_times:
            print(f"Average send time: {np.mean(series_times):.2f}s")
            print(f"Min send time: {np.min(series_times):.2f}s")
            print(f"Max send time: {np.max(series_times):.2f}s")
            print(f"Series/second: {len(series_times)/elapsed:.2f}")

    def test_orthanc_connection(self):
        """Test connection to Orthanc server"""
        try:
            response = requests.get(f"{self.orthanc_url}/system", auth=self.auth, timeout=10)
            if response.status_code == 200:
                system_info = response.json()
                print(f"[SUCCESS] Connected to Orthanc: {system_info.get('Name', 'Unknown')}")
                print(f"  Version: {system_info.get('Version', 'Unknown')}")
                return True
            else:
                print(f"[ERROR] Orthanc connection failed: HTTP {response.status_code}")
                return False
        except Exception as e:
            print(f"[ERROR] Orthanc connection failed: {e}")
            return False


def main():
    parser = argparse.ArgumentParser(description="DICOM Stress Test Generator for Orthanc")
    
    parser.add_argument('template', help="Path to template DICOM file or directory")
    parser.add_argument('-n', '--num-series', type=int, default=10, 
                       help="Number of fake series to generate (default: 10)")
    parser.add_argument('--noise', type=float, default=0, 
                       help="Noise level to add to pixel data (0-1, default: 0)")
    parser.add_argument('--workers', type=int, default=4, 
                       help="Number of concurrent workers (default: 4)")
    parser.add_argument('--max-series', type=int, default=50, 
                       help="Maximum series to keep on disk (default: 50)")
    parser.add_argument('--delay', type=float, default=0, 
                       help="Delay between sends in seconds (default: 0)")
    parser.add_argument('--orthanc-url', default="http://localhost:8042", 
                       help="Orthanc server URL (default: http://localhost:8042)")
    parser.add_argument('--username', default="demo", 
                       help="Orthanc username (default: demo)")
    parser.add_argument('--password', default="demo", 
                       help="Orthanc password (default: demo)")
    parser.add_argument('--test-connection', action='store_true', 
                       help="Test connection to Orthanc and exit")
    
    args = parser.parse_args()
    
    # Test connection only
    if args.test_connection:
        stress_test = DicomStressTest(args.orthanc_url, args.username, args.password)
        stress_test.test_orthanc_connection()
        return
    
    # Validate arguments
    if not os.path.exists(args.template):
        print(f"Error: Template path '{args.template}' does not exist")
        sys.exit(1)
    
    if args.noise < 0 or args.noise > 1:
        print("Error: Noise level must be between 0 and 1")
        sys.exit(1)
    
    # Run stress test
    with DicomStressTest(args.orthanc_url, args.username, args.password) as stress_test:
        # Test connection first
        if not stress_test.test_orthanc_connection():
            print("Cannot connect to Orthanc server. Please check the URL and credentials.")
            sys.exit(1)
        
        # Load template series
        try:
            template_files = stress_test.load_template_series(args.template)
        except Exception as e:
            print(f"Error loading template: {e}")
            sys.exit(1)
        
        # Run the stress test
        try:
            stress_test.run_stress_test(
                template_files=template_files,
                num_series=args.num_series,
                noise_level=args.noise,
                max_workers=args.workers,
                max_series_on_disk=args.max_series,
                delay_between=args.delay
            )
        except KeyboardInterrupt:
            print("\nStress test interrupted by user")
        except Exception as e:
            print(f"Error during stress test: {e}")
        finally:
            stress_test.print_statistics()


if __name__ == "__main__":
    main()