#!/usr/bin/env python3

"""
CHO Analysis Fake Data Generator and Manager - Updated for New Schema

This script can generate realistic fake CHO analysis results for testing
and development purposes, and also clean them up when needed.

Usage:
    python fake_data_manager.py generate --count 50
    python fake_data_manager.py generate --count 20 --full-analysis-ratio 0.7
    python fake_data_manager.py list-fake
    python fake_data_manager.py delete-fake
    python fake_data_manager.py delete-all --confirm
"""

import argparse
import random
import uuid
from datetime import datetime, timedelta
import numpy as np
import json
import sys
import psycopg2
import psycopg2.extras
from faker import Faker

# Database connection settings
DB_CONFIG = {
    'host': 'localhost',  # or 'postgres' if running in container
    'port': 5433,         # or 5432 if running in container
    'database': 'orthanc',
    'user': 'postgres',
    'password': 'pgpassword'
}

# Marker to identify fake data
FAKE_DATA_MARKER = "FAKE_DATA_"

class CHOFakeDataGenerator:
    def __init__(self):
        self.fake = Faker()
        self.connection = None
        
        # Realistic data ranges based on medical literature
        self.scanner_manufacturers = ['SIEMENS', 'GE MEDICAL SYSTEMS', 'Philips', 'TOSHIBA', 'Canon Medical Systems']
        self.scanner_models = {
            'SIEMENS': ['SOMATOM Force', 'SOMATOM Definition Edge', 'SOMATOM Perspective', 'SOMATOM go.Top'],
            'GE MEDICAL SYSTEMS': ['Revolution CT', 'Discovery CT750 HD', 'LightSpeed VCT', 'BrightSpeed Elite'],
            'Philips': ['iCT 256', 'Brilliance CT', 'Ingenuity CT', 'MX8000 IDT'],
            'TOSHIBA': ['Aquilion ONE', 'Aquilion PRIME', 'Aquilion CXL'],
            'Canon Medical Systems': ['Aquilion ONE', 'Aquilion PRIME SP', 'Aquilion Lightning']
        }
        
        self.institutions = [
            'Mayo Clinic', 'Johns Hopkins Hospital', 'Cleveland Clinic', 'Massachusetts General Hospital',
            'UCLA Medical Center', 'Stanford Medical Center', 'Mount Sinai Hospital', 'Presbyterian Hospital',
            'Memorial Sloan Kettering', 'Brigham and Women\'s Hospital', 'Duke University Hospital',
            'University of Chicago Medical Center', 'Cedars-Sinai Medical Center', 'Houston Methodist Hospital'
        ]
        
        self.protocols = [
            'Abdomen/Pelvis w/ Contrast', 'Chest/Abdomen/Pelvis w/ Contrast', 'Abdomen w/o Contrast',
            'Liver Triple Phase', 'Pancreatic Protocol', 'Renal Mass Protocol', 'Trauma Abdomen',
            'CT Enterography', 'Liver Lesion Characterization', 'Abdominal Aorta CTA'
        ]
        
        self.convolution_kernels = ['Br40', 'Br44', 'Br36', 'I30f', 'I44f', 'Qr40', 'Standard', 'Soft', 'Sharp']
        
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**DB_CONFIG)
            return True
        except Exception as e:
            import traceback
            print(f"Failed to connect to database: {str(e)}")
            traceback.print_exc()
            return False
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
    
    def generate_patient_info(self):
        """Generate realistic patient information"""
        # Use fake data marker in patient ID to identify fake entries
        patient_id = f"{FAKE_DATA_MARKER}{self.fake.random_number(digits=8)}"
        
        return {
            'patient_id': patient_id,
            'name': self.fake.name(),
            'birth_date': self.fake.date_of_birth(minimum_age=18, maximum_age=95),
            'sex': random.choice(['M', 'F']),
            'weight_kg': round(random.uniform(50, 120), 1)
        }
    
    def generate_scanner_info(self):
        """Generate realistic scanner information"""
        manufacturer = random.choice(self.scanner_manufacturers)
        model = random.choice(self.scanner_models[manufacturer])
        institution = random.choice(self.institutions)
        
        return {
            'manufacturer': manufacturer,
            'model_name': model,
            'device_serial_number': f"{FAKE_DATA_MARKER}{self.fake.random_number(digits=8)}",
            'software_versions': f"VA{random.randint(10, 50)}.{random.randint(1, 9)}",
            'station_name': f"CT{random.randint(1, 10):02d}",
            'institution_name': institution,
        }
    
    def generate_study_info(self, patient_id_fk, institution_name):
        """Generate realistic study information"""
        study_date = self.fake.date_between(start_date='-2y', end_date='today')
        study_time = self.fake.time()
        
        return {
            'patient_id_fk': patient_id_fk,
            'study_instance_uid': f"{FAKE_DATA_MARKER}{uuid.uuid4()}",
            'study_id': str(random.randint(100000, 999999)),
            'accession_number': f"ACC{random.randint(100000, 999999)}",
            'study_date': study_date,
            'study_time': study_time,
            'description': 'CT ABDOMEN PELVIS W CONTRAST',
            'referring_physician': self.fake.name(),
            'institution_name': institution_name,
            'institution_address': self.fake.address(),
        }
    
    def generate_series_info(self, study_id_fk, scanner_id_fk, n_images=None):
        """Generate realistic series information"""
        if n_images is None:
            n_images = random.randint(150, 400)  # Typical slice count for abdominal CT
        
        scan_length_cm = round(random.uniform(25, 45), 2)
        series_date = self.fake.date_between(start_date='-2y', end_date='today')
        series_time = self.fake.time()
            
        return {
            'uuid': f"{FAKE_DATA_MARKER}{uuid.uuid4()}",
            'study_id_fk': study_id_fk,
            'series_instance_uid': f"{FAKE_DATA_MARKER}{uuid.uuid4()}",
            'series_number': random.randint(1, 10),
            'description': random.choice(['Abdomen/Pelvis', 'Portal Venous', 'Arterial Phase', 'Delayed Phase']),
            'modality': 'CT',
            'body_part_examined': 'ABDOMEN',
            'protocol_name': random.choice(self.protocols),
            'convolution_kernel': random.choice(self.convolution_kernels),
            'patient_position': random.choice(['HFS', 'HFP', 'FFS']),
            'series_date': series_date,
            'series_time': series_time,
            'frame_of_reference_uid': f"{FAKE_DATA_MARKER}{uuid.uuid4()}",
            'image_type': ['ORIGINAL', 'PRIMARY', 'AXIAL'],
            'slice_thickness_mm': round(random.uniform(0.5, 5.0), 1),
            'pixel_spacing_mm': [round(random.uniform(0.5, 1.0), 2), round(random.uniform(0.5, 1.0), 2)],
            'rows': 512,
            'columns': 512,
            'scanner_id_fk': scanner_id_fk,
            'image_count': n_images,
            'scan_length_cm': scan_length_cm,
        }
    
    def generate_ct_technique_info(self, series_id_fk):
        """Generate realistic CT technique information"""
        return {
            'series_id_fk': series_id_fk,
            'kvp': random.choice([80, 100, 120, 140]),
            'exposure_time_ms': random.randint(500, 2000),
            'generator_power_kw': round(random.uniform(40, 80), 1),
            'focal_spots_mm': [round(random.uniform(0.5, 1.5), 1)],
            'filter_type': random.choice(['Al', 'Cu', 'Sn', 'None']),
            'data_collection_diam_mm': round(random.uniform(400, 600), 1),
            'recon_diameter_mm': round(random.uniform(350, 450), 1),
            'dist_src_detector_mm': round(random.uniform(900, 1100), 1),
            'dist_src_patient_mm': round(random.uniform(500, 600), 1),
            'gantry_detector_tilt_deg': round(random.uniform(-30, 30), 1),
            'single_collimation_width_mm': round(random.uniform(0.5, 2.0), 1),
            'total_collimation_width_mm': round(random.uniform(10, 40), 1),
            'table_speed_mm_s': round(random.uniform(10, 50), 1),
            'table_feed_per_rot_mm': round(random.uniform(10, 40), 1),
            'spiral_pitch_factor': round(random.uniform(0.8, 1.5), 2),
            'exposure_modulation_type': random.choice(['NONE', 'XY', 'Z', 'XYZ']),
        }
    
    def generate_realistic_cho_results(self, full_analysis=False, n_images=300):
        """Generate realistic CHO analysis results"""
        # Base parameters that affect other values
        scan_length = random.uniform(25, 45)  # cm
        mean_ctdi = random.uniform(5, 25)  # mGy
        mean_dw = random.uniform(25, 35)  # cm
        
        # Calculate derived values
        para_a = 3.704369
        para_b = 0.03671937
        f = para_a * np.exp(-para_b * mean_dw)
        ssde = f * mean_ctdi
        dlp = scan_length * mean_ctdi
        dlp_ssde = scan_length * ssde
        
        # Generate location arrays
        location_array = np.linspace(0, scan_length, n_images).tolist()
        location_sparse = np.arange(10, scan_length - 10 + 1, 5).tolist()
        
        # Generate CTDI and Dw arrays with realistic variation
        ctdivol_array = np.random.normal(mean_ctdi, mean_ctdi * 0.1, n_images).tolist()
        dw_array = np.random.normal(mean_dw, mean_dw * 0.05, n_images).tolist()
        
        # Base results common to both analysis types
        base_results = {
            'ctdivol_avg': float(mean_ctdi),
            'dw_avg': float(mean_dw),
            'ssde': float(ssde),
            'dlp': float(dlp),
            'dlp_ssde': float(dlp_ssde),
            'spatial_resolution': None,
            'location': location_array,
            'dw': dw_array,
            'ctdivol': ctdivol_array,
            'processing_time': round(random.uniform(30, 120), 2),
            'peak_frequency': None,
            'average_frequency': None,
            'percent_10_frequency': None,
            'average_noise_level': None,
            'average_index_of_detectability': None,
            'location_sparse': None,
            'spatial_frequency': None,
            'nps': None,
            'noise_level': None,
            'cho_detectability': None,
        }
        
        if full_analysis:
            # Full analysis includes additional metrics
            n_sparse = len(location_sparse)
            noise_levels = np.random.normal(12, 2, n_sparse).tolist()
            cho_detectability = np.random.normal(2.5, 0.5, n_sparse).tolist()
            
            # Generate NPS data
            spatial_freq_length = random.randint(100, 200)
            spatial_freq = np.linspace(0, 2.0, spatial_freq_length).tolist()
            nps_values = np.exp(-np.linspace(0, 3, spatial_freq_length)) * random.uniform(50, 200)
            nps_values = nps_values.tolist()
            
            full_results = {
                'peak_frequency': round(random.uniform(0.2, 0.8), 3),
                'average_frequency': round(random.uniform(0.4, 1.2), 3),
                'percent_10_frequency': round(random.uniform(0.8, 1.8), 3),
                'average_noise_level': round(random.uniform(10, 16), 2),
                'spatial_resolution': round(random.uniform(6.5, 8.5), 2),
                'average_index_of_detectability': round(random.uniform(1.5, 3.5), 2),
                'location_sparse': location_sparse,
                'spatial_frequency': spatial_freq,
                'nps': nps_values,
                'noise_level': noise_levels,
                'cho_detectability': cho_detectability,
            }
            base_results.update(full_results)
            
        return base_results
    
    def save_fake_result(self, fake_data):
        """Save fake result to database using new schema"""
        try:
            with self.connection.cursor() as cursor:
                # Insert patient
                cursor.execute("""
                    INSERT INTO dicom.patient (patient_id, name, birth_date, sex, weight_kg)
                    VALUES (%(patient_id)s, %(name)s, %(birth_date)s, %(sex)s, %(weight_kg)s)
                    RETURNING id
                """, fake_data['patient_info'])
                patient_id = cursor.fetchone()[0]
                
                # Insert scanner
                cursor.execute("""
                    INSERT INTO dicom.scanner (manufacturer, model_name, device_serial_number, 
                                             software_versions, station_name, institution_name)
                    VALUES (%(manufacturer)s, %(model_name)s, %(device_serial_number)s, 
                           %(software_versions)s, %(station_name)s, %(institution_name)s)
                    RETURNING id
                """, fake_data['scanner_info'])
                scanner_id = cursor.fetchone()[0]
                
                # Update study info with patient_id_fk
                study_info = fake_data['study_info'].copy()
                study_info['patient_id_fk'] = patient_id
                
                # Insert study
                cursor.execute("""
                    INSERT INTO dicom.study (patient_id_fk, study_instance_uid, study_id, 
                                           accession_number, study_date, study_time, description,
                                           referring_physician, institution_name, institution_address)
                    VALUES (%(patient_id_fk)s, %(study_instance_uid)s, %(study_id)s, 
                           %(accession_number)s, %(study_date)s, %(study_time)s, %(description)s,
                           %(referring_physician)s, %(institution_name)s, %(institution_address)s)
                    RETURNING id
                """, study_info)
                study_id = cursor.fetchone()[0]
                
                # Update series info with foreign keys
                series_info = fake_data['series_info'].copy()
                series_info['study_id_fk'] = study_id
                series_info['scanner_id_fk'] = scanner_id
                
                # Insert series
                cursor.execute("""
                    INSERT INTO dicom.series (uuid, study_id_fk, series_instance_uid, series_number,
                                            description, modality, body_part_examined, protocol_name,
                                            convolution_kernel, patient_position, series_date, series_time,
                                            frame_of_reference_uid, image_type, slice_thickness_mm,
                                            pixel_spacing_mm, rows, columns, scanner_id_fk,
                                            image_count, scan_length_cm)
                    VALUES (%(uuid)s, %(study_id_fk)s, %(series_instance_uid)s, %(series_number)s,
                           %(description)s, %(modality)s, %(body_part_examined)s, %(protocol_name)s,
                           %(convolution_kernel)s, %(patient_position)s, %(series_date)s, %(series_time)s,
                           %(frame_of_reference_uid)s, %(image_type)s, %(slice_thickness_mm)s,
                           %(pixel_spacing_mm)s, %(rows)s, %(columns)s, %(scanner_id_fk)s,
                           %(image_count)s, %(scan_length_cm)s)
                    RETURNING id
                """, series_info)
                series_id = cursor.fetchone()[0]
                
                # Update CT technique info with series_id_fk
                ct_info = fake_data['ct_info'].copy()
                ct_info['series_id_fk'] = series_id
                
                # Insert CT technique
                cursor.execute("""
                    INSERT INTO dicom.ct_technique (series_id_fk, kvp, exposure_time_ms, generator_power_kw,
                                                  focal_spots_mm, filter_type, data_collection_diam_mm,
                                                  recon_diameter_mm, dist_src_detector_mm, dist_src_patient_mm,
                                                  gantry_detector_tilt_deg, single_collimation_width_mm,
                                                  total_collimation_width_mm, table_speed_mm_s,
                                                  table_feed_per_rot_mm, spiral_pitch_factor, exposure_modulation_type)
                    VALUES (%(series_id_fk)s, %(kvp)s, %(exposure_time_ms)s, %(generator_power_kw)s,
                           %(focal_spots_mm)s, %(filter_type)s, %(data_collection_diam_mm)s,
                           %(recon_diameter_mm)s, %(dist_src_detector_mm)s, %(dist_src_patient_mm)s,
                           %(gantry_detector_tilt_deg)s, %(single_collimation_width_mm)s,
                           %(total_collimation_width_mm)s, %(table_speed_mm_s)s,
                           %(table_feed_per_rot_mm)s, %(spiral_pitch_factor)s, %(exposure_modulation_type)s)
                """, ct_info)
                
                # Update results info with series_id_fk
                results_info = fake_data['results_dict'].copy()
                results_info['series_id_fk'] = series_id
                
                # Insert analysis results
                cursor.execute("""
                    INSERT INTO analysis.results (series_id_fk, average_frequency, average_index_of_detectability,
                                                average_noise_level, cho_detectability, ctdivol, ctdivol_avg,
                                                dlp, dlp_ssde, dw, dw_avg, location, location_sparse,
                                                noise_level, nps, peak_frequency, percent_10_frequency,
                                                processing_time, spatial_frequency, spatial_resolution, ssde)
                    VALUES (%(series_id_fk)s, %(average_frequency)s, %(average_index_of_detectability)s,
                           %(average_noise_level)s, %(cho_detectability)s, %(ctdivol)s, %(ctdivol_avg)s,
                           %(dlp)s, %(dlp_ssde)s, %(dw)s, %(dw_avg)s, %(location)s, %(location_sparse)s,
                           %(noise_level)s, %(nps)s, %(peak_frequency)s, %(percent_10_frequency)s,
                           %(processing_time)s, %(spatial_frequency)s, %(spatial_resolution)s, %(ssde)s)
                """, results_info)
                
                
                self.connection.commit()
                return True
                
        except Exception as e:
            import traceback
            print(f"Error saving fake result: {str(e)}")
            print(f"Results info: {results_info}")
            traceback.print_exc()
            if self.connection:
                self.connection.rollback()
            return False
    
    def generate_fake_result(self, full_analysis=False):
        """Generate a complete fake CHO analysis result"""
        n_images = random.randint(150, 400)
        
        patient_info = self.generate_patient_info()
        scanner_info = self.generate_scanner_info()
        study_info = self.generate_study_info(None, scanner_info['institution_name'])  # patient_id_fk will be set later
        series_info = self.generate_series_info(None, None, n_images)  # foreign keys will be set later
        ct_info = self.generate_ct_technique_info(None)  # series_id_fk will be set later
        results_dict = self.generate_realistic_cho_results(full_analysis, n_images)
        
        return {
            'patient_info': patient_info,
            'scanner_info': scanner_info,
            'study_info': study_info,
            'series_info': series_info,
            'ct_info': ct_info,
            'results_dict': results_dict
        }
    
    def generate_multiple_fake_results(self, count=10, full_analysis_ratio=0.5):
        """Generate multiple fake results"""
        if not self.connect():
            return False
        
        print(f"Generating {count} fake CHO analysis results...")
        print(f"Full analysis ratio: {full_analysis_ratio:.1%}")
        
        successful = 0
        failed = 0
        
        for i in range(count):
            # Determine if this should be a full analysis
            is_full_analysis = random.random() < full_analysis_ratio
            analysis_type = "Full" if is_full_analysis else "Global Noise"
            
            print(f"  Creating result {i+1}/{count} ({analysis_type})...", end=' ')
            
            try:
                fake_data = self.generate_fake_result(is_full_analysis)
                if self.save_fake_result(fake_data):
                    print("✓")
                    successful += 1
                else:
                    print("✗ (save failed)")
                    failed += 1
            except Exception as e:
                import traceback
                print(f"✗ (error: {str(e)})")
                traceback.print_exc()
                failed += 1
        
        self.close()
        
        print(f"\nCompleted: {successful} successful, {failed} failed")
        return successful > 0
    
    def list_fake_results(self):
        """List all fake results in the database"""
        if not self.connect():
            return False
        
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        p.patient_id,
                        p.name as patient_name,
                        s.series_instance_uid,
                        CASE 
                            WHEN r.average_index_of_detectability IS NOT NULL THEN 'Full Analysis'
                            ELSE 'Global Noise'
                        END as analysis_type,
                        r.created_at,
                        st.institution_name
                    FROM analysis.results r
                    JOIN dicom.series s ON r.series_id_fk = s.id
                    JOIN dicom.study st ON s.study_id_fk = st.id
                    JOIN dicom.patient p ON st.patient_id_fk = p.id
                    WHERE p.patient_id LIKE %s
                    ORDER BY r.created_at DESC
                """, (f"{FAKE_DATA_MARKER}%",))
                
                results = cursor.fetchall()
                
                if not results:
                    print("No fake results found in database.")
                    return True
                
                print(f"Found {len(results)} fake CHO analysis results:")
                print("\n" + "="*120)
                print(f"{'Patient ID':<20} {'Patient Name':<25} {'Analysis Type':<15} {'Institution':<25} {'Created':<20}")
                print("="*120)
                
                for result in results:
                    created_str = result['created_at'].strftime('%Y-%m-%d %H:%M')
                    patient_id_clean = result['patient_id'].replace(FAKE_DATA_MARKER, '')
                    patient_name = result['patient_name']
                    analysis_type = result['analysis_type']
                    institution_name = result['institution_name'] or 'Unknown'
                    
                    print(f"{patient_id_clean:<20} {patient_name:<25} {analysis_type:<15} "
                          f"{institution_name:<25} {created_str:<20}")

                return True
                
        except Exception as e:
            import traceback
            print(f"Error listing fake results: {str(e)}")
            traceback.print_exc()
            return False
        finally:
            self.close()
    
    def delete_fake_results(self, confirm=False):
        """Delete all fake results from the database"""
        if not self.connect():
            return False
        
        try:
            with self.connection.cursor() as cursor:
                # First, count how many we'll delete
                cursor.execute("""
                    SELECT COUNT(*)
                    FROM analysis.results r
                    JOIN dicom.series s ON r.series_id_fk = s.id
                    JOIN dicom.study st ON s.study_id_fk = st.id
                    JOIN dicom.patient p ON st.patient_id_fk = p.id
                    WHERE p.patient_id LIKE %s
                """, (f"{FAKE_DATA_MARKER}%",))
                
                count = cursor.fetchone()[0]
                
                if count == 0:
                    print("No fake results found to delete.")
                    return True
                
                if not confirm:
                    print(f"Found {count} fake results that would be deleted.")
                    print("Use --confirm flag to actually delete them.")
                    return True
                
                print(f"Deleting {count} fake CHO analysis results...")
                
                # Delete fake patients (cascading should handle related records)
                cursor.execute("""
                    DELETE FROM dicom.patient WHERE patient_id LIKE %s
                """, (f"{FAKE_DATA_MARKER}%",))
                
                deleted_patients = cursor.rowcount
                
                # Delete fake scanners
                cursor.execute("""
                    DELETE FROM dicom.scanner WHERE device_serial_number LIKE %s
                """, (f"{FAKE_DATA_MARKER}%",))
                
                deleted_scanners = cursor.rowcount
                
                self.connection.commit()
                
                print(f"Successfully deleted:")
                print(f"  - {deleted_patients} fake patients (and their associated data)")
                print(f"  - {deleted_scanners} fake scanners")
                
                return True
                
        except Exception as e:
            import traceback
            print(f"Error deleting fake results: {str(e)}")
            traceback.print_exc()
            if self.connection:
                self.connection.rollback()
            return False
        finally:
            self.close()

def main():
    parser = argparse.ArgumentParser(description="CHO Analysis Fake Data Generator and Manager - Updated for New Schema")
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Generate command
    gen_parser = subparsers.add_parser('generate', help='Generate fake CHO analysis results')
    gen_parser.add_argument('--count', '-c', type=int, default=10, help='Number of fake results to generate')
    gen_parser.add_argument('--full-analysis-ratio', '-r', type=float, default=0.5, 
                           help='Ratio of full analysis vs global noise (0.0-1.0)')
    
    # List command
    list_parser = subparsers.add_parser('list-fake', help='List all fake results in database')
    
    # Delete command
    del_parser = subparsers.add_parser('delete-fake', help='Delete all fake results')
    del_parser.add_argument('--confirm', action='store_true', help='Actually delete the data')
    
    # Quick generation presets
    quick_parser = subparsers.add_parser('quick-demo', help='Generate a quick demo dataset')
    quick_parser.add_argument('--size', choices=['small', 'medium', 'large'], default='medium',
                             help='Demo size: small (10), medium (50), large (100)')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    generator = CHOFakeDataGenerator()
    
    if args.command == 'generate':
        if args.count <= 0:
            print("Count must be positive")
            return
        if not 0 <= args.full_analysis_ratio <= 1:
            print("Full analysis ratio must be between 0 and 1")
            return
            
        generator.generate_multiple_fake_results(args.count, args.full_analysis_ratio)
    
    elif args.command == 'list-fake':
        generator.list_fake_results()
    
    elif args.command == 'delete-fake':
        generator.delete_fake_results(args.confirm)
    
    elif args.command == 'quick-demo':
        size_map = {'small': 10, 'medium': 50, 'large': 100}
        count = size_map[args.size]
        print(f"Generating {args.size} demo dataset ({count} results)...")
        generator.generate_multiple_fake_results(count, 0.6)  # 60% full analysis

if __name__ == "__main__":
    main()