#!/usr/bin/env python3

"""
Database management utilities for CHO analysis system - Updated for New Schema
Provides functions for data migration, cleanup, and reporting
"""

import psycopg2
import psycopg2.extras
import json
from datetime import datetime, timedelta
import pandas as pd
import argparse
import sys

class CHODatabaseManager:
    def __init__(self):
        self.db_config = {
            'host': 'postgres',
            'port': 5432,
            'database': 'orthanc',
            'user': 'postgres',
            'password': 'pgpassword'
        }
        self.connection = None
    
    def connect(self):
        """Establish database connection"""
        try:
            self.connection = psycopg2.connect(**self.db_config)
            return True
        except Exception as e:
            print(f"Failed to connect to database: {str(e)}")
            return False
    
    def close(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
    
    def get_database_stats(self):
        """Get comprehensive database statistics"""
        if not self.connect():
            return None
        
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                stats = {}
                
                # Basic counts
                cursor.execute("SELECT COUNT(*) FROM analysis.results")
                stats['total_analyses'] = cursor.fetchone()[0]
                
                cursor.execute("SELECT COUNT(DISTINCT series_id_fk) FROM analysis.results")
                stats['unique_series'] = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT COUNT(DISTINCT p.id) 
                    FROM dicom.patient p
                    JOIN dicom.study st ON p.id = st.patient_id_fk
                    JOIN dicom.series s ON st.id = s.study_id_fk
                    JOIN analysis.results r ON s.id = r.series_id_fk
                """)
                stats['unique_patients'] = cursor.fetchone()[0]
                
                # Analysis type breakdown based on available data
                cursor.execute("""   
                    SELECT 
                        CASE 
                            WHEN average_index_of_detectability IS NOT NULL THEN 'full_analysis'
                            ELSE 'global_noise'
                        END as analysis_type,
                        COUNT(*) 
                    FROM analysis.results 
                    GROUP BY 
                        CASE 
                            WHEN average_index_of_detectability IS NOT NULL THEN 'full_analysis'
                            ELSE 'global_noise'
                        END
                """)
                stats['by_calculation_type'] = dict(cursor.fetchall())
                
                # Series completion status
                cursor.execute("""
                    SELECT 
                        CASE 
                            WHEN COUNT(CASE WHEN r.average_index_of_detectability IS NOT NULL THEN 1 END) > 0 
                                 AND COUNT(CASE WHEN r.average_index_of_detectability IS NULL THEN 1 END) > 0 
                                 THEN 'both'
                            WHEN COUNT(CASE WHEN r.average_index_of_detectability IS NOT NULL THEN 1 END) > 0 
                                 THEN 'full_only'
                            WHEN COUNT(CASE WHEN r.average_index_of_detectability IS NULL THEN 1 END) > 0 
                                 THEN 'global_only'
                            ELSE 'none'
                        END as status,
                        COUNT(DISTINCT s.id) as series_count
                    FROM dicom.series s
                    LEFT JOIN analysis.results r ON s.id = r.series_id_fk
                    GROUP BY 
                        CASE 
                            WHEN COUNT(CASE WHEN r.average_index_of_detectability IS NOT NULL THEN 1 END) > 0 
                                 AND COUNT(CASE WHEN r.average_index_of_detectability IS NULL THEN 1 END) > 0 
                                 THEN 'both'
                            WHEN COUNT(CASE WHEN r.average_index_of_detectability IS NOT NULL THEN 1 END) > 0 
                                 THEN 'full_only'
                            WHEN COUNT(CASE WHEN r.average_index_of_detectability IS NULL THEN 1 END) > 0 
                                 THEN 'global_only'
                            ELSE 'none'
                        END
                """)
                stats['by_completion_status'] = dict(cursor.fetchall())
                
                # Recent activity (last 30 days)
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM analysis.results 
                    WHERE created_at >= NOW() - INTERVAL '30 days'
                """)
                stats['recent_analyses'] = cursor.fetchone()[0]
                
                # Average processing times by type
                cursor.execute("""
                    SELECT 
                        CASE 
                            WHEN average_index_of_detectability IS NOT NULL THEN 'full_analysis'
                            ELSE 'global_noise'
                        END as calculation_type,
                        AVG(processing_time) as avg_time,
                        MIN(processing_time) as min_time,
                        MAX(processing_time) as max_time,
                        COUNT(*) as count
                    FROM analysis.results 
                    WHERE processing_time IS NOT NULL
                    GROUP BY 
                        CASE 
                            WHEN average_index_of_detectability IS NOT NULL THEN 'full_analysis'
                            ELSE 'global_noise'
                        END
                """)
                stats['processing_times'] = {row['calculation_type']: dict(row) for row in cursor.fetchall()}
                
                # Storage usage
                cursor.execute("""
                    SELECT 
                        pg_size_pretty(pg_total_relation_size('analysis.results')) as results_size,
                        pg_size_pretty(pg_total_relation_size('dicom.patient')) as patient_size,
                        pg_size_pretty(pg_total_relation_size('dicom.series')) as series_size,
                        pg_size_pretty(pg_total_relation_size('dicom.study')) as study_size,
                        pg_size_pretty(pg_total_relation_size('dicom.scanner')) as scanner_size,
                        pg_size_pretty(pg_total_relation_size('dicom.ct_technique')) as ct_technique_size
                """)
                stats['storage_usage'] = dict(cursor.fetchone())
                
                return stats
        except Exception as e:
            print(f"Error getting database stats: {str(e)}")
            return None
        finally:
            self.close()
    
    def cleanup_old_results(self, days_old=365, dry_run=True):
        """Clean up results older than specified days"""
        if not self.connect():
            return False
        
        try:
            with self.connection.cursor() as cursor:
                # Find old results
                cursor.execute("""
                    SELECT r.id, s.series_instance_uid, p.name as patient_name, 
                           CASE 
                               WHEN r.average_index_of_detectability IS NOT NULL THEN 'full_analysis'
                               ELSE 'global_noise'
                           END as calculation_type,
                           r.created_at
                    FROM analysis.results r
                    JOIN dicom.series s ON r.series_id_fk = s.id
                    JOIN dicom.study st ON s.study_id_fk = st.id
                    JOIN dicom.patient p ON st.patient_id_fk = p.id
                    WHERE r.created_at < NOW() - INTERVAL '%s days'
                    ORDER BY r.created_at
                """, (days_old,))
                
                old_results = cursor.fetchall()
                
                if not old_results:
                    print(f"No results older than {days_old} days found.")
                    return True
                
                print(f"Found {len(old_results)} results older than {days_old} days:")
                for result in old_results:
                    print(f"  - {result[1]} ({result[2]}) - {result[3]} - {result[4]}")
                
                if dry_run:
                    print("\nDRY RUN: No data was deleted. Use --no-dry-run to actually delete.")
                    return True
                
                # Confirm deletion
                confirm = input(f"\nDelete these {len(old_results)} results? (yes/no): ")
                if confirm.lower() != 'yes':
                    print("Deletion cancelled.")
                    return True
                
                # Delete old results
                deleted_count = 0
                for result in old_results:
                    cursor.execute("DELETE FROM analysis.results WHERE id = %s", (result[0],))
                    deleted_count += 1
                
                self.connection.commit()
                print(f"Successfully deleted {deleted_count} old results.")
                return True
                
        except Exception as e:
            print(f"Error during cleanup: {str(e)}")
            if self.connection:
                self.connection.rollback()
            return False
        finally:
            self.close()
    
    def find_duplicate_analyses(self):
        """Find series with duplicate analysis types"""
        if not self.connect():
            return None
        
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                # Check for duplicate results for the same series
                # Since we now have unique constraint on series_id_fk, this should not return results
                cursor.execute("""
                    SELECT s.series_instance_uid, COUNT(*) as count
                    FROM analysis.results r
                    JOIN dicom.series s ON r.series_id_fk = s.id
                    GROUP BY s.series_instance_uid, r.series_id_fk
                    HAVING COUNT(*) > 1
                    ORDER BY count DESC, s.series_instance_uid
                """)
                
                duplicates = cursor.fetchall()
                
                if duplicates:
                    print(f"Found {len(duplicates)} series with duplicate analyses:")
                    for dup in duplicates:
                        print(f"  - Series {dup['series_instance_uid']}: {dup['count']} analyses")
                else:
                    print("No duplicate analyses found.")
                
                return duplicates
        except Exception as e:
            print(f"Error finding duplicates: {str(e)}")
            return None
        finally:
            self.close()
    
    def export_analysis_summary(self, output_file=None):
        """Export analysis summary to CSV"""
        if not self.connect():
            return False
        
        try:
            with self.connection.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
                cursor.execute("""
                    SELECT 
                        s.series_instance_uid as series_id,
                        p.patient_id,
                        p.name as patient_name,
                        st.study_instance_uid as study_id,
                        st.institution_name,
                        sc.manufacturer,
                        sc.model_name as scanner_model,
                        sc.station_name,
                        s.protocol_name,
                        s.modality,
                        s.body_part_examined,
                        CASE 
                            WHEN r.average_index_of_detectability IS NOT NULL THEN 'Full Analysis'
                            ELSE 'Global Noise'
                        END as analysis_type,
                        r.ctdivol_avg,
                        r.ssde,
                        r.dlp,
                        r.dlp_ssde,
                        r.dw_avg,
                        r.spatial_resolution,
                        r.average_noise_level,
                        r.peak_frequency,
                        r.average_frequency,
                        r.percent_10_frequency,
                        r.average_index_of_detectability,
                        r.processing_time,
                        r.created_at
                    FROM analysis.results r
                    JOIN dicom.series s ON r.series_id_fk = s.id
                    JOIN dicom.study st ON s.study_id_fk = st.id
                    JOIN dicom.patient p ON st.patient_id_fk = p.id
                    JOIN dicom.scanner sc ON s.scanner_id_fk = sc.id
                    ORDER BY r.created_at DESC
                """)
                
                results = cursor.fetchall()
                
                if not results:
                    print("No analysis data found to export.")
                    return False
                
                # Convert to pandas DataFrame for easy CSV export
                df = pd.DataFrame([dict(row) for row in results])
                
                # Generate filename if not provided
                if not output_file:
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_file = f"cho_analysis_summary_{timestamp}.csv"
                
                df.to_csv(output_file, index=False)
                print(f"Exported {len(results)} series to {output_file}")
                return True
                
        except Exception as e:
            print(f"Error exporting data: {str(e)}")
            return False
        finally:
            self.close()
    
    def migrate_legacy_data(self, dry_run=True):
        """Check for any data integrity issues in new schema"""
        if not self.connect():
            return False
        
        try:
            with self.connection.cursor() as cursor:
                print("Checking data integrity in new schema...")
                
                # Check for orphaned records
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM analysis.results r
                    LEFT JOIN dicom.series s ON r.series_id_fk = s.id
                    WHERE s.id IS NULL
                """)
                orphaned_results = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM dicom.series s
                    LEFT JOIN dicom.study st ON s.study_id_fk = st.id
                    WHERE st.id IS NULL
                """)
                orphaned_series = cursor.fetchone()[0]
                
                cursor.execute("""
                    SELECT COUNT(*) 
                    FROM dicom.study st
                    LEFT JOIN dicom.patient p ON st.patient_id_fk = p.id
                    WHERE p.id IS NULL
                """)
                orphaned_studies = cursor.fetchone()[0]
                
                print(f"Data integrity check results:")
                print(f"  Orphaned analysis results: {orphaned_results}")
                print(f"  Orphaned series: {orphaned_series}")
                print(f"  Orphaned studies: {orphaned_studies}")
                
                if orphaned_results + orphaned_series + orphaned_studies == 0:
                    print("✓ No data integrity issues found")
                else:
                    print("⚠ Data integrity issues detected")
                
                return True
                
        except Exception as e:
            print(f"Error during integrity check: {str(e)}")
            return False
        finally:
            self.close()
    
    def generate_report(self):
        """Generate a comprehensive analysis report"""
        stats = self.get_database_stats()
        if not stats:
            print("Could not generate report - failed to get database stats.")
            return
        
        print("=" * 60)
        print("CHO ANALYSIS DATABASE REPORT - NEW SCHEMA")
        print("=" * 60)
        print(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print()
        
        print("OVERVIEW:")
        print(f"  Total Analyses: {stats['total_analyses']}")
        print(f"  Unique Series: {stats['unique_series']}")
        print(f"  Unique Patients: {stats['unique_patients']}")
        print(f"  Recent Activity (30 days): {stats['recent_analyses']}")
        print()
        
        print("ANALYSIS TYPES:")
        for calc_type, count in stats['by_calculation_type'].items():
            print(f"  {calc_type}: {count}")
        print()
        
        print("SERIES COMPLETION STATUS:")
        for status, count in stats['by_completion_status'].items():
            status_display = {
                'both': 'Both Complete',
                'global_only': 'Global Only',
                'full_only': 'Full Only',
                'none': 'No Analysis'
            }.get(status, status)
            print(f"  {status_display}: {count}")
        print()
        
        print("PROCESSING PERFORMANCE:")
        for calc_type, times in stats['processing_times'].items():
            print(f"  {calc_type}:")
            print(f"    Average Time: {times['avg_time']:.2f}s")
            print(f"    Min Time: {times['min_time']:.2f}s")
            print(f"    Max Time: {times['max_time']:.2f}s")
            print(f"    Total Runs: {times['count']}")
        print()
        
        print("STORAGE USAGE:")
        for table, size in stats['storage_usage'].items():
            print(f"  {table}: {size}")
        print()
        
        # Check for any issues
        duplicates = self.find_duplicate_analyses()
        if duplicates:
            print("ISSUES FOUND:")
            print(f"  {len(duplicates)} series have duplicate analyses")
            print("  Run with --find-duplicates for details")
        else:
            print("DATA INTEGRITY: No duplicate issues found")
        
        print("=" * 60)

def main():
    parser = argparse.ArgumentParser(description="CHO Analysis Database Management Utilities - New Schema")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Stats command
    stats_parser = subparsers.add_parser('stats', help='Show database statistics')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate comprehensive report')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export analysis summary to CSV')
    export_parser.add_argument('--output', '-o', help='Output file name')
    
    # Cleanup command
    cleanup_parser = subparsers.add_parser('cleanup', help='Clean up old results')
    cleanup_parser.add_argument('--days', '-d', type=int, default=365, help='Delete results older than this many days')
    cleanup_parser.add_argument('--no-dry-run', action='store_true', help='Actually delete data (default is dry run)')
    
    # Integrity check command
    integrity_parser = subparsers.add_parser('check-integrity', help='Check data integrity')
    
    # Find duplicates command
    duplicates_parser = subparsers.add_parser('find-duplicates', help='Find duplicate analyses')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    manager = CHODatabaseManager()
    
    if args.command == 'stats':
        stats = manager.get_database_stats()
        if stats:
            print(json.dumps(stats, indent=2, default=str))
    
    elif args.command == 'report':
        manager.generate_report()
    
    elif args.command == 'export':
        manager.export_analysis_summary(args.output)
    
    elif args.command == 'cleanup':
        manager.cleanup_old_results(args.days, not args.no_dry_run)
    
    elif args.command == 'check-integrity':
        manager.migrate_legacy_data()
    
    elif args.command == 'find-duplicates':
        manager.find_duplicate_analyses()

if __name__ == "__main__":
    main()