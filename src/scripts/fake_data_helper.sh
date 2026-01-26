#!/bin/bash

# CHO Fake Data Helper Script - Updated for New Schema
# Makes it easy to generate and manage fake CHO analysis data

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_DIR="$SCRIPT_DIR/../python"
FAKE_DATA_SCRIPT="$PYTHON_DIR/fake_data_manager.py"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if Python script exists
if [ ! -f "$FAKE_DATA_SCRIPT" ]; then
    print_error "Fake data script not found at $FAKE_DATA_SCRIPT"
    exit 1
fi

# Install requirements if needed
install_requirements() {
    print_status "Installing required Python packages..."
    pip install faker psycopg2-binary numpy
    print_success "Requirements installed"
}

# Test database connection
test_connection() {
    print_status "Testing database connection..."
    python -c "
import psycopg2
try:
    conn = psycopg2.connect(
        host='localhost',
        port=5433,
        database='orthanc',
        user='postgres',
        password='pgpassword',
        connect_timeout=5
    )
    conn.close()
    print('[SUCCESS] Database connection successful')
except Exception as e:
    print(f'[ERROR] Database connection failed: {e}')
    exit(1)
"
}

# Show usage
usage() {
    echo "CHO Fake Data Helper Script - Updated for New Schema"
    echo ""
    echo "Usage: $0 [command] [options]"
    echo ""
    echo "Commands:"
    echo "  install-reqs           Install required Python packages"
    echo "  test-db               Test database connection"
    echo "  demo-small            Generate 10 fake results (quick demo)"
    echo "  demo-medium           Generate 50 fake results (medium demo)"
    echo "  demo-large            Generate 100 fake results (large demo)"
    echo "  generate <count>      Generate specified number of fake results"
    echo "  generate-mixed <count> Generate mixed analysis types (70% full, 30% global)"
    echo "  list                  List all fake results in database"
    echo "  count                 Count fake results in database"
    echo "  delete                Delete all fake results (with confirmation)"
    echo "  delete-confirm        Delete all fake results (no confirmation)"
    echo "  status                Show database status"
    echo "  check-schema          Verify database schema is correct"
    echo ""
    echo "Examples:"
    echo "  $0 demo-medium        # Generate 50 fake results for testing"
    echo "  $0 generate 25        # Generate 25 fake results (50/50 mix)"
    echo "  $0 generate-mixed 40  # Generate 40 results (70% full analysis)"
    echo "  $0 list               # Show all fake results"
    echo "  $0 delete-confirm     # Delete all fake data immediately"
}

# Check database schema
check_schema() {
    print_status "Checking database schema..."
    python -c "
import psycopg2
try:
    conn = psycopg2.connect(
        host='localhost', port=5433, database='orthanc',
        user='postgres', password='pgpassword'
    )
    cursor = conn.cursor()
    
    # Check for required schemas
    cursor.execute(\"\"\"
        SELECT schema_name FROM information_schema.schemata 
        WHERE schema_name IN ('dicom', 'analysis')
    \"\"\")
    schemas = [row[0] for row in cursor.fetchall()]
    
    if 'dicom' in schemas and 'analysis' in schemas:
        print('[SUCCESS] Required schemas (dicom, analysis) found')
    else:
        print(f'[ERROR] Missing schemas. Found: {schemas}')
        exit(1)
        
    # Check for key tables
    tables_to_check = [
        ('dicom', 'patient'),
        ('dicom', 'scanner'), 
        ('dicom', 'study'),
        ('dicom', 'series'),
        ('dicom', 'ct_technique'),
        ('analysis', 'results')
    ]
    
    for schema, table in tables_to_check:
        cursor.execute(f\"\"\"
            SELECT EXISTS (
                SELECT FROM information_schema.tables 
                WHERE table_schema = '{schema}' AND table_name = '{table}'
            )
        \"\"\")
        if cursor.fetchone()[0]:
            print(f'[SUCCESS] Table {schema}.{table} exists')
        else:
            print(f'[ERROR] Table {schema}.{table} missing')
            exit(1)
    
    conn.close()
    print('[SUCCESS] Database schema verification complete')
    
except Exception as e:
    print(f'[ERROR] Schema check failed: {e}')
    exit(1)
"
}

# Count fake results
count_fake() {
    print_status "Counting fake results..."
    python "$FAKE_DATA_SCRIPT" list-fake | grep -E "Found [0-9]+ fake" || echo "No fake results found"
}

# Generate fake data with custom parameters
generate_custom() {
    local count=$1
    local ratio=${2:-0.5}
    
    if [ -z "$count" ]; then
        print_error "Count is required for generate command"
        echo "Usage: $0 generate <count> [full_analysis_ratio]"
        exit 1
    fi
    
    if ! [[ "$count" =~ ^[0-9]+$ ]] || [ "$count" -le 0 ]; then
        print_error "Count must be a positive integer"
        exit 1
    fi
    
    print_status "Generating $count fake CHO analysis results..."
    print_status "Full analysis ratio: $(echo "$ratio * 100" | bc -l 2>/dev/null | cut -d. -f1 || echo "$(python -c "print(int($ratio * 100))"))")%"
    
    python "$FAKE_DATA_SCRIPT" generate --count "$count" --full-analysis-ratio "$ratio"
    
    if [ $? -eq 0 ]; then
        print_success "Generated $count fake results successfully"
    else
        print_error "Failed to generate fake results"
        exit 1
    fi
}

# Main command handling
case "$1" in
    install-reqs)
        install_requirements
        ;;
    test-db)
        test_connection
        ;;
    check-schema)
        check_schema
        ;;
    demo-small)
        print_status "Generating small demo dataset (10 results)..."
        python "$FAKE_DATA_SCRIPT" quick-demo --size small
        print_success "Small demo dataset created"
        ;;
    demo-medium)
        print_status "Generating medium demo dataset (50 results)..."
        python "$FAKE_DATA_SCRIPT" quick-demo --size medium
        print_success "Medium demo dataset created"
        ;;
    demo-large)
        print_status "Generating large demo dataset (100 results)..."
        python "$FAKE_DATA_SCRIPT" quick-demo --size large
        print_success "Large demo dataset created"
        ;;
    generate)
        generate_custom "$2" "$3"
        ;;
    generate-mixed)
        generate_custom "$2" "0.7"  # 70% full analysis
        ;;
    list)
        print_status "Listing all fake results..."
        python "$FAKE_DATA_SCRIPT" list-fake
        ;;
    count)
        count_fake
        ;;
    delete)
        print_warning "This will show what would be deleted (dry run)"
        python "$FAKE_DATA_SCRIPT" delete-fake
        echo ""
        print_status "To actually delete, use: $0 delete-confirm"
        ;;
    delete-confirm)
        print_warning "This will permanently delete ALL fake data!"
        read -p "Are you sure? (y/N): " -n 1 -r
        echo
        if [[ $REPLY =~ ^[Yy]$ ]]; then
            print_status "Deleting all fake data..."
            python "$FAKE_DATA_SCRIPT" delete-fake --confirm
            print_success "All fake data deleted"
        else
            print_status "Deletion cancelled"
        fi
        ;;
    status)
        print_status "CHO Fake Data Status:"
        echo ""
        test_connection
        echo ""
        check_schema
        echo ""
        count_fake
        echo ""
        print_status "To generate demo data: $0 demo-medium"
        print_status "To delete fake data: $0 delete-confirm"
        ;;
    help|--help|-h)
        usage
        ;;
    "")
        print_error "No command specified"
        echo ""
        usage
        exit 1
        ;;
    *)
        print_error "Unknown command: $1"
        echo ""
        usage
        exit 1
        ;;
esac