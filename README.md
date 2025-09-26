# ORTHANC CHO Analysis Platform

A comprehensive Docker-based DICOM server with advanced image quality analysis capabilities, built on ORTHANC with custom Python plugins for medical imaging research and clinical applications.

[![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)](https://docker.com)
[![ORTHANC](https://img.shields.io/badge/ORTHANC-Medical%20DICOM-green.svg)](https://www.orthanc-server.com/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-Database-blue.svg)](https://postgresql.org)
[![MinIO](https://img.shields.io/badge/MinIO-S3%20Storage-orange.svg)](https://min.io)

## 🌟 Key Features

- **🏥 DICOM Server**: Full-featured ORTHANC DICOM server with C-STORE, C-FIND, C-MOVE support
- **🔬 CHO Analysis**: Advanced Channelized Hotelling Observer for CT image quality assessment
- **📊 Real-time Dashboard**: Modern web interface with live progress tracking and statistics
- **🗄️ Database Integration**: PostgreSQL for metadata and results storage
- **☁️ Object Storage**: MinIO S3-compatible storage for DICOM files and analysis artifacts
- **🖥️ OHIF Viewer**: Integrated advanced DICOM viewer for image visualization
- **📈 Analytics**: Comprehensive filtering, export, and statistical analysis tools
- **⚡ Performance Testing**: Built-in stress testing tools for system validation

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Web Dashboard │    │  ORTHANC Server │    │   PostgreSQL    │
│  (Port 8042)    │◄──►│  - DICOM C-*    │◄──►│   Database      │
│  - Statistics   │    │  - Python API   │    │  (Port 5433)    │
│  - CHO Analysis │    │  - REST API     │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                        │                        │
         │              ┌─────────────────┐                │
         └─────────────►│     MinIO       │◄───────────────┘
                        │  S3 Storage     │
                        │  (Port 9000)    │
                        └─────────────────┘
```

## 🚀 Quick Start

### Prerequisites

- Docker and Docker Compose
- 8GB+ RAM recommended
- 50GB+ storage space

### 1. Clone and Setup

```bash
git clone <repository-url>
cd orthanc-cho-platform
```

### 2. Configure Environment

Update the volume paths in `docker-compose.yml` to match your system:

```yaml
volumes:
  - D:/docker/postgres-data:/var/lib/postgresql/data # Windows
  - ./data/postgres:/var/lib/postgresql/data # Linux/Mac
```

### 3. Start the Platform

```bash
# Start all services
docker-compose up -d

# View logs
docker-compose logs -f

# Check service health
docker-compose ps
```

### 4. Access the Platform

- **ORTHANC Web Interface**: http://localhost:8042 (demo/demo)
- **CHO Analysis Dashboard**: http://localhost:8042/cho-dashboard
- **OHIF DICOM Viewer**: http://localhost:8042/ohif
- **MinIO Console**: http://localhost:9001 (minio/minio12345)
- **PostgreSQL**: localhost:5433 (postgres/pgpassword)

## 📖 Usage Guide

### DICOM Operations

**Upload DICOM Files:**

```bash
# Using ORTHANC REST API
curl -X POST http://localhost:8042/instances \
     -u demo:demo \
     -H "Content-Type: application/dicom" \
     --data-binary @your-dicom-file.dcm

# Using C-STORE (from DICOM workstation)
# AE Title: ORTHANC, Port: 4242
```

**Query/Retrieve:**

```bash
# Get all patients
curl -u demo:demo http://localhost:8042/patients

# Get patient studies
curl -u demo:demo http://localhost:8042/patients/{patient-id}/studies
```

### CHO Image Quality Analysis

The platform provides advanced Channelized Hotelling Observer (CHO) analysis for CT image quality assessment:

**Features:**

- **Detectability Analysis**: Low-contrast object detection performance
- **Global Noise Analysis**: Image noise characteristics and uniformity
- **Patient-Specific Parameters**: Customizable analysis parameters per series
- **Real-time Progress**: Live updates during analysis execution

**Starting CHO Analysis:**

1. Navigate to the CHO Dashboard: http://localhost:8042/cho-dashboard
2. Select a CT series from the results table
3. Click "CHO Analysis" button
4. Configure analysis parameters:
   - **Test Type**: Detectability, Global Noise, or Both
   - **ROI Parameters**: Region of interest dimensions and spacing
   - **Noise Parameters**: Frequency bands and analysis regions
5. Monitor progress in real-time
6. Export results as CSV for further analysis

**Programmatic Access:**

```bash
# Check analysis status
curl -u demo:demo http://localhost:8042/cho-calculation-status?series_id={series-id}

# Get results
curl -u demo:demo http://localhost:8042/cho-results/{series-id}
```

### Dashboard Features

**Filtering and Search:**

- Patient name/ID search
- Institute, scanner model, and protocol filters
- Date range filtering
- Advanced multi-field filtering

**Data Export:**

- CSV export of filtered results
- Comprehensive metadata inclusion
- Analysis results and parameters

**Statistics and Monitoring:**

- Real-time system status
- Analysis completion rates
- Error tracking and reporting

## ⚡ Performance Testing

The platform includes comprehensive stress testing tools:

```bash
# Quick connectivity test
./src/scripts/stress_test_runner.sh test-connection

# Predefined test scenarios
./src/scripts/stress_test_runner.sh quick-test      # 10 series, 2 workers
./src/scripts/stress_test_runner.sh medium-test     # 50 series, 4 workers
./src/scripts/stress_test_runner.sh heavy-test      # 100 series, 8 workers

# Custom test
./src/scripts/stress_test_runner.sh custom \
  --template /path/to/dicom/template \
  --num-series 25 \
  --workers 4 \
  --noise 0.1 \
  --delay 2
```

**Test Parameters:**

- `--template`: Path to DICOM template file
- `--num-series`: Number of synthetic series to generate
- `--workers`: Concurrent upload workers
- `--noise`: Noise level for synthetic data (0.0-1.0)
- `--delay`: Delay between uploads (seconds)
- `--max-series`: Maximum series to keep on disk

## 🔧 Configuration

### ORTHANC Configuration

Edit `orthanc/orthanc.json`:

```json
{
  "Name": "orthanc-python",
  "RemoteAccessAllowed": true,
  "AuthenticationEnabled": true,
  "DicomPort": 4242,
  "HttpPort": 8042,
  "RegisteredUsers": {
    "demo": "demo"
  },
  "Plugins": [
    "/usr/local/share/orthanc/plugins/libOrthancPython.so",
    "/usr/local/share/orthanc/plugins/libOrthancOHIF.so",
    "/usr/local/share/orthanc/plugins/libOrthancDicomWeb.so"
  ],
  "PythonScript": "/src/python/main.py"
}
```

### Database Schema

The platform automatically creates database schemas:

- **DICOM Schema**: Patient, study, series, and scan metadata
- **Analysis Schema**: CHO results, parameters, and execution logs

### Environment Variables

Key environment variables in `docker-compose.yml`:

```yaml
environment:
  # ORTHANC Settings
  ORTHANC__STABLE_AGE: 10
  VERBOSE_ENABLED: "true"
  DEBUG_MODE: "true"

  # MinIO Integration
  MINIO_ENDPOINT: "minio:9000"
  MINIO_ACCESS_KEY: "minio"
  MINIO_SECRET_KEY: "minio12345"

  # Analysis Settings
  AUTO_ANALYZE: "true"
```

## 🔍 API Reference

### CHO Analysis Endpoints

| Endpoint                   | Method | Description              |
| -------------------------- | ------ | ------------------------ |
| `/cho-results/{series-id}` | GET    | Get analysis results     |
| `/cho-results/{series-id}` | DELETE | Delete results           |
| `/cho-calculation-status`  | GET    | Check analysis status    |
| `/cho-active-calculations` | GET    | List active analyses     |
| `/cho-results-export`      | GET    | Export results as CSV    |
| `/cho-dashboard`           | GET    | Main dashboard interface |

### DICOM Web Endpoints

| Endpoint                               | Method | Description       |
| -------------------------------------- | ------ | ----------------- |
| `/dicom-web/studies`                   | GET    | QIDO-RS studies   |
| `/dicom-web/studies/{study}/series`    | GET    | QIDO-RS series    |
| `/dicom-web/studies/{study}/instances` | GET    | QIDO-RS instances |
| `/dicom-web/studies/{study}`           | GET    | WADO-RS study     |

### Standard ORTHANC REST API

Full ORTHANC REST API available at `/patients`, `/studies`, `/series`, `/instances`

## 🔧 Development

### Custom Python Plugins

Add custom functionality by extending `/src/python/main.py`:

```python
import orthanc

def CustomRestCallback(output, url, **request):
    """Custom REST endpoint handler"""
    # Your custom logic here
    output.AnswerBuffer("Custom response", "text/plain")

# Register the callback
orthanc.RegisterRestCallback('/custom-endpoint', CustomRestCallback)
```

### Frontend Extensions

Extend the dashboard by modifying:

- `/src/static/js/dashboard.js` - Main dashboard logic
- `/src/static/css/dashboard.css` - Styling
- `/src/templates/dashboard.html` - HTML structure

### Database Extensions

Add custom tables by extending:

- `/src/python/results_storage.py` - Database operations
- SQL schema files for new table definitions

## 📊 Monitoring and Troubleshooting

### Health Checks

Monitor service health:

```bash
# Check all services
docker-compose ps

# View service logs
docker-compose logs orthanc
docker-compose logs postgres
docker-compose logs minio

# Database connectivity test
curl -u demo:demo http://localhost:8042/system
```

### Common Issues

**ORTHANC won't start:**

- Check PostgreSQL connectivity
- Verify volume mount permissions
- Review ORTHANC logs for Python errors

**CHO Analysis fails:**

- Ensure sufficient memory (8GB+ recommended)
- Check series contains valid CT images
- Verify analysis parameters are within valid ranges

**Database connection errors:**

- Confirm PostgreSQL is healthy: `docker-compose exec postgres pg_isready`
- Check database credentials in environment variables
- Verify network connectivity between containers

### Performance Optimization

**For High-Volume Deployments:**

1. **Database Tuning**:

   ```bash
   # Increase PostgreSQL memory settings
   shared_buffers = 256MB
   effective_cache_size = 1GB
   work_mem = 4MB
   ```

2. **ORTHANC Optimization**:

   ```json
   {
     "MaximumStorageSize": 0,
     "MaximumPatientCount": 0,
     "StorageCompression": true
   }
   ```

3. **Resource Allocation**:
   ```yaml
   services:
     orthanc:
       deploy:
         resources:
           limits:
             memory: 4G
             cpus: "2.0"
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Submit a pull request

### Code Style

- Python: Follow PEP 8
- JavaScript: Use ES6+ features
- CSS: Follow BEM methodology
- Documentation: Use clear, concise language

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

- **Issues**: Report bugs and feature requests via GitHub Issues
- **Documentation**: Additional documentation available in `/docs`
- **Community**: Join our discussion forum for questions and support

## 🎯 Roadmap

- [ ] Multi-tenant support
- [ ] Advanced ML-based image analysis
- [ ] Cloud deployment templates (AWS, Azure, GCP)
- [ ] Enhanced DICOM SR support
- [ ] Integration with additional viewers (Cornerstone3D)
- [ ] Automated report generation

---

**Built with ❤️ for the medical imaging community**
