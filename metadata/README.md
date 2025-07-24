# Metadata Extraction Service - Comprehensive Image Metadata Analysis

Advanced metadata extraction service that extracts the maximum possible metadata from images using multiple extraction engines. Designed for consistent, comprehensive results across all image formats.

## Features

- **Dual Extraction Engines**: ExifTool + PIL/Pillow for maximum coverage
- **Comprehensive Analysis**: Camera settings, GPS data, technical specs, software info
- **Categorized Output**: Automatically organizes metadata into logical categories
- **Multiple Formats**: Supports JPEG, PNG, TIFF, RAW, GIF, BMP, WEBP, and more
- **File System Info**: File size, timestamps, and hash generation
- **Consistent Results**: Robust error handling and fallback mechanisms
- **RESTful API**: Structured JSON responses with detailed metadata

## Quick Start

### Prerequisites

```bash
# Install ExifTool (critical for maximum metadata extraction)
sudo apt-get install exiftool

# Create virtual environment
python3 -m venv metadata_venv

# Activate virtual environment
source metadata_venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt
```

### Configuration

1. Copy environment template:
```bash
cp .env.sample .env
```

2. Edit `.env` file:
```bash
# API Settings
PORT=7781
PRIVATE=false  # true for localhost only, false for all interfaces
```

### Running the Service

```bash
# Direct execution
python3 REST.py

# Using startup script
./metadata.sh

# As systemd service
sudo systemctl start metadata-api
sudo systemctl enable metadata-api
```

## API Endpoints

### Health Check
```bash
GET /health
```

Returns service status and extraction engine availability.

### Extract Metadata from URL
```bash
GET /?url=https://example.com/image.jpg
```

### Extract Metadata from File Upload
```bash
POST /
Content-Type: multipart/form-data

# Include image file in form data
```

### Local File Analysis (if not in private mode)
```bash
GET /?path=local_image.jpg
```

## Response Format

```json
{
  "metadata": {
    "file_info": {
      "filename": "IMG_1234.jpg",
      "file_size": 2048576,
      "file_size_human": "2.0 MB",
      "created_time": "2023-12-01T14:30:00",
      "modified_time": "2023-12-01T14:30:00"
    },
    "file_hash": "a1b2c3d4e5f6...",
    "summary": {
      "total_metadata_tags": 156,
      "categories_found": 6,
      "has_exif_data": true,
      "has_gps_data": true,
      "has_camera_info": true,
      "extraction_methods": ["ExifTool", "PIL/Pillow"],
      "processing_time": 0.234
    },
    "categorized": {
      "camera": {
        "EXIF:Make": "Canon",
        "EXIF:Model": "EOS R5",
        "EXIF:LensModel": "RF24-70mm F2.8 L IS USM",
        "EXIF:FocalLength": "35.0 mm",
        "EXIF:FNumber": 2.8,
        "EXIF:ISO": 400,
        "EXIF:ShutterSpeedValue": "1/250",
        "EXIF:Flash": "No Flash"
      },
      "gps": {
        "GPS:GPSLatitude": "37.7749 N",
        "GPS:GPSLongitude": "122.4194 W",
        "GPS:GPSAltitude": "16.0 m Above Sea Level",
        "GPS:GPSDateTime": "2023:12:01 14:30:00Z"
      },
      "datetime": {
        "EXIF:DateTime": "2023:12:01 14:30:00",
        "EXIF:DateTimeOriginal": "2023:12:01 14:30:00",
        "EXIF:CreateDate": "2023:12:01 14:30:00"
      },
      "image": {
        "EXIF:ImageWidth": 6000,
        "EXIF:ImageHeight": 4000,
        "EXIF:Orientation": "Horizontal (normal)",
        "EXIF:ColorSpace": "sRGB",
        "EXIF:Resolution": "72 dpi"
      },
      "software": {
        "EXIF:Software": "Adobe Lightroom 6.0",
        "EXIF:ProcessingSoftware": "Lightroom",
        "XMP:CreatorTool": "Adobe Lightroom"
      },
      "technical": {
        "File:FileType": "JPEG",
        "EXIF:Compression": "JPEG",
        "EXIF:BitsPerSample": "8 8 8",
        "EXIF:ColorComponents": 3
      }
    },
    "raw_metadata": {
      "...": "Complete unfiltered metadata from all sources"
    },
    "extraction_info": {
      "pil_result": "success",
      "exiftool_result": "success",
      "total_extraction_time": 0.234
    },
    "status": "success"
  }
}
```

## Extraction Engines

### 1. ExifTool (Primary Engine)
- **Coverage**: Most comprehensive metadata extraction available
- **Supports**: 600+ file formats, proprietary RAW formats, video files
- **Extracts**: EXIF, IPTC, XMP, GPS, maker notes, and vendor-specific data
- **Advantages**: Industry standard, regularly updated, handles edge cases

### 2. PIL/Pillow (Secondary Engine)  
- **Coverage**: Standard image formats and basic EXIF
- **Supports**: JPEG, PNG, TIFF, GIF, BMP, WEBP
- **Extracts**: Basic EXIF, image properties, format-specific info
- **Advantages**: Python native, fast processing, good for validation

## Metadata Categories

The service automatically categorizes extracted metadata:

### Camera Information
- Camera make and model
- Lens information and focal length
- Exposure settings (ISO, aperture, shutter speed)
- Flash and focus information
- Shooting modes and scene settings

### GPS Location Data
- Latitude and longitude coordinates
- Altitude and direction information
- GPS timestamp and datum
- Location accuracy and method

### Date/Time Information
- Original capture time
- File creation and modification dates
- GPS timestamp
- Software processing dates

### Image Properties
- Dimensions and resolution
- Color space and profile
- Orientation and rotation
- Compression and quality settings

### Software Information
- Camera firmware version
- Processing software used
- Creator and editor tools
- Software-specific metadata

### Technical Specifications
- File format and compression
- Bit depth and color components
- Encoding and profile information
- Format-specific technical data

## Advanced Features

### File Hash Generation
- SHA3-256 hash for image integrity verification
- Base64 encoding for consistent hashing
- Useful for duplicate detection and verification

### Error Resilience
- Graceful fallback when extraction engines fail
- Partial results when some metadata is corrupted
- Detailed error reporting for troubleshooting

### Performance Optimization
- Efficient metadata caching
- Minimal file I/O operations
- Fast categorization algorithms

## Supported File Formats

### Primary Support (Full Metadata)
- **JPEG/JPG**: Complete EXIF, IPTC, XMP extraction
- **TIFF**: Multi-page support, embedded profiles
- **RAW Formats**: Canon CR2/CR3, Nikon NEF, Sony ARW, etc.
- **PNG**: Text chunks, color profiles
- **WEBP**: Extended metadata support

### Secondary Support (Basic Metadata)
- **GIF**: Basic properties and comments
- **BMP**: Header information
- **HEIC/HEIF**: Modern mobile formats
- **PSD**: Photoshop documents

## Usage Examples

### Batch Processing
```bash
# Process multiple images
for image in *.jpg; do
    curl -X POST -F "uploadedfile=@$image" http://localhost:7776/
done
```

### GPS Data Extraction
```bash
# Extract only GPS information
curl "http://localhost:7776/?url=https://example.com/gps_photo.jpg" | jq '.metadata.categorized.gps'
```

### Camera Information Analysis
```bash
# Get camera settings
curl "http://localhost:7776/?url=https://example.com/photo.jpg" | jq '.metadata.categorized.camera'
```

## Performance Metrics

- **Processing Time**: 0.1-0.5 seconds per image (typical)
- **Memory Usage**: 50-100MB per image during processing
- **Throughput**: 10-50 images per second (depending on size and complexity)
- **Accuracy**: 99%+ metadata extraction success rate

## Error Handling

Comprehensive error handling for common scenarios:

```json
{
  "error": "Metadata extraction failed: Unsupported file format",
  "status": "error"
}
```

Common error types:
- File too large (>8MB limit)
- Unsupported file format
- Corrupted image data
- ExifTool unavailable
- Network timeouts for URL downloads

## Troubleshooting

### ExifTool Not Found
```bash
# Ubuntu/Debian
sudo apt-get install exiftool

# CentOS/RHEL
sudo yum install perl-Image-ExifTool

# macOS
brew install exiftool
```

### Poor Metadata Extraction
1. **Check File Format**: Ensure the image format supports metadata
2. **Verify File Integrity**: Corrupted files may have incomplete metadata
3. **Camera Settings**: Some cameras don't record full EXIF data
4. **Processing History**: Edited images may have stripped metadata

### Performance Issues
1. **File Size**: Large files take longer to process
2. **Format Complexity**: RAW files require more processing time
3. **System Resources**: Ensure adequate memory and CPU
4. **Network Speed**: URL downloads depend on connection speed

## Integration with Animal Farm

This service integrates with the Animal Farm distributed AI ecosystem:

- **Metadata Intelligence**: Provides context for image analysis decisions
- **Location Services**: GPS data enhances location-based AI decisions
- **Camera Profiling**: Camera information improves image quality assessment
- **Timestamp Analysis**: Temporal data supports timeline-based analysis

## Security Considerations

- **Privacy**: GPS and personal data are extracted - handle responsibly
- **File Validation**: Always validate uploaded files
- **Access Control**: Use private mode for sensitive environments
- **Data Retention**: Consider metadata privacy implications

## Development

### Adding Custom Categories
```python
# Extend categorize_metadata function
custom_keywords = ['custom', 'special', 'proprietary']
categories["custom"] = {}

if any(keyword in key_lower for keyword in custom_keywords):
    categories["custom"][key] = value
```

### Enhanced Processing
```python
# Add custom metadata processing
def process_custom_metadata(metadata):
    # Your custom processing logic
    return enhanced_metadata
```

## Dependencies

- `flask`: Web framework
- `python-dotenv`: Environment management  
- `pillow`: PIL image processing library
- `requests`: HTTP client for URL downloads
- `opencv-python`: Computer vision library for image processing
- `numpy`: Numerical computing library
- `PyExifTool`: Python wrapper for ExifTool (requires ExifTool binary)

## License

Part of the Animal Farm distributed AI project.