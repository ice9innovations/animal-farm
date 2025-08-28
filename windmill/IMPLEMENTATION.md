# Implementation Guide: Complete ML Processing Pipeline

**Quick start guide for deploying the distributed queue-based ML processing system**

## Prerequisites

### Infrastructure Requirements
- **RabbitMQ Server**: For message queuing (default: k2.local:192.168.0.122)
- **PostgreSQL Database**: For results storage (default: k1.local:192.168.0.121)  
- **MySQL Database**: For worker monitoring (default: localhost:3306)
- **ML Services**: 13 services running on various ports (see service_config.json)

### Software Requirements
```bash
# Python dependencies
pip install psycopg2-binary pika python-dotenv pillow requests mysql-connector-python

# Or from requirements.txt (if available)
pip install -r requirements.txt
```

## Step 1: Infrastructure Setup

### 1.1 Database Setup

#### **PostgreSQL Setup** (Main ML Pipeline)
```sql
-- Connect to PostgreSQL and create required tables
-- (Schema provided in distributed-database-strategy.md)

-- Core tables needed:
CREATE TABLE images (image_id SERIAL PRIMARY KEY, image_filename VARCHAR(255), image_path TEXT, image_url TEXT);
CREATE TABLE results (result_id SERIAL PRIMARY KEY, image_id INT, service VARCHAR(50), data JSONB, status VARCHAR(20), processing_time FLOAT, result_created TIMESTAMP DEFAULT NOW(), worker_id VARCHAR(50));
CREATE TABLE merged_boxes (merged_id SERIAL PRIMARY KEY, image_id INT, source_result_ids INT[], merged_data JSONB, status VARCHAR(20), created TIMESTAMP DEFAULT NOW(), worker_id VARCHAR(50));
CREATE TABLE consensus (consensus_id SERIAL PRIMARY KEY, image_id INT, consensus_data JSONB, processing_time FLOAT, consensus_created TIMESTAMP DEFAULT NOW());
CREATE TABLE postprocessing (postprocessing_id SERIAL PRIMARY KEY, image_id INT, merged_box_id INT, service VARCHAR(50), data JSONB, status VARCHAR(20), result_created TIMESTAMP DEFAULT NOW());
```

#### **MySQL Setup** (Worker Monitoring)
```sql
-- Create monitoring database
CREATE DATABASE monitoring;
USE monitoring;

-- Create monitoring user
CREATE USER 'worker_monitor'@'%' IDENTIFIED BY 'your_secure_monitoring_password';
GRANT SELECT, INSERT, UPDATE ON monitoring.* TO 'worker_monitor'@'%';
FLUSH PRIVILEGES;

-- Create monitoring table
CREATE TABLE worker_heartbeats (
    id INT AUTO_INCREMENT PRIMARY KEY,
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    worker_id VARCHAR(50),
    service_name VARCHAR(50),
    node_hostname VARCHAR(50),
    status ENUM('alive', 'starting', 'stopping', 'error'),
    jobs_completed INT,
    jobs_per_minute FLOAT,
    last_job_time TIMESTAMP,
    queue_depth INT,
    error_message TEXT,
    INDEX idx_worker_time (worker_id, timestamp),
    INDEX idx_service_time (service_name, timestamp)
);
```

### 1.2 RabbitMQ Setup
```bash
# Install and start RabbitMQ
sudo systemctl start rabbitmq-server
sudo systemctl enable rabbitmq-server

# Create user for the system
sudo rabbitmqctl add_user animal_farm your_secure_queue_password
sudo rabbitmqctl set_permissions -p / animal_farm ".*" ".*" ".*"

# Enable management console (optional)
sudo rabbitmq-plugins enable rabbitmq_management
```

### 1.3 ML Services Setup
Ensure all 13 ML services are running:
- BLIP (port 7777), CLIP (port 7778), Colors (port 7770)
- Detectron2 (port 7771), Face (port 7772), Inception_v3 (port 7779)
- Metadata (port 7781), OCR (port 7775), NSFW2 (port 7774)  
- Ollama (port 7782), Pose (port 7786), RT-DETR (port 7780), YOLOv8 (port 7773)

## Step 2: Configuration

### 2.1 Environment Configuration
```bash
# Copy example configuration
cp .env.example .env

# Edit .env with your infrastructure details
nano .env
```

**Required .env variables:**
```bash
# Infrastructure only (no service-specific config needed)
QUEUE_HOST=192.168.0.122
DB_HOST=192.168.0.121
DB_NAME=animal_farm
DB_USER=animal_farm_user
DB_PASSWORD=your_secure_db_password
QUEUE_USER=animal_farm
QUEUE_PASSWORD=your_secure_queue_password

# Worker configuration (optional)
WORKER_ID=worker_unique_name
LOG_LEVEL=INFO

# Optional monitoring (disabled by default)
# If ENABLE_MONITORING=true, ALL monitoring variables are REQUIRED
ENABLE_MONITORING=true
MONITORING_DB_HOST=192.168.0.101
MONITORING_DB_NAME=monitoring
MONITORING_DB_USER=worker_monitor
MONITORING_DB_PASSWORD=your_secure_monitoring_password
```

### 2.2 Service Configuration
Each box has its own `service_config.json` defining which services run locally:
```bash
# Example: Main workstation service_config.json
{
  "services": {
    "colors": {
      "host": "localhost",
      "port": 7770,
      "endpoint": "/analyze",
      "category": "primary",
      "description": "Color analysis"
    },
    "face": {
      "host": "localhost", 
      "port": 7772,
      "endpoint": "/analyze",
      "category": "spatial_only",
      "description": "Face detection"
    }
  }
}

# Check your service configuration
cat service_config.json
python generic_producer.py --list-services
```

**Box-Specific Configuration**: Each box only defines services it runs locally. No complex .env service variables needed!

### 2.3 Creating Box-Specific Configurations

**Production Pi Cluster Example:**
```bash
# k1.local (Database Pi) - service_config.json
{
  "services": {
    "consensus": {"host": "localhost", "port": 0, "category": "post_processing"}
  }
}

# k2.local (Queue Pi) - service_config.json  
{
  "services": {
    "spatial_enrichment": {"host": "localhost", "port": 0, "category": "post_processing"}
  }
}

# k3.local (GPU Pi) - service_config.json
{
  "services": {
    "blip": {"host": "localhost", "port": 7777, "category": "primary"},
    "clip": {"host": "localhost", "port": 7778, "category": "primary"},
    "yolov8": {"host": "localhost", "port": 7773, "category": "primary"}
  }
}

# Main workstation - service_config.json (most services)
{
  "services": {
    "colors": {"host": "localhost", "port": 7770, "category": "primary"},
    "face": {"host": "localhost", "port": 7772, "category": "spatial_only"},
    "pose": {"host": "localhost", "port": 7786, "category": "spatial_only"},
    "metadata": {"host": "localhost", "port": 7781, "category": "primary"}
  }
}
```

**Benefits:**
- ✅ Each box manages only its services  
- ✅ Same .env file across all boxes
- ✅ No service-specific environment variables
- ✅ Version control per deployment environment

## Step 3: Deployment Patterns

### 3.1 Single Machine Development
```bash
# Terminal 1: Start ML services (13 services on various ports)
# Terminal 2: Submit jobs (limit optional, mainly for testing)
# NOTE: 'all' excludes face/pose - they run via spatial_enrichment_worker  
python generic_producer.py --services all --group coco2017 --limit 1000

# Terminal 3-6: Start specialized workers
python bbox_merger_worker.py &
python consensus_worker.py &  
python spatial_enrichment_worker.py &

# Terminal 7+: Start ML service workers (as many as needed)
SERVICE_NAME=blip python generic_worker.py &
SERVICE_NAME=clip python generic_worker.py &
SERVICE_NAME=colors python generic_worker.py &
# ... repeat for all services
```

### 3.2 Pi Cluster Production
```bash
# k1.local (Database Pi): PostgreSQL + consensus worker
python consensus_worker.py

# k2.local (Queue Pi): RabbitMQ + spatial enrichment  
python spatial_enrichment_worker.py

# k3.local (GPU Pi): GPU-heavy services
SERVICE_NAME=blip python generic_worker.py &
SERVICE_NAME=clip python generic_worker.py &
SERVICE_NAME=yolov8 python generic_worker.py &

# k4.local (NPU Pi): Specialized services
SERVICE_NAME=rtdetr python generic_worker.py &
SERVICE_NAME=detectron2 python generic_worker.py &

# Main workstation: Remaining services + bbox merger
python bbox_merger_worker.py &
SERVICE_NAME=colors python generic_worker.py &
SERVICE_NAME=metadata python generic_worker.py &
# ... etc
```

### 3.3 Cloud Scaling
- **Auto-scaling groups** for generic_worker.py with different SERVICE_NAME
- **Managed RabbitMQ** (AWS MQ, Google Cloud Pub/Sub)
- **Managed PostgreSQL** (RDS, Cloud SQL)
- **GPU instances** for compute-heavy services (BLIP, CLIP)

## Step 4: Operation

### 4.1 Submit Processing Jobs
```bash
# Process ALL COCO2017 images through primary services (production)
# NOTE: 'all' excludes face/pose - they run via spatial_enrichment_worker
python generic_producer.py --services all --group coco2017

# Process specific image groups
python generic_producer.py --services all --group imagenet

# Service categories
python generic_producer.py --services primary --group coco2017      # Same as 'all'
python generic_producer.py --services full_catalog --group coco2017 # Includes face/pose (not recommended)

# Testing with limited images (limit is optional, for testing only)
python generic_producer.py --services blip,clip,colors --group coco2017 --limit 100

# Process specific service combinations for testing
python generic_producer.py --services yolov8,rtdetr,detectron2 --group coco2017 --limit 1000

# List available services by category
python generic_producer.py --list-services
```

### 4.2 Monitor Progress

#### **Worker Health Dashboard**
```bash
# Check overall worker health and performance
python monitor_workers.py

# Custom monitoring windows
python monitor_workers.py --dead-threshold 10 --performance-window 30 --errors-window 120

# Monitor specific aspects
python monitor_workers.py --dead-threshold 3    # Stricter dead worker detection
python monitor_workers.py --performance-window 60  # Longer performance window
```

#### **Queue and Database Status**
```bash
# Check queue depths
sudo rabbitmqctl list_queues name messages

# Check processing status in database
psql -h 192.168.0.121 -U animal_farm_user -d animal_farm -c "
SELECT service, COUNT(*) as total, 
       COUNT(*) FILTER (WHERE status = 'success') as success_count,
       COUNT(*) FILTER (WHERE status = 'error') as error_count
FROM results GROUP BY service ORDER BY service;
"

# Check post-processing status  
psql -h 192.168.0.121 -U animal_farm_user -d animal_farm -c "
SELECT COUNT(*) as total_merged_boxes FROM merged_boxes;
SELECT COUNT(*) as total_consensus FROM consensus;  
SELECT COUNT(*) as total_spatial_enrichments FROM postprocessing;
"
```

#### **Direct Database Monitoring Queries**
```bash
# Connect to monitoring database
mysql -h 192.168.0.101 -u worker_monitor -p monitoring

# Find dead workers
SELECT worker_id, service_name, node_hostname, 
       MAX(timestamp) as last_seen,
       TIMESTAMPDIFF(MINUTE, MAX(timestamp), NOW()) as minutes_silent
FROM worker_heartbeats 
GROUP BY worker_id 
HAVING minutes_silent > 5;

# Service performance summary
SELECT service_name, 
       AVG(jobs_per_minute) as avg_rate,
       COUNT(DISTINCT worker_id) as active_workers
FROM worker_heartbeats 
WHERE timestamp > DATE_SUB(NOW(), INTERVAL 10 MINUTE)
GROUP BY service_name;
```

### 4.3 Worker Management
```bash
# Check worker logs
tail -f worker.log

# Restart workers
pkill -f "python.*_worker.py"
# Then restart as needed

# Scale specific services
# Add more workers for bottleneck services:
SERVICE_NAME=yolov8 python generic_worker.py &
SERVICE_NAME=yolov8 python generic_worker.py &
SERVICE_NAME=yolov8 python generic_worker.py &
```

## Step 5: Troubleshooting

### 5.1 Common Issues

**Workers not processing:**
```bash
# Check RabbitMQ connection
curl -u animal_farm:your_secure_queue_password http://192.168.0.122:15672/api/queues

# Check service connectivity
curl http://localhost:7770/analyze  # Colors service example
```

**Database connection issues:**
```bash
# Test database connection
psql -h 192.168.0.121 -U animal_farm_user -d animal_farm -c "SELECT NOW();"
```

**Post-processing not running:**
```bash
# Check for new results
psql -h 192.168.0.121 -U animal_farm_user -d animal_farm -c "
SELECT COUNT(*) FROM results WHERE service IN ('yolov8','rtdetr','detectron2');
SELECT COUNT(*) FROM merged_boxes;
"

# Restart post-processing workers
python bbox_merger_worker.py &
python consensus_worker.py &
python spatial_enrichment_worker.py &
```

### 5.2 Performance Tuning

**Queue optimization:**
- Increase `WORKER_PREFETCH_COUNT` for faster workers
- Add more workers for bottleneck services
- Use `--delay` parameter in producer for rate limiting

**Database optimization:**
- Add indexes on frequently queried columns
- Monitor connection pool usage
- Use read replicas for monitoring queries

**Service optimization:**
- Distribute GPU-heavy services across hardware
- Use different timeout values per service type
- Implement service-specific retry strategies

## Step 6: Scaling Considerations

### 6.1 Horizontal Scaling
- **Worker scaling**: Add more `generic_worker.py` instances for bottleneck services
- **Queue partitioning**: Use multiple queue instances for high throughput
- **Database sharding**: Partition by image_id for massive scale

### 6.2 Performance Monitoring
- Queue depths and processing rates
- Worker success/failure rates  
- Database query performance
- Service response times

### 6.3 Fault Tolerance
- Worker auto-restart on failure
- Message acknowledgment for reliable processing
- Database transaction rollback on errors
- Service health checks and circuit breakers

---

## Quick Start Checklist

- [ ] Infrastructure running (RabbitMQ + PostgreSQL + 13 ML services)
- [ ] Database schema created
- [ ] `.env` file configured
- [ ] `service_config.json` verified
- [ ] Images loaded into database
- [ ] Test job submitted: `python generic_producer.py --services colors --limit 10`
- [ ] Workers started: `SERVICE_NAME=colors python generic_worker.py`
- [ ] Post-processing workers running
- [ ] Results monitored in database

**Success indicators:**
- Jobs appearing in RabbitMQ queues
- Results appearing in `results` table
- Merged boxes appearing in `merged_boxes` table
- Consensus appearing in `consensus` table
- No errors in worker logs

For detailed architecture information, see the main [README.md](README.md).