#!/usr/bin/env python3
"""
Bounding Box Merger Worker - Harmonizes bbox results from yolo, rtdetr, detectron2
Triggers immediately after each bbox service completes for continuous harmonization
"""
import os
import json
import time
import logging
import socket
import psycopg2
import mysql.connector
from datetime import datetime
from dotenv import load_dotenv

class BoundingBoxMergerWorker:
    """Continuous bounding box harmonization worker"""
    
    def __init__(self):
        # Load configuration
        if not load_dotenv():
            raise ValueError("Could not load .env file")
        
        # Database configuration
        self.db_host = self._get_required('DB_HOST')
        self.db_name = self._get_required('DB_NAME')
        self.db_user = self._get_required('DB_USER')
        self.db_password = self._get_required('DB_PASSWORD')
        
        # Worker configuration
        self.worker_id = os.getenv('WORKER_ID', f'bbox_merger_{int(time.time())}')
        self.scan_interval = int(os.getenv('SCAN_INTERVAL', '3'))  # seconds
        self.batch_size = int(os.getenv('BATCH_SIZE', '50'))
        
        # Bounding box services to harmonize
        self.bbox_services = ['yolov8', 'rtdetr', 'detectron2']
        
        # Monitoring configuration  
        self.enable_monitoring = os.getenv('ENABLE_MONITORING', 'false').lower() == 'true'
        if self.enable_monitoring:
            self.monitoring_db_host = self._get_required('MONITORING_DB_HOST')
            self.monitoring_db_user = self._get_required('MONITORING_DB_USER') 
            self.monitoring_db_password = self._get_required('MONITORING_DB_PASSWORD')
            self.monitoring_db_name = self._get_required('MONITORING_DB_NAME')
        else:
            self.monitoring_db_host = None
            self.monitoring_db_user = None
            self.monitoring_db_password = None
            self.monitoring_db_name = None
        
        # Logging
        self.setup_logging()
        self.db_conn = None
        
        # Initialize monitoring
        self.setup_monitoring()
    
    def _get_required(self, key):
        """Get required environment variable or raise error"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} not set")
        return value
        
    def setup_logging(self):
        """Configure logging"""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('bbox_merger')
    
    def setup_monitoring(self):
        """Initialize monitoring connection"""
        self.mysql_conn = None
        self.last_heartbeat = 0
        self.jobs_completed = 0
        self.start_time = time.time()
        self.hostname = socket.gethostname()
        
        if self.enable_monitoring and self.monitoring_db_password:
            try:
                self.mysql_conn = mysql.connector.connect(
                    host=self.monitoring_db_host,
                    user=self.monitoring_db_user,
                    password=self.monitoring_db_password,
                    database=self.monitoring_db_name,
                    autocommit=True
                )
                self.send_heartbeat('starting')
            except Exception as e:
                self.logger.warning(f"Could not connect to monitoring database: {e}")
    
    def send_heartbeat(self, status, error_msg=None):
        """Send heartbeat to monitoring database"""
        if not self.enable_monitoring or not self.mysql_conn:
            return
        
        try:
            runtime_minutes = max((time.time() - self.start_time) / 60, 0.1)
            jobs_per_minute = self.jobs_completed / runtime_minutes
            
            cursor = self.mysql_conn.cursor()
            cursor.execute("""
                INSERT INTO worker_heartbeats 
                (worker_id, service_name, node_hostname, status, jobs_completed, 
                 jobs_per_minute, last_job_time, error_message)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """, (
                self.worker_id,
                'bbox_merger',
                self.hostname,
                status,
                self.jobs_completed,
                round(jobs_per_minute, 2),
                datetime.now() if self.jobs_completed > 0 else None,
                error_msg
            ))
            
            self.last_heartbeat = time.time()
            
        except Exception as e:
            self.logger.warning(f"Failed to send heartbeat: {e}")
    
    def maybe_send_heartbeat(self):
        """Send heartbeat if enough time has passed"""
        if time.time() - self.last_heartbeat > 120:  # 2 minutes
            self.send_heartbeat('alive')
    
    def connect_to_database(self):
        """Connect to PostgreSQL database"""
        try:
            self.db_conn = psycopg2.connect(
                host=self.db_host,
                database=self.db_name,
                user=self.db_user,
                password=self.db_password
            )
            self.db_conn.autocommit = False
            self.logger.info(f"Connected to PostgreSQL at {self.db_host}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    def find_images_needing_bbox_merge(self):
        """Find images with any new bbox results since last merge"""
        try:
            cursor = self.db_conn.cursor()
            
            # Much simpler: find ANY image with bbox results that are newer than merged_boxes
            # This runs harmonization after EVERY bbox service completion, not waiting for "all"
            query = """
                SELECT DISTINCT r.image_id, i.image_filename
                FROM results r
                JOIN images i ON r.image_id = i.image_id
                WHERE r.service IN %s
                AND r.status = 'success'
                AND (
                    -- No merged boxes exist yet - harmonize whatever we have
                    NOT EXISTS (
                        SELECT 1 FROM merged_boxes mb 
                        WHERE mb.image_id = r.image_id
                        AND mb.status = 'success'
                    )
                    -- OR any bbox result is newer than merged boxes - reharmonize immediately
                    OR r.result_created > (
                        SELECT COALESCE(MAX(mb.created), '1970-01-01'::timestamp)
                        FROM merged_boxes mb
                        WHERE mb.image_id = r.image_id
                        AND mb.status = 'success'
                    )
                )
                ORDER BY r.image_id  -- Order by something in SELECT for DISTINCT
                LIMIT %s
            """
            
            bbox_services_tuple = tuple(self.bbox_services)
            cursor.execute(query, (bbox_services_tuple, self.batch_size))
            results = cursor.fetchall()
            cursor.close()
            
            return [(row[0], row[1]) for row in results]
            
        except Exception as e:
            self.logger.error(f"Error finding images needing bbox merge: {e}")
            return []
    
    def get_bbox_results_for_image(self, image_id):
        """Get all bbox service results for an image"""
        try:
            cursor = self.db_conn.cursor()
            
            query = """
                SELECT service, data, result_id
                FROM results
                WHERE image_id = %s 
                AND service IN %s 
                AND status = 'success'
                ORDER BY service
            """
            
            cursor.execute(query, (image_id, tuple(self.bbox_services)))
            results = cursor.fetchall()
            cursor.close()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting bbox results for image {image_id}: {e}")
            return []
    
    def harmonize_bounding_boxes(self, bbox_results):
        """Harmonize bounding boxes using BoundingBoxService logic"""
        if not bbox_results:
            return None
        
        # Convert database results to service format
        all_detections = []
        source_result_ids = []
        
        for service, data, result_id in bbox_results:
            source_result_ids.append(result_id)
            
            if not isinstance(data, dict) or 'predictions' not in data:
                continue
            
            predictions = data['predictions']
            for prediction in predictions:
                if not prediction.get('bbox'):
                    continue
                
                # Extract detection with harmonized format
                detection = {
                    'service': service,
                    'label': prediction.get('label', ''),
                    'emoji': prediction.get('emoji', ''),
                    'bbox': prediction['bbox'],
                    'confidence': prediction.get('confidence', 0.0),
                    'type': prediction.get('type', 'object_detection')
                }
                all_detections.append(detection)
        
        if not all_detections:
            return None
        
        # Apply BoundingBoxService harmonization logic
        grouped_objects = self.group_by_label_with_cross_service_clustering(all_detections)
        
        # Package harmonized results
        harmonized_data = {
            'all_detections': all_detections,
            'grouped_objects': grouped_objects,
            'source_services': list(set(r[0] for r in bbox_results)),
            'total_detections': len(all_detections),
            'harmonization_algorithm': 'cross_service_clustering_v1',
            'source_result_ids': source_result_ids
        }
        
        return harmonized_data
    
    def group_by_label_with_cross_service_clustering(self, detections):
        """Simplified version of BoundingBoxService clustering logic"""
        groups = {}
        
        # Step 1: Group by label/emoji
        for detection in detections:
            key = detection['type'] if detection['type'] == 'face_detection' else detection['label']
            if key not in groups:
                groups[key] = {
                    'label': detection['label'],
                    'emoji': detection['emoji'],
                    'type': detection['type'],
                    'detections': [],
                    'instances': []
                }
            groups[key]['detections'].append(detection)
        
        # Step 2: Create cross-service instances for each group
        for group_key, group in groups.items():
            if group['detections']:
                group['instances'] = self.create_cross_service_instances(
                    group['detections'], 
                    group['emoji']
                )
        
        return groups
    
    def create_cross_service_instances(self, detections, emoji):
        """Create instances with cross-service clustering"""
        if not detections:
            return []
        
        # Find clusters of overlapping detections
        clusters = self.find_cross_service_clusters(detections)
        instances = []
        
        for i, cluster in enumerate(clusters):
            # Clean cluster (remove duplicates, filter weak detections)
            cleaned_cluster = self.clean_cluster(cluster)
            if not cleaned_cluster:
                continue
            
            # Calculate merged bounding box
            if len(cleaned_cluster) == 1:
                merged_bbox = cleaned_cluster[0]['bbox']
            else:
                boxes = [d['bbox'] for d in cleaned_cluster]
                x1 = min(b['x'] for b in boxes)
                y1 = min(b['y'] for b in boxes)
                x2 = max(b['x'] + b['width'] for b in boxes)
                y2 = max(b['y'] + b['height'] for b in boxes)
                
                merged_bbox = {
                    'x': x1,
                    'y': y1,
                    'width': x2 - x1,
                    'height': y2 - y1
                }
            
            # Create instance
            services = list(set(d['service'] for d in cleaned_cluster))
            avg_confidence = sum(d['confidence'] for d in cleaned_cluster) / len(cleaned_cluster)
            
            instance = {
                'cluster_id': f"{cleaned_cluster[0]['label']}_{i+1}",
                'emoji': emoji,
                'label': cleaned_cluster[0]['label'],
                'merged_bbox': merged_bbox,
                'detection_count': len(cleaned_cluster),
                'avg_confidence': round(avg_confidence, 3),
                'contributing_services': services,
                'detections': [{'service': d['service'], 'confidence': d['confidence']} 
                             for d in cleaned_cluster]
            }
            instances.append(instance)
        
        return instances
    
    def find_cross_service_clusters(self, detections):
        """Find clusters using IoU overlap"""
        clusters = []
        used = set()
        
        for i, detection in enumerate(detections):
            if i in used:
                continue
            
            cluster = [detection]
            used.add(i)
            
            # Find overlapping detections
            for j in range(i + 1, len(detections)):
                if j in used:
                    continue
                
                overlap = self.calculate_overlap_ratio(detection['bbox'], detections[j]['bbox'])
                if overlap > 0.3:  # 30% overlap threshold
                    cluster.append(detections[j])
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def clean_cluster(self, cluster):
        """Remove same-service duplicates and filter weak single detections"""
        if not cluster:
            return None
        
        # Group by service, keep highest confidence per service
        service_groups = {}
        for detection in cluster:
            service = detection['service']
            if service not in service_groups:
                service_groups[service] = []
            service_groups[service].append(detection)
        
        cleaned = []
        for service, detections in service_groups.items():
            if len(detections) > 1:
                # Keep highest confidence
                best = max(detections, key=lambda d: d['confidence'])
                cleaned.append(best)
            else:
                cleaned.append(detections[0])
        
        # Filter single weak detections
        if len(cleaned) == 1 and cleaned[0]['confidence'] < 0.85:
            return None
        
        return cleaned
    
    def calculate_overlap_ratio(self, box1, box2):
        """Calculate IoU overlap ratio"""
        x1 = max(box1['x'], box2['x'])
        y1 = max(box1['y'], box2['y'])
        x2 = min(box1['x'] + box1['width'], box2['x'] + box2['width'])
        y2 = min(box1['y'] + box1['height'], box2['y'] + box2['height'])
        
        if x1 >= x2 or y1 >= y2:
            return 0
        
        intersection = (x2 - x1) * (y2 - y1)
        area1 = box1['width'] * box1['height']
        area2 = box2['width'] * box2['height']
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def update_merged_boxes_for_image(self, image_id, image_filename):
        """Update merged boxes for a single image using safe DELETE+INSERT pattern"""
        try:
            # Get bbox results
            bbox_results = self.get_bbox_results_for_image(image_id)
            if not bbox_results:
                self.logger.debug(f"No bbox results for image {image_id}, skipping")
                return False
            
            # Harmonize bounding boxes
            start_time = time.time()
            merged_data = self.harmonize_bounding_boxes(bbox_results)
            processing_time = time.time() - start_time
            
            if not merged_data:
                self.logger.warning(f"Could not harmonize boxes for image {image_id}")
                return False
            
            # Safe atomic DELETE + INSERT with foreign key handling
            cursor = self.db_conn.cursor()
            
            # Step 1: Clear postprocessing references to avoid FK constraint violation
            cursor.execute("""
                UPDATE postprocessing 
                SET merged_box_id = NULL 
                WHERE merged_box_id IN (
                    SELECT merged_id FROM merged_boxes WHERE image_id = %s
                )
            """, (image_id,))
            
            # Step 2: Delete old merged boxes (now safe)
            cursor.execute("DELETE FROM merged_boxes WHERE image_id = %s", (image_id,))
            deleted_count = cursor.rowcount
            
            # Step 3: Insert new merged boxes
            cursor.execute("""
                INSERT INTO merged_boxes (image_id, source_result_ids, merged_data, worker_id, status)
                VALUES (%s, %s, %s, %s, %s)
                RETURNING merged_id
            """, (
                image_id, 
                merged_data['source_result_ids'],
                json.dumps(merged_data),
                self.worker_id,
                'success'
            ))
            
            # Get the new merged_id for potential postprocessing re-linking
            new_merged_id = cursor.fetchone()[0]
            
            cursor.close()
            
            services = merged_data['source_services']
            detection_count = merged_data['total_detections']
            
            if deleted_count > 0:
                self.logger.info(f"Reharmonized boxes for {image_filename} ({detection_count} detections from {services})")
            else:
                self.logger.info(f"Harmonized boxes for {image_filename} ({detection_count} detections from {services})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating merged boxes for image {image_id}: {e}")
            return False
    
    def run_bbox_merge_batch(self):
        """Process one batch of images needing bbox harmonization"""
        try:
            # Find images needing merge
            images = self.find_images_needing_bbox_merge()
            if not images:
                return 0
            
            self.logger.info(f"Processing bbox merge for {len(images)} images")
            
            # Process each image in a transaction
            success_count = 0
            for image_id, image_filename in images:
                try:
                    if self.update_merged_boxes_for_image(image_id, image_filename):
                        self.db_conn.commit()
                        success_count += 1
                        self.jobs_completed += 1
                    else:
                        self.db_conn.rollback()
                except Exception as e:
                    self.db_conn.rollback()
                    self.logger.error(f"Transaction failed for image {image_id}: {e}")
            
            self.logger.info(f"Merged boxes for {success_count}/{len(images)} images")
            return success_count
            
        except Exception as e:
            self.logger.error(f"Error in bbox merge batch: {e}")
            return 0
    
    def run_continuous(self):
        """Main continuous processing loop"""
        self.logger.info(f"Starting bounding box merger worker ({self.worker_id})")
        self.logger.info(f"Scan interval: {self.scan_interval}s, Batch size: {self.batch_size}")
        self.logger.info(f"Monitoring services: {', '.join(self.bbox_services)}")
        
        try:
            while True:
                processed = self.run_bbox_merge_batch()
                
                # Send monitoring heartbeat periodically
                self.maybe_send_heartbeat()
                
                # Adaptive scanning based on workload
                if processed > 0:
                    time.sleep(1)  # Quick scan when busy
                else:
                    time.sleep(self.scan_interval)  # Normal scan when idle
                    
        except KeyboardInterrupt:
            self.logger.info("Stopping bbox merger worker...")
        except Exception as e:
            self.logger.error(f"Fatal error in continuous loop: {e}")
            raise
        finally:
            # Clean shutdown monitoring
            if self.enable_monitoring and self.mysql_conn:
                self.send_heartbeat('stopping')
                self.mysql_conn.close()
            
            if self.db_conn:
                self.db_conn.close()
            self.logger.info("Bbox merger worker stopped")
    
    def run(self):
        """Main entry point"""
        if not self.connect_to_database():
            return 1
        
        self.run_continuous()
        return 0

def main():
    """Main entry point"""
    try:
        worker = BoundingBoxMergerWorker()
        return worker.run()
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Bbox merger worker error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())