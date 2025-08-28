#!/usr/bin/env python3
"""
Spatial Enrichment Worker - Post-processes harmonized bounding boxes
Scans merged_boxes and adds face + pose data for person boxes, colors for all boxes
"""
import os
import json
import time
import logging
import psycopg2
import requests
from datetime import datetime
from dotenv import load_dotenv
from PIL import Image
import io

class SpatialEnrichmentWorker:
    """Post-processing worker for spatial enrichment of harmonized bounding boxes"""
    
    def __init__(self):
        # Load configuration
        if not load_dotenv():
            raise ValueError("Could not load .env file")
        
        # Load service definitions
        with open('service_config.json', 'r') as f:
            self.service_definitions = json.load(f)['services']
        
        # Database configuration
        self.db_host = self._get_required('DB_HOST')
        self.db_name = self._get_required('DB_NAME')
        self.db_user = self._get_required('DB_USER')
        self.db_password = self._get_required('DB_PASSWORD')
        
        # Services for enrichment (from service_config.json)
        face_config = self.service_definitions.get('face', {})
        pose_config = self.service_definitions.get('pose', {})
        colors_config = self.service_definitions.get('colors', {})
        
        self.face_service_url = f"http://{face_config.get('host', 'localhost')}:{face_config.get('port', 7772)}{face_config.get('endpoint', '/analyze')}"
        self.pose_service_url = f"http://{pose_config.get('host', 'localhost')}:{pose_config.get('port', 7786)}{pose_config.get('endpoint', '/analyze')}"
        self.colors_service_url = f"http://{colors_config.get('host', 'localhost')}:{colors_config.get('port', 7770)}{colors_config.get('endpoint', '/analyze')}"
        
        # Worker configuration
        self.worker_id = os.getenv('WORKER_ID', f'spatial_enrichment_{int(time.time())}')
        self.scan_interval = int(os.getenv('SCAN_INTERVAL', '5'))  # seconds
        self.batch_size = int(os.getenv('BATCH_SIZE', '20'))
        self.request_timeout = int(os.getenv('REQUEST_TIMEOUT', '15'))
        
        # Person detection labels that should get face/pose enrichment
        self.person_labels = ['person', 'human', 'people', 'man', 'woman', 'child', 'boy', 'girl']
        self.person_emojis = ['🧑', '🙂', '👤', '👥']
        
        # Logging
        self.setup_logging()
        self.db_conn = None
        
    def setup_logging(self):
        """Configure logging"""
        log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
        logging.basicConfig(
            level=getattr(logging, log_level),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('spatial_enrichment')
    
    def _get_required(self, key):
        """Get required environment variable with no fallback"""
        value = os.getenv(key)
        if not value:
            raise ValueError(f"Required environment variable {key} not set")
        return value
    
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
    
    def find_merged_boxes_needing_enrichment(self):
        """Find merged_boxes that need spatial enrichment"""
        try:
            cursor = self.db_conn.cursor()
            
            # Find merged boxes that haven't been processed for enrichment yet
            # or have been updated since last enrichment
            query = """
                SELECT mb.image_id, mb.merged_id, i.image_filename, i.image_path, mb.merged_data
                FROM merged_boxes mb
                JOIN images i ON mb.image_id = i.image_id
                WHERE mb.status = 'success'
                AND (
                    -- No enrichment exists yet
                    NOT EXISTS (
                        SELECT 1 FROM postprocessing p
                        WHERE p.merged_box_id = mb.merged_id
                        AND p.service = 'spatial_enrichment'
                    )
                    -- OR enrichment is older than merged boxes
                    OR mb.created > (
                        SELECT COALESCE(MAX(p.result_created), '1970-01-01'::timestamp)
                        FROM postprocessing p
                        WHERE p.merged_box_id = mb.merged_id
                        AND p.service = 'spatial_enrichment'
                    )
                )
                ORDER BY mb.created DESC
                LIMIT %s
            """
            
            cursor.execute(query, (self.batch_size,))
            results = cursor.fetchall()
            cursor.close()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error finding merged boxes needing enrichment: {e}")
            return []
    
    def extract_bbox_instances(self, merged_data):
        """Extract individual bbox instances from harmonized data"""
        instances = []
        
        if not isinstance(merged_data, dict):
            return instances
        
        grouped_objects = merged_data.get('grouped_objects', {})
        for group_key, group_data in grouped_objects.items():
            group_instances = group_data.get('instances', [])
            for instance in group_instances:
                instances.append({
                    'cluster_id': instance.get('cluster_id', ''),
                    'emoji': instance.get('emoji', ''),
                    'label': instance.get('label', ''),
                    'merged_bbox': instance.get('merged_bbox', {}),
                    'detection_count': instance.get('detection_count', 0),
                    'avg_confidence': instance.get('avg_confidence', 0.0)
                })
        
        return instances
    
    def is_person_box(self, instance):
        """Check if a bounding box represents a person"""
        label = instance.get('label', '').lower()
        emoji = instance.get('emoji', '')
        
        # Check label
        if any(person_label in label for person_label in self.person_labels):
            return True
        
        # Check emoji
        if emoji in self.person_emojis:
            return True
        
        return False
    
    def crop_bbox_from_image(self, image_path, bbox):
        """Crop bbox region from image and return as bytes for POST request"""
        try:
            # Load image
            with Image.open(image_path) as img:
                # Extract bbox coordinates
                x = bbox['x']
                y = bbox['y'] 
                width = bbox['width']
                height = bbox['height']
                
                # Crop the bbox region
                crop_box = (x, y, x + width, y + height)
                cropped_img = img.crop(crop_box)
                
                # Convert to bytes for POST request
                img_buffer = io.BytesIO()
                cropped_img.save(img_buffer, format='JPEG', quality=90)
                img_buffer.seek(0)
                
                return img_buffer.getvalue()
                
        except Exception as e:
            self.logger.error(f"Failed to crop bbox from {image_path}: {e}")
            return None
    
    def enrich_person_bbox(self, image_path, instance):
        """Add face and pose data for person bounding boxes"""
        enrichment_data = {
            'face_data': None,
            'pose_data': None,
            'enrichment_type': 'person_analysis'
        }
        
        # Crop bbox region in memory
        cropped_image_data = self.crop_bbox_from_image(image_path, instance['merged_bbox'])
        if not cropped_image_data:
            return enrichment_data
        
        # Get face data via POST with cropped image
        try:
            files = {'file': ('bbox_crop.jpg', io.BytesIO(cropped_image_data), 'image/jpeg')}
            face_response = requests.post(
                self.face_service_url,
                files=files,
                timeout=self.request_timeout
            )
            
            if face_response.status_code == 200:
                face_data = face_response.json()
                if face_data.get('status') == 'success' and face_data.get('predictions'):
                    enrichment_data['face_data'] = face_data
                    self.logger.debug(f"Added face data for {instance['cluster_id']}")
            
        except Exception as e:
            self.logger.warning(f"Failed to get face data for {instance['cluster_id']}: {e}")
        
        # Get pose data via POST with cropped image
        try:
            files = {'file': ('bbox_crop.jpg', io.BytesIO(cropped_image_data), 'image/jpeg')}
            pose_response = requests.post(
                self.pose_service_url,
                files=files,
                timeout=self.request_timeout
            )
            
            if pose_response.status_code == 200:
                pose_data = pose_response.json()
                if pose_data.get('status') == 'success' and pose_data.get('predictions'):
                    enrichment_data['pose_data'] = pose_data
                    self.logger.debug(f"Added pose data for {instance['cluster_id']}")
            
        except Exception as e:
            self.logger.warning(f"Failed to get pose data for {instance['cluster_id']}: {e}")
        
        return enrichment_data
    
    def enrich_bbox_colors(self, image_path, instance):
        """Add color analysis for any bounding box"""
        enrichment_data = {
            'color_data': None,
            'enrichment_type': 'color_analysis'
        }
        
        # Crop bbox region in memory
        cropped_image_data = self.crop_bbox_from_image(image_path, instance['merged_bbox'])
        if not cropped_image_data:
            return enrichment_data
        
        # Get color data via POST with cropped image
        try:
            files = {'file': ('bbox_crop.jpg', io.BytesIO(cropped_image_data), 'image/jpeg')}
            colors_response = requests.post(
                self.colors_service_url,
                files=files,
                timeout=self.request_timeout
            )
            
            if colors_response.status_code == 200:
                colors_data = colors_response.json()
                if colors_data.get('status') == 'success' and colors_data.get('predictions'):
                    enrichment_data['color_data'] = colors_data
                    self.logger.debug(f"Added color data for {instance['cluster_id']}")
            
        except Exception as e:
            self.logger.warning(f"Failed to get color data for {instance['cluster_id']}: {e}")
        
        return enrichment_data
    
    def process_merged_box_enrichment(self, merged_box_record):
        """Process spatial enrichment for a single merged box"""
        image_id, merged_id, image_filename, image_path, merged_data = merged_box_record
        
        try:
            # Extract bbox instances from harmonized data
            instances = self.extract_bbox_instances(merged_data)
            if not instances:
                self.logger.debug(f"No instances found in merged box {merged_id}")
                return False
            
            enrichment_results = []
            
            # Process each bbox instance
            for instance in instances:
                cluster_id = instance.get('cluster_id', '')
                
                # Face/pose enrichment for person boxes
                if self.is_person_box(instance):
                    person_enrichment = self.enrich_person_bbox(image_path, instance)
                    person_enrichment['cluster_id'] = cluster_id
                    person_enrichment['bbox'] = instance['merged_bbox']
                    enrichment_results.append(person_enrichment)
                
                # Color enrichment for all boxes
                color_enrichment = self.enrich_bbox_colors(image_path, instance)
                color_enrichment['cluster_id'] = cluster_id
                color_enrichment['bbox'] = instance['merged_bbox']
                enrichment_results.append(color_enrichment)
            
            # Store enrichment results
            if enrichment_results:
                self.save_enrichment_results(image_id, merged_id, enrichment_results)
                
                person_count = sum(1 for r in enrichment_results if r.get('enrichment_type') == 'person_analysis')
                color_count = sum(1 for r in enrichment_results if r.get('enrichment_type') == 'color_analysis')
                
                self.logger.info(f"Enriched {image_filename}: {person_count} person boxes (face+pose), {color_count} color analyses")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"Error processing enrichment for merged box {merged_id}: {e}")
            return False
    
    def save_enrichment_results(self, image_id, merged_box_id, enrichment_results):
        """Save spatial enrichment results to postprocessing table"""
        try:
            cursor = self.db_conn.cursor()
            
            # Delete old enrichment for this merged box (atomic replacement)
            cursor.execute("""
                DELETE FROM postprocessing 
                WHERE merged_box_id = %s AND service = 'spatial_enrichment'
            """, (merged_box_id,))
            
            # Insert new enrichment results
            enrichment_data = {
                'enrichment_results': enrichment_results,
                'total_instances': len(enrichment_results),
                'processing_algorithm': 'spatial_enrichment_v1'
            }
            
            cursor.execute("""
                INSERT INTO postprocessing (image_id, merged_box_id, service, data, status)
                VALUES (%s, %s, %s, %s, %s)
            """, (
                image_id,
                merged_box_id,
                'spatial_enrichment',
                json.dumps(enrichment_data),
                'success'
            ))
            
            cursor.close()
            
        except Exception as e:
            self.logger.error(f"Error saving enrichment results: {e}")
            raise
    
    def run_enrichment_batch(self):
        """Process one batch of merged boxes for enrichment"""
        try:
            # Find merged boxes needing enrichment
            merged_boxes = self.find_merged_boxes_needing_enrichment()
            if not merged_boxes:
                return 0
            
            self.logger.info(f"Processing spatial enrichment for {len(merged_boxes)} merged boxes")
            
            # Process each merged box in a transaction
            success_count = 0
            for merged_box_record in merged_boxes:
                try:
                    if self.process_merged_box_enrichment(merged_box_record):
                        self.db_conn.commit()
                        success_count += 1
                    else:
                        self.db_conn.rollback()
                except Exception as e:
                    self.db_conn.rollback()
                    merged_id = merged_box_record[1]
                    self.logger.error(f"Transaction failed for merged box {merged_id}: {e}")
            
            self.logger.info(f"Enriched {success_count}/{len(merged_boxes)} merged boxes")
            return success_count
            
        except Exception as e:
            self.logger.error(f"Error in enrichment batch: {e}")
            return 0
    
    def run_continuous(self):
        """Main continuous processing loop"""
        self.logger.info(f"Starting spatial enrichment worker ({self.worker_id})")
        self.logger.info(f"Scan interval: {self.scan_interval}s, Batch size: {self.batch_size}")
        self.logger.info(f"Face service: {self.face_service_url}")
        self.logger.info(f"Colors service: {self.colors_service_url}")
        
        try:
            while True:
                processed = self.run_enrichment_batch()
                
                # Adaptive scanning based on workload
                if processed > 0:
                    time.sleep(2)  # Quick scan when busy
                else:
                    time.sleep(self.scan_interval)  # Normal scan when idle
                    
        except KeyboardInterrupt:
            self.logger.info("Stopping spatial enrichment worker...")
        except Exception as e:
            self.logger.error(f"Fatal error in continuous loop: {e}")
            raise
        finally:
            if self.db_conn:
                self.db_conn.close()
            self.logger.info("Spatial enrichment worker stopped")
    
    def run(self):
        """Main entry point"""
        if not self.connect_to_database():
            return 1
        
        self.run_continuous()
        return 0

def main():
    """Main entry point"""
    try:
        worker = SpatialEnrichmentWorker()
        return worker.run()
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Spatial enrichment worker error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())