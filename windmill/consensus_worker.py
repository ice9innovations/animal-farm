#!/usr/bin/env python3
"""
Continuous Consensus Worker - Scans for new ML service results and updates consensus
Implements the DELETE+INSERT pattern for atomic consensus updates
"""
import os
import json
import time
import logging
import psycopg2
from datetime import datetime
from dotenv import load_dotenv

class ConsensusWorker:
    """Continuous consensus/voting worker"""
    
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
        self.worker_id = os.getenv('WORKER_ID', f'consensus_worker_{int(time.time())}')
        self.scan_interval = int(os.getenv('SCAN_INTERVAL', '5'))  # seconds
        self.batch_size = int(os.getenv('BATCH_SIZE', '100'))
        
        # V3 Voting configuration
        self.service_names = {
            'blip': 'blip',
            'clip': 'clip', 
            'yolov8': 'yolo',
            'colors': 'colors',
            'detectron2': 'detectron2',
            'face': 'face',
            'nsfw2': 'nsfw',
            'ocr': 'ocr',
            'inception_v3': 'inception',
            'rtdetr': 'rtdetr',
            'metadata': 'metadata',
            'ollama': 'llama',
            'pose': 'pose'
        }
        
        # Special emojis that auto-promote
        self.special_emojis = ['🔞', '💬']
        
        # Democratic voting configuration
        self.default_confidence = float(os.getenv('DEFAULT_CONFIDENCE', '0.75'))
        self.low_confidence_threshold = float(os.getenv('LOW_CONFIDENCE_THRESHOLD', '0.4'))
        
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
        self.logger = logging.getLogger('consensus_worker')
    
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
            self.db_conn.autocommit = False  # Use transactions
            self.logger.info(f"Connected to PostgreSQL at {self.db_host}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    def find_images_needing_consensus_update(self):
        """Find images that have new service results since last consensus"""
        try:
            cursor = self.db_conn.cursor()
            
            # Find images with service results but no consensus, or outdated consensus
            query = """
                SELECT DISTINCT r.image_id, i.image_filename
                FROM results r
                JOIN images i ON r.image_id = i.image_id
                WHERE r.status = 'success'
                AND (
                    -- No consensus exists yet
                    NOT EXISTS (
                        SELECT 1 FROM consensus c 
                        WHERE c.image_id = r.image_id
                    )
                    -- OR consensus is older than newest service result
                    OR EXISTS (
                        SELECT 1 FROM consensus c
                        WHERE c.image_id = r.image_id
                        AND c.consensus_created < (
                            SELECT MAX(result_created) 
                            FROM results r2 
                            WHERE r2.image_id = r.image_id 
                            AND r2.status = 'success'
                        )
                    )
                )
                ORDER BY r.image_id
                LIMIT %s
            """
            
            cursor.execute(query, (self.batch_size,))
            results = cursor.fetchall()
            cursor.close()
            
            return [(row[0], row[1]) for row in results]
            
        except Exception as e:
            self.logger.error(f"Error finding images needing consensus: {e}")
            return []
    
    def get_service_results_for_image(self, image_id):
        """Get all successful service results for an image"""
        try:
            cursor = self.db_conn.cursor()
            
            query = """
                SELECT service, data, processing_time, result_created
                FROM results
                WHERE image_id = %s AND status = 'success'
                ORDER BY service
            """
            
            cursor.execute(query, (image_id,))
            results = cursor.fetchall()
            cursor.close()
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error getting results for image {image_id}: {e}")
            return []
    
    def calculate_consensus(self, service_results):
        """Calculate consensus using full V3 voting algorithm (ported from V3VotingService.js)"""
        if not service_results:
            return None
        
        # Convert database results to V3VotingService format
        service_results_dict = {}
        for service, data, proc_time, created in service_results:
            service_results_dict[service] = {
                'success': True,
                'predictions': data.get('predictions', []) if isinstance(data, dict) else []
            }
        
        # Get harmonized bounding box data (if available)
        bounding_box_data = self.get_bounding_box_data_for_image(service_results_dict)
        
        # Implement V3 voting algorithm
        votes_result = self.process_votes(service_results_dict, bounding_box_data)
        
        # Package results for storage
        consensus = {
            'services_count': len(service_results),
            'services_list': [row[0] for row in service_results],
            'total_processing_time': sum(row[2] or 0 for row in service_results),
            'latest_result_time': max(row[3] for row in service_results).isoformat(),
            'consensus_algorithm': 'v3_voting_full_port',
            'votes': votes_result['votes'],
            'special': votes_result['special'],
            'debug': votes_result['debug']
        }
        
        return consensus
    
    def get_bounding_box_data_for_image(self, service_results_dict):
        """Get bounding box data for this image (if available)"""
        # For now, return None - bounding box integration will be added later
        # This would query the merged_boxes table for harmonized bbox data
        return None
    
    def process_votes(self, service_results, bounding_box_data=None):
        """Main V3 voting algorithm entry point (ported from V3VotingService.js)"""
        # Step 1: Extract all detections from all services
        all_detections = self.extract_all_detections(service_results, bounding_box_data)
        
        # Step 2: Group detections by emoji (democratic voting)
        emoji_groups = self.group_detections_by_emoji(all_detections)
        
        # Step 3: Analyze evidence for each emoji
        emoji_analysis = self.analyze_emoji_evidence(emoji_groups, service_results)
        
        # Step 4: Calculate evidence weights and final ranking
        ranked_consensus = self.calculate_final_ranking(emoji_analysis)
        
        # Step 5: Apply post-processing curation (quality adjustments)
        self.apply_post_processing_curation(ranked_consensus)
        
        return {
            'votes': {
                'consensus': ranked_consensus
            },
            'special': self.extract_special_detections(service_results),
            'debug': {
                'detection_count': len(all_detections),
                'emoji_groups': len(emoji_groups)
            }
        }
    
    def extract_all_detections(self, service_results, bounding_box_data=None):
        """Extract all detections from all services with metadata (ported from V3VotingService.js)"""
        all_detections = []

        for service_name, result in service_results.items():
            if not result.get('success') or not result.get('predictions'):
                continue

            service_display_name = self.service_names.get(service_name, service_name)
            seen_emojis = set()  # Deduplicate within service

            for prediction in result['predictions']:
                # Handle emoji_mappings format (BLIP, Ollama v3)
                if prediction.get('emoji_mappings') and isinstance(prediction['emoji_mappings'], list):
                    for mapping in prediction['emoji_mappings']:
                        if mapping.get('emoji') and mapping['emoji'] not in seen_emojis:
                            seen_emojis.add(mapping['emoji'])
                            all_detections.append({
                                'emoji': mapping['emoji'],
                                'service': service_display_name,
                                'evidence_type': self.get_evidence_type(service_name),
                                'confidence': self.default_confidence,
                                'context': {
                                    'word': mapping.get('word', ''),
                                    'source': 'caption_mapping'
                                },
                                'shiny': mapping.get('shiny', False)
                            })
                
                # Handle direct emoji format (CLIP, object detection, etc.)
                elif prediction.get('emoji') and prediction.get('type') != 'color_analysis':
                    emoji = prediction['emoji']
                    
                    if emoji and emoji not in seen_emojis:
                        seen_emojis.add(emoji)
                        all_detections.append({
                            'emoji': emoji,
                            'service': service_display_name,
                            'evidence_type': self.get_evidence_type(service_name),
                            'confidence': prediction.get('confidence', self.default_confidence),
                            'context': self.extract_context(prediction, service_name),
                            'shiny': prediction.get('shiny', False)
                        })

        # Extract spatial detections from clustered bounding box data (if available)
        if bounding_box_data and bounding_box_data.get('winning_objects', {}).get('grouped'):
            for key, group in bounding_box_data['winning_objects']['grouped'].items():
                if group.get('emoji') and group.get('instances'):
                    for instance in group['instances']:
                        all_detections.append({
                            'emoji': group['emoji'],
                            'service': 'spatial_clustering',
                            'evidence_type': 'spatial',
                            'confidence': instance.get('avg_confidence', 0.75),
                            'context': {'source': 'clustered_bounding_box'},
                            'shiny': False,
                            'spatial_data': {
                                'cluster_id': instance.get('cluster_id'),
                                'detection_count': instance.get('detection_count', 1),
                                'avg_confidence': instance.get('avg_confidence', 0.75),
                                'bbox': instance.get('merged_bbox', {}),
                                'individual_detections': instance.get('detections', [])
                            }
                        })

        return all_detections
    
    def get_evidence_type(self, service_name):
        """Determine evidence type based on service name (ported from V3VotingService.js)"""
        spatial_services = ['yolov8', 'detectron2', 'rtdetr']
        semantic_services = ['blip', 'ollama']  # Smart captioning services
        classification_services = ['clip', 'inception_v3']  # Image classification services
        specialized_services = ['face', 'nsfw2', 'ocr', 'pose']

        if service_name in spatial_services:
            return 'spatial'
        if service_name in semantic_services:
            return 'semantic'
        if service_name in classification_services:
            return 'classification'
        if service_name in specialized_services:
            return 'specialized'
        return 'other'

    def extract_context(self, prediction, service_name):
        """Extract context information from prediction (ported from V3VotingService.js)"""
        context = {}
        
        if service_name == 'face':
            context['pose'] = prediction.get('pose')
        if service_name == 'nsfw2':
            context['nsfw_confidence'] = prediction.get('confidence')
        if service_name == 'ocr':
            context['text_detected'] = prediction.get('has_text', False)
            context['text_content'] = prediction.get('text')
        
        return context

    def group_detections_by_emoji(self, all_detections):
        """Group detections by emoji for democratic voting (ported from V3VotingService.js)"""
        groups = {}
        
        for detection in all_detections:
            emoji = detection['emoji']
            if emoji not in groups:
                groups[emoji] = []
            groups[emoji].append(detection)
        
        return groups

    def analyze_emoji_evidence(self, emoji_groups, service_results):
        """Analyze evidence for each emoji group (ported from V3VotingService.js)"""
        analysis = []
        
        for emoji, detections in emoji_groups.items():
            voting_services = list(set(d['service'] for d in detections if d['service'] != 'spatial_clustering'))
            
            evidence_analysis = {
                'emoji': emoji,
                'total_votes': len(voting_services),
                'voting_services': voting_services,
                'detections': detections,
                'evidence': {
                    'spatial': self.analyze_spatial_evidence(detections),
                    'semantic': self.analyze_semantic_evidence(detections),
                    'classification': self.analyze_classification_evidence(detections),
                    'specialized': self.analyze_specialized_evidence(detections)
                },
                'instances': self.extract_instance_information(detections),
                'shiny': any(d.get('shiny', False) for d in detections)
            }
            
            analysis.append(evidence_analysis)
        
        return analysis

    def analyze_spatial_evidence(self, detections):
        """Analyze spatial evidence from object detection services (ported from V3VotingService.js)"""
        spatial_detections = [d for d in detections if d.get('evidence_type') == 'spatial']
        if not spatial_detections:
            return None
        
        clusters = [d['spatial_data'] for d in spatial_detections if d.get('spatial_data')]
        
        if not clusters:
            return None
        
        return {
            'service_count': len(spatial_detections),
            'clusters': clusters,
            'max_detection_count': max(c.get('detection_count', 1) for c in clusters),
            'avg_confidence': sum(c.get('avg_confidence', 0.75) for c in clusters) / len(clusters),
            'total_instances': len(clusters)
        }

    def analyze_semantic_evidence(self, detections):
        """Analyze semantic evidence from captioning services (ported from V3VotingService.js)"""
        semantic_detections = [d for d in detections if d.get('evidence_type') == 'semantic']
        if not semantic_detections:
            return None
        
        return {
            'service_count': len(semantic_detections),
            'words': [d['context'].get('word') for d in semantic_detections if d.get('context', {}).get('word')],
            'sources': [d['service'] for d in semantic_detections]
        }

    def analyze_classification_evidence(self, detections):
        """Analyze classification evidence from image classification services (ported from V3VotingService.js)"""
        classification_detections = [d for d in detections if d.get('evidence_type') == 'classification']
        if not classification_detections:
            return None
        
        return {
            'service_count': len(classification_detections),
            'sources': [d['service'] for d in classification_detections]
        }

    def analyze_specialized_evidence(self, detections):
        """Analyze specialized evidence (Face, NSFW, OCR) (ported from V3VotingService.js)"""
        specialized_detections = [d for d in detections if d.get('evidence_type') == 'specialized']
        if not specialized_detections:
            return None
        
        by_type = {}
        for d in specialized_detections:
            service_type = d['service'].lower()
            if service_type not in by_type:
                by_type[service_type] = []
            by_type[service_type].append(d)
        
        return by_type

    def extract_instance_information(self, detections):
        """Extract instance information (ported from V3VotingService.js)"""
        spatial_detections = [d for d in detections if d.get('spatial_data')]
        
        if not spatial_detections:
            return {'count': 1, 'type': 'non_spatial'}
        
        return {
            'count': len(spatial_detections),
            'type': 'spatial'
        }

    def calculate_evidence_weight(self, analysis):
        """Calculate evidence weight using consensus bonus system (ported from V3VotingService.js)"""
        weight = 0
        
        # Base democratic weight: 1 vote per service (pure democracy)
        base_votes = analysis['total_votes']
        
        # Spatial consensus bonus: Agreement on location
        spatial_consensus_bonus = 0
        if analysis['evidence']['spatial']:
            # Consensus = detection_count - 1 (one vote doesn't count as consensus)
            spatial_consensus_bonus = max(0, analysis['evidence']['spatial']['max_detection_count'] - 1)
        
        # Content consensus bonus: Agreement across semantic + classification services
        content_consensus_bonus = 0
        semantic_count = analysis['evidence']['semantic']['service_count'] if analysis['evidence']['semantic'] else 0
        classification_count = analysis['evidence']['classification']['service_count'] if analysis['evidence']['classification'] else 0
        total_content_services = semantic_count + classification_count
        
        if total_content_services >= 2:
            # Consensus = total_content_services - 1 (one vote doesn't count as consensus)
            content_consensus_bonus = total_content_services - 1
        
        # Total weight = democratic votes + consensus bonuses
        weight = base_votes + spatial_consensus_bonus + content_consensus_bonus
        
        return max(0, weight)  # Don't go negative

    def calculate_final_ranking(self, emoji_analysis):
        """Calculate final ranking with democratic voting + evidence weighting (ported from V3VotingService.js)"""
        # Calculate evidence weights
        for analysis in emoji_analysis:
            analysis['evidence_weight'] = self.calculate_evidence_weight(analysis)
            analysis['final_score'] = analysis['total_votes'] + analysis['evidence_weight']
            analysis['should_include'] = self.should_include_in_results(analysis)
        
        # Filter and sort
        filtered_analysis = [a for a in emoji_analysis if a['should_include']]
        
        # Sort by total votes (primary), then evidence weight (secondary)
        filtered_analysis.sort(key=lambda a: (a['total_votes'], a['evidence_weight']), reverse=True)
        
        # Convert to final result format
        results = []
        for analysis in filtered_analysis:
            result = {
                'emoji': analysis['emoji'],
                'votes': analysis['total_votes'],
                'evidence_weight': round(analysis['evidence_weight'], 2),
                'final_score': round(analysis['final_score'], 2),
                'instances': analysis['instances'],
                'evidence': {
                    'spatial': self.format_spatial_evidence(analysis['evidence']['spatial']) if analysis['evidence']['spatial'] else None,
                    'semantic': analysis['evidence']['semantic'],
                    'classification': analysis['evidence']['classification'],
                    'specialized': list(analysis['evidence']['specialized'].keys()) if analysis['evidence']['specialized'] else None
                },
                'services': analysis['voting_services']
            }
            
            # Add bounding boxes if available
            if analysis['evidence']['spatial'] and analysis['evidence']['spatial']['clusters']:
                result['bounding_boxes'] = self.format_bounding_boxes(analysis['evidence']['spatial']['clusters'], analysis['emoji'])
            
            # Add validation/correlation if they exist
            if analysis.get('validation'):
                result['validation'] = analysis['validation']
            if analysis.get('correlation'):
                result['correlation'] = analysis['correlation']
            if analysis.get('shiny'):
                result['shiny'] = True
            
            results.append(result)
        
        return results

    def format_spatial_evidence(self, spatial_evidence):
        """Format spatial evidence for output"""
        if not spatial_evidence:
            return None
        
        return {
            'detection_count': spatial_evidence['max_detection_count'],
            'avg_confidence': round(spatial_evidence['avg_confidence'], 3),
            'instance_count': spatial_evidence['total_instances']
        }

    def format_bounding_boxes(self, clusters, emoji):
        """Format bounding box data for output"""
        bounding_boxes = []
        for cluster in clusters:
            bounding_boxes.append({
                'cluster_id': cluster.get('cluster_id'),
                'merged_bbox': cluster.get('bbox', {}),
                'emoji': emoji,
                'label': cluster.get('cluster_id', '').split('_')[0] if cluster.get('cluster_id') else emoji,
                'detection_count': cluster.get('detection_count', 1),
                'avg_confidence': cluster.get('avg_confidence', 0.75),
                'detections': cluster.get('individual_detections', [])
            })
        return bounding_boxes

    def should_include_in_results(self, analysis):
        """Determine if emoji should be included in results (ported from V3VotingService.js)"""
        # Only include if has multiple votes (filter out single-vote emojis)
        return analysis['total_votes'] > 1

    def apply_post_processing_curation(self, ranked_consensus):
        """Apply post-processing curation (ported from V3VotingService.js)"""
        # Build lookup for cross-emoji validation
        emoji_map = {item['emoji']: item for item in ranked_consensus}
        
        for item in ranked_consensus:
            curation_adjustment = 0
            
            # Face validates Person (+1 confidence boost)
            if item['emoji'] == '🧑' and '🙂' in emoji_map:
                curation_adjustment += 1
                if 'validation' not in item:
                    item['validation'] = []
                item['validation'].append('face_confirmed')
            
            # Pose validates Person (+1 confidence boost)  
            has_pose_detection = any(
                other.get('evidence', {}).get('specialized') and 'pose' in other['evidence']['specialized']
                for other in ranked_consensus
            )
            if item['emoji'] == '🧑' and has_pose_detection:
                curation_adjustment += 1
                if 'validation' not in item:
                    item['validation'] = []
                item['validation'].append('pose_confirmed')
            
            # NSFW requires human context (quality filter)
            if item['emoji'] == '🔞':
                if '🧑' in emoji_map:
                    curation_adjustment += 1
                    if 'validation' not in item:
                        item['validation'] = []
                    item['validation'].append('human_context_confirmed')
                else:
                    curation_adjustment -= 1
                    if 'validation' not in item:
                        item['validation'] = []
                    item['validation'].append('suspicious_no_humans')
            
            # Apply curation adjustment
            if curation_adjustment != 0:
                item['evidence_weight'] += curation_adjustment
                item['final_score'] += curation_adjustment
                # Ensure we don't go negative
                item['evidence_weight'] = max(0, item['evidence_weight'])
                item['final_score'] = max(0, item['final_score'])

    def extract_special_detections(self, service_results):
        """Extract special detections (non-competing) (ported from V3VotingService.js)"""
        special = {}
        
        # Text detection from OCR
        if service_results.get('ocr', {}).get('predictions'):
            has_text = any(pred.get('has_text') for pred in service_results['ocr']['predictions'])
            if has_text:
                text_pred = next(pred for pred in service_results['ocr']['predictions'] if pred.get('has_text'))
                special['text'] = {
                    'emoji': '💬',
                    'detected': True,
                    'confidence': text_pred.get('confidence', 1.0),
                    'content': text_pred.get('text')
                }
            else:
                special['text'] = {'detected': False}
        else:
            special['text'] = {'detected': False}
        
        # Face detection from Face service
        if service_results.get('face', {}).get('predictions'):
            face_pred = next((pred for pred in service_results['face']['predictions'] if pred.get('emoji') == '🙂'), None)
            if face_pred:
                special['face'] = {
                    'emoji': '🙂',
                    'detected': True,
                    'confidence': face_pred.get('confidence', 1.0),
                    'pose': face_pred.get('pose')
                }
            else:
                special['face'] = {'detected': False}
        else:
            special['face'] = {'detected': False}
        
        # NSFW detection from NSFW service
        if service_results.get('nsfw2', {}).get('predictions'):
            nsfw_pred = next((pred for pred in service_results['nsfw2']['predictions'] if pred.get('emoji') == '🔞'), None)
            if nsfw_pred:
                special['nsfw'] = {
                    'emoji': '🔞',
                    'detected': True,
                    'confidence': nsfw_pred.get('confidence', 1.0)
                }
            else:
                special['nsfw'] = {'detected': False}
        else:
            special['nsfw'] = {'detected': False}
        
        return special
    
    def update_consensus_for_image(self, image_id, image_filename):
        """Update consensus for a single image using DELETE+INSERT pattern"""
        try:
            # Get all service results
            service_results = self.get_service_results_for_image(image_id)
            if not service_results:
                self.logger.debug(f"No results for image {image_id}, skipping")
                return False
            
            # Calculate consensus
            start_time = time.time()
            consensus_data = self.calculate_consensus(service_results)
            processing_time = time.time() - start_time
            
            if not consensus_data:
                self.logger.warning(f"Could not calculate consensus for image {image_id}")
                return False
            
            # Atomic DELETE + INSERT
            cursor = self.db_conn.cursor()
            
            # Delete old consensus
            cursor.execute("DELETE FROM consensus WHERE image_id = %s", (image_id,))
            deleted_count = cursor.rowcount
            
            # Insert new consensus
            cursor.execute("""
                INSERT INTO consensus (image_id, consensus_data, processing_time)
                VALUES (%s, %s, %s)
            """, (image_id, json.dumps(consensus_data), processing_time))
            
            cursor.close()
            
            if deleted_count > 0:
                self.logger.debug(f"Updated consensus for {image_filename} ({len(service_results)} services)")
            else:
                self.logger.debug(f"Created consensus for {image_filename} ({len(service_results)} services)")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error updating consensus for image {image_id}: {e}")
            return False
    
    def run_consensus_batch(self):
        """Process one batch of images needing consensus updates"""
        try:
            # Find images needing updates
            images = self.find_images_needing_consensus_update()
            if not images:
                return 0
            
            self.logger.info(f"Processing consensus for {len(images)} images")
            
            # Process each image in a transaction
            success_count = 0
            for image_id, image_filename in images:
                try:
                    if self.update_consensus_for_image(image_id, image_filename):
                        self.db_conn.commit()
                        success_count += 1
                    else:
                        self.db_conn.rollback()
                except Exception as e:
                    self.db_conn.rollback()
                    self.logger.error(f"Transaction failed for image {image_id}: {e}")
            
            self.logger.info(f"Updated consensus for {success_count}/{len(images)} images")
            return success_count
            
        except Exception as e:
            self.logger.error(f"Error in consensus batch: {e}")
            return 0
    
    def run_continuous(self):
        """Main continuous processing loop"""
        self.logger.info(f"Starting continuous consensus worker ({self.worker_id})")
        self.logger.info(f"Scan interval: {self.scan_interval}s, Batch size: {self.batch_size}")
        
        try:
            while True:
                processed = self.run_consensus_batch()
                
                # Sleep shorter if we processed items (more might be available)
                # Sleep longer if no items (less frequent scanning)
                if processed > 0:
                    time.sleep(min(self.scan_interval, 2))  # Quick scan when busy
                else:
                    time.sleep(self.scan_interval)  # Normal scan when idle
                    
        except KeyboardInterrupt:
            self.logger.info("Stopping consensus worker...")
        except Exception as e:
            self.logger.error(f"Fatal error in continuous loop: {e}")
            raise
        finally:
            if self.db_conn:
                self.db_conn.close()
            self.logger.info("Consensus worker stopped")
    
    def run(self):
        """Main entry point"""
        if not self.connect_to_database():
            return 1
        
        self.run_continuous()
        return 0

def main():
    """Main entry point"""
    try:
        worker = ConsensusWorker()
        return worker.run()
        
    except ValueError as e:
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        print(f"Consensus worker error: {e}")
        return 1

if __name__ == "__main__":
    import sys
    sys.exit(main())