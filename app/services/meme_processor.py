import pytesseract
from PIL import Image, ImageEnhance, ImageFilter, ImageOps, ImageDraw
from deep_translator import GoogleTranslator
from io import BytesIO
import boto3
from app.core.config import get_settings
from loguru import logger
from typing import Tuple, List
import re

settings = get_settings()


class MemeProcessor:
    def __init__(self):
        self.translator = GoogleTranslator(source='auto', target='ru')
        self.s3_client = boto3.client(
            's3',
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY
        )
        # Configure Tesseract with more specific options
        self.tesseract_config = {
            'default': '--oem 3 --psm 3 -c tessedit_create_txt=1',  # Default: Multi-line text
            'meme': '--oem 3 --psm 6 -c preserve_interword_spaces=1 -c tessedit_create_txt=1',  # Uniform block of text
            'single': '--oem 3 --psm 7 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?\'\" " -c tessedit_create_txt=1',  # Single line with limited chars
            'word': '--oem 3 --psm 8 -c tessedit_create_txt=1',  # Single word
            'sparse': '--oem 3 --psm 11 -c tessedit_do_invert=0 -c tessedit_create_txt=1',  # Sparse text with no rotation
            'vertical': '--oem 3 --psm 5 -c tessedit_create_txt=1',  # Vertical text
            'caption': '--oem 3 --psm 6 -c tessedit_char_blacklist="|" -c preserve_interword_spaces=1 -c tessedit_create_txt=1',  # For meme captions
            'clean': '--oem 3 --psm 3 -c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789.,!?\'\" " -c tessedit_create_txt=1 -c textord_heavy_nr=1 -c textord_min_linesize=3'  # Clean text with noise removal
        }

    def detect_text_regions(self, image: Image.Image) -> List[Tuple[int, int, int, int]]:
        """Detect potential text regions in the image"""
        # Convert to grayscale
        gray = image.convert('L')
        
        # Apply edge detection
        edges = gray.filter(ImageFilter.FIND_EDGES)
        
        # Threshold the edges
        threshold = edges.point(lambda x: 255 if x > 128 else 0, '1')
        
        # Find connected components (potential text regions)
        bbox = threshold.getbbox()
        if not bbox:
            return []
        
        # Split into horizontal regions (for meme captions)
        regions = []
        height = bbox[3] - bbox[1]
        if height > 100:  # If image is tall enough, try to split into top/middle/bottom
            top_region = (bbox[0], bbox[1], bbox[2], bbox[1] + height//3)
            middle_region = (bbox[0], bbox[1] + height//3, bbox[2], bbox[1] + 2*height//3)
            bottom_region = (bbox[0], bbox[1] + 2*height//3, bbox[2], bbox[3])
            regions.extend([top_region, middle_region, bottom_region])
        else:
            regions.append(bbox)
        
        return regions

    def clean_text(self, text: str) -> str:
        """Clean and normalize extracted text"""
        if not text:
            return ""
            
        # Remove non-printable characters
        text = ''.join(char for char in text if char.isprintable())
        
        # Remove repeated whitespace
        text = ' '.join(text.split())
        
        # Fix common OCR mistakes
        replacements = {
            '|': 'I',
            '[': 'I',
            ']': 'I',
            '{': '(',
            '}': ')',
            '0': 'O',
            '1': 'I',
            'l': 'I',
            'rn': 'm',
            'vv': 'w'
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # Remove single characters (likely noise)
        text = ' '.join(word for word in text.split() if len(word) > 1)
        
        # Fix capitalization
        text = '. '.join(s.capitalize() for s in text.split('. '))
        
        return text.strip()

    def enhance_for_text(self, image: Image.Image) -> Image.Image:
        """Apply text-specific enhancements"""
        # Convert to grayscale
        gray = image.convert('L')
        
        # Increase contrast
        contrast = ImageEnhance.Contrast(gray)
        enhanced = contrast.enhance(2.0)
        
        # Apply unsharp mask for better edge definition
        enhanced = enhanced.filter(
            ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3)
        )
        
        # Remove small noise
        enhanced = enhanced.filter(ImageFilter.MinFilter(3))
        
        return enhanced

    def scale_for_ocr(self, image: Image.Image) -> Image.Image:
        """Scale image to optimal size for OCR"""
        # Tesseract works best with images about 300 DPI
        # Assuming standard screen resolution of 96 DPI
        scale_factor = 300 / 96
        
        width, height = image.size
        new_width = int(width * scale_factor)
        new_height = int(height * scale_factor)
        
        # Don't make images too large
        max_dimension = 4000
        if new_width > max_dimension or new_height > max_dimension:
            scale = max_dimension / max(new_width, new_height)
            new_width = int(new_width * scale)
            new_height = int(new_height * scale)
        
        return image.resize(
            (new_width, new_height),
            Image.Resampling.LANCZOS
        )

    def preprocess_image(self, image: Image.Image) -> List[Tuple[str, Image.Image]]:
        """Preprocess image to improve OCR accuracy"""
        processed_images = []
        
        # Original image
        processed_images.append(("original", image))
        
        # Detect text regions
        regions = self.detect_text_regions(image)
        for i, region in enumerate(regions):
            cropped = image.crop(region)
            processed_images.append((f"region_{i}", cropped))
        
        # Basic conversions
        gray = image.convert('L')
        processed_images.append(("grayscale", gray))
        
        # Enhanced version
        enhanced = self.enhance_for_text(image)
        processed_images.append(("enhanced", enhanced))
        
        # Scaled version
        scaled = self.scale_for_ocr(enhanced)
        processed_images.append(("scaled", scaled))
        
        # Auto contrast
        auto_contrast = ImageOps.autocontrast(gray, cutoff=0.5)
        processed_images.append(("auto_contrast", auto_contrast))
        
        # Multiple threshold levels
        for threshold in [100, 128, 150]:
            threshold_img = gray.point(lambda x: 255 if x > threshold else 0, '1')
            processed_images.append((f"threshold_{threshold}", threshold_img))
        
        # Edge enhanced
        edge_enhanced = gray.filter(ImageFilter.EDGE_ENHANCE_MORE)
        processed_images.append(("edge_enhanced", edge_enhanced))
        
        # Inverted (for white text)
        inverted = ImageOps.invert(gray)
        processed_images.append(("inverted", inverted))
        
        # Inverted with auto contrast
        inv_auto_contrast = ImageOps.autocontrast(inverted, cutoff=0.5)
        processed_images.append(("inverted_auto_contrast", inv_auto_contrast))
        
        return processed_images

    def post_process_results(self, results: List[Tuple[str, str, float]]) -> List[Tuple[str, str, float]]:
        """Post-process and filter OCR results"""
        processed_results = []
        seen_texts = set()
        
        for text, method, confidence in results:
            # Clean the text
            cleaned_text = self.clean_text(text)
            if not cleaned_text:
                continue
                
            # Skip if we've seen this text before
            if cleaned_text.lower() in seen_texts:
                continue
                
            # Skip very short texts unless they're very confident
            if len(cleaned_text) < 3 and confidence < 90:
                continue
                
            # Skip nonsense strings (too many consonants or numbers)
            consonants = len(re.findall(r'[bcdfghjklmnpqrstvwxyzBCDFGHJKLMNPQRSTVWXYZ]', cleaned_text))
            if consonants / len(cleaned_text) > 0.7:
                continue
                
            processed_results.append((cleaned_text, method, confidence))
            seen_texts.add(cleaned_text.lower())
        
        return processed_results

    def image_to_bytes(self, image: Image.Image, format: str = 'JPEG') -> bytes:
        """Convert PIL Image to bytes"""
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format=format)
        return img_byte_arr.getvalue()

    def extract_text(self, image: bytes) -> Tuple[str, List[Tuple[str, str, float]]]:
        """Extract text from image using OCR with multiple preprocessing steps"""
        try:
            img = Image.open(BytesIO(image))
            logger.info(f"Processing image of size {img.size}, mode {img.mode}")
            
            # Convert RGBA to RGB if needed
            if img.mode == 'RGBA':
                img = Image.new('RGB', img.size, (255, 255, 255))
                img.paste(image, mask=image.split()[3])
                logger.debug("Converted RGBA image to RGB")
            
            # Resize if image is too large
            max_dimension = 1920
            if max(img.size) > max_dimension:
                ratio = max_dimension / max(img.size)
                new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                logger.debug(f"Resized image to {new_size}")
            
            # Process image with different methods
            processed_images = self.preprocess_image(img)
            
            # Store all results with confidence
            all_results = []
            
            for proc_name, proc_img in processed_images:
                for config_name, config in self.tesseract_config.items():
                    try:
                        # Get text and confidence
                        data = pytesseract.image_to_data(
                            proc_img,
                            config=config,
                            output_type=pytesseract.Output.DICT
                        )
                        
                        # Calculate average confidence for words
                        confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                        if not confidences:
                            continue
                            
                        avg_confidence = sum(confidences) / len(confidences)
                        text = ' '.join([
                            word for i, word in enumerate(data['text'])
                            if int(data['conf'][i]) > 60  # Filter low confidence words
                        ])
                        
                        if text.strip():
                            method = f"{proc_name}/{config_name}"
                            all_results.append((text.strip(), method, avg_confidence))
                            logger.debug(
                                f"Method {method}:\n"
                                f"Text: {text}\n"
                                f"Confidence: {avg_confidence:.1f}%"
                            )
                    
                    except Exception as e:
                        logger.warning(
                            f"Error with {proc_name}/{config_name}: {str(e)}"
                        )
                        continue
            
            if not all_results:
                logger.warning("No text found in any processing method")
                return "", []
            
            # Sort results by confidence and get top 3
            all_results.sort(key=lambda x: x[2], reverse=True)
            top_results = self.post_process_results(all_results[:3])
            
            # Use the highest confidence result as the main result
            best_text = top_results[0][0]
            
            logger.info(
                f"Top 3 results:\n" + "\n".join([
                    f"{i+1}. {text} ({method}, {conf:.1f}%)"
                    for i, (text, method, conf) in enumerate(top_results)
                ])
            )
            
            return best_text.strip(), top_results
            
        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            raise

    def translate_text(self, text: str, target_lang: str = 'ru') -> str:
        """Translate text to target language"""
        try:
            if target_lang != self.translator.target:
                self.translator = GoogleTranslator(
                    source='auto',
                    target=target_lang
                )
            return self.translator.translate(text)
        except Exception as e:
            logger.error(f"Error translating text: {e}")
            raise

    def overlay_text(self, image: bytes, text: str) -> bytes:
        """Overlay translated text on image"""
        try:
            # TODO: Implement text overlay logic using PIL
            # This is a placeholder that returns the original image
            return image
        except Exception as e:
            logger.error(f"Error overlaying text: {e}")
            raise

    def upload_to_s3(self, image: bytes, filename: str) -> str:
        """Upload image to S3 and return URL"""
        try:
            self.s3_client.put_object(
                Bucket=settings.S3_BUCKET_NAME,
                Key=filename,
                Body=image,
                ContentType='image/jpeg'
            )
            return (
                f"https://{settings.S3_BUCKET_NAME}.s3.amazonaws.com/",
                f"{filename}",
            )
        except Exception as e:
            logger.error(f"Error uploading to S3: {e}")
            raise
