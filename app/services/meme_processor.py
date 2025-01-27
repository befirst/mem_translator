import re
from io import BytesIO
from typing import List, Tuple

import boto3
import pytesseract
from deep_translator import GoogleTranslator
from loguru import logger
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

from app.core.config import get_settings

settings = get_settings()


class MemeProcessor:
    def __init__(self):
        self.translator = GoogleTranslator(source="auto", target="ru")
        self.s3_client = boto3.client(
            "s3",
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )

        # Define whitelist as a string instead of tuple
        tessedit_char_whitelist = (
            "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
            + "abcdefghijklmnopqrstuvwxyz"
            + "0123456789.,!?\\'"
        )

        # Configure Tesseract with more specific options
        base_config = "--oem 3"
        self.tesseract_config = {
            "default": f"{base_config} --psm 3 -c tessedit_create_txt=1",
            "meme": (
                f"{base_config} --psm 6 "
                "-c preserve_interword_spaces=1 "
                "-c tessedit_create_txt=1"
            ),
            "single": (
                f"{base_config} --psm 7 "
                f"-c tessedit_char_whitelist={tessedit_char_whitelist} "
                "-c tessedit_create_txt=1"
            ),
            "word": f"{base_config} --psm 8 -c tessedit_create_txt=1",
            "sparse": (
                f"{base_config} --psm 11 "
                "-c tessedit_do_invert=0 "
                "-c tessedit_create_txt=1"
            ),
            "vertical": f"{base_config} --psm 5 -c tessedit_create_txt=1",
            "caption": (
                f"{base_config} --psm 6 "
                "-c tessedit_char_blacklist='|' "
                "-c preserve_interword_spaces=1 "
                "-c tessedit_create_txt=1"
            ),
            "clean": (
                f"{base_config} --psm 3 "
                f"-c tessedit_char_whitelist={tessedit_char_whitelist} "
                "-c tessedit_create_txt=1 "
                "-c textord_heavy_nr=1 "
                "-c textord_min_linesize=3"
            ),
        }

    def detect_text_regions(
        self, image: Image.Image
    ) -> List[Tuple[int, int, int, int]]:
        """Detect potential text regions in the image"""
        # Convert to grayscale
        gray = image.convert("L")

        # Apply edge detection
        edges = gray.filter(ImageFilter.FIND_EDGES)

        # Threshold the edges
        threshold = edges.point(lambda x: 255 if x > 128 else 0, "1")

        # Find connected components (potential text regions)
        bbox = threshold.getbbox()
        if not bbox:
            return []

        # Split into horizontal regions (for meme captions)
        regions = []
        height = bbox[3] - bbox[1]

        # Always add the full image region first
        regions.append(bbox)

        # If image is tall enough, add additional regions
        if height > 100:
            # Top region (header/title)
            top_region = (bbox[0], bbox[1], bbox[2], bbox[1] + height // 3)
            # Middle region (main content)
            middle_region = (
                bbox[0],
                bbox[1] + height // 3,
                bbox[2],
                bbox[1] + 2 * height // 3,
            )
            # Bottom region (footer)
            bottom_region = (
                bbox[0],
                bbox[1] + 2 * height // 3,
                bbox[2],
                bbox[3],
            )
            regions.extend([top_region, middle_region, bottom_region])

        return regions

    def clean_text(self, text: str) -> str:
        """Clean and normalize text while preserving numbers and structure"""
        if not text:
            return ""

        # Replace multiple spaces with single space
        text = " ".join(text.split())

        # Fix basic character issues
        text = text.replace("|", "I")

        # Split into sentences and clean each
        sentences = text.split(". ")
        cleaned_sentences = []

        for sentence in sentences:
            if sentence:
                # Capitalize first letter if it's a letter
                if sentence[0].isalpha():
                    sentence = sentence[0].upper() + sentence[1:]
                cleaned_sentences.append(sentence)

        # Rejoin sentences
        text = ". ".join(cleaned_sentences)

        # Ensure proper spacing around punctuation
        text = re.sub(r"\s*([.,!?])\s*", r"\1 ", text)
        text = re.sub(r"\s+", " ", text)

        return text.strip()

    def enhance_for_text(self, image: Image.Image) -> Image.Image:
        """Apply text-specific enhancements"""
        # Convert to grayscale
        gray = image.convert("L")

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

        return image.resize((new_width, new_height), Image.Resampling.LANCZOS)

    def preprocess_image(
        self,
        image: Image.Image,
    ) -> List[Tuple[str, Image.Image],]:
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
        gray = image.convert("L")
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

    def post_process_results(
        self, results: List[Tuple[str, str, float]]
    ) -> List[Tuple[str, str, float]]:
        """Post-process and filter OCR results, returning up to top 3"""
        logger.info(f"Input results: {results}")

        # First, get the full-image results (highest confidence)
        full_image_results = [
            (text, method, conf)
            for text, method, conf in results
            if not method.startswith("region_") and conf > 90
        ]

        # Sort by confidence and length
        full_image_results.sort(key=lambda x: (x[2], len(x[0])), reverse=True)

        if full_image_results:
            # If we have good full-image results, return up to 3 best ones
            return full_image_results[:3]

        # If no good full-image results, try to combine region results
        region_results = {}
        for text, method, conf in results:
            if method.startswith("region_"):
                region_num = int(method.split("/")[0].split("_")[1])
                if (
                    region_num not in region_results
                    or conf > region_results[region_num][2]
                ):
                    region_results[region_num] = (text, method, conf)

        # Combine regions in order
        if region_results:
            combined_text = []
            for i in sorted(region_results.keys()):
                text, method, conf = region_results[i]
                cleaned = self.clean_text(text)
                if cleaned:
                    combined_text.append(cleaned)

            if combined_text:
                full_text = " ".join(combined_text)
                avg_conf = sum(r[2] for r in region_results.values()) / len(
                    region_results
                )
                combined_result = [(full_text, "combined_regions", avg_conf)]

                # Add up to 2 more best individual results
                results.sort(key=lambda x: (x[2], len(x[0])), reverse=True)
                additional_results = [
                    (self.clean_text(text), method, conf)
                    for text, method, conf in results
                    if len(self.clean_text(text)) > 20
                    and (text, method, conf) not in combined_result
                ][:2]

                return combined_result + additional_results

        # If all else fails, return up to 3 best single results
        results.sort(key=lambda x: (x[2], len(x[0])), reverse=True)
        final_results = []

        for text, method, conf in results:
            cleaned = self.clean_text(text)
            # Minimum length for meaningful text
            if cleaned and len(cleaned) > 20:
                if (cleaned, method, conf) not in final_results:
                    final_results.append((cleaned, method, conf))
                    if len(final_results) == 3:
                        break

        return final_results

    def image_to_bytes(
        self,
        image: Image.Image,
        format: str = "JPEG",
    ) -> bytes:
        """Convert PIL Image to bytes"""
        img_byte_arr = BytesIO()
        image.save(img_byte_arr, format=format)
        return img_byte_arr.getvalue()

    def extract_text(self, image: bytes) -> Tuple[
        str,
        List[Tuple[str, str, float]],
    ]:
        """Extract text from image using OCR"""
        try:
            img = Image.open(BytesIO(image))
            logger.info(
                f"Processing image of size {img.size}, mode {img.mode}",
            )

            # Basic image preprocessing
            if img.mode == "RGBA":
                background = Image.new("RGB", img.size, (255, 255, 255))
                background.paste(img, mask=img.split()[3])
                img = background

            processed_images = self.preprocess_image(img)
            all_results = []

            for proc_name, proc_img in processed_images:
                for config_name, config in self.tesseract_config.items():
                    try:
                        # Convert tuple config to string
                        if isinstance(config, (tuple, list)):
                            config = " ".join(str(x) for x in config)
                        # Escape any remaining quotes
                        config = config.replace('"', '\\"')

                        data = pytesseract.image_to_data(
                            proc_img,
                            config=config,
                            output_type=pytesseract.Output.DICT,
                        )

                        # Get confident words
                        words = []
                        confidences = []

                        for i, (word, conf) in enumerate(
                            zip(data["text"], data["conf"])
                        ):
                            if conf != -1 and float(conf) > 60:
                                if word and word.strip():
                                    words.append(word.strip())
                                    confidences.append(float(conf))

                        if words:
                            text = " ".join(words)
                            avg_confidence = sum(
                                confidences,
                            ) / len(
                                confidences,
                            )

                            if len(text) > 10:  # Only keep substantial text
                                all_results.append(
                                    (
                                        text,
                                        f"{proc_name}/{config_name}",
                                        avg_confidence,
                                    )
                                )
                                logger.debug(
                                    f"Method {proc_name}/{config_name}:\n"
                                    f"Text: {text}\n"
                                    f"Confidence: {avg_confidence:.1f}%"
                                )

                    except Exception as e:
                        logger.warning(
                            f"Error with {proc_name}/{config_name}: {str(e)}"
                        )
                        continue

            if not all_results:
                return "", []

            # Sort by confidence and length
            all_results.sort(key=lambda x: (x[2], len(x[0])), reverse=True)

            # Try to get complete text results first
            complete_results = [
                (text, method, conf)
                for text, method, conf in all_results
                if len(text) > 100
                and conf > 90
                and not method.startswith(
                    "region_",
                )
            ]

            if complete_results:
                best_text = complete_results[0][0]
                top_results = self.post_process_results(complete_results)
            else:
                # Process all results including regions
                top_results = self.post_process_results(all_results)
                best_text = top_results[0][0] if top_results else ""

            return best_text.strip(), top_results

        except Exception as e:
            logger.error(f"Error extracting text: {e}")
            raise

    def translate_text(self, text: str, target_lang: str = "ru") -> str:
        """Translate text to target language"""
        try:
            if target_lang != self.translator.target:
                self.translator = GoogleTranslator(
                    source="auto",
                    target=target_lang,
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
                ContentType="image/jpeg",
            )
            result = [
                f"https://{settings.S3_BUCKET_NAME}.s3.amazonaws.com/",
                f"{filename}",
            ]
            return result.join("")
        except Exception as e:
            logger.error(f"Error uploading to S3: {e}")
            raise
