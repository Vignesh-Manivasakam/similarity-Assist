import re
import logging
from app.config import MAX_TOKENS_FOR_TRUNCATION
from pint import UnitRegistry, UndefinedUnitError

logger = logging.getLogger(__name__)
ureg = UnitRegistry()

def lowercase_text(text):
    return text.lower()

def clean_whitespace(text):
    text = re.sub(r'[^\w\s-]', '', text)
    return re.sub(r'\s+', ' ', text).strip()

def add_leading_zero(text):
    return re.sub(r'(?<!\d)(?<!\.)\.(\d+)', r'0.\1', text)

def truncate_tokens(text, tokenizer, max_tokens):
    """Truncate text to max_tokens, return text and truncation flag."""
    tokens = tokenizer.encode(text, add_special_tokens=False)
    if len(tokens) > max_tokens:
        logger.warning(f"Text truncated: {text[:50]}... ({len(tokens)} tokens > {max_tokens})")
        truncated_tokens = tokens[:max_tokens]
        text = tokenizer.decode(truncated_tokens)
        return text, True
    return text, False

def detect_empty_invalid(text, identifier):
    if not isinstance(text, str) or not text.strip():
        logger.warning(f"Invalid/empty text for ID: {identifier}. Skipping.")
        return None
    return text

def remove_hierarchy(text):
    return re.sub(r'^\d+(?:\.\d+)*(?:-\d+)?\s*', '', text)

def normalize_units(text):
    """Normalize units using Pint, with improved error logging."""
    try:
        # Add space between number and unit (e.g., 5V -> 5 V)
        text_with_spaces = re.sub(r'(\d)([a-zA-Z])', r'\1 \2', text)
        unit_pattern = r'(\d+\.?\d*)\s*([a-zA-Z/]+)'
        
        # Using a function for replacement to handle errors gracefully
        def repl(match):
            num, unit = match.groups()
            try:
                quantity = ureg(f"{num} {unit}")
                # Use compact representation '~'
                return f"{quantity.to_base_units():~}"
            except (UndefinedUnitError, DimensionalityError):
                # Log the specific unit that failed instead of silently continuing
                logging.warning(f"Could not normalize unit '{unit}' in text: '{text}'")
                return match.group(0) # Return original on failure
        
        return re.sub(unit_pattern, repl, text_with_spaces)
    except Exception as e:
        logger.error(f"Unexpected error during unit normalization: {e}")
        return text # Return original text on unexpected failure

def preprocess_sentence(entry, tokenizer):
    """Preprocess a single sentence, return cleaned text and metadata."""
    identifier = entry.get("Object_Identifier", "Unknown")
    original_text = entry.get("Object_Text", "")

    text = detect_empty_invalid(original_text, identifier)
    if text is None: return None

    metadata = {"Object_Identifier": identifier, "Original_Text": original_text}
    text = normalize_units(text)
    text = add_leading_zero(text)
    text = lowercase_text(text)
    text = clean_whitespace(text)
    text = remove_hierarchy(text)
    
    # Use centralized token limit from config
    text, is_truncated = truncate_tokens(text, tokenizer, max_tokens=MAX_TOKENS_FOR_TRUNCATION)
    metadata["Cleaned_Text"] = text
    metadata["Truncated"] = is_truncated
    return metadata

def preprocess_data(data, tokenizer):
    """Preprocess a list of sentences, return processed results and count of skipped empty texts."""
    results = []
    skipped_count = 0
    for entry in data:
        processed = preprocess_sentence(entry, tokenizer)
        if processed:
            # Check for emptiness *after* all cleaning steps
            if not processed["Cleaned_Text"].strip():
                logger.warning(f"Empty 'Cleaned_Text' after processing for ID: {processed['Object_Identifier']}")
                skipped_count += 1
                continue
            results.append(processed)
        else:
            skipped_count += 1
    return results, skipped_count