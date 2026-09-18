"""
ASR Corrections for Sundanese Speech Recognition
=================================================
Post-processing corrections for Google Speech-to-Text ASR results.
Fixes common misrecognitions in Sundanese language.
"""

import re

# Common ASR mistakes and corrections for Sundanese
# Format: (wrong_pattern, correct_text)
ASR_CORRECTIONS = [
    # Greetings - very common mistakes
    (r'\bwilu\s+jeung\s+enjing\b', 'wilujeng enjing'),
    (r'\bwilu\s+jeung\b', 'wilujeng'),
    (r'\bwilujeng\s+enjing\b', 'wilujeng enjing'),  # Already correct, but ensure spacing
    (r'\bwilu\s+jeng\b', 'wilujeng'),
    (r'\bwiluh\s+jeng\b', 'wilujeng'),

    # Common Sundanese words often misheard
    (r'\bkumaha\s+damang\b', 'kumaha damang'),  # How are you
    (r'\bhatur\s+nuhun\b', 'hatur nuhun'),  # Thank you
    (r'\bhator\s+nuhun\b', 'hatur nuhun'),
    (r'\bha\s+tur\s+nuhun\b', 'hatur nuhun'),

    # Numbers and common words
    (r'\bsa\s+rebu\b', 'sarebu'),  # thousand
    (r'\bsa\s+juta\b', 'sajuta'),  # million
    (r'\bti\s+ga\b', 'tiga'),
    (r'\bdu\s+a\b', 'dua'),

    # Place-related
    (r'\bka\s+sakola\b', 'ka sakola'),  # to school (correct spacing)
    (r'\bdi\s+bumi\b', 'di bumi'),  # at home

    # Action verbs commonly misheard
    (r'\bnga\s+ring\b', 'ngiring'),  # accompany
    (r'\bnga\s+dahar\b', 'ngadahar'),  # eat
    (r'\bnga\s+leueut\b', 'ngaleueut'),  # drink

    # Common phrases
    (r'\babdi\s+te\b', 'abdi teh'),  # I am
    (r'\bman\s+eh\b', 'maneh'),  # you
    (r'\bma\s+neh\b', 'maneh'),

    # Fix double spaces
    (r'\s{2,}', ' '),
]

# Word-level corrections (exact match)
WORD_CORRECTIONS = {
    'wilu': 'wilujeng',
    'wilujeung': 'wilujeng',
    'wiluh': 'wilujeng',
    'kumha': 'kumaha',
    'dmang': 'damang',
    'hator': 'hatur',
    'nuhunn': 'nuhun',
    'sakla': 'sakola',
    'bde': 'bade',
    'ayena': 'ayeuna',
    'ieu': 'ieu',
    'teh': 'teh',
    'mah': 'mah',
}

# Phrase corrections (for context-aware fixes)
PHRASE_CORRECTIONS = {
    'wilu jeung enjing': 'wilujeng enjing',
    'wilu jeng enjing': 'wilujeng enjing',
    'selamat pagi sunda': 'wilujeng enjing',
    'kumaha kabar': 'kumaha damang',
    'apa kabar sunda': 'kumaha damang',
}


def post_process_asr(text: str) -> str:
    """
    Apply post-processing corrections to ASR output.

    Args:
        text: Raw ASR transcription

    Returns:
        Corrected transcription
    """
    if not text:
        return text

    original = text
    result = text.lower().strip()

    # Apply phrase corrections first (longer matches)
    for wrong, correct in PHRASE_CORRECTIONS.items():
        if wrong in result:
            result = result.replace(wrong, correct)

    # Apply regex-based corrections
    for pattern, replacement in ASR_CORRECTIONS:
        result = re.sub(pattern, replacement, result, flags=re.IGNORECASE)

    # Apply word-level corrections
    words = result.split()
    corrected_words = []
    for word in words:
        corrected = WORD_CORRECTIONS.get(word.lower(), word)
        corrected_words.append(corrected)
    result = ' '.join(corrected_words)

    # Clean up
    result = re.sub(r'\s+', ' ', result).strip()

    # Capitalize first letter
    if result:
        result = result[0].upper() + result[1:]

    if result != original:
        print(f"[ASR_CORRECTION] '{original}' -> '{result}'")

    return result


def correct_sundanese_greeting(text: str) -> str:
    """
    Specifically fix common Sundanese greetings.
    """
    corrections = {
        'wilu jeung enjing': 'wilujeng enjing',
        'wilu jeng enjing': 'wilujeng enjing',
        'wilujeung enjing': 'wilujeng enjing',
        'wilu jeung siang': 'wilujeng siang',
        'wilu jeung sonten': 'wilujeng sonten',
        'wilu jeung wengi': 'wilujeng wengi',
    }

    result = text.lower()
    for wrong, correct in corrections.items():
        result = result.replace(wrong, correct)

    return result.capitalize() if result else result
