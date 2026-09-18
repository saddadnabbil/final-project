# backend/src/corrections.py
"""
Koreksi otomatis untuk transkripsi dan terjemahan Sundanese.
File ini berisi semua dictionary koreksi yang bisa ditambahkan manual.

=== STRUKTUR FILE ===
1. TRANSCRIPTION CORRECTIONS - Koreksi output Whisper (Sundanese)
2. TRANSLATION CORRECTIONS - Koreksi output NLLB (Indonesian)

=== CARA MENAMBAH KOREKSI ===
1. Cari section yang sesuai (TRANSCRIPTION atau TRANSLATION)
2. Tambahkan entry baru dengan format: "kesalahan": "koreksi",
3. Restart backend untuk apply perubahan

=== PERFORMANCE ===
~1-5ms per proses (tidak signifikan)
"""

import re
import numpy as np

# =============================================================================
# BAGIAN 1: TRANSCRIPTION CORRECTIONS (Koreksi output Whisper)
# =============================================================================

# -----------------------------------------------------------------------------
# PHRASE-LEVEL TRANSCRIPTION CORRECTIONS (Multi-word)
# Prioritas lebih tinggi - diproses duluan
# -----------------------------------------------------------------------------

TRANSCRIPTION_PHRASE_CORRECTIONS = {
    # --- Sapaan / Greetings ---
    "wilu jeung enjing": "wilujeng enjing",
    "wilu jeng enjing": "wilujeng enjing",
    "wilujung enjing": "wilujeng enjing",
    "wilu jung enjing": "wilujeng enjing",
    "wilujeung enjing": "wilujeng enjing",
    "wilu jeung wengi": "wilujeng wengi",
    "wilu jeng wengi": "wilujeng wengi",
    "wilujung wengi": "wilujeng wengi",
    "wilu jeung sonten": "wilujeng sonten",
    "wilu jeng sonten": "wilujeng sonten",
    "wilujung sonten": "wilujeng sonten",

    # --- Damang / Kabar ---
    "kuma hadamang": "kumaha damang",
    "kumaha da mang": "kumaha damang",
    "kuma ha damang": "kumaha damang",

    # --- Babaturan / Teman ---
    "jengba baturana": "jeung babaturana",
    "jeng mba babaturana": "jeung babaturana",
    "jeng ba baturana": "jeung babaturana",
    "jeng babaturana": "jeung babaturana",
    "babaturan na": "babaturanna",
    "babaturan anna": "babaturanna",
    "ba baturan na": "babaturanna",

    # --- Haturnuhun / Terima kasih ---
    "hatur nuhun": "haturnuhun",
    "ha tur nuhun": "haturnuhun",
    "hatur nu hun": "haturnuhun",

    # --- Punten / Permisi ---
    "pun ten": "punten",

    # --- Mangga / Silakan ---
    "mang ga": "mangga",

    # --- Abdi / Saya ---
    "ab di": "abdi",

    # --- Common phrases ---
    "sok maen": "sok maen",
    "sok main": "sok maen",

    # ==========================================================================
    # TAMBAHKAN KOREKSI TRANSCRIPTION PHRASE BARU DI BAWAH INI
    # Format: "kesalahan dari whisper": "koreksi yang benar",
    # ==========================================================================

    # --- Hallucinated names (Whisper sering salah dengar sebagai nama) ---
    "anulu masung": "anu lumangsung",
    "anulu masuneng": "anu lumangsung",
    "anu lumasung": "anu lumangsung",

    "linten minggon": "dinten minggon",
    "linten mingon": "dinten minggon",

    # --- Common word errors ---
    "we nye": "wengi",
    "wenye": "wengi",
    "kawuruan": "kahuruan",
    "kemana-mana": "kamana-mana",
    "kemana mana": "kamana mana",
    "cerita kahuruan": "carita kahuruan",

    # --- Geus/Ges ---
    " ges ": " geus ",
    "ges sumebar": "geus sumebar",
}

# -----------------------------------------------------------------------------
# WORD-LEVEL TRANSCRIPTION CORRECTIONS (Single word)
# -----------------------------------------------------------------------------

TRANSCRIPTION_WORD_CORRECTIONS = {
    # --- Hallucinated greetings (remove) ---
    "hai ": "",
    "halo ": "",
    "helo ": "",

    # --- Common word fixes ---
    "jengba": "jeung",
    "baturana": "babaturana",

    # ==========================================================================
    # TAMBAHKAN KOREKSI TRANSCRIPTION WORD BARU DI BAWAH INI
    # Format: "kesalahan": "koreksi",
    # ==========================================================================

    # --- Common Sundanese word fixes ---
    "cerita": "carita",
    "kawuruan": "kahuruan",
    "wenye": "wengi",
    "ges": "geus",
    "kemana": "kamana",
}

# -----------------------------------------------------------------------------
# HALLUCINATED WORDS (Words to remove at start of sentence)
# -----------------------------------------------------------------------------

HALLUCINATED_STARTS = ['hai', 'halo', 'helo', 'hello', 'hi']


# =============================================================================
# BAGIAN 2: TRANSLATION CORRECTIONS (Koreksi output NLLB)
# =============================================================================

# -----------------------------------------------------------------------------
# SUNDA TO INDONESIAN WORD CORRECTIONS
# Untuk memperbaiki kata-kata Sunda yang tidak diterjemahkan NLLB
# -----------------------------------------------------------------------------

SUNDA_TO_INDO_CORRECTIONS = {
    # --- Kata Kerja / Verbs ---
    "tingkalang": "senang",
    "ternyanyi": "bernyanyi",
    "kernyanyi": "bernyanyi",
    "meninggali": "melihat",
    "ningali": "melihat",
    "kerneangan": "lagi mencari",
    "keurneangan": "lagi mencari",
    "neangan": "mencari",
    "milarian": "mencari",
    "nyiar": "mencari",
    "nyusud": "mengikuti",
    "ngiring": "mengikuti",
    "ngumbara": "berjalan-jalan",
    "ngajak": "mengajak",
    "ngomong": "berbicara",
    "ngadug": "menunggu",
    "ngantosan": "menunggu",
    "dahar": "makan",
    "tuang": "makan",
    "ngebon": "bertani",
    "ngajer": "bekerja",
    "ngalakukeun": "melakukan",
    "ngalakonan": "melakukan",
    "ngalaksanakeun": "melakukan",

    # --- Kata Benda / Nouns ---
    "sawah": "sawah",
    "ladang": "ladang",
    "kebon": "kebun",
    "imah": "rumah",
    "bapa": "ayah",
    "mama": "ibu",
    "lanceuk": "kakak",
    "adi": "adik",
    "kaka": "kakak",

    # --- Kata Ganti / Pronouns ---
    "kuring": "saya",
    "abdi": "saya",
    "urang": "kami",
    "manéh": "kamu",
    "anjeun": "anda",
    "manehna": "dia",
    "aranjeun": "kalian",

    # --- Kata Sifat / Adjectives ---
    "hadé": "bagus",
    "goréng": "buruk",
    "badag": "besar",
    "leutik": "kecil",
    "panjang": "panjang",
    "pondok": "pendek",
    "bageur": "sehat",
    "damang": "sehat",
    "cageur": "sehat",

    # --- Kata Keterangan / Adverbs ---
    "ayeuna": "sekarang",
    "kamari": "kemarin",
    "énjing": "besok",
    "isuk": "pagi",
    "siang": "siang",
    "sontén": "sore",
    "wengi": "malam",

    # --- Kata Umum / Common words ---
    "tipi": "televisi",
    "duit": "uang",
    "pemanjar": "muka",
    "pamangjar": "muka",
    "jangkredit": "kredit",
    "kudu": "harus",
    "kedah": "harus",
    "babaturana": "teman-temannya",
    "babaturanana": "teman-temannya",

    # --- Kata Tanya / Question words ---
    "naha": "apa",
    "kumaha": "bagaimana",
    "saha": "siapa",
    "dimana": "di mana",
    "iraaha": "kapan",
    "sabaraha": "berapa",

    # --- Kata Depan / Prepositions ---
    "ka": "ke",
    "ti": "dari",
    "jeung": "dengan",
    "atawa": "atau",
    "sareng": "dengan",

    # ==========================================================================
    # TAMBAHKAN KOREKSI TRANSLATION WORD BARU DI BAWAH INI
    # Format: "kata_sunda": "terjemahan_indonesia",
    # ==========================================================================

}

# -----------------------------------------------------------------------------
# TRANSLATION PHRASE CORRECTIONS (Multi-word)
# Koreksi untuk frasa yang salah diterjemahkan NLLB
# -----------------------------------------------------------------------------

TRANSLATION_PHRASE_CORRECTIONS = {
    "duit pemanjar": "uang muka",
    "duit pamangjar": "uang muka",
    "uang pinjaman": "uang muka",
    "jang kredit": "kredit",
    "jangkredit": "kredit",
    "kredit mobil": "kredit mobil",
    "jeng babaturana": "dengan teman-temannya",
    "main bowling": "bermain bowling",
    "sok main": "sering bermain",
    "ting haruleng": "senang",

    # ==========================================================================
    # TAMBAHKAN KOREKSI TRANSLATION PHRASE BARU DI BAWAH INI
    # Format: "frasa_salah": "frasa_benar",
    # ==========================================================================

    # --- Fix NLLB wrong translations ---
    # NLLB sering salah translate kata Indonesia yang sudah benar
    "saya harus mengatakan": "saya harus begini",
    "harus mengatakan ini": "harus begini",
    "harus mengatakan": "harus begini",

    # Common Indonesian words wrongly translated
    "mengatakan ini": "begini",
    "mengatakan itu": "begitu",
}

# -----------------------------------------------------------------------------
# REGEX PATTERNS untuk cleaning
# -----------------------------------------------------------------------------

CLEANUP_PATTERNS = [
    (r'\b(\w+)\s+\1\b', r'\1'),  # "nonton nonton" → "nonton"
    (r'\bting\b', ''),           # Remove standalone "ting"
    (r'\s+', ' '),               # Multiple spaces → single space
]


# =============================================================================
# FUNGSI-FUNGSI HELPER
# =============================================================================

# Pre-compile regex patterns for better performance
_compiled_transcription_patterns = None
_compiled_translation_patterns = None


def _compile_transcription_patterns():
    """Compile transcription regex patterns once."""
    global _compiled_transcription_patterns

    if _compiled_transcription_patterns is None:
        phrase_patterns = [
            (re.compile(re.escape(wrong), re.IGNORECASE), correct)
            for wrong, correct in TRANSCRIPTION_PHRASE_CORRECTIONS.items()
        ]
        word_patterns = [
            (re.compile(re.escape(wrong), re.IGNORECASE), correct)
            for wrong, correct in TRANSCRIPTION_WORD_CORRECTIONS.items()
        ]
        _compiled_transcription_patterns = (phrase_patterns, word_patterns)

    return _compiled_transcription_patterns


def _compile_translation_patterns():
    """Compile translation regex patterns once."""
    global _compiled_translation_patterns

    if _compiled_translation_patterns is None:
        _compiled_translation_patterns = [
            (re.compile(r'\b' + re.escape(wrong) + r'\b', re.IGNORECASE), correct)
            for wrong, correct in {**SUNDA_TO_INDO_CORRECTIONS, **TRANSLATION_PHRASE_CORRECTIONS}.items()
        ]

    return _compiled_translation_patterns


# =============================================================================
# FUNGSI UTAMA - TRANSCRIPTION
# =============================================================================

def apply_transcription_corrections(text: str) -> str:
    """
    Apply all transcription corrections to Whisper output.

    Args:
        text: Raw transcription from Whisper

    Returns:
        Corrected transcription
    """
    if not text:
        return ""

    normalized = text.strip()
    phrase_patterns, word_patterns = _compile_transcription_patterns()

    # Apply phrase corrections first (higher priority)
    for pattern, correct in phrase_patterns:
        normalized = pattern.sub(correct, normalized)

    # Apply word corrections
    for pattern, correct in word_patterns:
        normalized = pattern.sub(correct, normalized)

    # Clean up multiple spaces
    normalized = re.sub(r'\s+', ' ', normalized)

    return normalized.strip()


def remove_hallucinated_greeting(text: str, confidence_score: float = None) -> str:
    """
    Remove hallucinated greetings at the start of text.

    Args:
        text: Transcription text
        confidence_score: Confidence score for first word (optional)

    Returns:
        Text with hallucinated greeting removed
    """
    if not text:
        return ""

    words = text.split()
    if not words:
        return ""

    first_word = words[0].lower()

    if first_word in HALLUCINATED_STARTS:
        should_remove = True
        if confidence_score is not None and confidence_score > 0.3:
            should_remove = False

        if should_remove:
            words = words[1:]
            return ' '.join(words)

    return text


# =============================================================================
# FUNGSI UTAMA - TRANSLATION
# =============================================================================

def post_process_translation(text: str, source_lang: str = "sun", target_lang: str = "ind", original_text: str = None) -> str:
    """
    Apply rule-based corrections to improve NLLB translation quality.

    Args:
        text: Translation result from NLLB model
        source_lang: Source language code
        target_lang: Target language code
        original_text: Original input text (for fallback processing)

    Returns:
        Corrected translation
    """
    if not text or len(text.strip()) == 0:
        return text

    result = text

    # Apply all translation corrections
    patterns = _compile_translation_patterns()
    for pattern, correct in patterns:
        result = pattern.sub(correct, result)

    # If translation is identical to original (NLLB failed), apply word-by-word
    if original_text and text.strip() == original_text.strip() and source_lang == "sun" and target_lang == "ind":
        print(f"[INFO] NLLB translation identical to input, applying rule-based fallback")
        result = _translate_word_by_word(original_text)

    # Apply cleanup patterns
    for pattern, replacement in CLEANUP_PATTERNS:
        result = re.sub(pattern, replacement, result)

    # Clean up and capitalize
    result = result.strip()
    if result:
        result = result[0].upper() + result[1:]

    return result


def _translate_word_by_word(text: str) -> str:
    """
    Fallback word-by-word translation when NLLB fails.

    Args:
        text: Sundanese text

    Returns:
        Indonesian translation
    """
    if not text:
        return text

    result = text.lower()

    # Apply phrase corrections first
    for wrong, correct in TRANSLATION_PHRASE_CORRECTIONS.items():
        result = re.sub(r'\b' + re.escape(wrong) + r'\b', correct, result, flags=re.IGNORECASE)

    # Word-by-word translation
    words = result.split()
    translated = []

    for word in words:
        clean_word = re.sub(r'[^\w]', '', word)
        punctuation = word[len(clean_word):]
        translation = SUNDA_TO_INDO_CORRECTIONS.get(clean_word, clean_word)
        translated.append(translation + punctuation)

    result = ' '.join(translated)

    # Fix double words
    result = re.sub(r'\blagi mencari lagi\b', 'lagi mencari', result)
    result = re.sub(r'\bsaya saya\b', 'saya', result)

    return result


# =============================================================================
# STATISTIK (untuk debugging)
# =============================================================================

def get_correction_stats():
    """Get statistics about available corrections."""
    return {
        "transcription": {
            "phrase_corrections": len(TRANSCRIPTION_PHRASE_CORRECTIONS),
            "word_corrections": len(TRANSCRIPTION_WORD_CORRECTIONS),
            "hallucinated_words": len(HALLUCINATED_STARTS),
        },
        "translation": {
            "word_corrections": len(SUNDA_TO_INDO_CORRECTIONS),
            "phrase_corrections": len(TRANSLATION_PHRASE_CORRECTIONS),
        },
        "total": (
            len(TRANSCRIPTION_PHRASE_CORRECTIONS) +
            len(TRANSCRIPTION_WORD_CORRECTIONS) +
            len(SUNDA_TO_INDO_CORRECTIONS) +
            len(TRANSLATION_PHRASE_CORRECTIONS)
        )
    }


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    stats = get_correction_stats()
    print("=" * 60)
    print("CORRECTIONS LOADED")
    print("=" * 60)
    print(f"\nTranscription Corrections:")
    print(f"  - Phrase: {stats['transcription']['phrase_corrections']}")
    print(f"  - Word: {stats['transcription']['word_corrections']}")
    print(f"  - Hallucinated: {stats['transcription']['hallucinated_words']}")
    print(f"\nTranslation Corrections:")
    print(f"  - Word: {stats['translation']['word_corrections']}")
    print(f"  - Phrase: {stats['translation']['phrase_corrections']}")
    print(f"\nTotal: {stats['total']} corrections")

    print("\n" + "=" * 60)
    print("TEST TRANSCRIPTION CORRECTIONS")
    print("=" * 60)
    test_transcriptions = [
        "wilu jeung enjing kumaha damang",
        "Kang Indri sok main bowling jeng babaturana",
        "hai hatur nuhun",
    ]
    for test in test_transcriptions:
        corrected = apply_transcription_corrections(test)
        print(f"  '{test}'")
        print(f"  -> '{corrected}'")
        print()

    print("=" * 60)
    print("TEST TRANSLATION CORRECTIONS")
    print("=" * 60)
    test_translations = [
        "Kang Indri sok main bowling jeung babaturana",
        "Abdi kudu tuang ayeuna",
    ]
    for test in test_translations:
        corrected = post_process_translation(test)
        print(f"  '{test}'")
        print(f"  -> '{corrected}'")
        print()
