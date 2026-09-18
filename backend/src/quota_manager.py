"""
Quota Management for API Usage
===============================
Tracks API usage and provides fallback when limits are reached.
"""

import os
import json
import datetime
from pathlib import Path

class QuotaManager:
    """Manage API quotas and provide fallback strategies."""
    
    def __init__(self, quota_file='quota_usage.json'):
        self.quota_file = Path(quota_file)
        self.usage_data = self._load_usage_data()
        
        # Monthly limits (Google Free Tier)
        self.monthly_limits = {
            'speech_seconds': 3600,  # 60 minutes = 3600 seconds
            'translate_chars': 500000  # 500K characters
        }
        
    def _load_usage_data(self):
        """Load existing usage data or create new."""
        if self.quota_file.exists():
            try:
                with open(self.quota_file, 'r') as f:
                    return json.load(f)
            except:
                pass
        
        return {
            'current_month': datetime.datetime.now().strftime('%Y-%m'),
            'speech_seconds_used': 0,
            'translate_chars_used': 0,
            'fallback_mode': False
        }
    
    def _save_usage_data(self):
        """Save current usage data."""
        try:
            with open(self.quota_file, 'w') as f:
                json.dump(self.usage_data, f, indent=2)
        except Exception as e:
            print(f"[QUOTA] Warning: Could not save usage data: {e}")
    
    def _reset_if_new_month(self):
        """Reset counters if we're in a new month."""
        current_month = datetime.datetime.now().strftime('%Y-%m')
        if self.usage_data['current_month'] != current_month:
            print(f"[QUOTA] New month detected, resetting counters")
            self.usage_data = {
                'current_month': current_month,
                'speech_seconds_used': 0,
                'translate_chars_used': 0,
                'fallback_mode': False
            }
            self._save_usage_data()
    
    def can_use_speech_api(self, duration_seconds=30):
        """Check if we can use Speech API for given duration."""
        self._reset_if_new_month()
        
        if self.usage_data['fallback_mode']:
            return False
            
        remaining = self.monthly_limits['speech_seconds'] - self.usage_data['speech_seconds_used']
        return remaining >= duration_seconds
    
    def can_use_translate_api(self, text_length=100):
        """Check if we can use Translate API for given text length."""
        self._reset_if_new_month()
        
        if self.usage_data['fallback_mode']:
            return False
            
        remaining = self.monthly_limits['translate_chars'] - self.usage_data['translate_chars_used']
        return remaining >= text_length
    
    def record_speech_usage(self, duration_seconds):
        """Record speech API usage."""
        self.usage_data['speech_seconds_used'] += duration_seconds
        
        # Check if we're approaching limit (90%)
        used_percentage = self.usage_data['speech_seconds_used'] / self.monthly_limits['speech_seconds']
        if used_percentage >= 0.9:
            print(f"[QUOTA] WARNING: Speech API usage at {used_percentage*100:.1f}%")
            if used_percentage >= 1.0:
                self.usage_data['fallback_mode'] = True
                print(f"[QUOTA] Speech API limit reached, enabling fallback mode")
        
        self._save_usage_data()
    
    def record_translate_usage(self, text_length):
        """Record translate API usage."""
        self.usage_data['translate_chars_used'] += text_length
        
        # Check if we're approaching limit (90%)
        used_percentage = self.usage_data['translate_chars_used'] / self.monthly_limits['translate_chars']
        if used_percentage >= 0.9:
            print(f"[QUOTA] WARNING: Translate API usage at {used_percentage*100:.1f}%")
            if used_percentage >= 1.0:
                self.usage_data['fallback_mode'] = True
                print(f"[QUOTA] Translate API limit reached, enabling fallback mode")
        
        self._save_usage_data()
    
    def get_usage_stats(self):
        """Get current usage statistics."""
        self._reset_if_new_month()
        
        speech_pct = (self.usage_data['speech_seconds_used'] / self.monthly_limits['speech_seconds']) * 100
        translate_pct = (self.usage_data['translate_chars_used'] / self.monthly_limits['translate_chars']) * 100
        
        return {
            'speech_usage_percent': speech_pct,
            'translate_usage_percent': translate_pct,
            'speech_remaining_minutes': (self.monthly_limits['speech_seconds'] - self.usage_data['speech_seconds_used']) / 60,
            'translate_remaining_chars': self.monthly_limits['translate_chars'] - self.usage_data['translate_chars_used'],
            'fallback_mode': self.usage_data['fallback_mode']
        }
    
    def force_fallback_mode(self, enabled=True):
        """Manually enable/disable fallback mode."""
        self.usage_data['fallback_mode'] = enabled
        self._save_usage_data()
        print(f"[QUOTA] Fallback mode {'enabled' if enabled else 'disabled'} manually")

# Global quota manager instance
quota_manager = QuotaManager()