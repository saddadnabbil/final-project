#!/usr/bin/env python3
"""
Script untuk monitoring quota Google API dan status sistem.
Gunakan script ini untuk check status sebelum sidang.
"""

import sys
import os
sys.path.append('src')

from quota_manager import quota_manager
from datetime import datetime

def check_system_status():
    """Check comprehensive system status."""
    print("🎓 SKRIPSI TRANSLATOR - SYSTEM STATUS")
    print("=" * 60)
    print(f"📅 Check Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    
    # Quota Status
    stats = quota_manager.get_usage_stats()
    print("📊 API QUOTA STATUS:")
    print(f"   🎤 Speech Recognition: {stats['speech_usage_percent']:.1f}% used")
    print(f"      ├─ Remaining: {stats['speech_remaining_minutes']:.1f} minutes")
    print(f"      └─ Max per month: 60 minutes")
    print()
    print(f"   🌍 Translation: {stats['translate_usage_percent']:.1f}% used")
    print(f"      ├─ Remaining: {stats['translate_remaining_chars']:,} characters")
    print(f"      └─ Max per month: 500,000 characters")
    print()
    
    # Status Assessment
    speech_pct = stats['speech_usage_percent']
    translate_pct = stats['translate_usage_percent']
    max_usage = max(speech_pct, translate_pct)
    
    if stats['fallback_mode']:
        status = "🚨 FALLBACK MODE ACTIVE"
        color = "RED"
    elif max_usage > 90:
        status = "⚠️  CRITICAL - Very Close to Limit"
        color = "RED"
    elif max_usage > 80:
        status = "⚠️  WARNING - Approaching Limit"
        color = "YELLOW"
    elif max_usage > 50:
        status = "📊 MODERATE - Normal Usage"
        color = "BLUE"
    else:
        status = "✅ EXCELLENT - Low Usage"
        color = "GREEN"
    
    print(f"🎯 OVERALL STATUS: {status}")
    print()
    
    # Recommendations
    print("💡 RECOMMENDATIONS:")
    if stats['fallback_mode']:
        print("   ⚠️  Fallback mode is active!")
        print("   📝 To disable: python -c \"from src.quota_manager import quota_manager; quota_manager.force_fallback_mode(False)\"")
        print("   💰 Or consider upgrading to paid API")
    elif max_usage > 90:
        print("   🚨 URGENT: Enable fallback mode before sidang!")
        print("   📝 Command: python -c \"from src.quota_manager import quota_manager; quota_manager.force_fallback_mode(True)\"")
    elif max_usage > 80:
        print("   ⚠️  Monitor usage closely")
        print("   🎯 Consider limiting demo recordings")
    else:
        print("   ✅ System ready for sidang")
        print("   🎯 You can safely demo the system")
    
    print()
    
    # Usage Estimation
    print("📈 USAGE ESTIMATION FOR SIDANG:")
    print("   🎤 1 recording (30 sec) ≈ 0.5 minute quota")
    print("   🌍 1 translation (50 chars) ≈ 50 characters quota")
    print(f"   🎯 Estimated demo capacity: ~{int(stats['speech_remaining_minutes'] * 2)} recordings")
    print()
    
    # File Status
    if os.path.exists('quota_usage.json'):
        print("📄 Quota tracking file: ✅ Active")
    else:
        print("📄 Quota tracking file: ⚠️  Will be created on first use")
    
    print("=" * 60)
    return stats

def force_fallback(enable=True):
    """Force enable/disable fallback mode."""
    quota_manager.force_fallback_mode(enable)
    status = "ENABLED" if enable else "DISABLED"
    print(f"✅ Fallback mode {status}")

def force_true_offline(enable=True):
    """Force enable/disable true offline mode."""
    from enhanced_backend_service import get_enhanced_backend_service
    service = get_enhanced_backend_service()
    service.enable_true_offline_mode(enable)
    status = "ENABLED" if enable else "DISABLED"
    print(f"✅ True offline mode {status}")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        if sys.argv[1] == "fallback-on":
            force_fallback(True)
        elif sys.argv[1] == "fallback-off":
            force_fallback(False)
        elif sys.argv[1] == "true-offline-on":
            force_true_offline(True)
        elif sys.argv[1] == "true-offline-off":
            force_true_offline(False)
        else:
            print("Usage: python check_quota.py [fallback-on|fallback-off|true-offline-on|true-offline-off]")
    else:
        check_system_status()