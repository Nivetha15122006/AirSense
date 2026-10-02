"""
Health Advisory and AQI Categorization Module based on CPCB (Central Pollution Control Board) standards.
Provides color codes, severity categorization, and plain-language health guidance.
"""

from typing import Dict, Any

def get_season(month: int) -> int:
    """
    Returns Indian meteorological season:
    1: Winter (Dec, Jan, Feb)
    2: Summer / Pre-monsoon (Mar, Apr, May)
    3: Monsoon (Jun, Jul, Aug, Sep)
    4: Post-monsoon / Autumn (Oct, Nov)
    """
    if month in [12, 1, 2]:
        return 1
    elif month in [3, 4, 5]:
        return 2
    elif month in [6, 7, 8, 9]:
        return 3
    else:
        return 4

def get_aqi_category(aqi: float) -> str:
    """Categorizes numerical AQI value into official CPCB bands."""
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Satisfactory"
    elif aqi <= 200:
        return "Moderate"
    elif aqi <= 300:
        return "Poor"
    elif aqi <= 400:
        return "Very Poor"
    else:
        return "Severe"

def get_aqi_advisory(aqi: float) -> Dict[str, Any]:
    """
    Returns category, theme color, badge style, and tailored plain-language
    health advisories for the general public, sensitive groups, and outdoor workers.
    """
    category = get_aqi_category(aqi)
    
    if category == "Good":
        return {
            "aqi": round(aqi, 1),
            "category": "Good",
            "range": "0 - 50",
            "color": "#10B981",         # Emerald green
            "bg_color": "#ECFDF5",
            "border_color": "#A7F3D0",
            "text_color": "#065F46",
            "summary": "Air quality is ideal. Minimal or no health risk.",
            "general_population": "Enjoy your normal outdoor activities and open windows for ventilation.",
            "sensitive_groups": "Safe for everyone, including individuals with asthma and respiratory conditions.",
            "outdoor_workers": "Normal work routine with no special precautions required.",
            "actions": [
                "Great day for outdoor exercise, jogging, and sports.",
                "Natural indoor ventilation is recommended."
            ]
        }
    elif category == "Satisfactory":
        return {
            "aqi": round(aqi, 1),
            "category": "Satisfactory",
            "range": "51 - 100",
            "color": "#84CC16",         # Lime green
            "bg_color": "#F7FEE7",
            "border_color": "#D9F99D",
            "text_color": "#365314",
            "summary": "Air quality is acceptable. Minor discomfort possible for highly sensitive individuals.",
            "general_population": "Acceptable air quality. Unusually sensitive people should consider reducing prolonged outdoor exertion.",
            "sensitive_groups": "People with persistent respiratory symptoms or asthma should monitor comfort levels.",
            "outdoor_workers": "Normal activities. Stay hydrated and take regular rest breaks.",
            "actions": [
                "Safe for most outdoor activities.",
                "Sensitive individuals should keep standard inhalers or medications nearby."
            ]
        }
    elif category == "Moderate":
        return {
            "aqi": round(aqi, 1),
            "category": "Moderate",
            "range": "101 - 200",
            "color": "#F59E0B",         # Amber/Yellow
            "bg_color": "#FFFBEB",
            "border_color": "#FDE68A",
            "text_color": "#92400E",
            "summary": "Breathing discomfort possible for people with lung disease, asthma, and heart conditions.",
            "general_population": "Most people can continue normal routine, but consider limiting intense long-duration outdoor workouts.",
            "sensitive_groups": "Children, the elderly, and asthma patients should avoid prolonged outdoor physical exertion.",
            "outdoor_workers": "Take intermittent breaks in well-ventilated or shaded areas.",
            "actions": [
                "Wear a basic protective mask (cloth or surgical) during peak traffic hours.",
                "Keep windows closed during morning and evening rush hours."
            ]
        }
    elif category == "Poor":
        return {
            "aqi": round(aqi, 1),
            "category": "Poor",
            "range": "201 - 300",
            "color": "#F97316",         # Orange
            "bg_color": "#FFF7ED",
            "border_color": "#FFEDD5",
            "text_color": "#9A3412",
            "summary": "Unhealthy air. Breathing discomfort to most people upon prolonged exposure.",
            "general_population": "Reduce heavy outdoor exertion. Shift strenuous cardio workouts indoors.",
            "sensitive_groups": "Children, elderly, and respiratory patients should strictly avoid outdoor physical activity.",
            "outdoor_workers": "Wear an N95/KN95 respirator mask and reduce heavy manual labor where possible.",
            "actions": [
                "Wear an N95 mask outdoors.",
                "Run air purifiers indoors if available.",
                "Avoid burning garbage or biomass."
            ]
        }
    elif category == "Very Poor":
        return {
            "aqi": round(aqi, 1),
            "category": "Very Poor",
            "range": "301 - 400",
            "color": "#EF4444",         # Red
            "bg_color": "#FEF2F2",
            "border_color": "#FECACA",
            "text_color": "#991B1B",
            "summary": "Health warning: Significant respiratory distress upon prolonged exposure.",
            "general_population": "Avoid outdoor activities. Keep doors and windows tightly closed.",
            "sensitive_groups": "Remain strictly indoors. High risk of asthma attacks and cardiovascular strain.",
            "outdoor_workers": "Mandatory N95 respirator mask usage; restrict outdoor work shifts.",
            "actions": [
                "Wear a certified N95 respirator when outdoors is unavoidable.",
                "Keep indoor air filtered; wet mop floors to capture settling dust.",
                "Seek medical advice if experiencing persistent cough or chest tightness."
            ]
        }
    else:  # Severe
        return {
            "aqi": round(aqi, 1),
            "category": "Severe",
            "range": "401 - 500+",
            "color": "#7F1D1D",         # Dark Red / Maroon
            "bg_color": "#450A0A",
            "border_color": "#991B1B",
            "text_color": "#FEE2E2",
            "summary": "Emergency conditions. Serious health impact on entire population.",
            "general_population": "Complete avoidance of outdoor physical activity. Stay indoors with air purification.",
            "sensitive_groups": "Severe medical risk. Strict confinement to clean indoor environments.",
            "outdoor_workers": "Outdoor manual work should be suspended or heavily curtailed under medical oversight.",
            "actions": [
                "Do NOT exercise or walk outdoors.",
                "Seal doors and windows; operate HEPA air purifiers.",
                "Consult emergency healthcare promptly for any breathing difficulty."
            ]
        }
