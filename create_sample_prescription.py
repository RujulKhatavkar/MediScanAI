"""
Creates a sample prescription PNG image for testing MediScanAI.
Run: python create_sample_prescription.py
"""

from PIL import Image, ImageDraw, ImageFont
import os


def create_sample_prescription():
    # Create white image
    img = Image.new("RGB", (800, 600), color="white")
    draw = ImageDraw.Draw(img)

    # Try to use a system font, fall back to default
    try:
        title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 22)
        body_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
        small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 13)
    except Exception:
        title_font = ImageFont.load_default()
        body_font = ImageFont.load_default()
        small_font = ImageFont.load_default()

    # Header
    draw.rectangle([0, 0, 800, 80], fill="#1e3a5f")
    draw.text((30, 15), "CHAMPAIGN MEDICAL CENTER", font=title_font, fill="white")
    draw.text((30, 48), "Dr. Sarah Johnson, MD  |  License: IL-98765  |  (217) 555-0192", font=small_font, fill="#94b8d9")

    # Border
    draw.rectangle([20, 100, 780, 580], outline="#cccccc", width=1)

    # Rx symbol
    draw.text((30, 110), "Rx", font=title_font, fill="#1e3a5f")
    draw.line([20, 145, 780, 145], fill="#cccccc", width=1)

    # Patient info
    draw.text((30, 155), "Patient Name:", font=small_font, fill="#666666")
    draw.text((160, 155), "John Smith", font=body_font, fill="#111111")
    draw.text((420, 155), "DOB:", font=small_font, fill="#666666")
    draw.text((470, 155), "04/12/1985", font=body_font, fill="#111111")

    draw.text((30, 180), "Date:", font=small_font, fill="#666666")
    draw.text((100, 180), "03/14/2026", font=body_font, fill="#111111")
    draw.text((420, 180), "DEA #:", font=small_font, fill="#666666")
    draw.text((480, 180), "BJ1234563", font=body_font, fill="#111111")

    draw.line([20, 210, 780, 210], fill="#eeeeee", width=1)

    # Drug 1
    draw.text((30, 225), "Drug 1:", font=small_font, fill="#666666")
    draw.text((30, 248), "Amoxicillin 500mg Capsules", font=title_font, fill="#1e3a5f")
    draw.text((30, 278), "Sig:", font=small_font, fill="#666666")
    draw.text((65, 278), "Take 1 capsule by mouth THREE times daily for 10 days", font=body_font, fill="#111111")
    draw.text((30, 303), "Disp:", font=small_font, fill="#666666")
    draw.text((70, 303), "#30 capsules", font=body_font, fill="#111111")
    draw.text((220, 303), "Refills:", font=small_font, fill="#666666")
    draw.text((275, 303), "0 (zero)", font=body_font, fill="#cc0000")

    draw.line([30, 325, 770, 325], fill="#eeeeee", width=1)

    # Drug 2
    draw.text((30, 335), "Drug 2:", font=small_font, fill="#666666")
    draw.text((30, 358), "Lisinopril 10mg Tablets", font=title_font, fill="#1e3a5f")
    draw.text((30, 388), "Sig:", font=small_font, fill="#666666")
    draw.text((65, 388), "Take 1 tablet by mouth ONCE daily in the morning", font=body_font, fill="#111111")
    draw.text((30, 413), "Disp:", font=small_font, fill="#666666")
    draw.text((70, 413), "#30 tablets", font=body_font, fill="#111111")
    draw.text((220, 413), "Refills:", font=small_font, fill="#666666")
    draw.text((275, 413), "3", font=body_font, fill="#111111")

    draw.line([30, 435, 770, 435], fill="#eeeeee", width=1)

    # Warnings box
    draw.rectangle([30, 445, 770, 500], fill="#fff8f0", outline="#fcd34d", width=1)
    draw.text((40, 452), "⚠  WARNINGS:", font=small_font, fill="#92400e")
    draw.text((40, 472), "Complete full antibiotic course. Take Lisinopril with caution — monitor blood pressure.", font=small_font, fill="#92400e")

    # Footer / signature
    draw.line([30, 520, 770, 520], fill="#cccccc", width=1)
    draw.text((30, 530), "Prescriber Signature:", font=small_font, fill="#666666")
    draw.text((200, 525), "Dr. Sarah Johnson", font=title_font, fill="#1e3a5f")
    draw.text((550, 530), "NPI: 1234567890", font=small_font, fill="#666666")

    # Watermark diagonal text (light)
    draw.text((250, 560), "FOR INFORMATIONAL / TESTING PURPOSES ONLY", font=small_font, fill="#dddddd")

    img.save("sample_prescription.png")
    print("✅ Created sample_prescription.png — upload this in MediScanAI to test!")


if __name__ == "__main__":
    create_sample_prescription()
