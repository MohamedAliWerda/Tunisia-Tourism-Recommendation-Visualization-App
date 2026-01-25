# SOS Children's Village Workshop Poster

This directory contains the poster design for the **PUBLIC SPEAKING Workshop** organized by SOS Children's Village.

## Files

- **sos_public_speaking_workshop.svg** - Editable vector source file (recommended for editing and scaling)
- **sos_public_speaking_workshop.png** - Print-ready raster image (300 DPI, 2480×3508 pixels)

## Poster Details

**Workshop Information:**
- **Organization:** SOS Children's Village
- **Workshop:** PUBLIC SPEAKING
- **Date:** 28 January
- **Time:** 9:00 AM
- **Location:** IPSI

**Design Specifications:**
- **Size:** A4 Portrait (210×297mm / 8.27×11.69 inches)
- **Resolution:** 300 DPI (print-ready)
- **Color Theme:** Cyan and White
- **Style:** Clean, modern layout

## How to Use

### For Printing
1. Use the **PNG file** (`sos_public_speaking_workshop.png`) for direct printing
2. The file is sized at 300 DPI, which is print-ready quality
3. Print on A4 paper (210×297mm) in portrait orientation
4. Recommended: Use glossy or high-quality paper for best results

### For Editing
1. Open the **SVG file** (`sos_public_speaking_workshop.svg`) in a vector graphics editor:
   - **Inkscape** (free, open-source) - https://inkscape.org/
   - **Adobe Illustrator** (commercial)
   - **Figma** (online, free tier available) - https://figma.com/
   - Any SVG-compatible editor

2. Edit text, colors, or layout as needed
3. Export to PNG or PDF for printing

### Export Options

**From SVG to PNG (using command line):**
```bash
# Install rsvg-convert if not already installed
sudo apt-get install librsvg2-bin  # On Ubuntu/Debian
brew install librsvg               # On macOS

# Convert to 300 DPI PNG
rsvg-convert -w 2480 -h 3508 -d 300 -p 300 -f png sos_public_speaking_workshop.svg -o output.png
```

**From SVG to PDF (using command line):**
```bash
rsvg-convert -f pdf -o sos_public_speaking_workshop.pdf sos_public_speaking_workshop.svg
```

**Using Inkscape GUI:**
1. Open the SVG file in Inkscape
2. Go to File → Export PNG Image or File → Save As → PDF
3. Set DPI to 300 for print quality
4. Choose A4 size (210×297mm)

## Digital Display
For digital displays (screens, projectors, social media):
- You can use the PNG file directly
- For web use, you may want to create a lower-resolution version:
  ```bash
  rsvg-convert -w 1240 -h 1754 sos_public_speaking_workshop.svg -o web_version.png
  ```

## Color Information
- **Primary Cyan:** #00BCD4
- **Dark Cyan:** #0097A7, #00838F
- **White:** #FFFFFF
- **Light Cyan Background:** #F0F9FA

## License
This poster design is part of the Tunisia Tourism Recommendation & Visualization App repository and follows the repository's license terms.

---
Created: January 2026
For: SOS Children's Village PUBLIC SPEAKING Workshop
