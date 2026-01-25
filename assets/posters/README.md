# Poster Assets

This directory contains poster designs created for various events and workshops.

## SOS Children's Village - Public Speaking Workshop Poster

**Location:** `assets/posters/sos_public_speaking_workshop.svg` (editable source)

### Poster Details
- **Organization:** SOS Children's Village
- **Event:** Public Speaking Workshop
- **Date:** 28 January
- **Time:** 9:00 AM
- **Location:** IPSI
- **Design Theme:** Cyan and white, modern clean layout
- **Format:** A4 Portrait (210×297mm)

### Files Available
1. **SVG Source File:** `sos_public_speaking_workshop.svg` - Editable vector format
2. **PNG Export:** `sos_public_speaking_workshop.png` - Print-ready at 300 DPI (2480×3508 pixels)

### How to Edit the Poster

The SVG file can be edited with:
- **Inkscape** (free, open-source): https://inkscape.org/
- **Adobe Illustrator**
- **Any SVG-compatible vector editor**
- **Text editors** for simple text changes

### How to Export/Print

#### Option 1: Use the PNG File (Ready to Print)
The PNG file is already exported at 300 DPI and ready for professional printing:
- Simply send `sos_public_speaking_workshop.png` to your printer
- Print at actual size (A4, 210×297mm)
- No scaling required

#### Option 2: Re-export from SVG (if you made changes)

**Using Inkscape:**
1. Open `sos_public_speaking_workshop.svg` in Inkscape
2. Go to File → Export PNG Image
3. Set width to 2480 pixels and height to 3508 pixels (or DPI to 300)
4. Export

**Using command line (rsvg-convert):**
```bash
rsvg-convert -w 2480 -h 3508 sos_public_speaking_workshop.svg -o sos_public_speaking_workshop.png
```

**Using ImageMagick (if available):**
```bash
convert -density 300 sos_public_speaking_workshop.svg -resize 2480x3508 sos_public_speaking_workshop.png
```

### Printing Tips
- **Paper Size:** A4 (210×297mm or 8.27×11.69 inches)
- **Orientation:** Portrait
- **Recommended Paper:** Glossy or matte photo paper for best results
- **Resolution:** 300 DPI ensures sharp, professional quality
- **Color Mode:** RGB for digital display, convert to CMYK for professional offset printing if needed

### Design Specifications
- **Color Palette:**
  - Primary: Cyan (#00bcd4)
  - Secondary: White (#ffffff)
  - Text: Dark gray (#333333)
  - Accent: Light cyan (#f0f9fa)
- **Fonts:** Montserrat (Google Fonts) - weights: 400, 600, 700, 800
- **Layout:** Clean, modern with clear information hierarchy

### File Sizes
- SVG: ~3.5 KB (compact, scalable)
- PNG: ~216 KB (high-resolution, print-ready)
