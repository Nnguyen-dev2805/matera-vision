import sys
from pathlib import Path
from PIL import Image
from matera.data.extract import extract_pages
from matera.vision.alignment import align_page, AlignmentConfig
from matera.vision.reference import generate_median_reference

pdf_dir = Path('data/example')
pdf_files = list(pdf_dir.glob('*.pdf'))
print(f'Found {len(pdf_files)} PDF files')

aligned_pages = []
align_config = AlignmentConfig(inlier_threshold=0.2)

baseline_pdf = pdf_files[0]
baseline_pages = list(extract_pages(baseline_pdf))
baseline_img = baseline_pages[0].image.convert('RGB')

for f in pdf_files:
    pages = list(extract_pages(f))
    for p in pages:
        try:
            ap = align_page(p, baseline_img, align_config)
            aligned_pages.append(ap)
        except Exception as e:
            print(f'Failed to align {f}: {e}')

if aligned_pages:
    print(f'Generating median reference from {len(aligned_pages)} aligned pages...')
    ref_img = generate_median_reference(aligned_pages)
    ref_img.save('scratch/synthetic_median_reference.png')
    print('Saved to scratch/synthetic_median_reference.png')
else:
    print('No aligned pages generated')
