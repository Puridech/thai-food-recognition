"""
Check data quality and find problematic images
"""
import os
import cv2
import numpy as np
from PIL import Image
import json

def check_image_quality(image_path):
    """Check if image has good quality"""
    issues = []
    
    try:
        # Load image
        img = cv2.imread(image_path)
        pil_img = Image.open(image_path)
        
        # Check 1: Size
        if pil_img.size != (224, 224):
            issues.append(f"Wrong size: {pil_img.size}")
        
        # Check 2: Corrupted
        if img is None:
            issues.append("Corrupted file")
            return issues
        
        # Check 3: Blur detection (Laplacian variance)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 50:  # Very blurry
            issues.append(f"Blurry image (variance: {laplacian_var:.1f})")
        
        # Check 4: Too dark or too bright
        brightness = np.mean(img)
        if brightness < 30:
            issues.append(f"Too dark (brightness: {brightness:.1f})")
        elif brightness > 225:
            issues.append(f"Too bright (brightness: {brightness:.1f})")
        
        # Check 5: Grayscale (should be color)
        if len(img.shape) < 3 or img.shape[2] != 3:
            issues.append("Not RGB image")
        
        return issues
        
    except Exception as e:
        return [f"Error: {str(e)}"]

def scan_dataset(base_path='../data/training'):
    """Scan entire dataset for quality issues"""
    
    results = {
        'summary': {
            'total_images': 0,
            'good_images': 0,
            'problematic_images': 0
        },
        'issues_by_dish': {},
        'problematic_files': []
    }
    
    for split in ['train', 'valid', 'test']:
        split_path = os.path.join(base_path, split)
        
        if not os.path.exists(split_path):
            continue
        
        print(f"\n📂 Scanning {split}...")
        
        for dish in os.listdir(split_path):
            dish_path = os.path.join(split_path, dish)
            
            if not os.path.isdir(dish_path):
                continue
            
            print(f"   Checking: {dish}...", end=' ')
            
            if dish not in results['issues_by_dish']:
                results['issues_by_dish'][dish] = []
            
            for filename in os.listdir(dish_path):
                if not filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                    continue
                
                image_path = os.path.join(dish_path, filename)
                results['summary']['total_images'] += 1
                
                issues = check_image_quality(image_path)
                
                if issues:
                    results['summary']['problematic_images'] += 1
                    problem_info = {
                        'path': image_path,
                        'dish': dish,
                        'split': split,
                        'filename': filename,
                        'issues': issues
                    }
                    results['problematic_files'].append(problem_info)
                    results['issues_by_dish'][dish].append(problem_info)
                else:
                    results['summary']['good_images'] += 1
            
            issue_count = len([p for p in results['problematic_files'] 
                              if p['dish'] == dish and p['split'] == split])
            print(f"{'✅' if issue_count == 0 else '⚠️'} {issue_count} issues")
    
    return results

def generate_report(results):
    """Generate quality report"""
    print("\n" + "="*80)
    print("📊 DATA QUALITY REPORT")
    print("="*80)
    
    summary = results['summary']
    print(f"\n📈 Summary:")
    print(f"   Total images:        {summary['total_images']:,}")
    print(f"   Good images:         {summary['good_images']:,} ({summary['good_images']/summary['total_images']*100:.1f}%)")
    print(f"   Problematic images:  {summary['problematic_images']:,} ({summary['problematic_images']/summary['total_images']*100:.1f}%)")
    
    if results['problematic_files']:
        print(f"\n⚠️  Top 10 Problematic Images:")
        print("-"*80)
        for i, problem in enumerate(results['problematic_files'][:10], 1):
            print(f"{i}. {problem['dish']}/{problem['filename']}")
            print(f"   Issues: {', '.join(problem['issues'])}")
        
        if len(results['problematic_files']) > 10:
            print(f"\n   ... and {len(results['problematic_files'])-10} more")
    
    # Issues by type
    issue_types = {}
    for problem in results['problematic_files']:
        for issue in problem['issues']:
            issue_type = issue.split(':')[0].split('(')[0].strip()
            issue_types[issue_type] = issue_types.get(issue_type, 0) + 1
    
    if issue_types:
        print(f"\n📋 Issues by Type:")
        print("-"*80)
        for issue_type, count in sorted(issue_types.items(), key=lambda x: x[1], reverse=True):
            print(f"   {issue_type:30s}: {count:4d} images")
    
    # Save detailed report
    with open('../data/quality_report.json', 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Detailed report saved to: data/quality_report.json")
    
    # Recommendation
    print("\n" + "="*80)
    if summary['problematic_images'] / summary['total_images'] < 0.05:
        print("✅ EXCELLENT! Less than 5% problematic images. Ready to train!")
    elif summary['problematic_images'] / summary['total_images'] < 0.10:
        print("👍 GOOD! Less than 10% issues. Consider cleaning, but can train.")
    elif summary['problematic_images'] / summary['total_images'] < 0.20:
        print("⚠️  WARNING! 10-20% issues. Should clean before training.")
    else:
        print("❌ CRITICAL! More than 20% issues. Must clean before training!")
    
    print("="*80)

def main():
    print("="*80)
    print("🔍 Data Quality Checker")
    print("="*80)
    
    # Check if required packages are installed
    try:
        import cv2
        from PIL import Image
    except ImportError:
        print("\n❌ Missing required packages!")
        print("Please install: pip install opencv-python pillow")
        return
    
    results = scan_dataset()
    generate_report(results)

if __name__ == "__main__":
    main()