"""
Clean problematic images from dataset
"""
import os
import json
import shutil

def load_quality_report():
    """Load quality report"""
    with open('../data/quality_report.json', 'r', encoding='utf-8') as f:
        return json.load(f)

def backup_problematic_images(results, backup_dir='../data/training_backup'):
    """Backup problematic images before deleting"""
    
    os.makedirs(backup_dir, exist_ok=True)
    
    print(f"\n📦 Backing up {len(results['problematic_files'])} images...")
    
    for problem in results['problematic_files']:
        src = problem['path']
        
        # Create backup structure
        dish = problem['dish']
        split = problem['split']
        backup_path = os.path.join(backup_dir, split, dish)
        os.makedirs(backup_path, exist_ok=True)
        
        # Copy to backup
        dst = os.path.join(backup_path, problem['filename'])
        shutil.copy2(src, dst)
    
    print(f"✅ Backup complete: {backup_dir}")

def delete_problematic_images(results, threshold='critical'):
    """Delete problematic images based on threshold"""
    
    thresholds = {
        'critical': ['Corrupted', 'Error', 'Not RGB'],
        'high': ['Blurry', 'Too dark', 'Too bright'],
        'medium': ['Wrong size']
    }
    
    delete_keywords = thresholds['critical']
    if threshold == 'high':
        delete_keywords.extend(thresholds['high'])
    elif threshold == 'medium':
        delete_keywords.extend(thresholds['high'] + thresholds['medium'])
    
    to_delete = []
    for problem in results['problematic_files']:
        for issue in problem['issues']:
            if any(keyword in issue for keyword in delete_keywords):
                to_delete.append(problem)
                break
    
    if not to_delete:
        print("\n✅ No images to delete at this threshold!")
        return
    
    print(f"\n🗑️  Will delete {len(to_delete)} images:")
    for i, problem in enumerate(to_delete[:5], 1):
        print(f"   {i}. {problem['dish']}/{problem['filename']}: {problem['issues'][0]}")
    
    if len(to_delete) > 5:
        print(f"   ... and {len(to_delete)-5} more")
    
    confirm = input(f"\nDelete these {len(to_delete)} images? (yes/no): ").strip().lower()
    
    if confirm == 'yes':
        for problem in to_delete:
            try:
                os.remove(problem['path'])
                print(f"   ✅ Deleted: {problem['filename']}")
            except Exception as e:
                print(f"   ❌ Error deleting {problem['filename']}: {e}")
        
        print(f"\n✅ Deleted {len(to_delete)} problematic images")
    else:
        print("\n❌ Deletion cancelled")

def main():
    print("="*80)
    print("🧹 Data Cleaning Tool")
    print("="*80)
    
    # Load report
    try:
        results = load_quality_report()
    except FileNotFoundError:
        print("\n❌ Quality report not found!")
        print("Please run: python data_quality_check.py first")
        return
    
    print(f"\nFound {len(results['problematic_files'])} problematic images")
    
    if not results['problematic_files']:
        print("✅ No problematic images to clean!")
        return
    
    # Backup first
    print("\n1️⃣  Creating backup...")
    backup_problematic_images(results)
    
    # Choose threshold
    print("\n2️⃣  Select cleaning threshold:")
    print("   [1] Critical only (corrupted, errors)")
    print("   [2] High (critical + blurry + lighting issues)")
    print("   [3] Medium (all issues)")
    print("   [0] Cancel")
    
    choice = input("\nYour choice: ").strip()
    
    threshold_map = {'1': 'critical', '2': 'high', '3': 'medium'}
    threshold = threshold_map.get(choice)
    
    if threshold:
        delete_problematic_images(results, threshold)
    else:
        print("\n❌ Cleaning cancelled")

if __name__ == "__main__":
    main()