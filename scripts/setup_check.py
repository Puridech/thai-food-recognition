"""
Script to check if development environment is set up correctly
"""
import sys
import subprocess
import os
import platform

def check_command(command, name, windows_alternative=None):
    """Check if a command exists"""
    commands_to_try = [command]
    
    # On Windows, try .cmd extension
    if platform.system() == 'Windows' and windows_alternative:
        commands_to_try.append(windows_alternative)
    
    for cmd in commands_to_try:
        try:
            result = subprocess.run(
                [cmd, '--version'], 
                capture_output=True, 
                text=True,
                check=True,
                shell=True  # Important for Windows
            )
            version_output = result.stdout.strip().split('\n')[0]
            print(f"✅ {name}: {version_output}")
            return True
        except (subprocess.CalledProcessError, FileNotFoundError):
            continue
    
    print(f"❌ {name}: Not found or not working")
    return False

def check_directory(path, name):
    """Check if directory exists"""
    if os.path.exists(path):
        count = ""
        # Count files for specific directories
        if os.path.isdir(path):
            try:
                items = os.listdir(path)
                if items:
                    count = f" ({len(items)} items)"
            except:
                pass
        print(f"✅ {name}: Found{count}")
        return True
    else:
        print(f"❌ {name}: Not found")
        return False

def check_python_packages():
    """Check if important Python packages can be imported"""
    print("\n🐍 Checking Python environment:")
    packages = {
        'pip': 'pip',
        'venv': 'venv (virtual environment)'
    }
    
    checks = []
    for package, display_name in packages.items():
        try:
            __import__(package)
            print(f"✅ {display_name}: Available")
            checks.append(True)
        except ImportError:
            print(f"❌ {display_name}: Not available")
            checks.append(False)
    
    return all(checks)

def main():
    print("=" * 60)
    print("     Thai Food Recognition - Environment Check")
    print("=" * 60)
    print(f"Platform: {platform.system()} {platform.release()}")
    print(f"Python: {sys.version.split()[0]}")
    print()
    
    # Check commands
    print("📦 Checking installed software:")
    checks = []
    checks.append(check_command('python', 'Python'))
    checks.append(check_command('pip', 'Pip'))
    checks.append(check_command('node', 'Node.js'))
    checks.append(check_command('npm', 'NPM', 'npm.cmd'))  # Try npm.cmd on Windows
    checks.append(check_command('git', 'Git'))
    
    # Check Python environment
    checks.append(check_python_packages())
    
    print()
    
    # Check directories
    print("📁 Checking project structure:")
    checks.append(check_directory('data', 'Data folder'))
    checks.append(check_directory('data/training/raw', 'Raw images folder'))
    checks.append(check_directory('data/foods', 'Foods folder'))
    checks.append(check_directory('data/restaurants', 'Restaurants folder'))
    checks.append(check_directory('backend', 'Backend folder'))
    checks.append(check_directory('frontend', 'Frontend folder'))
    checks.append(check_directory('docs', 'Docs folder'))
    checks.append(check_directory('scripts', 'Scripts folder'))
    
    print()
    
    # Check files
    print("📄 Checking essential files:")
    checks.append(check_directory('.gitignore', '.gitignore'))
    checks.append(check_directory('README.md', 'README.md'))
    checks.append(check_directory('backend/requirements.txt', 'requirements.txt'))
    checks.append(check_directory('frontend/package.json', 'package.json'))
    
    # Check if virtual environment exists
    print()
    print("🔧 Checking Python virtual environment:")
    venv_exists = check_directory('venv', 'Virtual environment folder')
    checks.append(venv_exists)
    
    if venv_exists:
        # Check if we're running in venv
        in_venv = hasattr(sys, 'real_prefix') or (
            hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
        )
        if in_venv:
            print("✅ Currently running in virtual environment")
        else:
            print("⚠️  Not running in virtual environment (run: venv\\Scripts\\activate)")
    
    print()
    print("=" * 60)
    
    if all(checks):
        print("🎉 All checks passed! Your environment is ready!")
        print()
        print("Next steps:")
        print("  1. Start collecting images for training")
        print("  2. Create markdown files for food information")
        print("  3. Build restaurant database (JSON)")
    else:
        print("⚠️  Some checks failed. Please review the issues above.")
        print()
        print("Common fixes:")
        print("  - Make sure you're in the project root directory")
        print("  - Activate virtual environment: venv\\Scripts\\activate")
        print("  - Install missing software from the setup guide")
        sys.exit(1)
    
    print("=" * 60)

if __name__ == "__main__":
    main()