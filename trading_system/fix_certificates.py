#!/usr/bin/env python3
"""
Standalone script to fix SSL certificate issues on macOS.
Run this once to install the certificates properly.
"""
import sys
import subprocess
import os


def fix_macos_certificates():
    """
    Fix SSL certificate verification issues on macOS.
    """
    if sys.platform != 'darwin':
        print("This fix is only needed on macOS.")
        return

    print("Attempting to fix SSL certificates for Python on macOS...")

    try:
        # Check Python version
        python_version = '.'.join(sys.version.split('.')[:2])
        cert_command = f"/Applications/Python {python_version}/Install Certificates.command"

        if not os.path.exists(cert_command):
            # Try alternative locations
            cert_command = "/Applications/Python/Install Certificates.command"

            if not os.path.exists(cert_command):
                # Try to find the script in the current Python framework
                framework_path = sys._framework_path if hasattr(sys, '_framework_path') else None
                if framework_path:
                    cert_path = os.path.join(os.path.dirname(framework_path),
                                             "Resources/Install Certificates.command")
                    if os.path.exists(cert_path):
                        cert_command = cert_path

                if not os.path.exists(cert_command):
                    print(f"Certificate installation script not found.")
                    print("Manual instructions:")
                    print("1. Locate 'Install Certificates.command' in your Python installation folder")
                    print("2. Run it by double-clicking in Finder or using terminal")
                    print("\nAlternatively, you can use this command:")
                    print("   python -m pip install --upgrade certifi")
                    print("   or")
                    print("   /Applications/Python 3.x/Install Certificates.command")
                    print("   (replace 3.x with your Python version)")
                    return

        # Run the certificate installation script
        print(f"Running certificate installer: {cert_command}")
        subprocess.run(["/bin/bash", cert_command], check=True)
        print("Certificates successfully installed!")

    except Exception as e:
        print(f"Error fixing certificates: {e}")
        print("\nTry running these commands manually:")
        print("   python -m pip install --upgrade certifi")
        print("   or find and run 'Install Certificates.command' in your Python installation")


if __name__ == "__main__":
    fix_macos_certificates()