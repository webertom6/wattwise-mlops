# use subprocess to run the setup script instead of os due to security issues
# from subprocess import Popen
import os

print("Running setup script...")

# p = Popen("cmd_setup_dockerimg_windows.bat", cwd=r"C:\Users\Student11\Documents\git\MLOps\wattwise-mlops-project\wattwise/src")
# stdout, stderr = p.communicate()

os.system("cmd_setup_dockerimg_windows.bat")

print("Setup script completed.")
