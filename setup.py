from setuptools import setup, find_packages

setup(name="Open3D", version="1.0", author="TRAN DAI CHI", author_email="ctran743@gmail.com", description="README.md", url="", packages=find_packages(exclude=["envs*"]),
      py_modules=["DetAny3D", "OVM3D-Det-AIC", "Object_Detection", "Object_Tracking", "Draw_Bbox", "Format_Annotations"],
      license="LICENSE", python_requires=">=3.8", include_package_data=True, install_requires="requirements.txt")