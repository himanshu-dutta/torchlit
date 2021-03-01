from setuptools import find_packages, setup

with open("README.md") as f:
    long_description = f.read()


if __name__ == "__main__":
    setup(
        name="torchlit",
        version="0.1.0",
        description="torchlit - thin wrappers for Pytorch",
        long_description=long_description,
        long_description_content_type="text/markdown",
        author="Himanshu Dutta",
        author_email="meet.himanshu.dutta@gmail.com",
        url="https://github.com/himanshu-dutta/torchlit",
        license="MIT License",
        packages=find_packages(),
        include_package_data=True,
        install_requires=[
            "google-cloud-storage>=1.35.1",
            "selenium>=3.141.0",
            "torch>=1.6.0",
            "torchaudio>=0.7.2",
            "torchvision>=0.8.2",
            "tqdm>=4.55.0",
            "webdriver-manager>=3.2.2",
        ],
        platforms=["linux", "unix"],
        python_requires=">3.5.2",
    )