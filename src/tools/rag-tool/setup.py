from typing import List
from setuptools import find_packages, setup

PACKAGE_NAME = "rag_tool"


def parse_requirements(file_name: str) -> List[str]:
    with open(file_name) as f:
        return [
            require.strip() for require in f
            if require.strip() and not require.startswith('#')
        ]


setup(
    name=PACKAGE_NAME,
    version="1.0.0",
    description="This is my simple custom tool package",
    packages=find_packages(),
    install_requires=parse_requirements('requirements.txt'),
    entry_points={
        "package_tools": ["custom_tools = rag_tool.tools.utils:list_package_tools"],
    },
    include_package_data=True,   # This line tells setuptools to include files from MANIFEST.in
)
