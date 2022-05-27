import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='prompt_tools',
    version='0.0.1',
    author='Paul Christ',
    author_email='paul.l.christ@web.de',
    description='Tools for prompting KG from OPT-models',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/plc-dev/KnowledgePrompt',
    project_urls = {
        "Bug Tracker": "https://github.com/plc-dev/KnowledgePrompt/issues"
    },
    license='MIT',
    packages=['prompt_tools'],
    install_requires=[
        'torch',
        'transformers',
        'accelerate',
        'huggingface_hub'
    ],
)