from setuptools import setup


DEPENDENCIES = [
    "fastapi>=0.111",
    "uvicorn[standard]>=0.30",
    "jinja2>=3.1",
    "markdown>=3.6",
    "pymdown-extensions>=10.8",
    "pygments>=2.18",
    "mujoco>=3.2",
]


setup(
    name="mujoco-learning-docs",
    version="0.1.0",
    description="Local FastAPI documentation site and Python MuJoCo tutorial environment.",
    packages=["mujoco_learning_doc"],
    include_package_data=True,
    package_data={"mujoco_learning_doc": ["templates/*.html", "static/*.css", "static/*.js"]},
    python_requires=">=3.10",
    install_requires=DEPENDENCIES,
    extras_require={"dev": ["httpx>=0.27"]},
    entry_points={"console_scripts": ["mujoco_learning_doc=mujoco_learning_doc.main:run"]},
)
