from anchor_python_sphinx import configure_sphinx


# Importing Sphinx settings from the anchor_python_sphinx library
def setup(app):
    configure_sphinx.configure(app, "anchor_python_training", author="Owen Feehan")
