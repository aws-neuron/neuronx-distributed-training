# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.

import os
import sys
import datetime

sys.path.append(os.path.abspath("_ext"))

# get environment variables

project_name = ""
branch_name = ""


if os.environ.get("READTHEDOCS"):
    env_branch_name = os.environ.get("READTHEDOCS_VERSION_NAME")
    branch_name = env_branch_name
    if branch_name == "latest":
        branch_name = "master"
    if os.environ.get("READTHEDOCS_PROJECT") == "awsdocs-neuron":
        env_project_name = "neuronx-distributed-training"
        project_name = env_project_name
    elif os.environ.get("READTHEDOCS_PROJECT") == "awsdocs-neuron-staging":
        env_project_name = "neuronx-distributed-training-staging"
        project_name = env_project_name
else:
    env_project_name = os.environ.get("GIT_PROJECT_NAME")
    env_branch_name = os.environ.get("GIT_BRANCH_NAME")

    # set project name
    if env_project_name:
        project_name = env_project_name
    else:
        project_name = "neuronx-distributed-training"

    # set branch name
    if env_branch_name:
        branch_name = env_branch_name
        if branch_name == "latest":
            branch_name = "master"
    else:
        branch_name = "master"

# -- Project information -----------------------------------------------------

project = "NeuronxDistributedTraining"
copyright = "{}, Amazon.com".format(datetime.datetime.now().year)
author = "AWS"
master_doc = "index"
html_title = "NeuronxDistributedTraining Documentation"

# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinxcontrib.contentui",
    "nbsphinx",
    "sphinx.ext.extlinks",
    "sphinx.ext.intersphinx",
    "sphinx_plotly_directive",
    "sphinxcontrib.programoutput",
    "df_tables",
    "sphinx_design",
    "ablog",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
    "sphinx.ext.autodoc",
    "local_documenter",
    "sphinx_copybutton",
    "sphinx.ext.autosectionlabel",
]


html_sidebars = {"general/announcements/index": ["recentposts.html"]}

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
# exclude_patterns = ['_build','**.ipynb_checkpoints','.venv']
html_extra_path = ["_static"]

# remove bash/python/ipython/jupyter prompts and continuations
copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# nbsphinx_allow_errors = True
nbsphinx_execute = "never"

html_logo = "images/Site-Merch_Neuron-ML-SDK_Editorial.png"

napoleon_google_docstring = True

# -- more options -------------------------------------------------


projectblob = project_name + "/blob/" + branch_name
projecttree = project_name + "/tree/" + branch_name


# -- Options for Theme  -------------------------------------------------
html_theme = "sphinx_book_theme"
html_theme_options = {
    "repository_url": "https://github.com/aws-neuron/" + project_name,
    "use_issues_button": True,
    "use_repository_button": True,
    "use_download_button": True,
    "use_fullscreen_button": True,
    "use_edit_page_button": True,
    "home_page_in_toc": False,
    "repository_branch": branch_name,
    "body_max_width": "80ch",
}


# -- Options for HTML output -------------------------------------------------

html_css_files = ["css/custom.css", "styles/sphinx-book-theme.css"]

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

plotly_include_source = False
plotly_html_show_source_link = False
plotly_html_show_formats = False
plotly_include_directive_source = False


# -- ABlog config -------------------------------------------------
blog_feed_length = 5
fontawesome_included = True
post_show_prev_next = False
post_auto_image = 1
post_auto_excerpt = 2
execution_show_tb = "READTHEDOCS" in os.environ


# Exclude private github from linkcheck. Readthedocs only exposes the ssh-agent to the 'checkout' build step, which is too early for the linkchecker to run.
linkcheck_ignore = []
linkcheck_exclude_documents = []
nitpicky = True
