import sys

sys.path.append('../popdyn')


project = u'Popdyn'
copyright = u'2018. Devin Cairns'
master_doc = 'index'
templates_path = ['_templates']
extensions = ['sphinx.ext.autodoc', 'sphinx.ext.mathjax', 'sphinx.ext.intersphinx',
              'sphinx.ext.autosummary', 'sphinx.ext.extlinks']
autoclass_content = 'both'
source_suffix = '.rst'
version = 'X.Y.Z'
exclude_patterns = ['_build']

# -- HTML theme settings ------------------------------------------------

html_show_sourcelink = False
html_sidebars = {
    '**': ['logo-text.html',
           'globaltoc.html',
           'localtoc.html',
           'searchbox.html']
}

import guzzle_sphinx_theme

extensions.append("guzzle_sphinx_theme")
html_theme_path = guzzle_sphinx_theme.html_theme_path()
html_theme = 'guzzle_sphinx_theme'

# Guzzle theme options (see theme.conf for more information)
# html_theme_options = {
#     "base_url": "http://my-site.com/docs/"
# }
