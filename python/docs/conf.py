import sys
import os
import shlex
from pygments import styles

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),"../"))

extensions = [
    # 'sphinx.ext.autodoc',
    'gazetools_autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.viewcode',
    'sphinx.ext.inheritance_diagram',
    'sphinx.ext.graphviz'
]

templates_path = ['_templates']

source_suffix = '.rst'

master_doc = 'index'

project = u'Gazetools'
copyright = u'2015, Ryan M. Hope'
author = u'Ryan M. Hope'

version = '0.0.1'
release = '0.0.1'

language = None

today_fmt = '%B %d, %Y'

exclude_patterns = ['_build']
add_module_names = False
show_authors = False

pygments_style = 'default'

modindex_common_prefix = []

todo_include_todos = True

html_theme = 'sphinx_rtd_theme'
html_theme_options = {}

#html_logo = None
html_show_sourcelink = False
html_show_sphinx = False
html_show_copyright = True

html_search_language = 'en'

htmlhelp_basename = 'pydoc'

latex_elements = {
}

latex_documents = [
  (master_doc, 'Gazetools.tex', u'Gazetools Documentation',
   u'Ryan M. Hope', 'manual'),
]

man_pages = [
    (master_doc, 'gazetools', u'Gazetools Documentation',
     [author], 1)
]

texinfo_documents = [
  (master_doc, 'Gazetools', u'Gazetools Documentation',
   author, 'Gazetools', 'One line description of project.',
   'Miscellaneous'),
]

epub_title = project
epub_author = author
epub_publisher = author
epub_copyright = copyright
epub_exclude_files = ['search.html']

intersphinx_mapping = {'http://docs.python.org/': None}
