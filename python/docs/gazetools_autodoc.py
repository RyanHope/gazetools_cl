import sys,os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)),"../"))

from sphinx.ext import autodoc
from inspect import ArgSpec, getargspec, formatargspec
from gazetools.helpers import OCLWrapper
from itertools import count

try:
    from __builtin__ import unicode
except ImportError:
    unicode = lambda s, enc: s

def formatvalue(val):
    if type(val) is str:
        return '="' + unicode(val, 'utf-8').replace('"', '\\"').replace('\\', '\\\\') + '"'
    else:
        return '=' + repr(val)

def getconfigargspec(obj):
    args = []
    defaults = []

    if isinstance(obj, OCLWrapper):
		additional_args = obj.additional_args()
		argspecobjs = obj.argspecobjs()
		get_omitted_args = obj.omitted_args
    else:
		additional_args = ()
		argspecobjs = ((None, obj),)
		get_omitted_args = lambda *args: ()

    for arg in additional_args:
        args.append(arg[0])
        if len(arg) > 1:
            defaults.append(arg[1])

    for name, method in argspecobjs:
        argspec = getargspec(method)
        omitted_args = get_omitted_args(name, method)
        largs = len(argspec.args)
        for i, arg in enumerate(reversed(argspec.args)):
            if (
                largs - (i + 1) in omitted_args
            ):
                continue
            if argspec.defaults and len(argspec.defaults) > i:
                if arg in args:
                    idx = args.index(arg)
                    if len(args) - idx > len(defaults):
                        args.pop(idx)
                    else:
                        continue
                default = argspec.defaults[-(i + 1)]
                defaults.append(default)
                args.append(arg)
            else:
                if arg not in args:
                    args.insert(0, arg)

    return ArgSpec(args=args, varargs=None, keywords=None, defaults=tuple(defaults))

class OCLWrapperDocumenter(autodoc.FunctionDocumenter):
    '''Specialized documenter subclass for OCLWrapper subclasses.'''
    @classmethod
    def can_document_member(cls, member, membername, isattr, parent):
		return (isinstance(member, OCLWrapper) or
			super(OCLWrapperDocumenter, cls).can_document_member(member, membername, isattr, parent))

    def format_args(self):
		argspec = getconfigargspec(self.object)
		return formatargspec(*argspec, formatvalue=formatvalue).replace('\\', '\\\\')

def setup(app):
	autodoc.setup(app)
	app.add_autodocumenter(OCLWrapperDocumenter)
