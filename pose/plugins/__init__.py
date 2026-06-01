# SPDX-License-Identifier: Apache-2.0
'''
Plugin registry for Pose extension points.

Pose exposes four categories of extension. Authors of a separate package
register a plugin in two ways: by calling one of the `Register*` functions
at import time, or by declaring an entry point in their `pyproject.toml`
(preferred for installed plugins).

Categories:
	pose.parsers   - additional file-format readers/writers (e.g. cryo-EM
	                 map formats, custom in-house structure formats)
	pose.scorers   - alternative scoring functions / force fields
	pose.builders  - alternative builders (e.g. templated-loop builder)
	pose.exporters - additional export formats

Entry-point example (in a plugin package's pyproject.toml):

	[project.entry-points."pose.scorers"]
	my_ff = "my_package.module:MyScorerClass"

Programmatic example:

	from pose.plugins import RegisterScorer
	RegisterScorer("my_ff", MyScorerClass)

Discovery:

	pose.plugins.ListScorers() -> list of registered scorer names
	pose.plugins.GetScorer(name) -> the class
'''
from importlib.metadata import entry_points

_GROUPS = ('pose.parsers', 'pose.scorers',
	'pose.builders', 'pose.exporters')

_REGISTRY = {g: {} for g in _GROUPS}
_DISCOVERED = {g: False for g in _GROUPS}

def Discover(group):
	'''
	Populate the registry for one group from entry points.
	Arguments:
	----------
		group: one of 'pose.parsers', 'pose.scorers', 'pose.builders',
			'pose.exporters'
	Returns:
	--------
		None. The internal registry is mutated; names already registered
		programmatically are not overwritten.
	'''
	if _DISCOVERED[group]: return
	for ep in entry_points(group=group):
		if ep.name in _REGISTRY[group]: continue
		_REGISTRY[group][ep.name] = ep.load()
	_DISCOVERED[group] = True

def Register(group, name, cls):
	'''
	Register a plugin programmatically.
	Arguments:
	----------
		group: one of the four 'pose.*' groups
		name: string identifier
		cls: the plugin class (or callable)
	Returns:
	--------
		None. Raises ValueError on unknown group; raises KeyError if name
		is already registered (use Unregister first to replace).
	'''
	if group not in _REGISTRY:
		raise ValueError(f'unknown plugin group: {group!r}')
	if name in _REGISTRY[group]:
		raise KeyError(
			f'plugin {name!r} already registered in {group!r}')
	_REGISTRY[group][name] = cls

def Unregister(group, name):
	'''
	Remove a plugin from the registry.
	Arguments:
	----------
		group: one of the four 'pose.*' groups
		name: string identifier to remove
	Returns:
	--------
		None. Raises KeyError if name is not registered.
	'''
	del _REGISTRY[group][name]

def Get(group, name):
	'''
	Look up a registered plugin, populating the registry from entry
	points on first call for this group.
	Arguments:
	----------
		group: one of the four 'pose.*' groups
		name: string identifier
	Returns:
	--------
		The registered class/callable. Raises KeyError with the list of
		known names if name is not registered.
	'''
	Discover(group)
	if name not in _REGISTRY[group]:
		known = sorted(_REGISTRY[group])
		raise KeyError(
			f'no plugin {name!r} in {group!r}; known: {known}')
	return _REGISTRY[group][name]

def List(group):
	'''
	List registered plugin names for one group.
	Arguments:
	----------
		group: one of the four 'pose.*' groups
	Returns:
	--------
		Sorted list of strings.
	'''
	Discover(group)
	return sorted(_REGISTRY[group])

def RegisterParser(name, cls):
	'''
	Register a parser plugin. See Register() for details.
	Arguments:
	----------
		name: string identifier
		cls: parser class/callable
	Returns:
	--------
		None.
	'''
	Register('pose.parsers', name, cls)

def RegisterScorer(name, cls):
	'''
	Register a scorer plugin. See Register() for details.
	Arguments:
	----------
		name: string identifier
		cls: scorer class/callable
	Returns:
	--------
		None.
	'''
	Register('pose.scorers', name, cls)

def RegisterBuilder(name, cls):
	'''
	Register a builder plugin. See Register() for details.
	Arguments:
	----------
		name: string identifier
		cls: builder class/callable
	Returns:
	--------
		None.
	'''
	Register('pose.builders', name, cls)

def RegisterExporter(name, cls):
	'''
	Register an exporter plugin. See Register() for details.
	Arguments:
	----------
		name: string identifier
		cls: exporter class/callable
	Returns:
	--------
		None.
	'''
	Register('pose.exporters', name, cls)

def ListParsers():
	'''
	List registered parser plugins.
	Arguments:
	----------
		No arguments taken
	Returns:
	--------
		Sorted list of names.
	'''
	return List('pose.parsers')

def ListScorers():
	'''
	List registered scorer plugins.
	Arguments:
	----------
		No arguments taken
	Returns:
	--------
		Sorted list of names.
	'''
	return List('pose.scorers')

def ListBuilders():
	'''
	List registered builder plugins.
	Arguments:
	----------
		No arguments taken
	Returns:
	--------
		Sorted list of names.
	'''
	return List('pose.builders')

def ListExporters():
	'''
	List registered exporter plugins.
	Arguments:
	----------
		No arguments taken
	Returns:
	--------
		Sorted list of names.
	'''
	return List('pose.exporters')

def GetParser(name):
	'''
	Look up a parser plugin.
	Arguments:
	----------
		name: string identifier
	Returns:
	--------
		Registered class/callable. Raises KeyError if unknown.
	'''
	return Get('pose.parsers', name)

def GetScorer(name):
	'''
	Look up a scorer plugin.
	Arguments:
	----------
		name: string identifier
	Returns:
	--------
		Registered class/callable. Raises KeyError if unknown.
	'''
	return Get('pose.scorers', name)

def GetBuilder(name):
	'''
	Look up a builder plugin.
	Arguments:
	----------
		name: string identifier
	Returns:
	--------
		Registered class/callable. Raises KeyError if unknown.
	'''
	return Get('pose.builders', name)

def GetExporter(name):
	'''
	Look up an exporter plugin.
	Arguments:
	----------
		name: string identifier
	Returns:
	--------
		Registered class/callable. Raises KeyError if unknown.
	'''
	return Get('pose.exporters', name)
