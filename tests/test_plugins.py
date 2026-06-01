# SPDX-License-Identifier: Apache-2.0
'''
Tests for the pose.plugins extension-point registry.
'''
import pytest
from pose import plugins


class FakeScorer:
	def Score(self, p): return 0.0


class FakeParser:
	def Parse(self, path): return None


def test_register_and_get_scorer_roundtrip():
	plugins.RegisterScorer('roundtrip_scorer', FakeScorer)
	try:
		assert plugins.GetScorer('roundtrip_scorer') is FakeScorer
		assert 'roundtrip_scorer' in plugins.ListScorers()
	finally:
		plugins.Unregister('pose.scorers', 'roundtrip_scorer')


def test_double_register_raises_keyerror():
	plugins.RegisterScorer('dup_scorer', FakeScorer)
	try:
		with pytest.raises(KeyError, match='already registered'):
			plugins.RegisterScorer('dup_scorer', FakeScorer)
	finally:
		plugins.Unregister('pose.scorers', 'dup_scorer')


def test_unknown_name_lookup_raises_keyerror_with_known_list():
	with pytest.raises(KeyError, match='known:'):
		plugins.GetScorer('this_does_not_exist_xyzzy')


def test_unknown_group_register_raises_valueerror():
	with pytest.raises(ValueError, match='unknown plugin group'):
		plugins.Register('pose.not_a_group', 'x', FakeScorer)


def test_each_category_has_register_list_get_triple():
	# Parsers
	plugins.RegisterParser('p1', FakeParser)
	# Scorers
	plugins.RegisterScorer('s1', FakeScorer)
	# Builders + exporters use Register() directly for variety
	plugins.Register('pose.builders', 'b1', FakeScorer)
	plugins.Register('pose.exporters', 'e1', FakeScorer)
	try:
		assert 'p1' in plugins.ListParsers()
		assert 's1' in plugins.ListScorers()
		assert 'b1' in plugins.ListBuilders()
		assert 'e1' in plugins.ListExporters()
		assert plugins.GetParser('p1') is FakeParser
		assert plugins.GetScorer('s1') is FakeScorer
		assert plugins.GetBuilder('b1') is FakeScorer
		assert plugins.GetExporter('e1') is FakeScorer
	finally:
		plugins.Unregister('pose.parsers', 'p1')
		plugins.Unregister('pose.scorers', 's1')
		plugins.Unregister('pose.builders', 'b1')
		plugins.Unregister('pose.exporters', 'e1')


def test_list_returns_sorted():
	plugins.RegisterScorer('zzz_last', FakeScorer)
	plugins.RegisterScorer('aaa_first', FakeScorer)
	try:
		names = plugins.ListScorers()
		# The two we just registered are in sorted order relative to
		# each other.
		i = names.index('aaa_first')
		j = names.index('zzz_last')
		assert i < j
	finally:
		plugins.Unregister('pose.scorers', 'aaa_first')
		plugins.Unregister('pose.scorers', 'zzz_last')


def test_entry_point_discovery_is_idempotent():
	# Discover does not raise even when called twice for the same group
	# and even with no entry points installed.
	plugins.Discover('pose.scorers')
	plugins.Discover('pose.scorers')


def test_plugins_namespace_exposed_on_package():
	import pose
	assert hasattr(pose, 'plugins')
	assert pose.plugins is plugins
