"""Standard-library portability checks; no Torch, native solver or training."""
import ast
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from structured_kan.structure_builder import acuos2, pair_cache


class PortabilityTests(unittest.TestCase):
    def test_no_linux_only_execution_guard(self):
        root = Path(__file__).resolve().parents[1]
        for path in root.rglob('*.py'):
            if 'vendor' in path.relative_to(root).parts:
                continue
            code = path.read_text(encoding='utf-8-sig')
            for node in ast.walk(ast.parse(code)):
                if not isinstance(node, (ast.Assert, ast.If)):
                    continue
                condition = ast.get_source_segment(code, node.test) or ''
                if 'sys.platform' not in condition or 'linux' not in condition:
                    continue
                if isinstance(node, ast.Assert) or any(isinstance(x, ast.Raise) for x in node.body):
                    self.fail(f'Linux-only execution guard in {path.name}:{node.lineno}')

    def test_rss_unavailable_is_not_zero(self):
        with patch.object(acuos2, 'resource', None):
            self.assertIsNone(acuos2.peak_rss_mib())

    def test_spawn_reloads_native_signature(self):
        module = Mock()
        module.getModule.side_effect = ['test', 'spec']
        bootstrap = {'vendor': 'native', 'specification': 'fmod SPECIFICATION is endfm'}
        with patch.object(acuos2, 'load', return_value=(module, {})) as load:
            self.assertEqual(acuos2.restore_pair_modules(bootstrap), ('test', 'spec'))
        load.assert_called_once_with('native')
        module.input.assert_called_once_with(bootstrap['specification'])

    def test_spawn_never_pickles_native_objects(self):
        receiver, sender, process = Mock(), Mock(), Mock()
        receiver.poll.return_value = True
        receiver.recv.return_value = {'status': 'succeeded', 'choices': ['x'], 'info': {}}
        process.is_alive.return_value = False
        context = Mock()
        context.Pipe.return_value = receiver, sender
        context.Process.return_value = process
        bootstrap = {'vendor': 'native', 'specification': 'signature'}
        with patch.object(pair_cache.sys, 'platform', 'win32'), \
             patch.object(pair_cache.mp, 'get_context', return_value=context) as get_context, \
             patch.object(pair_cache.cache, 'get', return_value=None), \
             patch.object(pair_cache.cache, 'put'), \
             patch.object(acuos2, 'pair_bootstrap', return_value=bootstrap):
            result = pair_cache.pair_operation(object(), object(), 'a', 'b', 'signature', 3)
        self.assertEqual(result['status'], 'succeeded')
        get_context.assert_called_once_with('spawn')
        self.assertEqual(context.Process.call_args.kwargs['args'], (sender, None, None, 'a', 'b', bootstrap))
        sender.close.assert_called_once()
        receiver.close.assert_called_once()


if __name__ == '__main__':
    unittest.main()
