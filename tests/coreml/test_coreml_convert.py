import unittest
from onnxmltools.convert.coreml.convert import _resolve_name_conflicts
from onnxmltools.convert.coreml.CoremlConvertContext import CoremlConvertContext as ConvertContext

class TestCoremlConverter(unittest.TestCase):

    def test_resolve_name_conflicts(self):
        inputs = ['test', 'test1', 'test2']
        outputs = ['test', 'foo', 'bar']

        context = ConvertContext()
        # add inputs into the context
        for input in inputs:
            context.get_onnx_name(input)

        result = _resolve_name_conflicts(context, inputs, outputs)
        self.assertEqual(len(result), 3)
        expected = ['test3', 'foo', 'bar']
        self.assertEqual(result, expected)

    def test_resolve_name_conflicts_no_conflicts(self):
        inputs = ['test', 'test1', 'test2']
        outputs = ['test3', 'foo', 'bar']

        context = ConvertContext()
        # add inputs into the context
        for input in inputs:
            context.get_onnx_name(input)

        result = _resolve_name_conflicts(context, inputs, outputs)
        self.assertEqual(len(result), 3)
        expected = ['test3', 'foo', 'bar']
        self.assertEqual(result, expected)

    def test_resolve_name_conflicts_string(self):
        input = 'foo'
        output = input

        context = ConvertContext()
        context.get_onnx_name(input)

        result = _resolve_name_conflicts(context, input, output)
        expected = 'foo1'
        self.assertEqual(result, expected)

    def test_resolve_name_no_conflicts_string(self):
        input = 'foo'
        output = 'bar'

        context = ConvertContext()
        context.get_onnx_name(input)

        result = _resolve_name_conflicts(context, input, output)
        expected = 'bar'
        self.assertEqual(result, expected)
