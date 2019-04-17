import unittest
from onnxconverter_common.data_types import FloatTensorType


class TestTypes(unittest.TestCase):

    def test_to_onnx_type(self):
        dt = FloatTensorType((1, 5))
        assert str(dt) == 'FloatTensorType(shape=(1, 5))'
        onx = dt.to_onnx_type()
        assert "dim_value: 5" in str(onx)
        tt = onx.tensor_type
        assert "dim_value: 5" in str(tt)
        assert tt.elem_type == 1
        o = onx.sequence_type
        assert str(o) == ""
        

if __name__ == '__main__':
    unittest.main()
