"""
Unit tests for the utility functions. 

"""
import unittest
from unittest.mock import patch, MagicMock
from neuronx_distributed_training.utils.utils import get_platform_target, get_dtype, get_lnc_size
from parameterized import parameterized
import torch

class TestUtils(unittest.TestCase):
    def test_get_dtype(self):
        """
        Tests the get_dtype function.

        Args:
            dtype (str, optional): The data type to test. Defaults to 'BF16'.
        """
        assert get_dtype('fp32') == torch.float32
        assert get_dtype('fp16') == torch.float16
        assert get_dtype('bf16') == torch.bfloat16
        assert get_dtype('unknown') == torch.bfloat16

    @parameterized.expand([('trn2',None, 2), ('trn1', None, 1),('trn2', 1, 1),])
    def test_get_lnc_size(self, mock_return_value, mock_lnc, expected_lnc_size):
        """
        Tests the get_lnc_size function after mocking the get_platform_target

        """
        with patch('neuronx_distributed_training.utils.utils.get_platform_target', return_value=mock_return_value):
            result_lnc_size = get_lnc_size(mock_lnc)
            self.assertEqual(result_lnc_size, expected_lnc_size)


if __name__ == '__main__':
    unittest.main()