import json
import tempfile
import unittest
from pathlib import Path

from scripts.shard_model import shard_model


class ShardModelTests(unittest.TestCase):
    def test_preserves_mixed_dtype_tensors_and_revision(self):
        import torch
        from safetensors import safe_open
        from safetensors.torch import save_file

        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / "source"
            source.mkdir()
            expected = {"bf16": torch.arange(32, dtype=torch.bfloat16),
                        "fp32": torch.arange(24, dtype=torch.float32).reshape(4, 6),
                        "integer": torch.tensor([1, 2, 3], dtype=torch.int64)}
            save_file(expected, source / "model.safetensors", metadata={"format": "pt"})
            (source / "download_manifest.json").write_text('{"revision":"synthetic"}')
            output = Path(folder) / "shards"
            result = shard_model(source, output, max_bytes=70)
            self.assertEqual(result["shards"], 3)
            index = json.loads((output / "model.safetensors.index.json").read_text())
            for name, filename in index["weight_map"].items():
                with safe_open(output / filename, framework="pt") as checkpoint:
                    actual = checkpoint.get_tensor(name)
                    self.assertEqual(actual.dtype, expected[name].dtype)
                    self.assertTrue(torch.equal(actual, expected[name]))
            self.assertEqual(json.loads((output / "download_manifest.json").read_text())["revision"], "synthetic")
            with self.assertRaises(FileExistsError):
                shard_model(source, output)


if __name__ == "__main__":
    unittest.main()
