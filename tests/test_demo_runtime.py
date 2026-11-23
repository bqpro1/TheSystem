"""Native inference mechanics on tiny CPU models; these are not A100 results."""
from contextlib import ExitStack
from functools import wraps
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import torch
from transformers.models.qwen3_5 import modeling_qwen3_5 as qwen

from thesystem import inference
from thesystem import demo_runtime
from test_inference import tiny


class DemoRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def test_runtime_allows_24_gb_but_keeps_native_bf16_requirement(self):
        with patch.object(torch.cuda, "is_available", return_value=True), \
             patch.object(torch.cuda, "get_device_properties", return_value=SimpleNamespace(total_memory=24 * 1024**3)), \
             patch.object(torch.cuda, "get_device_name", return_value="NVIDIA L4"), \
             patch.object(torch.cuda, "get_device_capability", return_value=(8, 9)), \
             patch.object(torch.cuda, "is_bf16_supported", return_value=True) as bf16:
            demo_runtime.verify_runtime()
            bf16.return_value = False
            with self.assertRaisesRegex(RuntimeError, "native BF16 support"):
                demo_runtime.verify_runtime()

    def test_native_forward_cached_decode_and_generation_match_torch_fallback(self):
        _, reference, _ = tiny("probe")
        base = reference.base
        ids = torch.tensor([[1, 2, 3, 4]])
        with torch.no_grad():
            expected = base(ids, use_cache=True)
            expected_next = base(torch.tensor([[5]]), past_key_values=expected.past_key_values)
            expected_tokens = base.generate(ids, max_new_tokens=3, do_sample=False)
        globals_before = {name: getattr(qwen, name) for name in demo_runtime._NATIVE_FUNCTIONS}
        modules_before = tuple((name, id(module), type(module)) for name, module in base.named_modules())
        params_before = tuple((name, id(p), p._version) for name, p in base.named_parameters())
        self.assertEqual(demo_runtime.use_native_inference(base), 2)
        with torch.no_grad():
            actual = base(ids, use_cache=True)
            actual_next = base(torch.tensor([[5]]), past_key_values=actual.past_key_values)
            actual_tokens = base.generate(ids, max_new_tokens=3, do_sample=False)
        torch.testing.assert_close(actual.logits, expected.logits, rtol=0, atol=0)
        torch.testing.assert_close(actual_next.logits, expected_next.logits, rtol=0, atol=0)
        self.assertTrue(torch.equal(actual_tokens, expected_tokens))
        self.assertEqual(modules_before, tuple((n, id(m), type(m)) for n, m in base.named_modules()))
        self.assertEqual(params_before, tuple((n, id(p), p._version) for n, p in base.named_parameters()))
        for name, function in globals_before.items():
            self.assertIs(getattr(qwen, name), function)
        reference.assert_unchanged()

    def test_installed_optional_kernel_wrappers_are_bypassed_on_this_model_only(self):
        public, reference, _ = tiny("probe")
        _, other, _ = tiny("probe")
        with ExitStack() as stack:
            for name in demo_runtime._NATIVE_FUNCTIONS:
                original = getattr(qwen, name)

                @wraps(original)
                def forbidden(*args, **kwargs):
                    raise AssertionError("Optional extension must not run in the demo")

                stack.enter_context(patch.object(qwen, name, forbidden))
            demo_runtime.use_native_inference(reference.base)
            result = public.respond("Write a reply.", mode="generation", max_new_tokens=3)
            self.assertEqual(len(result["generated_ids"]), 3)
            with self.assertRaisesRegex(AssertionError, "Optional extension"):
                other.base(torch.tensor([[1, 2, 3, 4]]))
        reference.assert_unchanged()

    def test_reference_runtime_monkeypatch_requests_a_fresh_session(self):
        _, reference, _ = tiny("probe")
        with patch.object(qwen, "torch_chunk_gated_delta_rule", lambda *args, **kwargs: None):
            with self.assertRaisesRegex(RuntimeError, "Restart the Colab session"):
                demo_runtime.use_native_inference(reference.base)

    def test_native_probe_and_lora_keep_routing_and_generation_behavior(self):
        for variant in ("probe", "lora"):
            with self.subTest(variant=variant):
                public, reference, _ = tiny(variant)
                demo_runtime.use_native_inference(reference.base)
                with torch.no_grad():
                    reference.gate.weight.zero_()
                    reference.gate.bias.fill_(-100)
                generated = public.respond("Write a reply.", max_new_tokens=3)
                explicit = public.respond("Write a reply.", mode="generation", max_new_tokens=3)
                self.assertEqual(generated["mode"], "generation")
                self.assertEqual(generated["generated_ids"], explicit["generated_ids"])
                with torch.no_grad():
                    reference.gate.bias.fill_(100)
                decision = public.respond("Choose A or B.", {"A": "Yes", "B": "No"})
                self.assertEqual(decision["mode"], "decision")
                self.assertEqual(public.respond("Write a reply.")["mode"], "unsupported")
                if variant == "probe":
                    reference.assert_unchanged()
                    self.assertFalse(any("lora" in type(m).__name__.lower() for m in reference.base.modules()))

    def test_public_inference_dispatch_does_not_import_reference_runtime(self):
        reference = Mock()
        with patch.object(inference, "resolve_model_resources", return_value=Path("/snapshot")), \
             patch.object(inference, "import_module", side_effect=AssertionError("Training runtime imported")), \
             patch.object(demo_runtime, "load_model", return_value=(reference, Mock())) as loader, \
             patch("safetensors.torch.load_file", return_value={}):
            result = inference.load_model("lora", execution="inference")
        self.assertEqual(result.variant, "lora")
        self.assertIs(result._model, reference)
        self.assertEqual(loader.call_args.args, ("lora",))
        reference.stage.assert_called_once_with("evaluation")

    def test_loader_keeps_exact_backbone_tokenizer_and_wrapper_configuration(self):
        from transformers import PreTrainedTokenizerFast, Qwen3_5ForConditionalGeneration
        for variant in ("probe", "lora"):
            base = Mock()
            base.config.text_config.num_hidden_layers = 32
            base.config.text_config.hidden_size = 4096
            tokenizer, wrapper = Mock(), Mock()
            wrapper_name = ("thesystem._vendor.probe.model.FrozenQwen" if variant == "probe"
                            else "thesystem._vendor.lora.model.InternalFAST")
            order = []
            with patch.object(demo_runtime, "verify_runtime"), \
                 patch.object(PreTrainedTokenizerFast, "from_pretrained", return_value=tokenizer) as tokenizer_load, \
                 patch.object(Qwen3_5ForConditionalGeneration, "from_pretrained", return_value=base) as base_load, \
                 patch.object(demo_runtime, "use_native_inference", side_effect=lambda model: order.append("native") or 24), \
                 patch(wrapper_name, side_effect=lambda *a, **kw: order.append("wrapper") or wrapper) as wrap:
                actual, actual_tokenizer = demo_runtime.load_model(variant, token="test-only")
            self.assertIs(actual, wrapper)
            self.assertIs(actual_tokenizer, tokenizer)
            self.assertEqual(order, ["native", "wrapper"])
            self.assertEqual(base_load.call_args.args, ("Qwen/Qwen3.5-9B",))
            self.assertEqual(base_load.call_args.kwargs["revision"], "c202236235762e1c871ad0ccb60c8ee5ba337b9a")
            self.assertEqual(base_load.call_args.kwargs["revision"], tokenizer_load.call_args.kwargs["revision"])
            self.assertIs(base_load.call_args.kwargs["dtype"], torch.bfloat16)
            self.assertFalse(base_load.call_args.kwargs["use_kernels"])
            self.assertEqual(base_load.call_args.kwargs["device_map"], {"": 0})
            self.assertEqual(wrap.call_args.kwargs["gate_after"], 16)
            self.assertEqual(wrap.call_args.kwargs["max_length"], 2048)
            if variant == "lora":
                self.assertEqual(wrap.call_args.kwargs["lora_rank"], 16)
                self.assertEqual(wrap.call_args.kwargs["lora_alpha"], 32.)


if __name__ == "__main__":
    unittest.main()
