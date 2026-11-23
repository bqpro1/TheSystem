"""Public API checks with tiny random Qwen models, not 9B performance tests."""
import string
import subprocess
import sys
import unittest

import torch
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from thesystem import TheSystem
from thesystem._vendor.lora.model import InternalFAST
from thesystem._vendor.probe.model import FrozenQwen
from thesystem._vendor.probe.probe_model import ProbeModel


class Tokenizer:
    pad_token_id, eos_token_id = 0, 127
    pad_token, eos_token, padding_side = '<pad>', '<eos>', 'right'

    def encode(self, text, **kwargs):
        if len(text) == 1 and text in string.ascii_uppercase:
            return [ord(text) - ord('A') + 10]
        return [1] * 2049 if 'TOO_LONG' in text else [1, 2, 3, 4]

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs['enable_thinking'] is False
        self.messages = messages
        return 'chat:' + messages[-1]['content']

    def decode(self, tokens, **kwargs):
        return ' '.join(map(str, tokens))


def tiny(variant):
    torch.manual_seed(42)
    config = Qwen3_5TextConfig(hidden_size=32, intermediate_size=64, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=16,
        linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_conv_kernel_dim=4,
        vocab_size=128, max_position_embeddings=256,
        layer_types=['linear_attention','full_attention','linear_attention','full_attention'],
        rope_parameters={'rope_type':'default','rope_theta':10000.,'partial_rotary_factor':1.,'mrope_section':[2,3,3]},
        pad_token_id=0, eos_token_id=127)
    config._attn_implementation = 'eager'
    base = Qwen3_5ForCausalLM(config)
    base.generation_config.eos_token_id = None
    tokenizer = Tokenizer()
    if variant == 'lora':
        reference = InternalFAST(base, tokenizer, gate_after=2, lora_rank=2, lora_alpha=4, max_length=2048)
        reference.stage('evaluation')
        with torch.no_grad():
            for adapter in reference.adapters.values():
                adapter.up.weight.normal_(std=.05)
        model = reference
    else:
        reference = FrozenQwen(base, tokenizer, gate_after=2, max_length=2048)
        model = ProbeModel(reference)
    model.requires_grad_(False).eval()
    return TheSystem(variant, model), reference, tokenizer


class InferenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)

    def gate(self, reference, decision):
        with torch.no_grad():
            reference.gate.weight.zero_()
            reference.gate.bias.fill_(100 if decision else -100)

    def test_public_decisions_match_frozen_backend_and_keep_prompt(self):
        prompt = 'Choose a team: A: Billing; B: Technical support. API key rejected.'
        for variant in ('probe','lora'):
            with self.subTest(variant=variant):
                public, reference, tokenizer = tiny(variant)
                self.gate(reference, True)
                result = public.respond(prompt, {'B':'Technical support','A':'Billing'})
                row = {'prompt':prompt,'options':['Billing','Technical support']}
                expected = (public._model.decide(row, 'mlp256', automatic=True) if variant == 'probe'
                            else reference.score_requests([row], 'auto')[0])
                self.assertEqual(result['candidate_scores'], expected['candidate_scores'])
                self.assertEqual(result['mode'], 'decision')
                self.assertEqual(result['selected_code'], 'AB'[result['selected_index']])
                self.assertEqual(tokenizer.messages[-1]['content'], prompt)

    def test_automatic_missing_choices_stays_visible(self):
        for variant in ('probe','lora'):
            public, reference, _ = tiny(variant)
            self.gate(reference, True)
            result = public.respond('Write a short reply.')
            self.assertEqual(result['mode'], 'unsupported')
            self.assertEqual(result['route'], 1)
            self.assertIsNone(result['answer'])

    def test_writing_with_choices_matches_native_tokens(self):
        prompt = 'Choose A or B and explain your answer.'
        for variant in ('probe','lora'):
            with self.subTest(variant=variant):
                public, reference, _ = tiny(variant)
                self.gate(reference, False)
                automatic = public.respond(prompt, {'A':'Billing','B':'Support'}, max_new_tokens=3)
                explicit = public.respond(prompt, {'A':'Billing','B':'Support'}, mode='generation', max_new_tokens=3)
                row = {'prompt':prompt,'options':['Billing','Support']}
                native = (public._model.respond(row, 'original', mode='text', max_new_tokens=3) if variant == 'probe'
                          else reference.generate_request(row, mode='native', max_new_tokens=3))
                self.assertEqual(automatic['mode'], 'generation')
                self.assertEqual(automatic['generated_ids'], explicit['generated_ids'])
                self.assertEqual(automatic['generated_ids'], native['generated_ids'])
                self.assertTrue(automatic['hit_limit'])

    def test_forced_decision_overrides_route_only_when_requested(self):
        for variant in ('probe','lora'):
            public, reference, _ = tiny(variant)
            self.gate(reference, False)
            result = public.respond('Choose A or B.', {'A':'Yes','B':'No'}, mode='decision')
            self.assertEqual(result['mode'], 'decision')
            self.assertLess(result['gate_probability'], .5)

    def test_input_contract_and_no_truncation(self):
        for variant in ('probe','lora'):
            public, _, _ = tiny(variant)
            for choices in ({'B':'Yes','C':'No'}, {str(i):str(i) for i in range(19)}, {'A':'Yes'}, {'A':'Yes','B':'Yes'}):
                with self.assertRaises(ValueError): public.respond('A prompt', choices)
            with self.assertRaises(ValueError): public.respond('')
            with self.assertRaises(ValueError): public.respond('A prompt', mode='decision')
            with self.assertRaises(ValueError): public.respond('A prompt', max_new_tokens=True)
            with self.assertRaises(ValueError): public.respond('TOO_LONG')

    def test_import_is_lazy_and_does_not_pollute_module_names(self):
        subprocess.run([sys.executable, '-c', "import sys,thesystem; assert 'torch' not in sys.modules; assert 'model' not in sys.modules"], check=True)
        tiny('lora'); tiny('probe')
        self.assertNotIn('model', sys.modules)


if __name__ == '__main__':
    unittest.main()
