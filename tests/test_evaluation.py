import json
from pathlib import Path
import tempfile
import unittest

from thesystem.evaluation.evaluation_data import convert, SUBSETS, COUNTS
from thesystem.evaluation.evaluation_io import evaluate_stage, read_stage
from thesystem.evaluation.evaluation_metrics import decision_metrics, routing_metrics
from thesystem.evaluation.writing_checks import check_writing


class EvaluationTests(unittest.TestCase):
    def test_exact_five_benchmark_scope(self):
        self.assertEqual(SUBSETS, ("vitaminc-dev", "paws", "squad2", "aegis2", "pubmedqa"))
        self.assertEqual(sum(COUNTS), 1648)

    def test_gold_label_does_not_enter_prompt(self):
        record = {"id": "one", "family": "context", "input": {"state": "Context", "questions": {"decision": {"type": "noul", "criteria": {"false": "No", "true": "Yes"}, "instructions": "Is it supported?"}}}, "reference": {"target": False}}
        before = convert(record, "paws")
        record["reference"]["target"] = True
        after = convert(record, "paws")
        self.assertEqual(before["prompt"], after["prompt"])
        self.assertEqual((before["target_index"], after["target_index"]), (0, 1))
        writing = convert(record, "paws", writing=True)
        self.assertEqual(writing["mode_target"], 0)
        self.assertNotIn("target_index", writing)
        self.assertEqual(before["options"], writing["options"])

    def test_wrong_route_counts_even_when_answer_is_right(self):
        rows = [{"id": "one", "group_id": "one", "target": 0, "scores": {"lora": [2., 1.]},
                 "automatic_scores": {"lora": [2., 1.]}, "route": 0, "kind": "choice", "option_keys": ["a", "b"]}]
        self.assertEqual(decision_metrics(rows, "lora")["correct"], 1)
        self.assertEqual(decision_metrics(rows, "lora", automatic=True)["correct"], 0)

    def test_confusion_matrix_orientation(self):
        rows = [{"id":str(i), "group_id":str(i), "target": a, "route": b} for i,(a,b) in enumerate([(1,1),(1,1),(1,0),(0,1),(0,0)])]
        result = routing_metrics(rows)
        self.assertEqual(result["matrix"], [[2,1],[1,1]])
        self.assertEqual(result["matrix_order"], ["decision", "writing"])

    def test_interrupted_shards_resume_and_detect_tampering(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); rows=[{"id":str(i)} for i in range(3)]
            def interrupted(batch):
                if batch[0]["id"] == "2": raise RuntimeError("interrupted")
                return batch
            with self.assertRaises(RuntimeError):
                evaluate_stage(rows,folder,{"run":1},interrupted,{},shard_size=2)
            called=[]
            evaluate_stage(rows,folder,{"run":1},lambda batch: called.extend(batch) or batch,{},shard_size=2)
            self.assertEqual(called,[{"id":"2"}])
            self.assertEqual(read_stage(folder),rows)
            shard=folder/'part-000000.json'; payload=json.loads(shard.read_text());payload['rows'][0]['id']='changed';shard.write_text(json.dumps(payload))
            with self.assertRaisesRegex(ValueError,"Invalid results"):
                read_stage(folder)

    def test_generation_mismatch_caps_and_wrong_routes_stay_failures(self):
        class Backend:
            arm="probe"
            def __init__(self, capped=False): self.capped=capped
            def generate(self,row,mode,cap):
                return {"generated_ids":[1,2], "kind":"decision" if mode=="auto" else "text", "hit_limit":self.capped,"trace":{"head_calls":{"mlp256":0}}}
        row={"id":"one","group_id":"one","writing_suite":"historical","prompt":"write","input_sha256":"hash"}
        result=check_writing(Backend(),row)
        self.assertTrue(result['forced_matches_native'])
        self.assertFalse(result['automatic_matches_native'])
        self.assertFalse(result['passed'])
        capped=check_writing(Backend(True),row)
        self.assertEqual(len(capped['attempts']),2)
        self.assertTrue(capped['remaining_truncation'])
