"""P01-A contract tests. The real host audit is a separate integration run."""
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from dexgym.tasks.direct.wholebody_s05.data.historical_usage import history_events, root_use
from dexgym.tasks.direct.wholebody_s05.data.inventory import Inventory, calibration_witnesses
from dexgym.tasks.direct.wholebody_s05.data.lineage import Lineage, exact_aliases
from dexgym.tasks.direct.wholebody_s05.data.source_registry import (
    classify, final_test, legacy_groups, same_bytes, stable_json,
)


class SourceTest(unittest.TestCase):
    def test_egopro_and_simulation_are_excluded_even_with_human_topics(self):
        for origin in ('EGOPRO', 'SIM_REPLAY'):
            with self.subTest(origin=origin):
                self.assertEqual(classify(origin, True, ())[0], 'EXCLUDED')

    def test_unknown_origin_is_quarantined(self):
        self.assertEqual(classify('UNKNOWN', True, ())[0], 'QUARANTINED')

    def test_stamped_candidate_with_open_metadata_stays_pending(self):
        self.assertEqual(classify('LIVE_TELEOP', True, ('SEMANTIC_RELATION_UNRESOLVED',))[0],
                         'PENDING_DECODE_VERIFICATION')
        self.assertEqual(classify('LIVE_TELEOP', False, ())[0], 'PENDING_DECODE_VERIFICATION')
        self.assertEqual(classify('LIVE_TELEOP', True, ())[0], 'ADMITTED_CANDIDATE')

    def test_manifest_and_split_order(self):
        scans = ['bag_10.mcap', 'other_0.mcap', 'bag_2.mcap', 'bag_0.mcap']
        groups = legacy_groups(scans, ('bag_0.mcap', 'bag_2.mcap', 'bag_10.mcap', 'bag_11.mcap'))
        self.assertEqual(groups[0]['parts'], ['bag_0.mcap', 'bag_2.mcap', 'bag_10.mcap', 'bag_11.mcap'])
        self.assertEqual(groups[0]['missing'], ['bag_11.mcap'])
        self.assertEqual(groups[1]['parts'], ['other_0.mcap'])
        self.assertEqual(legacy_groups(scans)[0]['parts'], ['bag_0.mcap', 'bag_2.mcap', 'bag_10.mcap'])

    def test_file_rename_copies_and_symlink_have_same_hash(self):
        with tempfile.TemporaryDirectory() as temp:
            folder = Path(temp)
            original = folder / 'original_0.mcap'
            renamed = folder / 'renamed.mcap'
            original.write_bytes(b'one recording')
            renamed.write_bytes(original.read_bytes())
            digest1 = same_bytes(original)
            digest2 = same_bytes(renamed)
            self.assertEqual(digest1['sha256'], digest2['sha256'])
            self.assertNotEqual(digest1['physical_identity'], digest2['physical_identity'])
            link = folder / 'link.mcap'
            link.symlink_to(original)
            self.assertEqual(same_bytes(link)['physical_identity'], digest1['physical_identity'])

    def test_canonical_json_is_deterministic(self):
        self.assertEqual(stable_json({'z':[1,2],'a':3}), stable_json({'a':3,'z':[1,2]}))

    def test_calibration_requires_a_recorded_matching_configuration(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'config-2-tracker_calibration.yaml'
            payload = b'tracker_calibration_node:\n  ros__parameters:\n    version: 1\n'
            path.write_bytes(payload)
            entry = {'path': '/config/tracker_calibration.yaml', 'snapshot': str(path),
                     'sha256': sha256(payload).hexdigest()}
            unit = {'unit': 'tracker', 'startedConfigurations': [entry]}
            snapshot = {'units': [unit]}
            result = calibration_witnesses(snapshot)
            self.assertEqual(result[0]['sha256'], entry['sha256'])
            self.assertEqual(result[0]['path'], str(path))
            path.write_text('tracker_calibration_node: {}\n')
            self.assertEqual(calibration_witnesses(snapshot), [])

    def test_registry_is_deterministic_on_the_same_input(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root/'metadata.yaml').write_text(
                'rosbag2_bagfile_information:\n  relative_file_paths: [one_0.mcap]\n'
                '  topics_with_message_count:\n    - topic_metadata: {name: /manus_glove_left}\n'
                '      message_count: 1\n'
            )
            (root/'one_0.mcap').write_bytes(b'mock bytes; sidecar only')
            config = {'raw_roots': [{'path': str(root), 'mode': 'legacy', 'label': 'fixture'}],
                      'history_roots': [], 'positive_replays': {}}
            first = Inventory(config).run()
            second = Inventory(config).run()
            self.assertEqual(stable_json(first), stable_json(second))
            self.assertEqual(first['source_inventory.json']['counts']['canonical_recordings'], 1)

    def test_unknown_or_historical_root_cannot_enter_final_test(self):
        roots = ['native-1','native-2']
        self.assertEqual(final_test(roots, {'native-1':['TRAIN']})['status'], 'BLOCKED')
        self.assertEqual(final_test(roots, {})['eligible_root_ids'], [])


class LineageTest(unittest.TestCase):
    def test_alias_does_not_create_a_new_root(self):
        rows = [{'id':'z-old-name','signature':('f',), 'priority':2},
                {'id':'a-rename','signature':('f',), 'priority':1},
                {'id':'b-empty','signature':None,'priority':3},
                {'id':'c-empty','signature':None,'priority':3}]
        expected = {'a-rename':'z-old-name'}
        self.assertEqual(exact_aliases(rows), expected)
        self.assertEqual(exact_aliases(list(reversed(rows))), expected)

    def test_multilevel_and_multiple_parents(self):
        graph = Lineage({'left':[], 'right':[], 'crop':['left'],
                         'slow':['crop'], 'retarget':['slow','right']})
        self.assertEqual(graph.resolve('retarget'), (('left', 'right'), ()))

    def test_unknown_parent_does_not_become_known_root(self):
        graph = Lineage({'known':[], 'derived':['known','missing-parent']})
        self.assertEqual(graph.resolve('derived'), (('known',), ('missing-parent',)))

    def test_cycle_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'cycle'):
            Lineage({'x':['y'],'y':['x']}).resolve('x')

    def test_history_propagates_through_child_to_all_parent_roots(self):
        graph = Lineage({'a':[], 'b':[], 'merged':['a','b'], 'cropped':['merged']})
        events = [{'source_ref':'cropped','role':'MODEL_SELECTION'}]
        self.assertEqual(root_use(events,graph), {'a':['MODEL_SELECTION'],'b':['MODEL_SELECTION']})

    def test_manifest_assignment_is_distinct_from_observed_use(self):
        sample = {'clips':[{'source_episode':'episode1','split':'train','start_frame':3,'end_frame':12,
                            'source_frame_interval':[3,12], 'timing':{'speed':0.5}}],
                  'splits':{'test':{'clips':[{'episode_id':'episode2'}]}}}
        rows = history_events(sample,'artifact','s0')
        self.assertEqual({(x['source_ref'],x['role'],x['confidence']) for x in rows},
                         {('episode1','TRAIN','DECLARED_SPLIT'),('episode2','HISTORICAL_TEST_ASSIGNMENT','DECLARED_SPLIT')})
        self.assertEqual(rows[0]['interval']['source_frame_interval'], [3,12])
        self.assertEqual(rows[0]['interval']['timing']['speed'], 0.5)


if __name__ == '__main__':
    unittest.main()
