import unittest

from dexgym.tasks.direct.wholebody_s05.v1.task.parity import FREE_RESPONSE_LIMITS, free_response_ok, update_peak


class FreeResponseParityTest(unittest.TestCase):
    def test_matching_clean_run_passes(self):
        errors = dict.fromkeys(FREE_RESPONSE_LIMITS, 0.)
        self.assertTrue(free_response_ok(errors, clean=True, completed=10, planned=10))

    def test_final_substep_event_and_missing_substep_fail(self):
        errors = dict.fromkeys(FREE_RESPONSE_LIMITS, 0.)
        self.assertFalse(free_response_ok(errors, clean=False, completed=10, planned=10))
        self.assertFalse(free_response_ok(errors, clean=True, completed=9, planned=10))

    def test_bad_applied_target_fails_even_with_matching_state(self):
        errors = dict.fromkeys(FREE_RESPONSE_LIMITS, 0.)
        errors["applied_rad"] = 2 * FREE_RESPONSE_LIMITS["applied_rad"]
        self.assertFalse(free_response_ok(errors, clean=True, completed=10, planned=10))

    def test_invalid_diagnostic_cannot_be_erased_by_later_finite_sample(self):
        errors = dict.fromkeys(FREE_RESPONSE_LIMITS, 0.)
        errors["torque_Nm"] = update_peak(update_peak(0., float("nan")), 0.)
        self.assertFalse(free_response_ok(errors, clean=True, completed=10, planned=10))
