import unittest

from dexteroushands_lab.scenes.commands import BimanualCommand


class BimanualCommandTests(unittest.TestCase):
    def test_52_values_split_into_independent_26_value_hands(self):
        command = BimanualCommand.from_flat(tuple(range(52)))
        self.assertEqual(command.left.translation_world, (0, 1, 2))
        self.assertEqual(command.left.rotation_local, (3, 4, 5))
        self.assertEqual(command.left.finger_positions, tuple(range(6, 26)))
        self.assertEqual(command.right.translation_world, (26, 27, 28))
        self.assertEqual(command.right.finger_positions, tuple(range(32, 52)))

    def test_wrong_length_rejected(self):
        with self.assertRaises(ValueError):
            BimanualCommand.from_flat((0,) * 51)


if __name__ == "__main__":
    unittest.main()
