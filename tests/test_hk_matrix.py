import unittest
from vla_eval.hk_matrix import matrix_cells


class MatrixContractTests(unittest.TestCase):
    def test_requested_grid_does_not_invent_missing_actions(self):
        cells=matrix_cells([32,16,8,4],[16,8,4,2],50)
        self.assertEqual(len(cells),16)
        self.assertEqual(sum(c['applicable'] for c in cells),13)
        self.assertEqual({(c['H'],c['K']) for c in cells if not c['applicable']},
                         {(8,16),(4,16),(4,8)})

    def test_invalid_axes_cannot_silently_change_model_workload(self):
        for h,k in [([64],[4]),([0],[2]),([4,4],[2]),([4],[0]),([4],[2.5])]:
            with self.subTest(h=h,k=k),self.assertRaises(ValueError):matrix_cells(h,k,50)
