import os
os.environ.setdefault("QT_QPA_PLATFORM","offscreen")
import unittest
from unittest.mock import Mock, patch
import numpy as np
from PyQt5.QtWidgets import QApplication
from PyQt5.QtCore import QPointF
from lineup_dialog import LineupDialog, PlateView
import test_lineup_solver as fixtures


class LiteWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app=QApplication.instance() or QApplication([])

    def dialog(self, callback=lambda T:(True,"")):
        data=fixtures.SurveySolveTests()
        data.setUp()
        faces=np.array([[0,1,2],[3,4,5]],int)
        dlg=LineupDialog(np.zeros((480,848),np.uint8),data.K,data.world,faces,
                         np.eye(4),callback)
        self.addCleanup(dlg.close)
        return dlg,data

    def populate(self,dlg,data):
        for p,uv in zip(data.world,data.image):
            dlg._scan_pick(p)
            dlg._image_pick(uv)

    def test_point_pairs_gate_solve_and_apply(self):
        dlg,data=self.dialog()
        self.assertFalse(dlg.solve_button.isEnabled())
        self.assertFalse(dlg.apply_button.isEnabled())
        self.populate(dlg,data)
        self.assertTrue(dlg.solve_button.isEnabled())
        dlg._solve()
        self.assertTrue(dlg.apply_button.isEnabled(),dlg.summary.text())
        self.assertIn("RMS",dlg.summary.text())

    def test_solve_button_with_exactly_six_pairs_previews_before_apply(self):
        callback=Mock(return_value=(True,""))
        dlg,data=self.dialog(callback)
        for p,uv in zip(data.world[:6],data.image[:6]):
            dlg._scan_pick(p)
            dlg._image_pick(uv)
        self.assertTrue(dlg.solve_button.isEnabled())
        dlg.solve_button.click()
        self.assertTrue(dlg.apply_button.isEnabled(),dlg.summary.text())
        self.assertIsNotNone(dlg.plate.projected)
        self.assertIn("Apply verified lineup",dlg.instructions.text())
        callback.assert_not_called()

    def test_planar_rejection_is_visible_in_instruction_banner(self):
        dlg,data=self.dialog()
        data.world[:,2]=1
        self.populate(dlg,data)
        dlg.solve_button.click()
        self.assertIsNone(dlg.solution)
        self.assertFalse(dlg.apply_button.isEnabled())
        self.assertIn("Camera solve failed",dlg.instructions.text())
        self.assertIn("flat surface",dlg.instructions.text())

    def test_failed_retry_clears_previous_preview(self):
        dlg,data=self.dialog()
        self.populate(dlg,data)
        dlg.solve_button.click()
        self.assertIsNotNone(dlg.plate.projected)
        with patch("lineup_dialog.solve_lineup",
                   return_value=fixtures.solve_lineup(data.world[:5],data.image[:5],data.K)):
            dlg.solve_button.click()
        self.assertIsNone(dlg.solution)
        self.assertIsNone(dlg.plate.projected)
        self.assertFalse(dlg.apply_button.isEnabled())
        self.assertEqual(dlg.table.item(0,3).text(),"—")
        self.assertEqual(dlg.table.item(0,4).text(),"Pending")
        self.assertIn("at least 6",dlg.instructions.text())

    def test_solver_exception_is_shown_and_can_be_retried(self):
        dlg,data=self.dialog()
        self.populate(dlg,data)
        with patch("lineup_dialog.solve_lineup",side_effect=RuntimeError("solver unavailable")):
            with self.assertLogs("lineup_dialog",level="ERROR"):
                dlg.solve_button.click()
        self.assertIsNone(dlg.solution)
        self.assertFalse(dlg.apply_button.isEnabled())
        self.assertIn("RuntimeError: solver unavailable",dlg.instructions.text())
        self.assertTrue(dlg.solve_button.isEnabled())
        dlg.solve_button.click()
        self.assertTrue(dlg.apply_button.isEnabled(),dlg.summary.text())

    def test_failed_depth_check_leaves_dialog_open(self):
        dlg,data=self.dialog(lambda T:(False,"Depth mismatch"))
        self.populate(dlg,data); dlg._solve(); dlg._apply()
        self.assertEqual(dlg.solution.ok,True)
        self.assertEqual(dlg.summary.text(),"Depth mismatch")
        self.assertEqual(dlg.result(), 0)  # QDialog not accepted on a rejected check

    def test_edit_after_solve_invalidates_old_result(self):
        dlg,data=self.dialog()
        self.populate(dlg,data); dlg._solve(); dlg._undo()
        self.assertIsNone(dlg.solution)
        self.assertFalse(dlg.apply_button.isEnabled())

    def test_image_click_without_survey_point_adds_nothing(self):
        dlg,data=self.dialog()
        dlg._image_pick(data.image[0])
        self.assertEqual(len(dlg.world),0)

    def test_scan_transform_is_undone_for_solve_coordinates(self):
        dlg,data=self.dialog()
        dlg.T_display_map[:3,3]=[10,20,30]
        dlg._scan_pick(data.world[0]+[10,20,30])
        dlg._image_pick(data.image[0])
        np.testing.assert_allclose(dlg.world[0],data.world[0])

    def test_letterbox_click_and_mapping(self):
        plate=PlateView(np.zeros((480,848),np.uint8))
        plate.resize(848,848)
        self.assertIsNone(plate.to_image(QPointF(100,10)))
        np.testing.assert_allclose(plate.to_image(QPointF(424,424)),(424,240))


if __name__ == "__main__":
    unittest.main()
