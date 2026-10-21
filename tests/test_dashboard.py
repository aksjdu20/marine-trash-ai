"""대시보드는 데이터/모델 없이도 안내 화면을 열어야 합니다."""
import unittest

from streamlit.testing.v1 import AppTest

from src.common import ROOT


class DashboardTest(unittest.TestCase):
    def test_five_menus_without_models(self):
        app=AppTest.from_file(str(ROOT/"app.py"),default_timeout=20).run()
        self.assertEqual(len(app.exception),0)
        for menu in ["동영상 분석","AI 성능 비교","실험 결과","데이터셋 분석","AI 이미지 분석"]:
            app.sidebar.radio[0].set_value(menu).run()
            self.assertEqual(len(app.exception),0,menu)


if __name__=="__main__":
    unittest.main()
