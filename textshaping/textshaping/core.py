"""textshaping 核心 API（初始未实现）。"""


class Shaping:
    def shape(self, glyphs, features):
        """应用"""
        raise NotImplementedError

    def advances(self):
        """步进"""
        raise NotImplementedError
