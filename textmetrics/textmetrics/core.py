"""textmetrics 核心 API（初始未实现）。"""


class Metrics:
    def __init__(self, ascent, descent, line_gap):
        """构造"""
        raise NotImplementedError

    def line_height(self, size):
        """行高"""
        raise NotImplementedError

    def layout(self, lines, size):
        """布局"""
        raise NotImplementedError
