"""textellipsis 核心 API（初始未实现）。"""


class Ellipsis:
    def truncate(self, glyphs, max_width, ell='…'):
        """截断"""
        raise NotImplementedError

    def width(self, glyphs):
        """宽度"""
        raise NotImplementedError
