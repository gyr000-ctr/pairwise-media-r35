"""texstream 核心 API（初始未实现）。"""


class TexStream:
    def __init__(self, w, h, budget):
        """构造"""
        raise NotImplementedError

    def request(self, mip, frame):
        """请求"""
        raise NotImplementedError

    def resident(self):
        """驻留集合"""
        raise NotImplementedError

    def bytes_used(self):
        """占用字节"""
        raise NotImplementedError
