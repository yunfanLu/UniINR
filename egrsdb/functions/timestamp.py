from absl.logging import info
from absl.testing import absltest


def get_gs_sharp_timestamps(start, end, count, total, correct_offset):
    """
    We get "count" frames from indexes of "start" to "end" in "total" frames.
    :param start: the start index of the global shutter frames
    :param end: the end index of the global shutter frames
    :param count: the number gs frame we want to get.
    :param total: the total number of gs frames.
    :param correct_offset: if True, we will get the center of the gs frame.
    :return:
    """
    step = (end - start) / count
    if correct_offset:
        indexes = [int(start + i * step + step / 2) for i in range(count)]
    else:
        indexes = [int(start + i * step) for i in range(count)]
    timestamps = [i / total for i in indexes]
    return indexes, timestamps


def get_rs_shape_timestamps(rs_blur_timestamp, rs_integral):
    """

    :param rs_blur_timestamp: [start, end, exposure]
    :param rs_integral: the number of sharp frames we want to get.
    :return:
    """
    rs_start, rs_end, rs_exposure = rs_blur_timestamp
    rs_step = rs_exposure / rs_integral
    rs_sharp_timestamps = []
    for i in range(rs_integral):
        rs_sharp_timestamps.append((rs_start + i * rs_step, rs_end + i * rs_step))
    return rs_sharp_timestamps


class GlobalShutterTimestampTest(absltest.TestCase):
    def test_gs_sharp_timestamps(self):
        start = 0
        end = 520
        count = 16
        total = 520
        indexes, timestamps = get_gs_sharp_timestamps(start, end, count, total, False)
        info(f"  indexes: {indexes}")
        info(f"  timestamps: {timestamps}")
        self.assertEqual(len(indexes), count)
        self.assertEqual(len(timestamps), count)
        #
        indexes, timestamps = get_gs_sharp_timestamps(start, end, count, total, True)
        info(f"  indexes: {indexes}")
        info(f"  timestamps: {timestamps}")

    def test_rs_sharp_timestamps(self):
        rs_blur_timestamp = [0, 0.5, 0.5]
        rs_integral = 52
        rs_sharp_timestamps = get_rs_shape_timestamps(rs_blur_timestamp, rs_integral)
        info(f"  rs_sharp_timestamps: {rs_sharp_timestamps}")
        self.assertEqual(len(rs_sharp_timestamps), rs_integral)


if __name__ == "__main__":
    absltest.main()
