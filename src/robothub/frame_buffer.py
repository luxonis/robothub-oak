import datetime
import itertools
import logging
import threading
import time
import uuid
from collections import deque
from enum import Enum
from pathlib import Path
from queue import Empty, Queue
from typing import Callable, Optional, Union

import depthai as dai
from depthai_sdk.recorders.video_writers import AvWriter
from depthai_sdk import FramePacket
from robothub.events import send_video_event

try:
    import av
except ImportError:
    av = None

logger = logging.getLogger(__name__)

__all__ = ['FrameBuffer']


def _depthai_timestamp(frame: dai.ImgFrame) -> datetime.timedelta:
    return frame.getTimestampDevice()


def _sdk_timestamp(frame: dai.ImgFrame) -> datetime.timedelta:
    return frame.getTimestampDevice()


def _write_depthai_img_Frames(av_writer: AvWriter, packets: list[dai.ImgFrame]) -> None:
    for img_frame in packets:
        av_writer.write(img_frame)


def _write_sdk_packets(av_writer: AvWriter, packets: list) -> None:
    for p in packets:
        av_writer.write(p.msg)


class PacketType(Enum):
    DEPTHAI = 0
    SDK = 1


class FrameBuffer:
    __packet_timestamp: dict = {PacketType.DEPTHAI: _depthai_timestamp,
                                PacketType.SDK: _sdk_timestamp}
    __mux_in_av_writer: dict = {PacketType.DEPTHAI: _write_depthai_img_Frames,
                                PacketType.SDK: _write_sdk_packets}

    def __init__(self, maxlen: int = None):
        """
        A buffer for storing frames.

        :param maxlen: The maximum number of frames to store in the buffer. If None, the buffer will be unbounded.
        """
        self.__buffer = deque(maxlen=maxlen)
        self.__temporary_queues = set()
        self.__packet_type: Optional[PacketType] = None

    def _get_slice(self, start: int, end: int | None = None) -> list:
        """
        Get a slice of the buffer.

        :param start: Start index.
        :param end: End index. If None, return all elements from `start` to the end of the buffer.
        :return: Slice of the buffer.
        """
        if start < 0:
            start = 0
        if end is None:
            end = len(self.__buffer) - 1
        if end > len(self.__buffer) - 1:
            end = len(self.__buffer) - 1
        return list(self.__buffer)[start:end]

    def save_video_event(self,
                         before_seconds: int,
                         after_seconds: int,
                         title: str,
                         fps: int,
                         frame_width: int,
                         frame_height: int,
                         on_complete: Optional[Callable] = None,
                         delete_after_complete: bool = False,
                         metadata: Optional[dict] = None,
                         ) -> threading.Thread:
        """
        Saves a video event to the frame buffer, then calls `on_complete` when the video is ready.
        When the video is ready, the `on_complete` function will be called with the path to the video file.
        Note: When app is stopped, it is not guaranteed that the video will be saved.

        :param before_seconds: Number of seconds to save before the event occurred.
        :param after_seconds: Number of seconds to save after the event occurred.
        :param title: Title of the video event.
        :param fps: The FPS of the video.
        :param frame_width: Video frame width.
        :param frame_height: Video frame height.
        :param on_complete: Callback function to call when the video is ready. Path to the video file will be passed as the first argument.
        :param delete_after_complete: If True, delete the video file after the callback function is called. Default: False.
        :return: The 'threading.Thread' where the video is processed.
        """

        def on_complete_default(video_path):
            send_video_event(video_path, title, metadata)

        if on_complete is None:
            on_complete = on_complete_default

        t = threading.Thread(target=self._save_video,
                             args=(before_seconds, after_seconds, fps, frame_width, frame_height, on_complete, delete_after_complete),
                             daemon=True)
        t.start()
        return t

    def _save_video(self,
                    before_seconds: int,
                    after_seconds: int,
                    fps: int,
                    frame_width: int,
                    frame_height: int,
                    on_complete: Optional[Callable] = None,
                    delete_after_complete: bool = False
                    ) -> Optional[str]:
        """
        Save a video of the last `before_seconds` seconds and the next `after_seconds` seconds.

        :param before_seconds: Number of seconds to save before the current time.
        :param after_seconds: Number of seconds to save after the current time.
        :param fps: The FPS of the video.
        :param frame_width: Video frame width.
        :param frame_height: Video frame height.
        :param on_complete: Callback function to call when the video is ready. Path to the video file will be passed as the first argument.
        :param delete_after_complete: If True, delete the video file after the callback function is called. Default: False.
        """
        if not av:
            raise ImportError('av library is not installed. Cannot save video. '
                              'Please make sure PyAV is installed (`pip install pyav`).')

        if before_seconds < 0 or after_seconds < 0:
            raise ValueError('`before_seconds` and `after_seconds` must be positive.')
        if (isinstance(before_seconds, int) and isinstance(after_seconds, int)) is False:
            raise ValueError(f'`before_seconds`: {before_seconds} and `after_seconds`: {after_seconds} must be integers,'
                             f'but they are {type(before_seconds)} and {type(after_seconds)}.')
        if len(self.__buffer) == 0:
            logger.warning(f"There are no frames in the buffer. Cannot save video."
                           f"\nMake sure to use 'frame_buffer.add_frame()' on every_frame")
            return None

        logger.info(f"Initiating video capture: Starting {before_seconds} seconds before and ending {after_seconds} seconds after...")
        if before_seconds * fps > len(self.__buffer):
            logger.warning(f"`before_seconds` {before_seconds}  is too large. The buffer does not contain enough frames."
                           f" The video will start later.")

        # Get frames before the current time
        video_frames_before = self._get_slice(start=(len(self.__buffer) - 1) - before_seconds * fps)
        video_frames_after = []
        temp_queue = Queue()
        self.__temporary_queues.add(temp_queue)

        latest_t_before = self.__packet_timestamp[self.__packet_type](video_frames_before[-1])

        # Get frames after the current time
        logger.info(f"Video capture waiting for {after_seconds} seconds...")
        retries = 0
        while True:
            try:
                p = temp_queue.get(block=True, timeout=2.0)
                timestamp = self.__packet_timestamp[self.__packet_type](p)
                if timestamp > latest_t_before:
                    video_frames_after.append(p)
                if timestamp - latest_t_before > datetime.timedelta(seconds=after_seconds):
                    break
            except Empty:
                time.sleep(0.5)
                retries += 1
                if retries > 10:
                    logger.warning(f"Video capture timed out after 10 retries. "
                                   f"Make sure to use 'frame_buffer.add_frame()' on every_frame")
                    return None

        self.__temporary_queues.remove(temp_queue)
        video_path = self._mux_video(packets=video_frames_before + video_frames_after[:int(after_seconds * fps)],
                                     fps=fps,
                                     frame_width=frame_width,
                                     frame_height=frame_height)

        logger.info(f"Video capture complete. Video saved to {video_path}. Calling 'on_complete': {on_complete.__name__}.\n"
                    f"Video {'will' if delete_after_complete else 'will not'} be deleted after the callback function is called.")
        if on_complete:
            on_complete(video_path)
        if on_complete and delete_after_complete:
            Path(video_path).unlink()
            return None

        return video_path

    def _mux_video(self, packets: list,
                   fps: int,
                   frame_width: int,
                   frame_height: int,
                   ) -> str | None:
        """
        Mux a list of packets into a video file and return the path to the video file.
        """
        dir_path = Path(f'/tmp/robothub-videos/{uuid.uuid4().hex}')
        dir_path.mkdir(parents=True)
        name = str(uuid.uuid4().hex)
        av_writer = AvWriter(path=Path(dir_path),
                             name=name,
                             fourcc='h264',
                             fps=fps,
                             frame_shape=(frame_width, frame_height))

        self.__mux_in_av_writer[self.__packet_type](av_writer, packets)
        av_writer.close()

        video_path = Path(dir_path, name).with_suffix('.mp4')
        return str(video_path)

    def add_frame(self, packet: Union[FramePacket, dai.ImgFrame]) -> None:
        """
        Default callback for the frame buffer. It will append the packet to the buffer and put it in all temporary queues.
        """
        if self.__packet_type is None:
            self.__packet_type = PacketType.DEPTHAI if isinstance(packet, dai.ImgFrame) else PacketType.SDK

        self.__buffer.append(packet)
        for q in self.__temporary_queues:
            q.put(packet)

    @property
    def maxlen(self) -> int:
        return self.__buffer.maxlen
