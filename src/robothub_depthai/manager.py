import contextlib
import logging as log
import os
from typing import Optional

import robothub

from robothub_depthai import Device
from robothub_depthai.hub_camera import HubCamera

__all__ = ['DeviceManager', 'DEVICE_MANAGER']


class NoDevicesException(Exception):
    pass


class DeviceManager:
    """
    A manager class to handle multiple HubCamera instances.
    """
    REPORT_FREQUENCY = 10  # seconds
    POLL_FREQUENCY = 0.0005

    def __init__(self):
        self._devices = []  # Used to store all devices, shouldn't be modified in any way
        self._hub_cameras = []  # Used to store all HubCamera instances that are currently running

        self.running = False
        self.connecting_to_device = False  # Used to prevent multiple threads from connecting to the same device
        self.stop_event = robothub.threading.Event()

        self.reporting_thread = robothub.threading.Thread(target=self._report, name='ReportingThread', daemon=False)
        self.polling_thread = robothub.threading.Thread(target=self._poll, name='PollingThread', daemon=False)
        self.connection_thread = robothub.threading.Thread(target=self._connect, name='ConnectionThread', daemon=False)

    def __exit__(self):
        self.stop()

    def start(self) -> None:
        """
        Start the cameras, start reporting and polling threads.
        """
        self.running = True

        # Endless loop to prevent app from exiting if no devices are found
        while self.running:
            if self._devices:
                break

            self.stop_event.wait(5)

        self.connection_thread.start()
        log.info('Device connection thread: started successfully.')

        self.reporting_thread.start()
        log.info('Reporting thread: started successfully.')

        self.polling_thread.start()
        log.info('Polling thread: started successfully.')

    def stop(self) -> None:
        """
        Stop the cameras, stop reporting and polling threads.
        """
        log.debug('Threads: gracefully stopping...')
        self.stop_event.set()

        self.__graceful_thread_join(self.connection_thread)
        self.__graceful_thread_join(self.reporting_thread)
        self.__graceful_thread_join(self.polling_thread)

        try:
            robothub.STREAMS.destroy_all_streams()
        except BaseException as e:
            raise Exception(f'Destroy all streams excepted with: {e}.')

        for camera in self._hub_cameras:
            try:
                if camera.state != robothub.DeviceState.DISCONNECTED:
                    with open(os.devnull, 'w') as devnull:
                        with contextlib.redirect_stdout(devnull):
                            camera.oak_camera.__exit__(Exception, 'Device disconnected - app shutting down', None)
            except BaseException as e:
                raise Exception(f'Device {camera.device_mxid}: could not exit with exception: {e}.')

        log.info('App: stopped successfully.')

    def add_device(self, device: 'Device') -> None:
        """
        Add a camera to the list of cameras.
        """
        self._devices.append(device)

    def remove_device(self, device: 'Device') -> None:
        """
        Remove a camera from the list of cameras.
        """
        self._devices.remove(device)

    def _report(self) -> None:
        """
        Reports the state of the cameras to the agent. Active when app is running, inactive when app is stopped.
        Reporting frequency is defined by REPORT_FREQUENCY.
        """
        while self.running:
            for camera in self._hub_cameras:
                try:
                    device_info = camera.info_report()
                    device_stats = camera.stats_report()

                    robothub.AGENT.publish_device_info(device_info)
                    robothub.AGENT.publish_device_stats(device_stats)
                except Exception as e:
                    log.debug(f'Device {camera.device_mxid}: could not report info/stats with error: {e}.')

            self.stop_event.wait(self.REPORT_FREQUENCY)

    def _poll(self) -> None:
        """
        Polls the cameras for new detections. Polling frequency is defined by POLL_FREQUENCY.
        """
        while self.running:
            for camera in self._hub_cameras:
                if not camera.poll():
                    self._disconnect_camera(camera)
                    continue

            self.stop_event.wait(self.POLL_FREQUENCY)

    def _connect(self) -> None:
        """
        Reconnects the cameras that were disconnected or reconnected.
        """
        while self.running:
            if len(self._hub_cameras) == len(self._devices) or self.connecting_to_device:
                self.stop_event.wait(5)
                continue

            mxids = [camera.device_mxid for camera in self._hub_cameras]
            for device in self._devices:
                if device.mxid not in mxids:
                    self.connecting_to_device = True
                    self._connect_device(device)
                    self.connecting_to_device = False

    def _connect_device(self, device: 'Device') -> None:
        """
        Connect a device to the app.
        """
        hub_camera = HubCamera(device_mxid=device.mxid)
        if not device._start(hub_camera):  # Initialize the device (create streams, etc.)
            hub_camera.stop()
            return

        hub_camera.start()  # Start the pipeline

        device.connect_callback(hub_camera)
        self._hub_cameras.append(hub_camera)

    def _disconnect_camera(self, camera: HubCamera) -> None:
        """
        Disconnect a device from the app.
        """
        log.info(f'Device {camera.device_mxid}: disconnected.')
        camera.stop()
        self._hub_cameras.remove(camera)

        device = self._get_device_by_mxid(camera.device_mxid)
        if device:
            device.disconnect_callback(device)

    def _get_device_by_mxid(self, mxid: str) -> Optional[Device]:
        """
        Get a device by its mxid.
        """
        for device in self._devices:
            if device.mxid == mxid:
                return device

        return None

    @staticmethod
    def __graceful_thread_join(thread) -> None:
        """
        Gracefully stop a thread.
        """
        try:
            if thread.is_alive():
                thread.join()
        except BaseException as e:
            log.error(f'{thread.getName()}: join excepted with: {e}.')


DEVICE_MANAGER = DeviceManager()
