"""Injection registry; concrete robots and tasks register themselves externally."""


class PluginRegistry:
    def __init__(self):
        self._robots = {}
        self._tasks = {}

    def register_robot(self, robot_id: str, factory):
        if not robot_id or robot_id in self._robots or not callable(factory):
            raise ValueError("Robot registration is empty, duplicate or not callable")
        self._robots[robot_id] = factory

    def register_task(self, task_id: str, factory):
        if not task_id or task_id in self._tasks or not callable(factory):
            raise ValueError("Task registration is empty, duplicate or not callable")
        self._tasks[task_id] = factory

    def robot(self, robot_id: str, **configuration):
        try:
            instance = self._robots[robot_id](**configuration)
        except KeyError as error:
            raise KeyError(f"Unknown SuperMimic robot: {robot_id}") from error
        if instance.spec.robot_id != robot_id:
            raise ValueError("Robot factory returned a different RobotSpec ID")
        return instance

    def task(self, task_id: str, **configuration):
        try:
            instance = self._tasks[task_id](**configuration)
        except KeyError as error:
            raise KeyError(f"Unknown SuperMimic task: {task_id}") from error
        if instance.spec.task_id != task_id:
            raise ValueError("Task factory returned a different TaskSpec ID")
        return instance

    @property
    def robot_ids(self):
        return tuple(self._robots)

    @property
    def task_ids(self):
        return tuple(self._tasks)
