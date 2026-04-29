class BaseLayerScheduler:
    """
    Scheduler for managing active FBPINN submodels during training
    """

    def __init__(self, n: int):
        """
        Parameters
        ----------
        n: int
            Number of submodels
        """
        self.current_indices = list(range(0, n))

    def on_epoch_start(self) -> list[int]:
        """
        Called on epoch start and return current active models

        Returns
        -------
        indices: list[int]
            List of active submodels indices
        """
        return self.current_indices

    def get_frozen_indices(self) -> list[int]:
        """
        Returns
        -------
        indices: list[int]
            List of indices of frozen, but active submodels
        """
        return []

    def step(self):
        """
        Method for update inner state of scheduler
        """
        pass


class SequenceBlockScheduler(BaseLayerScheduler):
    """
    Scheduler in which each `left_bound_schedule` step index of the first active submodel
    is incremented by `left_bound_step` starts from `start_left_bound`. And index of last active submodel
    incremented by `right_bound_step` every `right_bound_step` step starts from `start_right_bound`.
    """

    def __init__(
        self,
        n: int,
        left_bound_step: int,
        right_bound_step: int,
        left_bound_schedule: int,
        right_bound_schedule: int,
        start_left_bound: int,
        start_right_bound: int,
    ):
        super().__init__(n=n)
        self.n = n
        self.left_bound_step = left_bound_step
        self.right_bound_step = right_bound_step
        self.left_bound_schedule = left_bound_schedule
        self.right_bound_schedule = right_bound_schedule
        self.current_step = 0
        self.current_left_bound = start_left_bound
        self.current_right_bound = start_right_bound
        self.current_indices = list(range(start_left_bound, start_right_bound))
        self.is_changes = True
        self.frozen_indices = []

    def on_epoch_start(self) -> list[int]:
        if self.is_changes:
            range_changed = False
            if self.current_step % self.right_bound_schedule == 0:
                if self.current_right_bound == self.n:
                    self.current_left_bound = 0
                    self.is_changes = False
                self.current_right_bound = min(
                    self.n, self.current_right_bound + self.right_bound_step
                )
                range_changed = True

            active_lb = self.current_left_bound
            if self.is_changes and self.current_step % self.left_bound_schedule == 0:
                curr_lb = self.current_left_bound
                new_lb = self.current_left_bound + self.left_bound_step
                self.current_left_bound = new_lb
                self.frozen_indices = [i for i in range(curr_lb, new_lb)]
                range_changed = True
            if range_changed:
                self.current_indices = list(range(active_lb, self.current_right_bound))
        res = self.current_indices
        return res

    def get_frozen_indices(self) -> list[int]:
        return self.frozen_indices

    def step(self):
        self.current_step += 1


class ReduceEpochsBlockScheduler(SequenceBlockScheduler):
    """
    Expands `SequenceBlockScheduler` with schedule decay. Idea is that first blocks may need to be trained longer.
    """

    def __init__(
        self, reduce_step: int, reduce_schedule: int, reduce_count: int, **kwargs
    ):
        super().__init__(**kwargs)
        self.reduce_step = reduce_step
        self.reduce_schedule = reduce_schedule
        self.reduce_count = reduce_count

    def on_epoch_start(self) -> list[int]:
        if self.reduce_count > 0 and self.current_step % self.reduce_schedule == 0:
            self.left_bound_schedule -= self.reduce_step
            self.right_bound_schedule -= self.reduce_step
            self.reduce_count -= 1

        return super().on_epoch_start()


class SequenceLayerScheduler(BaseLayerScheduler):
    """
    Scheduler for multidimensional tasks. Active models are whole layer.
    For example, if the decomposition contains blocks [4, 4, 4, 4, 4],
    the active models are [0..4], than [0, 8], [4, 12], ..., [12, 20].
    """

    def __init__(self, n: int, blocks_per_axis: list[int], schedule: int):
        """
        Parameters
        ----------
        n: int
            Number of submodels
        blocks_per_axis: list[int]
            Number of submodels per axis
        schedule: int
            Time before change active models
        """
        super().__init__(n=n)
        self.n = n
        self.schedule = schedule
        self.current_step = 0
        self.current_ax = 0
        self.blocks_per_axis = blocks_per_axis
        self.current_left_bound = 0
        self.current_right_bound = blocks_per_axis[self.current_ax]
        self.current_indices = list(
            range(self.current_left_bound, self.current_right_bound)
        )
        self.is_changes = True
        self.frozen_indices = []

    def on_epoch_start(self) -> list[int]:
        if self.is_changes:
            range_changed = False
            active_lb = self.current_left_bound
            if self.current_step % self.schedule == 0:
                if self.current_right_bound == self.n:
                    self.frozen_indices = []
                    self.current_left_bound = 0
                    self.is_changes = False
                    active_lb = self.current_left_bound
                else:
                    self.current_ax = min(
                        len(self.blocks_per_axis) - 1, self.current_ax + 1
                    )
                    self.current_right_bound = min(
                        self.n,
                        self.current_right_bound
                        + self.blocks_per_axis[self.current_ax],
                    )
                    curr_lb = self.current_left_bound
                    new_lb = (
                        self.current_left_bound
                        + self.blocks_per_axis[self.current_ax - 1]
                    )
                    self.current_left_bound = new_lb
                    self.frozen_indices = [i for i in range(curr_lb, new_lb)]
                range_changed = True
            if range_changed:
                self.current_indices = list(range(active_lb, self.current_right_bound))
        res = self.current_indices
        return res

    def get_frozen_indices(self) -> list[int]:
        return self.frozen_indices

    def step(self):
        self.current_step += 1


class GrowingColumnScheduler(BaseLayerScheduler):
    """
    Scheduler that increases the number of active submodels until all are active
    """

    def __init__(self, n: int, blocks_per_axis: list[int], schedule: int):
        """
        Parameters
        ----------
        n: int
            Number of submodels
        blocks_per_axis: list[int]
            Number of submodels per axis
        schedule: int
            Time before change active models
        """
        super().__init__(n=n)
        self.n = n
        self.blocks_per_axis = blocks_per_axis
        self.schedule = schedule
        self.current_step = 0
        self.current_col = 0

        first = blocks_per_axis[0]
        self.current_right_bound = first
        self.current_indices = list(range(0, first))

    def step(self):
        self.current_step += 1
        if self.current_step % self.schedule != 0 or self.current_right_bound >= self.n:
            return

        self.current_col = min(self.current_col + 1, len(self.blocks_per_axis) - 1)
        self.current_right_bound = min(
            self.n, self.current_right_bound + self.blocks_per_axis[self.current_col]
        )
        self.current_indices = list(range(0, self.current_right_bound))
