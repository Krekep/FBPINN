class PhysLoss:
    def __init__(self, description: str = ""):
        self.description = description
        self.full_losses: list[callable] = []  # on full domain
        self.sub_losses: list[
            tuple[callable, list[list[float]]]
        ] = []  # on subdomain (e.g. u(0, x) = sin(x))
