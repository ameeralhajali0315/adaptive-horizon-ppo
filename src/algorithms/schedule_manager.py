
class ScheduleManager:
    """
    Manages the discount factor gamma and GAE parameter lambda.

    Supported strategies:
        - fixed: parameters remain constant
        - linear: parameters change linearly over training
    """

    def __init__(
        self,
        strategy="fixed",
        gamma_start=0.99,
        gamma_end=0.99,
        lambda_start=0.95,
        lambda_end=0.95,
        total_updates=100,
    ):
        if strategy not in {"fixed", "linear"}:
            raise ValueError(
                "strategy must be 'fixed' or 'linear'"
            )

        if total_updates < 1:
            raise ValueError("total_updates must be at least 1")

        if not (0.0 <= gamma_start <= 1.0):
            raise ValueError("gamma_start must be between 0 and 1")

        if not (0.0 <= gamma_end <= 1.0):
            raise ValueError("gamma_end must be between 0 and 1")

        if not (0.0 <= lambda_start <= 1.0):
            raise ValueError("lambda_start must be between 0 and 1")

        if not (0.0 <= lambda_end <= 1.0):
            raise ValueError("lambda_end must be between 0 and 1")

        self.strategy = strategy
        self.gamma_start = gamma_start
        self.gamma_end = gamma_end
        self.lambda_start = lambda_start
        self.lambda_end = lambda_end
        self.total_updates = total_updates

    def get_values(self, update):
        """
        Return gamma and lambda for a 1-based training update.
        """
        if not 1 <= update <= self.total_updates:
            raise ValueError(
                f"update must be between 1 and {self.total_updates}"
            )

        if self.strategy == "fixed":
            progress = 0.0
        elif self.total_updates == 1:
            progress = 1.0
        else:
            progress = (
                (update - 1) / (self.total_updates - 1)
            )

        gamma = (
            self.gamma_start
            + progress * (self.gamma_end - self.gamma_start)
        )

        gae_lambda = (
            self.lambda_start
            + progress * (self.lambda_end - self.lambda_start)
        )

        return gamma, gae_lambda
